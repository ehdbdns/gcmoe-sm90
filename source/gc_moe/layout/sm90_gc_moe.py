"""SM90-owned workspace ABI, independent of the SM100 ring layout."""
import os
from dataclasses import dataclass, field
from functools import lru_cache
from types import MappingProxyType


def align(n, a=128):
    return (n + a - 1) // a * a


@dataclass(frozen=True)
class View:
    offset: int
    shape: tuple
    dtype: str
    nbytes: int


@dataclass(frozen=True)
class SM90GCBufferLayout:
    ranks: int
    experts: int
    capacity: int
    topk: int
    hidden: int
    intermediate: int
    shared_experts: int = 0
    # EPLB: total physical expert slots. `physical == experts` keeps the
    # upstream layout; a 10% redundancy makes it experts * 1.1 (rounded up to a
    # multiple of ranks), so each rank owns more slots than it has experts.
    physical: int = 0
    # [ECO-SM90] reserve the ECO region (quota / counts / EcoMeta / state).
    eco: bool = False
    # [ECO-TRACE] reserve the per-step timeline for ANY arm. The dispatch-side
    # rows (plan_wake / push / pull) are identical for RR and ECO, so tracing
    # RR uses the same table and the two can be compared column by column.
    trace: bool = False

    source_inbox: bool = field(init=False)
    single_push: bool = field(init=False)
    source_index_pull: bool = field(init=False)
    split_exchange: bool = field(init=False)
    stream_blocks: bool = field(init=False)
    n_layers: int = field(init=False)
    eco_per_layer: bool = field(init=False)
    eco_max_copies: int = field(init=False)
    ir_steps: int = field(init=False)
    pull_trace: bool = field(init=False)
    release_trace: bool = field(init=False)
    step_profile: bool = field(init=False)
    fine_records: int = field(init=False)
    fixed_route_order: bool = field(init=False)
    long_cost: bool = field(init=False)
    eplb_window: int = field(init=False)
    matrix: bool = field(init=False)

    def __post_init__(self):
        ranks, experts, capacity, topk = self.ranks, self.experts, self.capacity, self.topk
        hidden, intermediate, shared_experts = self.hidden, self.intermediate, self.shared_experts
        layers = int(os.environ.get('GC_MOE_LAYERS', '40'))
        if layers < 1:
            raise ValueError('GC_MOE_LAYERS must be positive')
        width = int(os.environ.get('GC_MOE_ECO_MAX_COPIES', str(ranks))) if self.eco else 0
        if self.eco and not 1 <= width <= 32:
            raise ValueError('GC_MOE_ECO_MAX_COPIES must fit the 32-entry EcoMeta row')
        object.__setattr__(self, 'n_layers', layers)
        object.__setattr__(self, 'eco_per_layer', self.eco and os.environ.get('GC_MOE_ECO_PER_LAYER', '1') == '1')
        object.__setattr__(self, 'eco_max_copies', min(width, ranks))
        def optional_capacity(switch, default, count, default_count):
            if os.environ.get(switch, default) != '1':
                return 0
            value = int(os.environ.get(count, default_count))
            if value <= 0:
                raise ValueError(f'{count} must be positive when {switch}=1')
            return value

        object.__setattr__(self, 'ir_steps', optional_capacity('GC_MOE_IR_STEP', '1', 'GC_IRSTEP_STEPS', '1024'))
        object.__setattr__(self, 'fine_records', optional_capacity('GC_MOE_FINE_TRACE', '0', 'GC_MOE_FINE_RECS', '262144'))
        object.__setattr__(self, 'pull_trace', os.environ.get('GC_MOE_PULL_TRACE') == '1')
        object.__setattr__(self, 'step_profile', os.environ.get('GC_MOE_STEP_PROFILE') == '1')
        object.__setattr__(self, 'release_trace', self.step_profile and self.pull_trace
                           and os.getenv('SM90_MEASURE_BLOCK_RELEASE') == '1')
        matrix = bool(os.environ.get('GC_MOE_MATRIX_ARM'))
        object.__setattr__(self, 'matrix', matrix)
        collect_history = (
            matrix or (os.environ.get('GC_MOE_EPLB') == '1' and os.environ.get('GC_MOE_EPLB_NO_REARRANGE') != '1'
                       and os.environ.get('GC_MOE_SLOT_PLACEMENT') != 'random'))
        window = int(os.environ.get('GC_MOE_EPLB_WINDOW', '1000')) if collect_history else 0
        if collect_history and window < 1:
            raise ValueError('invalid EPLB history length')
        object.__setattr__(self, 'eplb_window', window)
        object.__setattr__(self, 'fixed_route_order', (matrix or self.eco) and bool(os.environ.get('GC_MOE_FIXED_DIR')))
        object.__setattr__(self, 'long_cost', os.environ.get('GC_MOE_LONG_COST') == '1')
        inbox = self.eco and os.getenv('GC_MOE_SM90_SOURCE_INBOX') == '1'
        object.__setattr__(self, 'source_inbox', inbox)
        object.__setattr__(self, 'single_push', inbox or (self.eco and os.getenv('GC_MOE_SM90_SINGLE_PUSH') == '1'))
        object.__setattr__(self, 'source_index_pull', self.eco and os.getenv('GC_MOE_SM90_SOURCE_INDEX_PULL') == '1')
        object.__setattr__(self, 'split_exchange', self.eco and os.getenv('GC_MOE_SM90_SPLIT_EXCHANGE') == '1')
        views = {}
        cursor = 0
        def reserve(name, shape, dtype, itemsize):
            nonlocal cursor
            cursor = align(cursor)
            size = itemsize
            for dim in shape:
                size *= dim
            views[name] = View(cursor, shape, dtype, size)
            object.__setattr__(self, name, cursor)
            cursor += size
        physical = self.physical or experts
        if physical < experts or physical % ranks:
            raise ValueError('physical must be >= experts and divisible by ranks')
        object.__setattr__(self, 'physical', physical)
        # `local` is how many slots each rank holds; logical experts per rank is
        # still experts // ranks. They differ only under replication.
        local = physical // ranks
        rows = ranks * capacity
        # epoch, input_ready, input_tokens, task_queue, finalize_count,
        # compute_done, finalized; all epochs monotonically increase.
        reserve('header', (35,), 'int32', 4)  # words 32..34 are role-6 diagnostics
        reserve('peer_table', (ranks,), 'int64', 8)
        reserve('expert_ready', (local,), 'int32', 4)
        reserve('expert_count', (local,), 'int32', 4)
        reserve('l1_done', (local,), 'int32', 4)

        # [MB-CLAIM] One ACTIVE TICKET PER M BLOCK, not per (expert, ntile).
        #
        # `head+7` hands out monotonic tickets, so a ticket can never be
        # revisited: with one ticket per tile, a tile's 2nd/3rd block could only
        # be reached by the same producer CTA looping inside one ticket, which
        # is exactly the serialisation that collapsed the tail (measured:
        # expert 46 = 184 tokens = 3 blocks of one ntile, one CTA busy
        # 655->729 us while 51 CTAs idled).
        #
        # Publishing one ticket per (expert, ntile, mblock) lets every block be
        # claimed independently by any free CTA. Bound: at most `capacity`
        # tokens per expert, i.e. ceil(capacity/max_m) blocks; 64 covers
        # cap=2048 at block_m=64. `active_count` is compared against these
        # tickets only, so the extra rows cost nothing when unused.
        stream_blocks = self.fixed_route_order or os.environ.get('GC_MOE_SM90_STREAM_M_BLOCKS') == '1'
        object.__setattr__(self, 'stream_blocks', stream_blocks)
        # Streaming supports M >= 16. Across local slots there are at most
        # ranks * capacity * topk routes plus one partial block per slot.
        min_block_m = 16 if stream_blocks else 8
        batches = max(local * 64, (rows * topk + min_block_m - 1) // min_block_m + local)
        reserve('active_expert', (batches,), 'int32', 4)
        reserve('active_ready', (batches,), 'int32', 4)
        reserve('active_mb', (batches,), 'int32', 4)
        if stream_blocks:
            reserve('stream_l1_done', (batches,), 'int32', 4)
            reserve('stream_l1_next', (batches,), 'int32', 4)
            reserve('stream_l2_next', (batches,), 'int32', 4)
            reserve('stream_l2_ready', (batches,), 'int32', 4)
        reserve('compute_ready', (ranks,), 'int32', 4)
        reserve('combine_block_done', (1024,), 'int32', 4)
        if self.fixed_route_order:
            reserve('replay_stream_mode', (1,), 'int32', 4)
            reserve('route_ordinal', (capacity, topk), 'int32', 4)
            reserve('used_route_ordinal', (capacity, topk), 'int32', 4)
        reserve('send_count', (physical,), 'int32', 4)
        reserve('source_index', (local, ranks, capacity), 'int32', 4)
        # [T17-PERLAYER] slot_logical[layer][g]: the logical expert served by
        # physical slot g IN THAT LAYER. DeepInfer plans every layer
        # separately, so this table cannot be shared across layers -- doing so
        # measured IR 1.2803 vs 1.1324 for the per-layer form.
        reserve('slot_logical', (layers, physical), 'int32', 4)
        # A logical expert may own every redundant slot, including multiple copies on one rank.
        reserve('expert_slots', (layers, experts, physical - experts + 1), 'int32', 4)
        if self.eplb_window:
            reserve('placement_history', (window * layers, local + 1), 'int32', 4)
        if self.matrix and self.eco:
            reserve('auction_counts', (1000 * layers, 7), 'int32', 4)
        # ── ECO (EPLB placement + online plan) ───────────────────────────
        # Only allocated when `eco` is on, so the plain path keeps its exact
        # byte layout. Sizes use PHYSICAL slots (physical = ranks*local).
        # [ECO-SM90]
        if self.eco:
            # [ECO-DBUF] two tokens only: [0]=counts, [1]=plan. The old
            # [2]="counts consumed" is unused since `eco_counts` became
            # double buffered (SM100's `gen = epoch & 1`).
            reserve('eco_ready', (4,), 'int32', 4)          # counts, quota, consumed, plan resident
            # [ECO-SM90] Dedicated per-LOGICAL-expert histogram for the plan.
            # SM100 keeps this in shared memory (`route_count[kE]`); on SM90 the
            # plan is a separate CTA, so it lives in gmem. Deliberately NOT
            # `send_count`: that one is the push's slot allocator and has to
            # march from 0, so sharing an array for both jobs needs an extra
            # clear + grid barrier per call -- that variant deadlocked.
            if self.single_push:
                if self.source_index_pull or self.split_exchange or not stream_blocks:
                    raise ValueError('single push requires streamed blocks and baseline replica transport')
                # Alternate epochs so early peers never race a destination reset.
                slot_words = ranks + 2 + (rows + min_block_m - 1) // min_block_m
                reserve('push_state', (2, local, slot_words), 'int32', 4)
                reserve('push_ticket', (1,), 'int32', 4)
                if self.source_inbox:
                    # Each route goes to at most one physical replica. Share payload
                    # capacity across experts, with a private region per source rank.
                    inbox_rows = rows * topk
                    reserve('inbox_x', (inbox_rows, hidden), 'uint8', 1)
                    reserve('inbox_sf', (inbox_rows, hidden // 128), 'float32', 4)
                    reserve('inbox_weight', (inbox_rows,), 'float32', 4)
                    reserve('inbox_route', (local, ranks, capacity), 'int32', 4)
                else:
                    reserve('push_base', (experts,), 'int32', 4)
                reserve('batch_valid', (batches,), 'int32', 4)
                if not self.fixed_route_order:
                    reserve('push_ordinal', (capacity, topk), 'int32', 4)
            if self.source_index_pull or self.single_push:
                if self.split_exchange:
                    raise ValueError('select one replica index transport')
                reserve('source_routes', (experts, capacity), 'int32', 4)
            if self.split_exchange and not self.fixed_route_order:
                reserve('replica_ordinal', (capacity, topk), 'int32', 4)
            reserve('eco_hist', (experts,), 'int32', 4)
            reserve('eco_quota', (physical,), 'int32', 4)
            reserve('eco_quota_start', (physical,), 'int32', 4)
            # [ECO-DBUF] TWO copies of the counts panel, indexed by
            # `epoch & 1` (SM100 eco_unit_kernel.cu L124 `gen = epoch & 1`).
            # Double buffering is what lets a plan write this call's panel
            # while the previous call's plan may still be reading the other
            # half, so no "previous call finished" token is needed. A single
            # buffer forced such a token and deadlocked on the 40-layer
            # sequence (each plan would have to wait for its own predecessor
            # to be completely done, in the same CTA).
            reserve('eco_counts', (2 * ranks * physical,), 'int32', 4)
            # [T17-PERLAYER] One EcoMeta row per MoE layer: DeepInfer plans
            # every layer separately (its map is [layers, physical]), so the
            # copy counts and the slot->expert table differ per layer.
            # `layers` comes from the compile-time geometry below.
            # The layer count comes from the same env the kernel reads
            # (`GC_MOE_LAYERS`, default 40) so the two cannot disagree.
            reserve('eco_meta',
                    (layers * (16 + experts * 16 + physical + ranks),),
                    'int32', 4)
            # [T17-PERLAYER] The warm start (`_eco_do_warm`) reads the PREVIOUS
            # call's per-slot quotas and per-expert counts and reuses as much of
            # them as it can. Under per-layer placement that previous call ran a
            # DIFFERENT layer, i.e. a different slot->expert map: the inherited
            # "units" would describe slots that hold other experts, so the warm
            # plan would hand a hot expert's tokens to a slot that cannot compute
            # them (or leave real copies with no quota). One row per layer keeps
            # the warm start meaningful -- call it on layer L and it sees layer
            # L's own previous plan, one decode step earlier.
            reserve('eco_state', (layers * 2 * physical,), 'int32', 4)
            # [ECO-MEAS] ECO timeline trace: 32 slots x 8B of globaltimer ns.
            #   [0..5]  stage timestamps (store)
            #   [24]    role-1 start of this call (graph-replay anchor)
            #   [27]    accumulated "replica slot waited for the plan" ns
            #   [30]    warm (1) / cold (0) this call
            #   [31]    auction rounds this call
            # Only reserved when ECO is on, so the plain path is byte-identical.
            reserve('eco_trace', (40,), 'int64', 8)

        # [ECO-TRACE] Per-step, per-rank RAW timeline, available to EVERY arm.
        # The dispatch-side rows (plan_wake / push / pull) are identical for RR
        # and ECO, so tracing RR uses this same table and the two can be
        # compared column by column. Deliberately NOT reduced: every call keeps
        # its own row, every rank its own copy, and the host averages over
        # steps afterwards. Microseconds, not nanoseconds (int32 overflows ~36min).
        if self.trace:
            reserve('eco_tl', (256 * 21,), 'int32', 4)
            reserve('eco_tl_ctr', (16,), 'int32', 4)
            reserve('eco_tl_step', (1,), 'int32', 4)
        elif self.pull_trace:
            # [PHASES] The compute role needs a NON-STALE name for the call it
            # is serving: `header[0]` is captured once at kernel entry by a
            # resident CTA, so on every later call it still holds an old epoch
            # and its "first task" stamp lands in another call's row
            # (measured: 18/256 rows off by ~2^31 us). Dispatch republishes
            # `eco_tl_step` at the top of every call, so reserving it for the
            # pull trace too gives the stamp an exact row without turning on
            # the whole ECO timeline.
            reserve('eco_tl_step', (1,), 'int32', 4)

        # [IR-STEP] [layer][step] per-rank token counts, filled by the
        # dispatch role ITSELF while the measured window runs.
        #
        # Its own switch, NOT tied to `trace`: the ECO timeline is a debugging
        # aid, whereas this table IS the measurement. Binding them together
        # silently produced `pairs=0` on a run that forgot GC_MOE_ECO_TRACE=1.
        #
        # Why in-kernel: decode replays a captured CUDA graph, so no Python
        # hook can sample per step, and an out-of-band RPC per step costs
        # ~60 ms (it would either pollute ms/step or force a SECOND window
        # that burns `max_tokens` budget -- at BS=256 that second window is
        # what made the run not fit in KV at all).
        #
        # Indexing: the dispatch epoch advances by ONE PER LAYER, verified on
        # consecutive calls. Epochs are relative to the marked window.
        # Each slot contributes its own row and M-block counts.
        if self.ir_steps:
            steps = self.ir_steps
            reserve('ir_step', (layers * steps,), 'int32', 4)
            reserve('ir_step_epoch', (layers * steps,), 'int32', 4)
            reserve('ir_step_mark', (2,), 'int32', 4)
            # [IR-STEP] Two companion tables, same (layer, step) indexing:
            #   ir_step_mb[layer][step] += ceil(owned / block_m)   per unit
            #   ir_step_ae[layer][step] += 1 if owned > 0           per unit
            # They answer "how many M blocks / how many ACTIVE slots does this
            # rank have in this layer of this step", which the token table
            # cannot: tokens per slot are below block_m, so a rank's block count
            # is NOT a function of its token count alone.
            reserve('ir_step_mb', (layers * steps,), 'int32', 4)
            reserve('ir_step_ae', (layers * steps,), 'int32', 4)

        # [PULL-TRACE] Pull-phase timing for EVERY arm, including the plain
        # (no replicas, no ECO) path -- the ECO trace above only exists when
        # `eco` is on, so the vanilla arm had no pull measurement at all.
        # Rows are keyed by the dispatch epoch modulo 256, exactly like eco_tl:
        #   [0] epoch tag   [1] wake    [2] push start
        #   [3] push end    [4] pull start  [5] pull end (all local slots done)
        #   [6] count of single-copy slots  [7] count of replica slots
        #   [8] first role-3 CTA in combine [9] combine done (epoch bump)
        # 256 x 10 x 4 B = 10 KB per rank, and only when the env switch is on.
        # ★ [8]/[9] exist because `eco_trace` (slots 14/15) is only allocated
        # in an ECO layout, so the vanilla arm had NO combine measurement.
        if self.pull_trace:
            # [RANK-TS] 14 slots now (was 10). The 4 extra ones hold the
            # per-rank MoE cost stamps the earlier measurement could not
            # provide:
            #   [10] dispatch kernel entry     [11] compute kernel exit
            #   [12] combine kernel entry      [13] combine kernel exit
            #   [15] first compute CTA got a task (min)
            #   [16]/[17] reserved for a shared-expert span.  NOTE: on this
            #         model the shared expert is NOT a gc_moe role at all --
            #         vLLM runs it as `vllm::moe_forward_shared` (nvjet + 2
            #         marlin GEMMs) on its own stream, so these two slots are
            #         never written and the span is read from the trace.
            #   [24..31] ECO plan's exported T[r] (one per rank).  This USED
            #         to be 14+r, which overwrote slots 15/16/17 and made the
            #         "first GEMM task" stamp read back as a plan unit count
            #         (measured: 33753 us instead of ~a few hundred).
            # plus two atomic counters used to elect "the last CTA to leave"
            # (rows 254 and 255, column 10/11), so no header slot is needed.
            reserve('pull_tl', (256 * max(32, 24 + ranks),), 'int32', 4)

        if self.long_cost:
            # One row for every layer in 1000 decode steps; reset outside the timed window.
            reserve('long_clocks', (1000 * layers, 6), 'int64', 8)
        if self.step_profile:
            # Match pull_tl's epoch ring. Every local slot is written once,
            # including empty slots, by its dispatch CTA.
            reserve('profile_slot_counts', (256, local), 'int32', 4)

        # [ECO-FINE] Fine-grained single-call event log. One flat array of
        # fixed 8-word records, appended through ONE global counter, so the
        # emission order IS the order (no ring, no per-type interleaving).
        #   [0] t_us   [1] type   [2] id   [3..6] args   [7] epoch
        # Types: 1 slot_begin 2 slot_end 3 slot_wait_plan 4 claim_spin
        #        5 claim_ok   6 task_begin 7 task_end
        # Only calls matching GC_MOE_FINE_EPOCH / GC_MOE_FINE_MOD record, and
        # the epoch is stored in every record, so several calls can be captured
        # without any reset (a reset would race resident CTAs).
        # 65536 x 8 x 4 B = 2 MB per rank, only when the switch is on.
        if self.fine_records:
            # [ABSTRACE] 10 words per record (was 8): the absolute-time figure
            # needs ntile and the M-block index next to expert/start/valid, so
            # every GEMM slice can be labelled "expert E, ntile T, M-block b".
            # Record: [0] t_us [1] type [2] id [3] a0 [4] a1 [5] a2
            #         [6] a3 [7] a4 [8] epoch [9] (reserved)
            reserve('fine_tl', (self.fine_records * 10,), 'int32', 4)
            reserve('fine_ctr', (1,), 'int32', 4)
            # [ECO-FINE] Per-CTA spin state (0 = busy, 1 = already logged a
            # spinstreak). Edge detection needs the PREVIOUS state of a CTA,
            # which no single record can carry. Sized for the largest role-2
            # grid in use (72 CTAs) with headroom.
            reserve('fine_spin', (256,), 'int32', 4)
            # [ECO-FINE] active_slot -> the layer epoch that dispatch published
            # for its expert. A resident consumer CTA cannot use `header[0]`
            # (stale) nor `eco_tl_step` (names the NEWEST call, not this task).
            # [MB-CLAIM] indexed by ACTIVE TICKET, and tickets are now
            # per-M-block, so this must match `active_expert`'s new size.
            reserve('fine_lay', (batches,), 'int32', 4)
            # [ECO-FINE] expert -> layer epoch (what `_consume_tasks` can see,
            # since its task record only carries the expert id).
            reserve('fine_epx', (experts,), 'int32', 4)
            # [ECO-FINE-FLOOR] First epoch of the MEASURED window, written by
            # the host in `irstep_begin()`. Without it the 2 MB log is consumed
            # by prefill: `fine_ctr` is monotonic and stops at 65536 records, so
            # with GC_MOE_FINE_MOD=0 the buffer was already full after the first
            # four prefill layers and the whole 1000-step measurement produced
            # ZERO records (verified: an "M blocks = 12996" plot for BS=32 whose
            # real value is ~128). Records below the floor are dropped in the
            # kernel, so the log holds decode calls only.
            reserve('fine_floor', (1,), 'int32', 4)
            # Low 32 bits of globaltimer in µs, plus a flag for a host-seeded
            # origin. Fine timestamps are offsets modulo 2^32 from this origin.
            reserve('fine_t0', (2,), 'int32', 4)

        reserve('x', (capacity, hidden), 'float8_e4m3fn', 1)
        reserve('x_sf', (capacity, hidden // 128), 'float32', 4)
        reserve('topk_idx', (capacity, topk), 'int64', 8)
        reserve('topk_weights', (capacity, topk), 'float32', 4)
        # [ECO-BLKRDY] Per-slot count of M blocks whose data is already in
        # `l1_acts`. Pull increments it after each block's last row lands, so
        # the producer can publish that block's GEMM task immediately instead
        # of waiting for the whole slot -- SM100 does the same with
        # `l1_full_count[ring_block]` (sm100_fp8_fp4_gc_moe.py L1495).
        reserve('l1_block_ready', (local,), 'int32', 4)
        reserve('route_meta', (local, rows, 3), 'int32', 4)
        reserve('route_weight', (local, rows), 'float32', 4)
        reserve('l1_acts', (local, rows, hidden), 'float8_e4m3fn', 1)
        reserve('l1_acts_sf', (local, hidden // 128, rows), 'float32', 4)
        reserve('l2_acts', (local, rows, intermediate), 'float8_e4m3fn', 1)
        reserve('l2_acts_sf', (local, intermediate // 128, rows), 'float32', 4)
        reserve('combine', (capacity, topk, hidden), 'bfloat16', 2)
        if shared_experts:
            reserve('shared_l1_acts', (capacity, hidden), 'float8_e4m3fn', 1)
            reserve('shared_l1_acts_sf', (capacity, hidden // 32), 'float32', 4)
            reserve('shared_l2_acts', (capacity, intermediate * shared_experts), 'float8_e4m3fn', 1)
            reserve('shared_l2_acts_sf', (capacity, intermediate * shared_experts // 32), 'float32', 4)
            reserve('shared_l1_done', ((capacity + 7) // 8,), 'int32', 4)
            reserve('shared_y', (capacity, hidden), 'bfloat16', 2)
        object.__setattr__(self, 'views', MappingProxyType(views))
        object.__setattr__(self, 'nbytes', align(cursor))


@dataclass(frozen=True)
class SharedStorageLayout:
    routed_x_offset: int
    sf_stage_offset: int
    stage_bytes: int
    stages: int
    full_offset: int
    empty_offset: int
    task_full_offset: int
    task_empty_offset: int
    task_record_offset: int
    command_offset: int


@lru_cache(maxsize=32)
def get_role_shared_layout(tile_m: int, warp_specialized: bool,
                           routed_n_factor: int) -> SharedStorageLayout:
    routed_x = 17408 * routed_n_factor
    stage_m = 8 if routed_n_factor == 2 else (tile_m if warp_specialized else 64)
    sf_stage = routed_x + stage_m * 128
    stage = align(sf_stage + stage_m * 4, 1024)
    stages = 5 if tile_m == 64 else (3 if tile_m == 32 else 4)
    full = stages * stage
    empty = full + stages * 8
    task_full = empty + stages * 8
    return SharedStorageLayout(routed_x, sf_stage, stage, stages, full, empty,
                               task_full, task_full + 16, task_full + 64,
                               task_full + 160)
