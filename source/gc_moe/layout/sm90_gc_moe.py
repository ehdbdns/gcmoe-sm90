"""SM90-owned workspace ABI, independent of the SM100 ring layout."""
import os
from dataclasses import dataclass
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

    def __post_init__(self):
        ranks, experts, capacity, topk = self.ranks, self.experts, self.capacity, self.topk
        hidden, intermediate, shared_experts = self.hidden, self.intermediate, self.shared_experts
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
        reserve('header', (32,), 'int32', 4)
        reserve('peer_table', (ranks,), 'int64', 8)
        reserve('expert_ready', (local,), 'int32', 4)
        reserve('expert_count', (local,), 'int32', 4)
        reserve('l1_done', (local,), 'int32', 4)
        reserve('active_expert', (local,), 'int32', 4)
        reserve('active_ready', (local,), 'int32', 4)
        reserve('compute_ready', (ranks,), 'int32', 4)
        reserve('combine_block_done', (1024,), 'int32', 4)
        reserve('send_count', (physical,), 'int32', 4)
        reserve('source_index', (local, ranks, capacity), 'int32', 4)
        # slot_logical[g]: the logical expert served by physical slot g.
        # Read-only; the host fills it once from the EPLB plan.
        reserve('slot_logical', (physical,), 'int32', 4)
        # expert_slots[e*max_rep + k]: physical slot ids holding a copy of
        # logical expert e (-1 = unused). The router picks one per token.
        reserve('expert_slots', (experts, 4), 'int32', 4)
        # ── ECO (EPLB placement + online plan) ───────────────────────────
        # Only allocated when `eco` is on, so the plain path keeps its exact
        # byte layout. Sizes use PHYSICAL slots (physical = ranks*local).
        # [ECO-SM90]
        if self.eco:
            # [ECO-DBUF] two tokens only: [0]=counts, [1]=plan. The old
            # [2]="counts consumed" is unused since `eco_counts` became
            # double buffered (SM100's `gen = epoch & 1`).
            reserve('eco_ready', (3,), 'int32', 4)          # [0]=counts [1]=plan
            # [ECO-SM90] Dedicated per-LOGICAL-expert histogram for the plan.
            # SM100 keeps this in shared memory (`route_count[kE]`); on SM90 the
            # plan is a separate CTA, so it lives in gmem. Deliberately NOT
            # `send_count`: that one is the push's slot allocator and has to
            # march from 0, so sharing an array for both jobs needs an extra
            # clear + grid barrier per call -- that variant deadlocked.
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
            reserve('eco_meta', (16 + experts * 16 + physical + ranks,), 'int32', 4)
            reserve('eco_state', (2 * physical,), 'int32', 4)
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
        # real traces (9305,9306,... contiguous across 8 ranks, and
        # epoch % 40 cycles 0..39 exactly). So
        #     layer = epoch %  layers
        #     step  = (epoch - epoch0) // layers        (epoch0 = window start)
        # The rows are OVERWRITTEN, not accumulated: one (layer,step) cell is
        # written exactly once per forward.
        #
        # 40 layers x 1024 steps x 4 B = 160 KB per rank.
        if os.environ.get('GC_MOE_IR_STEP', '1') == '1':
            reserve('ir_step', (40 * 1024,), 'int32', 4)
            reserve('ir_step_mark', (2,), 'int32', 4)
            # [IR-STEP] Two companion tables, same (layer, step) indexing:
            #   ir_step_mb[layer][step] += ceil(owned / block_m)   per unit
            #   ir_step_ae[layer][step] += 1 if owned > 0           per unit
            # They answer "how many M blocks / how many ACTIVE slots does this
            # rank have in this layer of this step", which the token table
            # cannot: tokens per slot are below block_m, so a rank's block count
            # is NOT a function of its token count alone.
            reserve('ir_step_mb', (40 * 1024,), 'int32', 4)
            reserve('ir_step_ae', (40 * 1024,), 'int32', 4)

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
        if os.environ.get('GC_MOE_PULL_TRACE') == '1':
            reserve('pull_tl', (256 * 10,), 'int32', 4)

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
        if os.environ.get('GC_MOE_FINE_TRACE') == '1':
            reserve('fine_tl', (65536 * 8,), 'int32', 4)
            reserve('fine_ctr', (1,), 'int32', 4)
            # [ECO-FINE] Per-CTA spin state (0 = busy, 1 = already logged a
            # spinstreak). Edge detection needs the PREVIOUS state of a CTA,
            # which no single record can carry. Sized for the largest role-2
            # grid in use (72 CTAs) with headroom.
            reserve('fine_spin', (256,), 'int32', 4)
            # [ECO-FINE] active_slot -> the layer epoch that dispatch published
            # for its expert. A resident consumer CTA cannot use `header[0]`
            # (stale) nor `eco_tl_step` (names the NEWEST call, not this task).
            reserve('fine_lay', (physical,), 'int32', 4)
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
