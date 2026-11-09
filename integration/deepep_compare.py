"""GC MoE as a vLLM MoE backend.

Two integration points are registered:

1. ``Mxfp4MoeBackend.GC_MOE`` -- an experts class for the MXFP4 checkpoints
   DeepSeek-V4.1 ships, selected with ``--kernel-config.moe-backend gc_moe``.
2. A prepare/finalize pair that only stages data: GC MoE does its own
   all-to-all, so vLLM must not dispatch anything itself.

Weight layout. DeepSeek-V4.1 checkpoints store routed experts as packed
MXFP4: ``[E, N, K/2]`` int8 payload + ``[E, N, K/32]`` e8m0 scales. That is
exactly what ``gc_moe.transform_weights_for_gc_moe_sm90`` consumes, so the
conversion is a permutation, not a requantization.
"""
from __future__ import annotations

import os

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.moe_output import UnfinalizedMoEOutput
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kMxfp4Dynamic,
    kMxfp4Static,
    kMxfp8Dynamic,
)

logger = init_logger(__name__)

_GLOBAL_TOPPING = []
# GCMoEExperts is not an nn.Module: vLLM creates one per layer inside the
# Mxfp4MoEMethod. The model-level hook only walks model.modules(), so keep a
# explicit registry of the live expert objects for it to find.
_EXPERT_REGISTRY: list = []
_SHARED_BUFFER = {'buf': None, 'key': None}
# EPLB layout is identical for every layer in this model (same expert geometry
# and the same uniform demand), so it is built once and shared.
_SHARED_EPLB = {'layout': None, 'key': None}
_PROF = {'ms': [], 'tokens': []}
_TRACE = {'n': [], 'capturing': False}
# [SME-IR] per-engine-step snapshot of `expert_count` (tokens per physical
# slot on this rank), used to compute the imbalance ratio
#     IR = max_rank(tokens) / mean_rank(tokens).
# [SME-MEAS] per-step per-layer/per-rank load snapshots
# [SME-MEAS] id(w1 payload) -> layer index, shared by all module instances
_LAYER_OF = {}
_LAYER_REFS = []
# Enabled from the environment at import time: the counter table has to exist
# before vLLM captures its CUDA graphs, which happens during LLM() construction
# -- earlier than any RPC the driver could send.
# [METRIC-FIX] per-call occupancy snapshots (header[7] and expert_count are
# reset/overwritten by every call, so they must be read right after a launch).
_ACTIVE_HDR = {'last': None, 'samples': [], 'counts': None}
_MEAS = {'on': False, 'steps_done': 0, 'has_stats': False,
         'table': None, 'vanilla_idx': {},
         'measure': os.environ.get('GC_MEASURE_LOAD', '1') == '1'}
_IR = {'on': False, 'steps': [], 'ir': -1.0}
# [RR-FIX] `eplb_logic` sits next to this module (integration/), which is
# already on sys.path when the driver imports us.
try:
    from eplb_logic import rr_pick
except Exception:  # pragma: no cover - only if integration/ is absent
    rr_pick = None
    rr_scratch_for = None


def _eplb_enabled() -> bool:
    """Opt-in switch for EPLB placement + real-time token plan."""
    return os.environ.get('GC_MOE_EPLB') == '1'


def _eplb_redundancy() -> float:
    return float(os.environ.get('GC_MOE_EPLB_REDUNDANCY', '0.10'))


def _slot_placement() -> str:
    """[SME-EPLB] Which static slot->expert map to install, EPLB or not.

    ``identity`` (default) keeps upstream behaviour: without EPLB logical id ==
    physical id, with EPLB the DeepInfer `initial_layout` is used.

    ``random`` scatters the same multiset of experts over the same slots with a
    fixed seed. That is the CONTROL arm: routing, slot count and the RR key all
    stay identical, so any difference in rank balance comes purely from WHICH
    experts share a rank.

    ``eplb`` forces the DeepInfer initial layout even when GC_MOE_EPLB is off.
    """
    v = os.environ.get('GC_MOE_SLOT_PLACEMENT', '').strip().lower()
    if v in ('identity', 'random', 'eplb'):
        return v
    return 'random' if _eplb_random_layout() else 'identity'


def _eplb_random_layout() -> bool:
    """[SME-EPLB] Use a RANDOM placement instead of the identity layout.

    ``GC_MOE_EPLB_RANDOM=1`` permutes the slot -> logical-expert map with a
    fixed seed, so every rank still holds `physical // ranks` slots but the
    experts they serve are scattered instead of contiguous. This is the
    control arm for "does the PLACEMENT matter at all?" -- with the same
    slot count, the same routing and the same RR key, the only difference to
    the identity arm is which experts share a rank.

    The seed is pinned so the arm is reproducible and every process builds
    the same layout (the placement must be identical on all ranks).
    """
    return os.environ.get('GC_MOE_EPLB_RANDOM') == '1'


def _eplb_random_seed() -> int:
    return int(os.environ.get('GC_MOE_EPLB_RANDOM_SEED', '0'))


# [ECO-WARP] Per-rank slot permutation produced by the single-copy-first
# reorder. Kept so the host can map a permuted local slot back to the original
# one when it has to (weights are moved with the placement, so normally nothing
# needs it).
_ECO_SLOT_ORDER = {'order': None}


def _eplb_extra_per_rank() -> int:
    """[SME-EPLB] Redundant slots PER RANK; overrides the fraction knob.

    ``GC_MOE_EPLB_EXTRA_PER_RANK=4`` means every rank holds
    ``experts // ranks + 4`` slots (EP8, 384 experts -> 52 slots/rank, P=416).
    The fraction knob cannot express that exactly: it rounds to a multiple of
    `ranks`, so 0.0833 also lands on 32 extras but is fragile to expert-count
    changes. -1 (default) keeps the fraction behaviour.
    """
    return int(os.environ.get('GC_MOE_EPLB_EXTRA_PER_RANK', '-1'))


def _eco_enabled() -> bool:
    """[ECO-SM90] Online split plan (role 5) instead of host round-robin.

    Requires EPLB placement: ECO splits each logical expert's route stream
    between its copies, so it only has anything to split when copies exist.
    """
    return os.environ.get('GC_MOE_ECO') == '1'


def _eco_weight(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _max_copies_for(experts: int, physical: int) -> int:
    """# [SME-EPLB] Largest copy count a layout with `physical` slots can give.

    The buffer's replica capacity and the layout's max_copies must agree, so
    derive the capacity from the same placement the buffer is sized for.
    """
    try:
        from gc_moe.layout.eplb_placement import build_eplb_layout
        lay = build_eplb_layout([1.0] * experts, experts, 8, _eplb_redundancy())
        return max(max(lay.logcnt), 2)
    except Exception:  # noqa: BLE001
        return 4


def _eplb_extra_floor(experts: int) -> int:
    """# [SME-EPLB] Redundant experts requested by GC_MOE_EPLB_REDUNDANCY.

    The B arm's config expresses redundancy as a minimum *count* of extra
    experts (DEEPINFER_OVERRIDE_NUM_EXTRA_EXPERTS); we keep the fraction knob
    but convert it the same way.
    """
    per_rank = _eplb_extra_per_rank()
    if per_rank >= 0:
        return max(1, per_rank * 8)
    return max(1, int(round(experts * _eplb_redundancy())))


def _physical_slots(experts: int) -> int:
    """Physical slot count: experts + a ~redundancy share, divisible by ranks.

    EPLB needs `physical > experts` so some experts get a second copy; the
    kernel walks exactly this many slots per rank. `GC_MOE_EPLB_EXTRA_PER_RANK`
    takes precedence and expresses the share directly per rank.
    """
    ranks = 8
    per_rank = _eplb_extra_per_rank()
    if per_rank >= 0:
        return experts + per_rank * ranks
    want = int(experts * _eplb_redundancy())
    return experts + want + (-(experts + want)) % ranks


def _shared_eplb_layout(gc, experts: int, ranks: int, redundancy: float):
    """The one static EPLB layout this model uses (built on first use).

    Demand is a uniform prior: the layout has to exist before any traffic is
    measured, and a uniform prior still exercises replication (the packer
    spreads physical experts evenly, and a 25% redundancy means some experts
    get copies). Swap in real per-expert counts to drive placement by load.
    """
    key = (experts, ranks, redundancy)
    if _SHARED_EPLB['layout'] is None or _SHARED_EPLB['key'] != key:
        demand = torch.ones(experts, dtype=torch.float32)
        _SHARED_EPLB['layout'] = gc.build_eplb_layout(demand, experts, ranks,
                                                     redundancy)
        _SHARED_EPLB['key'] = key
    return _SHARED_EPLB['layout']


def _capturing_enabled() -> bool:
    """True while a CUDA graph is being captured.

    Checks the CUDA stream first (authoritative for the current thread) and
    falls back to vLLM's global flag. Order matters: the global flag is also
    set during the *warmup* pass that vLLM runs before capture, so trusting it
    alone hides every eager call.
    """
    try:
        # Authoritative for this thread's stream: True only inside a capture.
        return bool(torch.cuda.is_current_stream_capturing())
    except Exception:  # noqa: BLE001
        pass
    # Fallback only when the stream query is unavailable. NOTE: vLLM's global
    # `monitor.cudagraph_capturing_enabled` defaults to True and is therefore
    # useless as a "we are capturing right now" test -- trusting it made every
    # eager call look like a capture.
    try:
        from vllm.compilation import monitor

        return bool(monitor.cudagraph_capturing_enabled)
    except Exception:  # noqa: BLE001
        return False


def _install_ptxas_override() -> None:
    """# [ENV] Point gc_moe's PTX assembler at the CUDA 13.4 ptxas.

    gc_moe/runtime/jit.py hardcodes /usr/local/cuda/bin/ptxas (13.2, PTX<=9.2),
    but CuTe DSL 4.8.1 emits PTX 9.4, so assembly fails with
    "Unsupported .version 9.4". The env ships ptxas 13.4 under nvidia/cu13.
    """
    try:
        import sys
        import os
        here = os.path.dirname(os.path.abspath(__file__))
        if here not in sys.path:
            sys.path.insert(0, here)
        import gc_ptxas_override  # noqa: F401
    except Exception:  # noqa: BLE001
        pass


def _import_gc_moe():
    """Import gc_moe with the PTX assembler override already applied."""
    try:
        import gc_ptxas_override  # noqa: F401
    except ImportError:
        pass
    import gc_moe

    return gc_moe


def _token_fp8_quant(x: torch.Tensor):
    """FP8 per-token K128-block quant with FP32 scales.

    ``deep_gemm.utils`` is GC MoE's shim (small math helpers); it may live
    outside the worker's import path, so fall back to an equivalent local
    implementation.
    """
    # Use GC MoE's own fused quantizer. A hand-rolled torch version costs ~8
    # elementwise kernels per layer (~180 us at batch 1), which dwarfs the
    # 103 us the MoE operator itself takes.
    try:
        from .fp8_quant_ref import per_token_cast_to_fp8 as _cast

        return _cast(x, use_ue8m0=False, gran_k=128)
    except ImportError:
        pass
    n, k = x.shape
    xf = x.float().reshape(n, k // 128, 128)
    amax = xf.abs().amax(dim=-1, keepdim=True).clamp_min(1e-10)
    sf = (amax / 448.0).to(torch.float32)      # e4m3 max
    q = (xf / sf).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return q.reshape(n, k), sf.reshape(n, k // 128)


class GCMoEPrepareFinalize(mk.FusedMoEPrepareAndFinalizeModular):
    """Stage tokens into GC MoE's symmetric buffer; no all-to-all here.

    GC MoE dispatches and combines inside its own graph, so vLLM's prepare step
    only has to (a) FP8-quantize the activations in the K128 block layout GC
    MoE expects and (b) hand the routing table through unchanged.
    """

    def __init__(self, max_tokens_per_rank: int, num_dispatchers: int):
        super().__init__()
        self.max_tokens_per_rank_ = max_tokens_per_rank
        self.num_dispatchers_ = num_dispatchers

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        # FusedMoEKernelModularImpl reads this off the instance.
        return mk.FusedMoEActivationFormat.Standard

    def max_num_tokens_per_rank(self) -> int | None:
        return None

    def topk_indices_dtype(self) -> torch.dtype | None:
        return torch.int64

    def num_dispatchers(self) -> int:
        return self.num_dispatchers_

    def output_is_reduced(self) -> bool:
        # GC MoE applies the router weights and sums the top-k contributions.
        return True

    def supports_async(self) -> bool:
        return False

    def prepare(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool = False,
    ) -> mk.PrepareResultType:
        if apply_router_weight_on_input:
            raise NotImplementedError(
                'GC MoE applies router weights after GEMM2; '
                'apply_router_weight_on_input is not supported.'
            )
        return a1, None, None, topk_ids.to(torch.int64), topk_weights

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl=None,
    ) -> None:
        # fused_expert_output already holds the reduced BF16 result.
        if output is not fused_expert_output:
            output.copy_(fused_expert_output)


class GCMoEExperts(mk.FusedMoEExpertsModular):
    """Runs ``gc_moe.sm90_fp8_fp4_gc_moe`` as the whole MoE operator."""

    # GC MoE is SM90-only and takes packed MXFP4 weights with FP8 activations.
    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        max_num_tokens: int | None = None,
        num_dispatchers: int | None = None,
    ):
        super().__init__(moe_config, quant_config, max_num_tokens, num_dispatchers)
        self._gc = None
        self._buffer = None
        self._buffer_key = None
        self._packed = {}  # id(layer) -> (w1, w2)
        self._last_packed = None
        self._warmed = False
        self._fixed_bound = None
        self._capture_sizes = None
        # EPLB + real-time plan (opt-in via GC_MOE_EPLB=1). Placement comes
        # from the B-arm EPLB algorithm; the plan only splits token dispatch.
        self._eplb = None
        # (layer, packed) stashed by the per-layer hook, resolved once all ranks
        # finish loading (see _install_eplb_all).
        self._pending_eplb = []
        self._eplb_done = False
        _EXPERT_REGISTRY.append(self)

    # ---- capability reporting -------------------------------------------
    @staticmethod
    def _supports_current_device() -> bool:
        from vllm.platforms import current_platform

        return current_platform.is_cuda() and current_platform.get_device_capability() == (9, 0)

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(weight_key, activation_key) -> bool:
        # Packed MXFP4 weights (whatever block layout the solver picked) with
        # activations GC MoE quantizes itself, so the activation key is either
        # BF16 (None) or one of the MX/FP8 forms the checker may report.
        if weight_key is not None and weight_key.dtype != torch.uint8:
            return False
        if activation_key is None or activation_key in (kMxfp4Dynamic, kMxfp8Dynamic):
            return True
        return activation_key.dtype == torch.bfloat16

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation == MoEActivation.SILU

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        return moe_parallel_config.use_ep

    @staticmethod
    def _supports_routing_method(routing_method, weight_key, activation_key) -> bool:
        # GC MoE consumes a finished top-k table, so any router feeding it
        # (DeepSeek sqrt-softplus + noaux_tc included) is fine.
        return True

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        # The solver calls this on the class, not an instance.
        return mk.FusedMoEActivationFormat.Standard

    def workspace_shapes(self, M, N, K, topk, global_num_experts, local_num_experts,
                         expert_tokens_meta, activation):
        # GC MoE allocates all of its scratch space in its own symmetric buffer,
        # so the two intermediate workspaces stay empty; only the fused output
        # has to exist, because vLLM copies it into the layer's result.
        return (0,), (0,), (M, K)

    # ---- weights ---------------------------------------------------------
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self._gc = _import_gc_moe()
        w13 = layer.w13_weight
        w2 = layer.w2_weight
        s13 = layer.w13_weight_scale
        s2 = layer.w2_weight_scale
        for name, t in (('w13', w13), ('w2', w2), ('s13', s13), ('s2', s2)):
            if t is None:
                raise RuntimeError(f'GC MoE needs {name} in the checkpoint')
        import time as _t
        _s = _t.time()
        packed = self._gc.transform_weights_for_gc_moe_sm90(
            (w13.data, s13.data), (w2.data, s2.data))
        print(f'[eplb-dbg] transform {_t.time()-_s:.2f}s', flush=True)
        self._packed[id(w13)] = packed
        self._packed[id(layer)] = packed
        self._last_packed = packed
        logger.info_once(
            'GC MoE SM90 weights packed: W1 %s / W2 %s',
            tuple(packed[0][0].shape), tuple(packed[1][0].shape))

        # The packed tiles have exactly the checkpoint's byte count, so keeping
        # both copies would double the expert footprint (70 GiB vs 36 GiB).
        # Release the originals: every forward goes through the packed form.
        def _stub_like(t):
            # Keep the original shape/stride contract (vLLM reads shapes to
            # size workspaces) but drop the storage: a zero-stride view of a
            # single byte per element costs nothing and is never read.
            return torch.nn.Parameter(
                torch.zeros(1, dtype=t.dtype, device=t.device).expand(t.shape),
                requires_grad=False)

        with torch.no_grad():
            layer.w13_weight = _stub_like(w13)
            layer.w2_weight = _stub_like(w2)
            layer.w13_weight_scale = _stub_like(s13)
            layer.w2_weight_scale = _stub_like(s2)
        # apply() only receives tensors, so hang the packed pair on the
        # placeholder that stands in for w13.
        # EPLB runs AFTER the checkpoint weights are stubbed: the physical-slot
        # view needs a second copy of the same bytes while it is built, and
        # doing it earlier doubles the peak (OOM on a 95 GiB device).
        # NOTE: EPLB's weight replication is a *collective* (all_gather), and
        # vLLM calls this hook once per layer while ranks are not in lockstep
        # (loading is disk-bound, ranks drift apart). Running a collective here
        # makes the first ranks block until the slowest one reaches the *same*
        # layer, which never happens. So this hook only stashes the packed
        # weights; all EPLB work happens once, from the model-level hook, after
        # every layer of every rank is loaded.
        layer.w13_weight._gc_packed = packed
        # [SME-EPLB] Also stash when a placement override is active (e.g. the
        # random control arm), otherwise `_install_eplb` never sees any layer.
        if _eplb_enabled() or _slot_placement() != 'identity':
            self._pending_eplb.append((layer, packed))
        del w13, w2, s13, s2
        # NOTE: `torch.cuda.empty_cache()` used to be called here. With
        # expandable_segments:True it deadlocks once the EPLB path has churned
        # ~7 transient 540 MiB all-gather buffers per layer: the last rank
        # blocks inside the allocator while the others wait in the next
        # collective (observed: GPU utilisation 0%, hang after all 36 layers
        # installed). The allocator already reuses those segments, so drop it.

        # NOTE: the symmetric buffer is allocated lazily on the first forward.
        # Allocating it here doubles the peak (checkpoint weights + packed
        # tiles still coexist for the layer being processed).

    # ---- execution -------------------------------------------------------
    def finalize_weight_and_reduce_impl(self):
        # GC MoE applies the router weights and reduces the top-k inside its
        # combine kernel, so the post-GEMM reduction must be a no-op.
        from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
            TopKWeightAndReduceNoOP,
        )

        return TopKWeightAndReduceNoOP()

    @property
    def expects_unquantized_inputs(self) -> bool:
        # GC MoE quantizes its own activations; prepare must not pre-quantize.
        return True

    def apply(self, output, hidden_states, w1, w2, topk_weights, topk_ids,
              activation, global_num_experts, expert_map, a1q_scale, a2_scale,
              workspace13, workspace2, expert_tokens_meta,
              apply_router_weight_on_input) -> UnfinalizedMoEOutput | None:
        """Run dispatch + expert GEMM + combine as one GC MoE launch.

        ``w1``/``w2`` are the parameters vLLM loaded; the packed form is
        cached per layer in ``process_weights_after_loading``.
        """
        packed = getattr(w1, '_gc_packed', None)
        if packed is None:
            packed = self._packed.get(id(w1))
        if packed is None:
            raise RuntimeError(
                'GC MoE weights were not packed; process_weights_after_loading '
                'must run before the first forward.')
        tokens = hidden_states.shape[0]
        if os.environ.get('GC_MOE_ROUTE_DBG') == '1':
            _TRACE['apply'] = _TRACE.get('apply', 0) + 1
            if _TRACE['apply'] % 200 == 0:
                import torch.distributed as _d
                print(f"[apply-dbg] n={_TRACE['apply']} rank={_d.get_rank()} "
                      f"tokens={tokens} eplb={self._eplb is not None}", flush=True)
        if os.environ.get('GC_MOE_TOKEN_TRACE') == '2' and not _capturing_enabled():
            # Raw per-call trace: rank, hidden-state row count handed to MoE.
            # `calls` counts this rank's MoE invocations (40 per engine step).
            _TRACE.setdefault('raw', 0)
            _TRACE['raw'] += 1
            import torch.distributed as _d
            n = _TRACE['raw']
            if n % 40 == 1 or n % 40 == 0:
                print(f'[tokraw] rank={_d.get_rank()} call={n} '
                      f'step={n // 40} tokens={tokens}', flush=True)
        if os.environ.get('GC_MOE_SLOT_STATS') == '1' and not _capturing_enabled():
            # Per-physical-slot route counts, straight from the kernel's own
            # `expert_count` field (written at impls/sm90_fp8_fp4_gc_moe.py:226
            # for each slot this rank owns). This shows how full the 48 (or 53
            # under EPLB) slots actually are during decode, which is what
            # decides GEMM tile occupancy -- no kernel change needed.
            _TRACE.setdefault('slot', [])
            _TRACE['slot'].append(self._get_buffer().expert_count.clone())
        if os.environ.get('GC_MOE_TOKEN_TRACE') == '1' and not _capturing_enabled():
            # Skip capture: it walks every CUDA-graph bucket (4, 8, 16, ...),
            # which says nothing about the steady-state decode batch.
            _TRACE['n'].append(tokens)
            if len(_TRACE['n']) == 1:
                try:
                    from vllm.config import get_current_vllm_config as _g

                    _cc = _g().compilation_config
                    print(f"[gc-tokens] capture_sizes={_cc.cudagraph_capture_sizes} "
                          f"max_num_seqs={_g().scheduler_config.max_num_seqs}", flush=True)
                except Exception as _e:  # noqa: BLE001
                    print('[gc-tokens] cfg unavailable', _e, flush=True)
            if len(_TRACE['n']) % 40 == 0:
                import collections

                print(f"[gc-tokens] calls={len(_TRACE['n'])} "
                      f"last40={collections.Counter(_TRACE['n'][-40:])}", flush=True)
                if len(_TRACE['n']) > 400:
                    _TRACE['n'].clear()
        self._run_into(output, packed, hidden_states, topk_weights, topk_ids, tokens)
        if os.environ.get('GC_MOE_LAUNCH_TRACE') == '1':
            import torch.distributed as _d
            print(f'[apply] exit rank={_d.get_rank()} tokens={tokens}', flush=True)
        return None

    def run(self, layer, output, x, topk_weights, topk_ids):
        """Direct entry point (used by the offline checks)."""
        return self._run_into(output, self._packed[id(layer)], x, topk_weights,
                              topk_ids, x.shape[0])

    def _run_into(self, output, packed, x, topk_weights, topk_ids, tokens):
        gc = self._gc
        buf = self._get_buffer()
        cap = buf.num_max_tokens_per_rank
        cap = buf.num_max_tokens_per_rank
        if tokens > cap:
            raise RuntimeError(
                f'GC MoE buffer holds {cap} tokens/rank but got {tokens}; '
                'raise --max-num-batched-tokens or the capacity passed to '
                'get_symm_buffer_for_gc_moe_sm90.')

        # Quantize with GC MoE's own kernel rather than vLLM's helper: the SM90
        # GC kernel wants a plain FP32 1x128 scale, while vLLM's block quantizer
        # rounds scales to UE8M0 (powers of two) whenever DeepGEMM E8M0 is on,
        # which loses the mantissa and corrupts the activations.
        if not _capturing_enabled():
            flush_pending_instantiates()
        xq, xsf = _token_fp8_quant(x)
        xsf = xsf.reshape(tokens, -1).to(torch.float32).contiguous()

        # NOTE: CUDA-event timing here is only safe outside capture, and GC's
        # role launches land on green-context streams where
        # is_current_stream_capturing() is not authoritative. Keep it opt-in.
        prof = os.environ.get('GC_MOE_PROFILE') == '1' and not _TRACE['capturing']
        if prof:
            _t0 = torch.cuda.Event(enable_timing=True)
            _t1 = torch.cuda.Event(enable_timing=True)
            _t0.record()
        buf.x[:tokens].copy_(xq.view(tokens, -1))
        buf.x_sf[:tokens].copy_(xsf.view(tokens, -1))
        if self._eplb is not None and _eco_enabled():
            # [ECO-SM90] ECO dispatch: `topk_idx` keeps the LOGICAL expert id.
            # The kernel pushes every route to each active copy of that expert
            # and the plan decides how many tokens each copy keeps, so doing
            # the slot choice here would defeat the plan (and make the quota
            # slices overlap).
            buf.topk_idx[:tokens].copy_(topk_ids.to(torch.int64).view(tokens, -1))
        elif self._eplb is not None:
            # [SME-EPLB] B-arm routing: logical id -> one of its physical slots,
            # handed out round-robin per route. That is the "实时分发" half of
            # EPLB and it is what the validated arm uses.
            #
            # Written with preallocated scratch only: this runs inside CUDA
            # graph capture, where fresh allocations corrupt the caching
            # allocator and deadlock the capture.
            e = self._eplb
            ids = topk_ids.view(-1)
            n = ids.numel()
            width = e['slot_width']
            cand = self._scratch(buf, 'cand', n * width, torch.int32)[:n * width]
            cand2 = cand.view(n, width)
            # slot_table is slots_of.reshape(-1): each expert owns a
            # CONTIGUOUS run of `width` entries, so the index must be
            # ids*width + arange(width).
            col = torch.arange(width, device=ids.device, dtype=torch.int64)
            gather_idx = (ids.to(torch.int64).unsqueeze(-1) + 0) * width + col
            torch.index_select(e['slot_table'], 0, gather_idx.reshape(-1),
                               out=cand2.view(-1))
            pick = self._scratch(buf, 'pick', n, torch.int64)[:n]
            # [RR-DEEPINFER] Replica choice is EXACTLY DeepInfer's gate line
            # (deepseek-ai/TileKernels,
            #  tile_kernels/moe/moe_topk_gate_forward_cuda.py:283):
            #
            #     num_duplicates = logical_count[logical_expert_idx]
            #     duplicate_idx  = (ep_rank + global_token_idx * 23333) % num_duplicates
            #     topk_idx_var   = to_physical_map[logical_expert_idx, duplicate_idx]
            #
            # The key is the TOKEN index inside this rank's batch, scaled by
            # the large prime the kernel uses ("Keep the same with the old
            # implementation") and phase-shifted by ep_rank. It is NOT the
            # route's position in the flattened list, and NOT a per-expert
            # occurrence counter.
            rr_pick(ids, e['count'], ep_rank=int(buf.rank_idx),
                    topk=int(buf.num_topk), out=pick)
            slots = self._scratch(buf, 'slots', n, torch.int32)[:n]
            torch.gather(cand2, 1, pick.unsqueeze(-1), out=slots.unsqueeze(-1))
            buf.topk_idx[:tokens].copy_(slots.view(tokens, -1))
        else:
            buf.topk_idx[:tokens].copy_(topk_ids.to(torch.int64).view(tokens, -1))
        buf.topk_weights[:tokens].copy_(topk_weights.to(torch.float32).view(tokens, -1))

        self._launch(packed, buf, output, tokens)
        # [METRIC-FIX] Snapshot the per-call occupancy of the call that just
        # finished. role 3 zeroes header[7] and the next call overwrites
        # expert_count, so a single read after the window would see zeros.
        #
        # [CAPTURE-FIX] This must NOT run while a CUDA graph is being
        # captured: `.item()` and `.cpu()` are device->host synchronisations,
        # which CUDA refuses inside a capture (cudaErrorStreamCaptureUnsupported).
        # That does not raise here -- it silently invalidates the capture, and
        # the engine only finds out later with the opaque
        # "operation failed due to a previous error during capture".
        # Timed decode runs replay a captured graph, so the Python body below
        # never executes there anyway: no observation is lost, and no host sync
        # can inflate the measured ms/step.
        if _MEAS['measure'] and not _capturing_enabled():
            try:
                hdr = int(buf.header[7].item())
                _ACTIVE_HDR['last'] = hdr
                if hdr > 0:
                    _ACTIVE_HDR['samples'].append(hdr)
                cnt = buf.expert_count.to(torch.float64) / max(1, buf.group.size())
                _ACTIVE_HDR['counts'] = cnt.cpu()
            except Exception:  # noqa: BLE001
                pass
        if prof:
            _t1.record()
            # elapsed_time() is illegal mid-capture; only sample eagerly.
            if torch.cuda.is_current_stream_capturing() is False:
                _PROF['ms'].append(_t0.elapsed_time(_t1))
                _PROF['tokens'].append(tokens)
            if len(_PROF['ms']) % 480 == 0:
                import statistics

                print(f"[gc-prof] calls={len(_PROF['ms'])} "
                      f"median={statistics.median(_PROF['ms']):.3f}ms "
                      f"max={max(_PROF['ms']):.3f}ms", flush=True)
        return output

    def warmup(self, capture_sizes=None) -> None:
        """JIT + child-graph construction for every batch size vLLM captures.

        Runs fully eagerly. The GC runtime compiles the role cubins, loads them
        and builds a child graph the first time a (role, argument) shape is
        seen; doing that inside vLLM's capture is what fails, so we pre-build
        them here for each token count the engine will replay.
        """
        if not self._packed:
            return
        # Remember the engine's own graph buckets: the compile-time bound is
        # fixed to the largest of them so one cubin set covers every batch.
        self._capture_sizes = sorted({int(x) for x in (capture_sizes or [])}) or None
        packed = next(iter(self._packed.values()))
        buf = self._get_buffer()
        device = buf.buffer.device
        hidden = self.moe_config.hidden_dim
        topk = self.moe_config.experts_per_token
        sizes = sorted({int(x) for x in (capture_sizes or []) if 1 <= int(x) <= buf.num_max_tokens_per_rank})
        if not sizes:
            sizes = [1]
        # One extra larger size so the warm-up covers prefill-ish batches too.
        for extra in (256, buf.num_max_tokens_per_rank):
            if extra <= buf.num_max_tokens_per_rank and extra not in sizes:
                sizes.append(extra)
        out = torch.empty((max(sizes), hidden), dtype=torch.bfloat16, device=device)
        for tokens in sizes:
            self._launch(packed, buf, out[:tokens], tokens)
        torch.cuda.synchronize(device)
        logger.info_once('GC MoE warmed %d child graphs (%s)', len(sizes),
                         ','.join(str(s) for s in sizes))

    def allocate_workspace(self) -> None:
        """Allocate the symmetric buffer after weights are fully loaded.

        vLLM checks free memory before loading and sizes the KV cache from a
        post-load profile, so the buffer has to exist before the profile run
        but after the checkpoint weights are gone (otherwise the peak doubles).
        """
        self._get_buffer()

    def _launch(self, packed, buf, output, tokens: int) -> None:
        """One GC MoE call with a compile-time-fixed bound.

        ``num_tokens_bound`` is NOT a per-call runtime hint: it selects the
        kernel geometry (tile_m / thread budget / grids) and therefore which
        cubins get compiled -- the kernel itself only receives ``tokens``,
        ``rank_idx`` and ``clamp``. Passing the live token count made bound
        vary per call, so every distinct batch size forced a fresh JIT and
        could land on a worse geometry.

        Fix it once per (buffer, capacity) to the largest bucket vLLM will
        replay, so one cubin set serves every decode batch.
        """
        if os.environ.get('GC_MOE_LAUNCH_TRACE') == '1':
            import torch.distributed as _d
            print(f'[launch] rank={_d.get_rank()} tokens={tokens} '
                  f'eplb={self._eplb is not None}', flush=True)
        # [SME-MEAS] The kernel accumulates per-slot token counts into this
        # layer's own row, so a rearrangement can attribute them per layer.
        stats = None
        if self._eplb is None and _MEAS['measure']:
            if _MEAS['table'] is None:
                buf0 = self._get_buffer()
                _MEAS['table'] = torch.zeros(
                    (int(os.environ.get('GC_MOE_LAYERS', '40')),
                     buf0.physical // buf0.group.size()),
                    dtype=torch.int32, device=buf0.buffer.device)
            if id(packed[0][0]) not in _MEAS['vanilla_idx']:
                _MEAS['vanilla_idx'][id(packed[0][0])] = len(_MEAS['vanilla_idx'])
                _LAYER_REFS.append(packed[0][0])
            stats = _MEAS['table'][_MEAS['vanilla_idx'][id(packed[0][0])]]
        if self._eplb is None and _MEAS['has_stats'] and _MEAS['table'] is None:
            # lazily size the vanilla counter table to (layers, slots/rank)
            buf0 = self._get_buffer()
            n_layers = int(os.environ.get('GC_MOE_LAYERS', '40'))
            per = buf0.physical // buf0.group.size()
            _MEAS['table'] = torch.zeros((n_layers, per), dtype=torch.int32,
                                         device=buf0.buffer.device)
        if self._eplb is None and _MEAS['has_stats']:
            if id(packed[0][0]) not in _MEAS['vanilla_idx']:
                _MEAS['vanilla_idx'][id(packed[0][0])] = len(_MEAS['vanilla_idx'])
                _LAYER_REFS.append(packed[0][0])
            # vanilla arm: still feed the kernel a per-layer counter table so
            # the same load measurement works without EPLB.
            _idx = _MEAS['vanilla_idx'].get(id(packed[0][0]), 0)
            stats = _MEAS['table'][_idx]
        if self._eplb is not None:
            table = self._eplb['stats']
            # keyed by the packed w1 payload object (strong refs are held in
            # _LAYER_REFS so the id cannot be recycled)
            idx = _LAYER_OF.get(id(packed[0][0]))
            if os.environ.get('GC_MOE_LAYER_DBG') == '1':
                _TRACE['lc'] = _TRACE.get('lc', 0) + 1
                if _TRACE['lc'] <= 80:
                    print(f"[layer-dbg] call={_TRACE['lc']} idx={idx} "
                          f"ptr={packed[0][0].data_ptr()}", flush=True)
            stats = table[idx] if idx is not None else None
        # [SME-EPLB] Same call shape as the plain arm: the validated B arm does
        # not touch the kernel, only the physical slot count and host weights.
        self._gc.sm90_fp8_fp4_gc_moe(
            output, packed[0], packed[1], buf,
            recipe=(1, 1, 32), activation='swiglu', activation_clamp=10.0,
            fast_math=True, num_tokens_bound=self._bound_for(buf, tokens),
            cumulative_local_expert_recv_stats=stats)
        if os.environ.get('GC_MOE_LAUNCH_TRACE') == '1':
            print('[launch] returned', flush=True)

    def _scratch(self, buf, name: str, n: int, dtype):
        """Persistent device buffer of at least `n` elements (no capture allocs)."""
        cache = self.__dict__.setdefault('_scratch_cache', {})
        key = (name, dtype)
        t = cache.get(key)
        if t is None or t.numel() < n:
            t = torch.empty(max(n, 1), dtype=dtype, device=buf.buffer.device)
            cache[key] = t
        return t

    def _slot_scratch(self, buf, tokens: int):
        """Persistent 1-D scratch sized for the largest batch seen."""
        need = tokens * buf.topk_idx.shape[1]
        s = getattr(self, '_slot_buf', None)
        # Same dtype as the mapping table: index_select(out=...) requires it.
        if s is None or s.numel() < need or s.dtype != buf.expert_primary.dtype:
            s = torch.empty(need, dtype=buf.expert_primary.dtype,
                            device=buf.buffer.device)
            self._slot_buf = s
        return s[:need]

    def _bound_for(self, buf, tokens: int) -> int:
        """One compile-time bound, chosen once and then never changed.

        ``GC_MOE_BOUND`` overrides it. The default is the buffer capacity,
        which selects the largest geometry; sweep it to trade geometry against
        the batch range you actually serve.
        """
        if self._fixed_bound is None:
            cap = buf.num_max_tokens_per_rank
            requested = int(os.environ.get('GC_MOE_BOUND', '0')) or cap
            self._fixed_bound = max(1, min(requested, cap))
            logger.info_once(
                'GC MoE compile-time bound fixed at %d (capacity %d)',
                self._fixed_bound, cap)
        if tokens > self._fixed_bound:
            raise RuntimeError(
                f'batch {tokens} exceeds the fixed bound {self._fixed_bound}; '
                'raise GC_MOE_BOUND (it is a compile-time kernel geometry)')
        return self._fixed_bound

    def _install_eplb(self, modules=None):
        """# [SME-EPLB] EPLB setup, ported 1:1 from the validated B arm.

        Source of truth: eplb_deepinfer_isolated_20261022/source/rtp_llm/eplb/
        deepinfer_periodic/ (policy.py + shuffle.py). That arm changes NO kernel
        dispatch structure -- only the host-side slot count (physical) -- and
        moves weights between GPUs with WeightShuffler (symmetric-memory IPC
        inside one node, NCCL P2P across nodes).
        """
        import socket
        import torch.distributed as dist
        from gc_moe.eplb.deepinfer_periodic.config import (
            initial_layout, resolve_num_extra_experts)
        from gc_moe.eplb.deepinfer_periodic.shuffle import WeightShuffler

        modules = list(modules or [self])
        if self._eplb_done:
            return
        self._eplb_done = True
        sm90 = self._gc.sm90_gc
        experts = self.moe_config.num_experts
        buf = self._get_buffer()
        ranks = buf.group.size()
        per_rank = buf.physical // ranks

        # B arm's initial layout (config.initial_layout): identity for the
        # logical experts, then the extra slots duplicate experts 0..extra-1.
        extra = resolve_num_extra_experts(experts, _eplb_extra_floor(experts),
                                          ranks)
        mode = _slot_placement()
        if mode == 'random':
            # [SME-EPLB] Control arm: same multiset of experts over the same
            # slots, scattered with a fixed seed. `extra` copies are drawn at
            # random so the redundancy level is unchanged too.
            g = torch.Generator(device='cpu').manual_seed(_eplb_random_seed())
            base = torch.randperm(experts, generator=g)
            if extra:
                base = torch.cat([base, torch.randint(0, experts, (extra,),
                                                     generator=g)])
            plan_list = base.tolist()
        else:
            # `identity`/`eplb` both start from DeepInfer's initial layout:
            # identity experts first, then `extra` duplicates of 0..extra-1.
            plan_list = initial_layout(experts, extra)
        if len(plan_list) != buf.physical:
            raise RuntimeError(
                f'initial layout has {len(plan_list)} slots, buffer has '
                f'{buf.physical}; raise GC_MOE_EPLB_REDUNDANCY so they agree')
        if _eco_enabled():
            # [ECO-WARP] Put every rank's single-copy slots in a prefix so the
            # dispatch cursor can walk them before it ever touches the plan.
            # SM100's EcoMeta contract requires exactly this ("num_fixed[R] is
            # a prefix of its local slots", eco_meta.cuh L17); the EPLB packer
            # does not produce it, so the permutation is applied HERE and the
            # permuted placement is used by every consumer below (weights,
            # routing, EcoMeta, push's source_index).
            from gc_moe.eco.meta_sm90 import (
                reorder_single_copy_first as _reorder_scf)
            plan_list, _slot_order = _reorder_scf(plan_list, ranks)
            _ECO_SLOT_ORDER['order'] = _slot_order
            print(f'[eco] single-copy-first permutation applied '
                  f'(rank0 first {sum(1 for x in _slot_order[0])} slots)', flush=True)
        plan, slots_of = sm90.install_eplb_plan(buf, plan_list)
        count = (slots_of >= 0).sum(-1).clamp_min(1)
        if _eco_enabled():
            # [ECO-SM90] Build the plan role's EcoMeta for this static
            # placement and write it into the buffer. It is host-side and
            # static: copies[E][32], phy2log[P], num_fixed[R].
            from gc_moe.eco.meta_sm90 import (
                install_into_buffer as _install_eco_meta)
            meta = _install_eco_meta(buf, [int(x) for x in plan_list], ranks,
                                     experts)
            print(f'[eco] meta installed: words={len(meta)} '
                  f'logical={experts} physical={buf.physical} ranks={ranks}',
                  flush=True)
        # [SME-MEAS] One statistics row per MoE layer: the kernel accumulates
        # per-slot token counts, and a rearrangement needs them per layer (the
        # B arm keeps a [window, layers, local] ring for the same reason).
        n_layers_total = sum(len(m._pending_eplb) for m in modules) or 1
        stats = torch.zeros((n_layers_total, per_rank), dtype=torch.int32,
                            device=buf.buffer.device)
        layer_of = {}     # id(w1 tensor) -> row in stats / index in components

        group = dist.group.WORLD
        _LAYER_OF.clear()
        _LAYER_REFS.clear()

        # [SME-EPLB] Expand each layer's 48 logical rows to this rank's 53
        # physical slot rows (replicas come from their owner ranks). The kernel
        # indexes weights by physical slot, so the plain 48-row view is invalid
        # under EPLB.
        for m in modules:
            m._eplb = dict(plan=plan, slots=slots_of, per_rank=per_rank,
                           slot_table=slots_of.reshape(-1).contiguous(),
                           slot_width=slots_of.shape[1], count=count,
                           stats=stats, layout=list(plan_list))
            m._packed = {}
            m._packed_layers = []
            while m._pending_eplb:
                layer, packed = m._pending_eplb.pop()
                pair = sm90.gather_slot_weights(packed, buf, plan, group)
                m._packed[id(layer)] = pair
                m._packed_layers.append((layer, pair))
                layer.w13_weight._gc_packed = pair
                m._last_packed = pair
                del packed, pair
            m._pending_eplb.clear()

        # WeightShuffler wants one component list per layer, holding the active
        # GEMM tensors for that layer. GC MoE's packed pair is
        #   ((w1 data, w1 sf, w1 secondary), (w2 data, w2 sf, w2 secondary))
        components, layer_keys = [], []
        for m in modules:
            for layer, pair in m._packed_layers:
                idx = len(components)
                # Keyed by the layer's w1 payload tensor: `_launch` only sees
                # `packed`, not the layer object.
                # Key on the layer's w13_weight *object id* and keep a strong
                # reference: data_ptr keys are not stable because the caching
                # allocator reuses freed addresses, which made several layers
                # share one stats row.
                _LAYER_REFS.append(pair[0][0])
                layer_of[id(pair[0][0])] = idx
                # Each layer gets its own window row.
                m._eplb.setdefault('stats', stats)
                components.append([pair[0][0], pair[0][1], pair[0][2],
                                   pair[1][0], pair[1][1], pair[1][2]])
                layer_keys.append((m, layer))

        hosts = [None] * ranks
        dist.all_gather_object(hosts, socket.gethostname(), group=group)
        intranode = len(set(hosts)) == 1
        # One map for the whole process: `apply()` runs on this same module
        # object, but other GCMoEExperts instances exist too.
        from eplb_logic import build_layer_index
        _LAYER_OF.update(build_layer_index([t for t in _LAYER_REFS]))
        self._eplb_layer_of = layer_of
        if len(set(_LAYER_OF)) != len(_LAYER_OF):
            raise RuntimeError('layer key collision')
        print(f'[eplb] layer map: {len(_LAYER_OF)} unique keys for '
              f'{len(components)} layers', flush=True)
        self._eplb_shuffler = WeightShuffler(group, components, intranode)
        self._eplb_components = components
        self._eplb_layer_keys = layer_keys
        self._eplb_layout = list(plan_list)
        print(f'[eplb] installed: physical={buf.physical} slots/rank={per_rank} '
              f'experts={experts} redundant={extra} '
              f'shuffle={"symm_mem_ipc" if intranode else "nccl_p2p"} '
              f'layers={len(components)}', flush=True)

    def _get_buffer(self):
        import torch.distributed as dist

        # Capacity has to be the same on every rank (the kernel's peer loop is
        # sized from it) and must cover the largest per-rank batch vLLM runs.
        cap = int(os.environ.get('GC_MOE_CAPACITY', 0)) or self.moe_config.max_num_tokens
        key = (self.moe_config.num_experts, cap, self.moe_config.experts_per_token,
               _eplb_enabled(), _eplb_redundancy(),
               # [SME-EPLB] the per-rank override changes `physical`, so it has
               # to key the buffer too -- otherwise a run that switched
               # GC_MOE_EPLB_EXTRA_PER_RANK would silently reuse a buffer sized
               # for the other redundancy.
               _eplb_extra_per_rank(),
               _slot_placement(),
               os.environ.get('GC_MOE_ECO_TRACE') == '1',
               # [ECO-FINE] The buffer grows by ~2 MB when the fine trace is
               # reserved, so it MUST key the buffer -- otherwise a buffer
               # built without `fine_*` is reused by a kernel compiled WITH
               # them, and the extra views land on top of `eco_meta`
               # (measured: phy2log read back as the identity 0,1,2,3...).
               os.environ.get('GC_MOE_FINE_TRACE') == '1',
               _eplb_enabled() and _eco_enabled())
        # Every layer of this model has identical expert shapes, so one
        # process-wide buffer serves all of them (calls are serialized).
        if _SHARED_BUFFER['buf'] is not None and _SHARED_BUFFER['key'] == key:
            self._buffer = _SHARED_BUFFER['buf']
            self._buffer_key = key
            return self._buffer
        if self._buffer is None or self._buffer_key != key:
            # [SME-EPLB] A placement override (the random control arm) also
            # needs the larger slot count, even with EPLB itself switched off.
            need_slots = _eplb_enabled() or _slot_placement() != 'identity'
            physical = _physical_slots(key[0]) if need_slots else 0
            self._buffer = self._gc.get_symm_buffer_for_gc_moe_sm90(
                dist.group.WORLD, num_experts=key[0],
                num_max_tokens_per_rank=key[1], num_topk=key[2],
                hidden=self.moe_config.hidden_dim,
                intermediate_hidden=self.moe_config.intermediate_size,
                activation='swiglu',
                # [SME-EPLB] only the slot count changes for EPLB.
                physical=physical,
                # [ECO-SM90] reserve the plan region + quota/counts views.
                eco=_eplb_enabled() and _eco_enabled(),
                # [ECO-TRACE] timeline for EVERY arm (see layout `trace`).
                trace=os.environ.get('GC_MOE_ECO_TRACE') == '1'
                      or os.environ.get('GC_MOE_PULL_TRACE') == '1')
            self._buffer_key = key
            _SHARED_BUFFER['buf'] = self._buffer
            _SHARED_BUFFER['key'] = key
        return self._buffer


_BUFFER_READY = {'flag': False}


def _sync_buffer_creation(module) -> None:
    """Create the shared symmetric buffer once, with all ranks in lockstep."""
    import torch.distributed as dist

    if _BUFFER_READY['flag']:
        return
    # Every rank must enter the rendezvous together; the buffer itself is
    # process-wide, so a single barrier around its first creation is enough.
    dist.barrier()
    module._get_buffer()
    dist.barrier()
    _BUFFER_READY['flag'] = True


def _run_eplb_replication(model) -> None:
    """Trigger the one-shot EPLB weight replication on every GC MoE module."""
    live = [m for m in _EXPERT_REGISTRY if m._pending_eplb]
    print(f'[eplb-dbg] hook reached eplb={_eplb_enabled()} '
          f'registry={len(_EXPERT_REGISTRY)} pending={len(live)}', flush=True)
    if not _eplb_enabled() and _slot_placement() == 'identity':
        return
    # One collective pass for every layer this rank holds.
    if live:
        live[0]._install_eplb(live)


def rearrange_now() -> bool:
    """# [SME-EPLB] Re-plan from the statistics collected during warm-up.

    Mirrors deepinfer_periodic/runtime.py::_rearrange:
      1. per-layer per-slot counters -> logical-expert demand (all-reduced);
      2. rank 0 runs rebalance_experts_global, the layout is broadcast;
      3. WeightShuffler.move() moves every layer's weights between GPUs;
      4. routing is repointed at the new placement.

    Fixed relative to the first port: `stats` is [layers, per_rank] and is
    indexed per layer, so the scatter no longer runs off the end of a 53-entry
    tensor with a 424-entry index (that was the `index [424] vs self [384]`
    crash).
    """
    live = [m for m in _EXPERT_REGISTRY if m._eplb is not None]
    if not live:
        return False
    if os.environ.get('GC_MOE_EPLB_NO_REARRANGE') == '1':
        import torch.distributed as _d
        if _d.get_rank() == 0:
            print('[eplb] rearrangement SKIPPED', flush=True)
        return True

    import torch.distributed as dist
    from gc_moe.eplb.deepinfer_periodic.policy import rebalance_experts_global

    m = live[0]
    buf = m._get_buffer()
    experts = buf.num_experts
    ranks = buf.group.size()
    per_rank = buf.physical // ranks

    # --- 1. demand per logical expert -------------------------------------
    # Shared with the CPU regression test (integration/eplb_logic.py) so the
    # tested code and the running code cannot drift apart.
    from eplb_logic import demand_from_stats
    layout = m._eplb['layout']
    stats = m._eplb['stats']               # [layers, per_rank]
    demand = demand_from_stats(stats, layout, buf.rank_idx, per_rank, experts)
    dist.all_reduce(demand, group=dist.group.WORLD)

    # B arm feeds [num_steps, num_moe_layers, num_logical_experts].
    hist = demand.view(1, 1, experts)

    # --- 2. plan on rank 0, broadcast -------------------------------------
    if buf.rank_idx == 0:
        new_plan, _, _ = rebalance_experts_global(hist, buf.physical, per_rank)
        new_plan = new_plan.contiguous()
    else:
        new_plan = torch.empty((1, buf.physical), dtype=torch.int32,
                               device=buf.buffer.device)
    dist.broadcast(new_plan, src=0, group=dist.group.WORLD)
    target = new_plan[0].cpu().tolist()

    # --- 3. single-copy-first permutation, BEFORE anything consumes target -
    # [ECO-WARP-FIX] The permutation MUST be applied to `target` before
    # `shuffler.move`, not after. `_ipc_layer` writes slot j with the weights
    # of `target[j]`, so moving to the RAW target and then installing the
    # PERMUTED one as the routing map pairs slot j's real weights with
    # `permuted[j]` -- measured on a CPU replay: 448/448 slots mismatched.
    # Applying it here means the weight move, the routing map, EcoMeta and
    # push's source_index all refer to the same permuted placement.
    if _eco_enabled():
        from gc_moe.eco.meta_sm90 import (
            reorder_single_copy_first as _reorder_scf)
        target, _slot_order = _reorder_scf(target, ranks)
        _ECO_SLOT_ORDER['order'] = _slot_order

    # --- 4. move weights between GPUs -------------------------------------
    shuffler = m._eplb_shuffler
    n_layers = len(shuffler.components)
    shuffler.move([list(layout) for _ in range(n_layers)],
                  [list(target) for _ in range(n_layers)])

    # --- 5. repoint routing ----------------------------------------------
    new_t, slots_of = m._gc.sm90_gc.install_eplb_plan(buf, target)
    count = (slots_of >= 0).sum(-1).clamp_min(1)
    if _eco_enabled():
        # [ECO-SM90] The plan reads EcoMeta's copies/phy2log, so a new
        # placement must rewrite it, and the warm start must not inherit
        # quotas from the previous (now different) placement.
        from gc_moe.eco.meta_sm90 import (
            install_into_buffer as _install_eco_meta)
        _install_eco_meta(buf, [int(x) for x in target], ranks, experts)
        with torch.inference_mode():
            # The warm start must not survive a placement change: the quotas
            # are indexed by physical slot and would describe a different
            # expert map. Zeroing `eco_state` makes the next plan call fall
            # back to the cold path (correct, just slower for one call).
            buf.eco_state.zero_()
            # [ECO-SM90] NEVER clear `eco_ready`: the three words are epoch
            # TOKENS, not flags.
            #   eco_ready[0] = counts published (dispatch -> plan)
            #   eco_ready[1] = plan published   (plan -> replica pull)
            #   eco_ready[2] = counts consumed  (plan -> next dispatch)
            # Dispatch's very first act is `wait_epoch(eco_ready[2],
            # epoch - 1)`, i.e. it waits for the PREVIOUS call's plan to
            # finish reading the broadcast panel. Only that plan can write the
            # token, and it is woken by `eco_ready[0]` -- which dispatch has
            # not written yet at that point. Clearing the three words
            # therefore deadlocks the first call after any rearrangement
            # (measured: 5 s of decode, then "RPC call to sample_tokens timed
            # out"). Leaving them alone keeps all three epoch chains intact;
            # the stale values are one epoch behind and can never be mistaken
            # for the new epoch.
        if buf.rank_idx == 0:
            print('[eco] meta re-installed after rearrangement', flush=True)
    for mod in live:
        mod._eplb['slot_table'].copy_(slots_of.reshape(-1))
        mod._eplb['count'].copy_(count)
        mod._eplb['layout'] = list(target)
        mod._eplb.update(plan=new_t, slots=slots_of)
        with torch.inference_mode():
            mod._eplb['stats'].zero_()
    if buf.rank_idx == 0:
        copies = torch.bincount(torch.tensor(target), minlength=experts)
        print(f'[eplb] rearranged: slots={buf.physical} per_rank={per_rank} '
              f'max_copies={int(copies.max())} '
              f'replicated={int((copies > 1).sum())}', flush=True)
    return True


_BLOCK_M = {'value': None}


def _block_m(buf) -> int:
    """# [SME-MEAS] [METRIC-FIX] The M chunk the SM90 routed role really runs.

    `get_block_m_for_gc_moe` is the SM100 selector: it answers a different
    question (how wide a GEMM tile that backend wants) and returned
    192/240 here, while the SM90 role processes `count` rows in chunks of
    `schedule.tile_m` (= 64 at this geometry, 16 at small batches). Using the
    wrong divisor under-reported the block count by ~4x, so the metric now
    comes from the same schedule the kernel was compiled with.
    """
    if _BLOCK_M['value'] is not None:
        return _BLOCK_M['value']
    value = None
    try:
        from gc_moe.apis.sm90_gc_moe import ValidatedCall
        from gc_moe.jit_kernels.heuristics.sm90_gc_moe import get_sm90_role_config
        from gc_moe.runtime.sm90_gc_moe import get_context
        # `bound` is the compile-time token bound this launch uses; the
        # capacity of the buffer is its upper limit.
        bound = buf.num_max_tokens_per_rank
        capped = None
        for m in _EXPERT_REGISTRY:
            if getattr(m, '_fixed_bound', None):
                capped = int(m._fixed_bound)
                break
        if capped:
            bound = capped
        cfg = ValidatedCall(ranks=buf.group.size(), experts=buf.num_experts,
                            capacity=buf.num_max_tokens_per_rank,
                            topk=buf.num_topk, hidden=buf.hidden,
                            intermediate=buf.intermediate_hidden, tokens=1,
                            bound=bound, clamp=10.0, fast_math=True,
                            shared_experts=buf.num_shared_experts,
                            physical=(buf.physical if buf.physical != buf.num_experts else 0),
                            eco_logical_experts=(buf.eco_logical
                                                 if getattr(buf, 'eco', False) else 0))
        value = int(get_sm90_role_config(cfg, get_context()).tile_m)
    except Exception as exc:  # noqa: BLE001
        logger.warning_once(f'GC MoE: cannot read tile_m ({exc}); using 64')
    if value is None:
        value = 64
    _BLOCK_M['value'] = value
    return value


def meas_enable() -> None:
    """# [SME-MEAS] Turn on the per-layer counter table BEFORE graph capture.

    A CUDA graph bakes in the kernel variant it captured. If the stats table
    only appears at measurement time, the captured graph never writes it and
    the reading is all zeros.
    """
    _MEAS['measure'] = True


def meas_begin(steps: int = 0) -> None:
    """# [SME-MEAS] Start per-step measurement.

    Clears the per-layer counters first: warm-up (and the rearrangement's own
    statistics window) already accumulated into them, and an uncleared table
    makes the first measured step look enormous.
    """
    _MEAS['on'] = True
    _MEAS['steps_done'] = int(steps)
    _MEAS['has_stats'] = True
    torch.cuda.synchronize()
    # vanilla arm has no EPLB table; build it here so the first measured step
    # already writes into it (the kernel only records when stats is provided).
    if _MEAS['table'] is None:
        live = [m for m in _EXPERT_REGISTRY
                if getattr(m, '_buffer', None) is not None]
        if live and live[0]._eplb is None:
            b = live[0]._get_buffer()
            _MEAS['table'] = torch.zeros(
                (int(os.environ.get('GC_MOE_LAYERS', '40')),
                 b.physical // b.group.size()),
                dtype=torch.int32, device=b.buffer.device)
    # zero_() must run inside inference_mode: these tensors are created under
    # it (the worker runs in inference mode), and torch refuses an in-place
    # update to an inference tensor from outside that mode.
    cleared = False
    with torch.inference_mode():
        for m in _EXPERT_REGISTRY:
            if m._eplb is not None:
                m._eplb['stats'].zero_()
                cleared = True
                break
        if not cleared and _MEAS['table'] is not None:
            _MEAS['table'].zero_()
    # [METRIC-FIX] drop the previous window's occupancy samples
    _ACTIVE_HDR['samples'] = []
    _ACTIVE_HDR['last'] = None
    _ACTIVE_HDR['counts'] = None


def meas_step() -> None:
    """# [SME-MEAS] DEPRECATED -- do not call per step.

    Kept only for ad-hoc debugging. One `collective_rpc` per engine step costs
    ~60 ms/step (measured), which is larger than every effect this experiment
    measures, so the driver no longer calls it.

    The per-step quantities it was meant to provide are already in the kernel:
      * `header[7]`       -- slots that got a token in THIS call (role 3 zeroes
                             it each call, role 1 increments it);
      * `expert_count[]`  -- tokens per slot for THIS call.
    One read after the window is enough; see `meas_end` / `aggregate_with_sample`.
    """
    if not _MEAS['on']:
        return
    torch.cuda.synchronize()
    live = [m for m in _EXPERT_REGISTRY if m._eplb is not None]
    if not live:
        return
    m = live[0]
    buf = m._get_buffer()
    infl = buf.group.size()
    # This rank's per-slot counts for the most recent kernel call.
    counts = buf.expert_count.to(torch.float64) / infl
    bm = _block_m(buf)
    row = counts.sum().reshape(1)                    # this rank, one layer
    blocks = ((counts + bm - 1) // bm).sum().reshape(1)
    active = (counts > 0).sum().to(torch.float64).reshape(1)

    if os.environ.get('GC_MEAS_DBG') == '1':
        print(f"[meas-dbg] rank={buf.rank_idx} tokens={row.item():.0f} "
              f"active={active.item():.0f} blocks={blocks.item():.0f}", flush=True)
    if buf.rank_idx == 0:
        _MEAS['steps'].append(dict(tokens=row.item(), active=active.item(),
                                   blocks=blocks.item(), block_m=bm))


def meas_end() -> dict:
    """# [SME-MEAS] Window metrics via the CPU-tested helper.

    Two kinds of number, deliberately kept apart:

    * per-step AVERAGES over the window (tokens, M blocks) -- exact, from the
      accumulated table divided by the step count;
    * a single-call SAMPLE (`expert_count`, overwritten every call) which gives
      the only honest per-step occupancy (`active_step`).

    The earlier version reported `active` from the accumulated table, which made
    it equal slots_per_rank every time: with 100 steps, every slot has seen at
    least one token. That was a definition bug, not a property of the run.
    """
    _MEAS['on'] = False
    live = [m for m in _EXPERT_REGISTRY if m._eplb is not None]
    if live:
        m = live[0]
        buf = m._get_buffer()
        stats = m._eplb['stats']
    else:
        allm = [m for m in _EXPERT_REGISTRY
                if getattr(m, '_buffer', None) is not None]
        if not allm or _MEAS['table'] is None:
            return {'ok': False, 'reason': 'no buffer'}
        m = allm[0]
        buf = m._get_buffer()
        stats = _MEAS['table']

    from eplb_logic import aggregate_with_sample
    bm = _block_m(buf)
    # `header[7]` is the dispatch role's count of slots that received a token in
    # THIS call, and role 3 zeroes it at the start of every call. So one read
    # after the window gives the exact per-step occupancy -- no per-step RPC.
    # [METRIC-FIX] `header[7]` is zeroed by role 3 at the start of EVERY call,
    # so reading it after the window always returned 0. `_run_into` snapshots
    # it right after the launch returns, which is the last completed call.
    active_now = _ACTIVE_HDR['last']
    if active_now is None:
        active_now = None
    r = aggregate_with_sample(stats, bm, _MEAS.get('steps_done', 0),
                              buf.group.size(), sample=buf.expert_count,
                              active_sample=active_now)
    samples = [x for x in _ACTIVE_HDR['samples'] if x > 0]
    import statistics as _st
    return dict(ok=True, ok_steps=int(_MEAS.get('steps_done', 0)),
                block_m=int(bm),
                # [METRIC-FIX] per-call occupancy, sampled after every launch
                active_experts_samples=len(samples),
                active_experts_sampled_median=(
                    float(_st.median(samples)) if samples else -1.0),
                # window averages
                tokens_per_rank_median=r['tokens_median'],
                tokens_per_rank_min=r['tokens_min'],
                tokens_per_rank_max=r['tokens_max'],
                m_blocks_median=r['blocks_median'],
                m_blocks_min=r['blocks_min'],
                m_blocks_max=r['blocks_max'],
                # one-call sample: the only exact per-step occupancy
                sample_tokens_per_rank=r['sample_tokens'],
                active_experts_median=r['active_step'],
                active_experts_kernel_header=r.get('active_header'),
                m_blocks_step=r['blocks_step'],
                # window-wide "slot ever used" (not occupancy)
                active_ever_median=r['active_ever_median'],
                # [METRIC-FIX] three DIFFERENT imbalance ratios, never mixed:
                #   ir_layer = max(per-layer tokens)/mean   -> placement has no
                #              effect on it (routing is unchanged), so it must be
                #              identical across arms
                #   ir_slot  = max(per-slot tokens)/mean    -> how evenly THIS
                #              rank's slots are used; directly comparable only
                #              between arms with the same slot count
                #   ir_rank  = max(per-RANK tokens)/mean    -> the run json's
                #              top-level `ir`, from ir_end():
                #              max(tokens_per_rank)/mean(tokens_per_rank)
                ir_window=r['ir_layer'],
                ir_layer=r['ir_layer'],
                ir_slot=r['ir_slot'],
                per_slot_token_sum=r['per_slot_tokens'],
                ir_rank=_IR.get('ir', -1.0),
                slots_per_rank=_slot_count(),
                per_layer_median_tokens=r['tokens'],
                per_layer_median_blocks=r['blocks'],
                per_layer_active_ever=r['active_ever'],
                notes=r['notes'])


def _slot_count() -> int:
    live = [m for m in _EXPERT_REGISTRY if m._eplb is not None]
    if live:
        return int(live[0]._eplb['stats'].shape[1])
    if _MEAS['table'] is not None:
        return int(_MEAS['table'].shape[1])
    return -1


def ir_begin() -> None:
    """# [SME-IR] Arm the imbalance measurement (zero per-call overhead)."""
    _IR['on'] = True


def ir_end(buf=None) -> dict:
    """# [SME-IR] IR = max_rank(tokens) / mean_rank(tokens), over ALL layers.

    Uses the per-layer counter table summed over layers, not a single kernel
    call. A one-layer sample carries Poisson noise of about 7%
    (sqrt(192)/192), which is the same size as the imbalance being measured --
    the earlier single-layer version produced an IR that swung with that noise
    instead of with the placement.

    `stats` also carries a world_size x inflation (the kernel sums every
    source rank's send_count, and send_count is a global counter replicated on
    each rank), so it is divided out before the gather.
    """
    import torch.distributed as _d

    _IR['on'] = False
    if buf is None:
        live = [m for m in _EXPERT_REGISTRY
                if getattr(m, '_buffer', None) is not None]
        if not live:
            return {'ir': -1.0, 'reason': 'no buffer'}
        buf = live[0]._get_buffer()

    world = _d.get_world_size()
    # per-layer table: EPLB arm keeps it in _eplb['stats'], vanilla in _MEAS
    live = [m for m in _EXPERT_REGISTRY if m._eplb is not None]
    if live:
        table = live[0]._eplb['stats']
        table = table.to(torch.float64).sum(dim=1)          # [layers]
        slots = int(live[0]._eplb['stats'].shape[1])
    elif _MEAS['table'] is not None:
        table = _MEAS['table'].to(torch.float64).sum(dim=1)
        slots = int(_MEAS['table'].shape[1])
    else:
        # fall back to the single-call counter (noisy)
        counts = buf.expert_count
        table = counts.to(torch.float64).sum().reshape(1)
        slots = int(counts.numel())

    local = (table.sum() / world).reshape(1)
    gathered = [torch.zeros_like(local) for _ in range(world)]
    _d.all_gather(gathered, local)
    loads = [float(x.item()) for x in gathered]

    mean = sum(loads) / len(loads)
    _IR['ir'] = (max(loads) / mean) if mean > 0 else -1.0
    return {'ir': _IR['ir'],
            'max_rank_tokens': max(loads), 'min_rank_tokens': min(loads),
            'mean_rank_tokens': mean,
            'tokens_per_rank': [int(x) for x in loads],
            'slots_per_rank': slots,
            'layers_averaged': int(table.numel()),
            'sum_slots_tokens': int(sum(loads))}


def dump_slot_stats(label: str) -> None:
    """Summarise per-slot route counts gathered during warm-up decode.

    Reads the kernel's own ``expert_count`` field (written per physical slot at
    ``impls/sm90_fp8_fp4_gc_moe.py:226``), so no kernel change is needed. The
    last 40 samples are one full engine step (40 MoE layers) on one batch.
    """
    rows = _TRACE.get('slot') or []
    if not rows:
        print('[slots] no samples', flush=True)
        return
    import statistics
    counts = [int(x) for r in rows[-40:] for x in r.tolist()]
    if not counts:
        return
    nz = [c for c in counts if c > 0]
    print(f'[slots] label={label} slots/sample={len(rows[-1])} '
          f'nonzero={len(nz)}/{len(counts)} '
          f'mean={statistics.mean(counts):.2f} '
          f'median={statistics.median(counts)} max={max(counts)} '
          f'zeros={sum(1 for c in counts if c == 0)} '
          f'le2={sum(1 for c in counts if c <= 2)}', flush=True)


def _install_worker_rpc() -> None:
    """Expose the replan as a worker method so the driver can reach it.

    ``collective_rpc`` resolves a *string* to ``getattr(worker, name)``. Passing
    the callable directly would need pickle-based RPC (vLLM refuses that by
    default), and the EPLB registry only exists inside the workers anyway.
    """
    from vllm.v1.worker.gpu_worker import Worker

    if getattr(Worker, '_gc_moe_rearrange', None) is not None:
        return

    def gc_moe_rearrange(self) -> bool:
        return rearrange_now()

    Worker._gc_moe_rearrange = gc_moe_rearrange

    if getattr(Worker, '_gc_moe_meas_begin', None) is None:

        def gc_moe_meas_begin(self, steps: int = 0) -> bool:
            meas_begin(steps)
            return True

        Worker._gc_moe_meas_begin = gc_moe_meas_begin

    if getattr(Worker, '_gc_moe_meas_enable', None) is None:

        def gc_moe_meas_enable(self) -> bool:
            meas_enable()
            return True

        Worker._gc_moe_meas_enable = gc_moe_meas_enable

    if getattr(Worker, '_gc_moe_meas_step', None) is None:

        def gc_moe_meas_step(self) -> bool:
            meas_step()
            return True

        Worker._gc_moe_meas_step = gc_moe_meas_step

    if getattr(Worker, '_gc_moe_meas_end', None) is None:

        def gc_moe_meas_end(self) -> dict:
            return meas_end()

        Worker._gc_moe_meas_end = gc_moe_meas_end

    if getattr(Worker, '_gc_moe_ir_begin', None) is None:

        def gc_moe_ir_begin(self) -> bool:
            """# [SME-IR] Start recording per-slot loads inside this worker."""
            ir_begin()
            return True

        Worker._gc_moe_ir_begin = gc_moe_ir_begin

    if getattr(Worker, '_gc_moe_ir_end', None) is None:

        def gc_moe_ir_end(self) -> dict:
            """# [SME-IR] Stop recording and return this rank's imbalance stats."""
            return ir_end()

        Worker._gc_moe_ir_end = gc_moe_ir_end

    if getattr(Worker, '_gc_moe_eco_trace_dump', None) is None:

        def gc_moe_eco_trace_dump(self) -> dict:
            """# [ECO-TRACE] Raw 30-slot timeline + per-step rows of THIS rank.

            Read once at the end of the measured window (never per step, which
            would cost a host sync per forward and inflate ms/step). The
            driver averages over steps afterwards.
            """
            import torch
            from gc_moe.layout import sm90_gc_moe as _lay  # noqa: F401

            allm = [m for m in _EXPERT_REGISTRY
                    if getattr(m, '_buffer', None) is not None]
            if not allm:
                return {'ok': False, 'reason': 'no buffer'}
            buf = allm[0]._get_buffer()
            out = {'ok': True, 'rank': int(buf.rank_idx)}
            tr = getattr(buf, 'eco_trace', None)
            if tr is not None:
                torch.cuda.synchronize()
                out['slots'] = [int(v) for v in tr.cpu().tolist()]
            ptl = getattr(buf, 'pull_tl', None)
            if ptl is not None:
                out['pull_rows'] = ptl.view(256, 10).cpu().tolist()
            ftl = getattr(buf, 'fine_tl', None)
            if ftl is not None:
                out['fine_rows'] = ftl.view(65536, 8).cpu().tolist()
            em = getattr(buf, 'eco_meta', None)
            if em is not None:
                out['eco_meta'] = [int(v) for v in em.cpu().tolist()]
                # [ECO-FINE] pointer audit: the kernel receives
                # `buffer.buffer.data_ptr()` and adds `off_eco_meta`, so this
                # difference MUST equal layout.eco_meta.
                out['ptr_eco_meta'] = int(em.data_ptr())
                out['ptr_workspace'] = int(buf.buffer.data_ptr())
                out['nbytes_workspace'] = int(buf.buffer.numel())
                out['ptr_delta'] = int(em.data_ptr()) - int(buf.buffer.data_ptr())
                # [ECO-FINE] Absolute-address audit from the HOST side: read the
                # exact bytes the kernel will read (workspace + off_eco_meta +
                # phy2log offset) with torch, so "the kernel sees zeros" can be
                # separated from "the memory really holds zeros".
                import torch as _t
                _lay = buf.layout
                _w = (_lay.eco_meta + (16 + 384 * 16) * 4) // 4
                out['host_read_phy2log'] = [
                    int(v) for v in
                    buf.buffer.view(_t.int32)[_w:_w + 8].cpu().tolist()]
            pl = getattr(self, '_eplb_layout', None)
            if pl is not None:
                out['eplb_layout'] = [int(v) for v in pl]
            tl = getattr(buf, 'eco_tl', None)
            if tl is not None:
                rows = tl.view(256, 21).cpu().tolist()
                steps = getattr(buf, 'eco_tl_step', None)
                out['tl_rows'] = rows
                out['tl_step'] = int(steps.cpu().item()) if steps is not None else -1
            return out

        Worker._gc_moe_eco_trace_dump = gc_moe_eco_trace_dump

    if getattr(Worker, '_gc_moe_sync', None) is None:

        def gc_moe_sync(self) -> bool:
            """Block until this worker's GPU work has actually finished.

            ``LLMEngine.step()`` only *submits* work: with vLLM's default
            ``async_scheduling=True`` the scheduler runs a batch ahead, so a
            ``perf_counter`` around ``step()`` measures queue latency, not GPU
            time. Timing a window of steps therefore needs one real device
            synchronize at the end.
            """
            import torch

            torch.cuda.synchronize()
            return True

        Worker._gc_moe_sync = gc_moe_sync


def _allocate_all_gc_workspaces(model) -> None:
    """Create the symmetric buffer and warm the Green Context pair.

    Both are one-time, thread-local resources. Building them during capture is
    illegal (CUDA forbids context/stream creation inside a capture), so they
    must exist before the engine captures its graphs.
    """
    for module in model.modules():
        if not isinstance(module, GCMoEExperts):
            continue
        if module._buffer is None:
            module.allocate_workspace()
    if any(isinstance(m, GCMoEExperts) for m in model.modules()):
        _warm_gc_child_graph(model)


def _warm_gc_child_graph(model) -> None:
    """Build the GC child graph once, outside any CUDA capture.

    The GC runtime JIT-compiles the role cubins, loads them with
    cuLibraryLoadData and builds a child graph on first use -- all of which are
    illegal inside a stream capture. Upstream's own benchmark therefore runs a
    few eager calls before capturing; vLLM needs the same, so this runs at the
    end of weight loading where no capture is active.
    """
    module = next(m for m in model.modules() if isinstance(m, GCMoEExperts))
    if module._warmed:
        return
    try:
        from vllm.config import get_current_vllm_config

        sizes = list(get_current_vllm_config().compilation_config.cudagraph_capture_sizes or [])
    except Exception:  # noqa: BLE001
        sizes = []
    try:
        print('[warm] start', flush=True)
        module.warmup(sizes)
        print('[warm] done', flush=True)
        module._warmed = True
    except Exception as exc:  # noqa: BLE001 - never block engine bring-up
        logger.warning('GC MoE warmup failed (%s); graphs will be disabled', exc)


_PENDING_INSTANTIATE: list = []
_ORIG_INSTANTIATE: list = [None]


def install_capture_safe_instantiate() -> None:
    """Let GC MoE build child graphs while vLLM is capturing.

    On the first call for a given (roles, arguments) key the GC runtime builds
    a child graph and immediately *instantiates* it. Inside a stream capture
    ``cuGraphInstantiate`` is illegal, which is why capturing failed even after
    the kernels were JIT-warmed: vLLM's capture uses its own static buffers, so
    the keys differ from anything a warm-up could have prepared.

    A parent capture only needs the child graph's *definition* (it is added as a
    node); the executable is only needed for a direct launch. So during capture
    we build the definition, skip instantiation, and finish it as soon as we are
    outside a capture again.
    """
    from gc_moe.runtime import sm90_gc_moe as rt

    if getattr(rt._ChildGraph, '_gc_capture_safe', False):
        return
    orig_instantiate = rt._ChildGraph.instantiate

    def instantiate(self):
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            _PENDING_INSTANTIATE.append(self)
            return
        orig_instantiate(self)

    rt._ChildGraph.instantiate = instantiate
    rt._ChildGraph._gc_capture_safe = True
    _ORIG_INSTANTIATE[0] = orig_instantiate


def flush_pending_instantiates() -> None:
    """Materialise child graphs deferred during capture (call outside capture)."""
    if not _PENDING_INSTANTIATE:
        return
    from gc_moe.runtime import sm90_gc_moe as rt

    orig = _ORIG_INSTANTIATE[0]
    if orig is None:
        return
    pending, _PENDING_INSTANTIATE[:] = list(_PENDING_INSTANTIATE), []
    for graph in pending:
        try:
            orig(graph)
        except Exception:  # noqa: BLE001
            pass


def register() -> None:
    """Teach vLLM that ``--moe-backend gc_moe`` means GC MoE (idempotent).

    vLLM resolves a backend name to a ``Mxfp4MoeBackend`` enum member and then
    to kernel classes through ``backend_to_kernel_cls``. Both are module-level
    tables, so patching them is the smallest change that leaves stock vLLM
    untouched on disk.
    """
    from vllm.config import kernel as kernel_cfg
    from vllm.model_executor.layers.fused_moe.oracle import mxfp4 as oracle

    if 'gc_moe' in _GLOBAL_TOPPING:
        return
    _GLOBAL_TOPPING.append('gc_moe')

    # 1. A MoE backend name vLLM will accept in KernelConfig / --moe-backend.
    enum_cls = oracle.Mxfp4MoeBackend
    if not hasattr(enum_cls, 'GC_MOE'):
        from enum import Enum
        member = object.__new__(enum_cls)
        member._name_ = 'GC_MOE'
        member._value_ = 'GC_MOE'
        type.__setattr__(enum_cls, 'GC_MOE', member)
        enum_cls._member_map_['GC_MOE'] = member
        enum_cls._value2member_map_['GC_MOE'] = member
    gc_member = enum_cls.GC_MOE

    # 2. backend name -> enum candidates.
    orig_map = oracle.map_mxfp4_backend

    def map_with_gc(runner_backend):
        if runner_backend == 'gc_moe':
            return [gc_member]
        return orig_map(runner_backend)

    oracle.map_mxfp4_backend = map_with_gc

    # 3. enum -> kernel classes (both the experts and the prepare/finalize).
    orig_to_cls = oracle.backend_to_kernel_cls

    def to_cls_with_gc(backend):
        if backend == gc_member:
            return [GCMoEExperts]
        return orig_to_cls(backend)

    oracle.backend_to_kernel_cls = to_cls_with_gc

    # 3b. GC MoE owns dispatch/combine, so vLLM must not build a DeepEP
    #     prepare/finalize for it. Force the pass-through pair instead.
    from vllm.model_executor.layers.fused_moe import all2all_utils

    orig_make_pf = all2all_utils.maybe_make_prepare_finalize

    def make_pf_with_gc(moe, quant_config, *args, **kwargs):
        if getattr(moe, 'moe_backend', None) == 'gc_moe':
            return GCMoEPrepareFinalize(
                max_tokens_per_rank=moe.max_num_tokens,
                num_dispatchers=1,
            )
        return orig_make_pf(moe, quant_config, *args, **kwargs)

    all2all_utils.maybe_make_prepare_finalize = make_pf_with_gc
    # make_mxfp4_moe_kernel captured the original name at import time.
    oracle.maybe_make_prepare_finalize = make_pf_with_gc

    # 4. Weight conversion is a passthrough for GC MoE.
    orig_convert = oracle.convert_weight_to_mxfp4_moe_kernel_format

    def convert_with_gc(mxfp4_backend, layer, w13_weight, w2_weight,
                        w13_weight_scale, w2_weight_scale, w13_bias=None,
                        w2_bias=None, **kwargs):
        if mxfp4_backend == gc_member:
            return (w13_weight.data, w2_weight.data, w13_weight_scale.data,
                    w2_weight_scale.data, w13_bias, w2_bias)
        return orig_convert(mxfp4_backend, layer, w13_weight, w2_weight,
                            w13_weight_scale, w2_weight_scale, w13_bias,
                            w2_bias, **kwargs)

    oracle.convert_weight_to_mxfp4_moe_kernel_format = convert_with_gc

    # 4b. Make GC MoE's child-graph instantiation legal inside a capture.
    install_capture_safe_instantiate()

    # 4a. Let the driver trigger an EPLB replan inside every worker.
    _install_worker_rpc()

    # 5. Allocate GC MoE's symmetric buffer at the end of weight loading.
    #    The model's own finalize hook runs once, after every layer is packed,
    #    which is the only point where the buffer can be created without
    #    doubling the weight peak.
    # vLLM runs one module-level ``process_weights_after_loading(model, ...)``
    # AFTER every layer's quant method was processed. That is the only hook
    # that is reached once per rank, with all ranks in the same place -- the
    # per-layer ``QuantizeMethodBase.process_weights_after_loading`` is not, so
    # collectives must not live there. The model classes do not define their
    # own hook at all, which is why patching them silently did nothing.
    from vllm.model_executor.model_loader import utils as _loader_utils

    if not getattr(_loader_utils, '_gc_moe_post_hook', False):
        _orig_pwal = _loader_utils.process_weights_after_loading

        def process_weights_after_loading(model, model_config, target_device,
                                         _orig=_orig_pwal):
            _orig(model, model_config, target_device)
            _run_eplb_replication(model)
            _allocate_all_gc_workspaces(model)

        _loader_utils.process_weights_after_loading = process_weights_after_loading
        _loader_utils._gc_moe_post_hook = True
        # base_loader imported the symbol directly; patch its binding too.
        from vllm.model_executor.model_loader import base_loader as _base_loader
        if getattr(_base_loader, 'process_weights_after_loading', None) is _orig_pwal:
            _base_loader.process_weights_after_loading = process_weights_after_loading

register()
