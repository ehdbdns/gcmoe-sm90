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
# [T17-PERLAYER] id(packed w1 payload) -> row index inside `plan`/`slots_of`.
# Distinct from `_LAYER_OF`, whose value is the STATS/component index (built in
# LIFO pop order, so it is NOT the plan row).
_LAYER_ROW = {}
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
# [IR-STEP] Per-(step, layer) rank-critical-path cost, for the IR the user
# actually wants: one max/mean over the 8 ranks PER LAYER PER STEP, averaged
# over all (step, layer) pairs afterwards -- NOT max/mean taken over a window
# sum (that number is diluted by averaging and cannot be recovered from it).
#
# Storage is one float per (layer, step) on every rank:
#   _IRSTEP['buf'][layer, step] = this rank's tokens routed to `layer` in `step`
# filled by differencing the per-layer accumulator right after each MoE
# launch (no RPC: the driver calls forward 40 times per engine step, so the
# table is complete by the time `_gc_moe_ir_end` runs).
_IRSTEP = {'on': False, 'rows': [], 'prev': None, 'acc': None}
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



# ══════════════════════════════════════════════════════════════════════════
# [ROLE4] Shared expert computed by gc_moe's own role 4 instead of vLLM.
#
# vLLM keeps the shared expert as a plain bf16 DeepseekV2MLP
# (gate_up_proj / down_proj) and computes it on its own aux stream. gc_moe's
# role 4 wants the SAME maths in its native layout: E4M3 data + FP32 scales
# with recipe (1,1,32). We quantise the bf16 weights ONCE at init (never in
# the hot path) and hand them to the kernel; combine then adds `shared_y`
# itself, so vLLM must NOT compute the shared expert as well (1+1=2).
# ══════════════════════════════════════════════════════════════════════════
_SHARED_W = {}
_SE_RUNNER = {}   # id(SharedExperts) -> MoERunner
_SHARED_MODULES = []   # SharedExperts in construction order


def _quant_row_k32(weight):
    """bf16 [n, k] -> (fp8_e4m3 [n,k], fp32 [n, k//32]) with recipe (1,1,32).

    Per-row, per-K32 scaling: exact for the (1,1,32) recipe `_transform_shared`
    expects. n and k must already be multiples of 256/128.
    """
    import torch
    w = weight.detach().to(torch.float32)
    n, k = w.shape
    blocks = w.reshape(n, k // 32, 32)
    amax = blocks.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
    scale = amax / 448.0
    q = (blocks / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return (q.reshape(n, k).contiguous(), scale.squeeze(-1).to(torch.float32).contiguous())


_CKPT_HDR = {}
_SHARD_CACHE = {}


def _ckpt_read(key):
    """Read one tensor straight out of the checkpoint's safetensors shards.

    WHY THE CHECKPOINT: vLLM holds the shared expert as MXFP4-packed `int32`
    (measured gate_up (320,2304) int32, down (20,20480) int32), sharded over TP
    and padded on the intermediate axis -- unusable for role 4 without a
    dequant+padded-removal pass. The checkpoint instead stores exactly what
    role 4 wants:

        shared_experts.w1.weight  F8_E4M3 [2304, 5120]  w1.scale F8_E8M0 [72,160]
        shared_experts.w2.weight  F8_E4M3 [5120, 2304]  w2.scale F8_E8M0 [160,72]

    i.e. full, unpadded, FP8 weights with 32x32 ue8m0 block scales. Only the
    scale encoding has to be expanded (see `_ue8m0_to_rowk32`), which is exact.
    """
    import json as _json
    import struct
    import torch
    root = os.environ.get('MODEL', '').rstrip('/')
    if not _CKPT_HDR:
        idx = os.path.join(root, 'model.safetensors.index.json')
        if not os.path.exists(idx):
            _CKPT_HDR['map'] = None
        else:
            with open(idx) as fh:
                _CKPT_HDR['map'] = _json.load(fh)['weight_map']
    wmap = _CKPT_HDR['map']
    if wmap is None or key not in wmap:
        return None
    path = os.path.join(root, wmap[key])
    cached = _SHARD_CACHE.get(path)
    if cached is None:
        with open(path, 'rb') as fh:
            n = struct.unpack('<Q', fh.read(8))[0]
            hdr = _json.loads(fh.read(n))
        cached = (hdr, 8 + n)
        _SHARD_CACHE[path] = cached
    hdr, base = cached
    info = hdr[key]
    dtype = {'F8_E4M3': torch.float8_e4m3fn,
             'F8_E8M0': torch.float8_e8m0fnu,
             'BF16': torch.bfloat16}[info['dtype']]
    count = 1
    for d in info['shape']:
        count *= d
    buf = torch.empty(count, dtype=torch.uint8)
    with open(path, 'rb') as fh:
        fh.seek(base + info['data_offsets'][0])
        fh.readinto(memoryview(buf.numpy()))
    return buf.view(dtype).reshape(info['shape'])


def _ue8m0_to_rowk32(scale):
    """(n/32, k/32) ue8m0 block scale -> (n, k/32) FP32, exact.

    ue8m0 stores the exponent only: value = 2^(code-127), which is what
    `(code << 23) as float32` produces directly. Each 32-row block is then
    replicated to its 32 rows, because `_transform_shared` with recipe
    (1,1,32) wants one scale per ROW per K32 group.

    (1,128,128) would be cheaper but is WRONG here: the four 32x32 sub-blocks
    inside a 128x128 tile do not share a value (verified on layer 6).
    """
    from gc_moe.sm90_gc import _scale_codes
    codes = _scale_codes(scale)
    f32 = (codes << 23).view(torch.float32)
    return f32.repeat_interleave(32, 0).contiguous()


def _checkpoint_shared(layer_idx, device):
    """(w1, s1), (w2, s2) as role 4 needs them, from the checkpoint.

    The tensors must end up on `device`: `validate_call` checks
    `tensor.device == buffer.buffer.device`, while the raw checkpoint read is
    host memory.
    """
    import torch as _torch
    base = f'layers.{layer_idx}.ffn.shared_experts'
    # gate_up needs [2*I, H]: the checkpoint stores gate (w1) and up (w3)
    # separately, so they must be concatenated on dim 0 (scales too).
    gw = _ckpt_read(f'{base}.w1.weight')
    if gw is None:
        return None
    gs = _ckpt_read(f'{base}.w1.scale')
    uw = _ckpt_read(f'{base}.w3.weight')
    us = _ckpt_read(f'{base}.w3.scale')
    w2 = _ckpt_read(f'{base}.w2.weight')
    s2 = _ckpt_read(f'{base}.w2.scale')
    if None in (gs, uw, us, w2, s2):
        return None
    w1 = _torch.cat([gw, uw], dim=0)
    s1 = _torch.cat([gs, us], dim=0)
    return ((w1.to(device), _ue8m0_to_rowk32(s1).to(device)),
            (w2.to(device), _ue8m0_to_rowk32(s2).to(device)))


def _shared_weights_for_se(se):
    """Quantise + transform the shared MLP behind a SharedExperts instance."""
    if se is None or getattr(se, '_gc_real_layer', None) is None:
        return None
    return _shared_weights_for(se)


def _shared_weights_for(se):
    """Return transformed (w1, w2) for gc_moe role 4, or None if unavailable.

    `se` is the SharedExperts instance itself (not the MoERunner): the real
    bf16 MLP was moved to `_gc_real_layer` when `_layer` became a zero stub.
    """
    try:
        from gc_moe.sm90_gc import transform_weights_for_gc_moe_sm90
        mlp = getattr(se, '_gc_real_layer', None) or getattr(se, '_layer', None)
        if not getattr(se, '_r4_shapes', False):
            se._r4_shapes = True
            for nm, obj in (('gate_up_proj', mlp.gate_up_proj),
                            ('down_proj', mlp.down_proj)):
                w = getattr(obj, 'weight', None)
                print(f'[ROLE4-W] {nm}: cls={type(obj).__name__} '
                      f'w={None if w is None else (tuple(w.shape), str(w.dtype))}',
                      flush=True)
                for k, v in vars(obj).items():
                    if isinstance(v, torch.Tensor):
                        print(f'   .{k} = {tuple(v.shape)} {v.dtype}', flush=True)
                for k, v in vars(getattr(obj, 'base_layer', obj)).items() if hasattr(obj,'base_layer') else []:
                    if isinstance(v, torch.Tensor):
                        print(f'   [base].{k} = {tuple(v.shape)} {v.dtype}', flush=True)
                # parameters registered on the module tree
                for pn, pv in obj.named_parameters(recurse=True):
                    print(f'   param {pn} = {tuple(pv.shape)} {pv.dtype}', flush=True)
                for bn, bv in obj.named_buffers(recurse=True):
                    print(f'   buf {bn} = {tuple(bv.shape)} {bv.dtype}', flush=True)
        gu = mlp.gate_up_proj.weight
        dn = mlp.down_proj.weight
        # Cache key on the SHARD identity (stable after load).
        key = (id(gu), id(dn), gu.data_ptr(), dn.data_ptr())
        hit = _SHARED_W.get(key)
        if hit is not None:
            return hit
        # [ROLE4-A3] Use the checkpoint's own FP8 weights (full, unpadded).
        lay = getattr(mlp, '_gc_layer_idx', None)
        if lay is None:
            import re as _re
            nm = getattr(mlp, 'prefix', '') or getattr(mlp, 'name', '') or ''
            if not nm:
                for attr in ('gate_up_proj', 'down_proj'):
                    sub = getattr(mlp, attr, None)
                    nm = getattr(sub, 'prefix', '') or ''
                    if nm:
                        break
            m = _re.search(r'layers[.](\d+)', str(nm))
            if m:
                mlp._gc_layer_idx = int(m.group(1))
            lay = getattr(mlp, '_gc_layer_idx', None)
            if not getattr(mlp, '_gc_nm_dbg', False):
                mlp._gc_nm_dbg = True
                print(f'[ROLE4-NM] prefix={nm!r} lay={lay}', flush=True)
        if lay is not None:
            pair = _checkpoint_shared(lay, gu.device)
            if pair is not None:
                out = transform_weights_for_gc_moe_sm90(
                    pair[0], pair[1], shared_recipe=(1, 1, 32))
                # [ROLE4-CACHE] Do NOT clear: each of the 40 layers has its own
                # key, and clearing meant every layer re-read the checkpoint and
                # re-ran the transform on EVERY forward (measured 190684 CKPT
                # hits in 590 steps = 40 layers x 8 ranks per step, i.e. a
                # permanent cache miss -> 1252 ms/step instead of ~50).
                _SHARED_W[key] = out
                print(f'[ROLE4-CKPT] layer{lay} w1={tuple(pair[0][0].shape)} '
                      f'w2={tuple(pair[1][0].shape)} ok', flush=True)
                return out
        # Fallback (should not be reached): vLLM's weights are MXFP4-packed
        # int32 and TP-padded, so they cannot be used directly. Say so once
        # instead of failing silently.
        print('[ROLE4] checkpoint path unavailable; vLLM weights are '
              f'{tuple(gu.shape)}/{gu.dtype} (MXFP4-packed, unusable) -> role 4 OFF',
              flush=True)
        return None
    except Exception as exc:                    # noqa: BLE001
        import traceback
        print(f'[ROLE4-ERR] shared weights unavailable: {type(exc).__name__}: {exc}',
              flush=True)
        traceback.print_exc()
        return None

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
            # [T17-PERLAYER] `slot_table` is `slots_of` flattened as
            # [layers, experts, width] and EVERY layer has its own placement
            # (DeepInfer plans each layer separately). Using one row for all 40
            # layers pairs layer L's WEIGHTS with another layer's expert map --
            # the token then lands in a slot that does not hold its expert.
            # `_LAYER_ROW` is the same row index `_install_eplb` used for
            # `gather_slot_weights`.
            _lid = _LAYER_ROW.get(id(packed[0][0]))
            if _lid is None:
                raise RuntimeError(
                    'RR arm: no plan row for this packed pair (layer map '
                    'was rebuilt?)')
            cand = self._scratch(buf, 'cand', n * width, torch.int32)[:n * width]
            cand2 = cand.view(n, width)
            # each expert owns a CONTIGUOUS run of `width` entries, so the
            # index is layer_base + ids*width + arange(width).
            col = torch.arange(width, device=ids.device, dtype=torch.int64)
            base = int(_lid) * int(e['slot_experts']) * width
            gather_idx = ((ids.to(torch.int64).unsqueeze(-1) + 0) * width
                          + col + base)
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
            rr_pick(ids, e['count'][_lid], ep_rank=int(buf.rank_idx),
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
        # [ROLE4] When the buffer reserves shared experts, role 4 computes
        # them and `combine` adds `shared_y`; vLLM's own shared MLP is then
        # skipped (see _install_role4) so the term is added exactly once.
        _sw = None
        if int(os.environ.get('GC_MOE_SHARED_IN_GC', '0') or 0):
            if not hasattr(self, '_r4_dbg'):
                self._r4_dbg = True
                print(f'[ROLE4-DBG] expert_layer={getattr(self,"_role4_layer",None) is not None} '
                      f'registered={len(_EXPERT_REGISTRY)} '
                      f'runner_map={len(_SE_RUNNER)}', flush=True)
            # [ROLE4] Pair by CONSTRUCTION ORDER. vLLM builds the shared MLPs
            # and the gc_moe experts in the same layer order, and at
            # MoERunner.__init__ time the expert registry is still empty
            # (measured reg=0), so ordering is the only reliable key here.
            k = next((i for i, e in enumerate(_EXPERT_REGISTRY) if e is self), -1)
            _se = _SHARED_MODULES[k] if 0 <= k < len(_SHARED_MODULES) else None
            _sw = _shared_weights_for_se(_se)
            if k == 0 and not getattr(self, '_r4_dbg2', False):
                self._r4_dbg2 = True
                print(f'[ROLE4-DBG] idx={k} shared_mods={len(_SHARED_MODULES)} '
                      f'weights={"ok" if _sw is not None else "NONE"}', flush=True)
        if _sw is None:
            self._gc.sm90_fp8_fp4_gc_moe(
                output, packed[0], packed[1], buf,
                recipe=(1, 1, 32), activation='swiglu', activation_clamp=10.0,
                fast_math=True, num_tokens_bound=self._bound_for(buf, tokens),
                cumulative_local_expert_recv_stats=stats)
        else:
            self._gc.sm90_fp8_fp4_gc_moe(
                output, packed[0], packed[1], buf,
                recipe=(1, 1, 32), activation='swiglu', activation_clamp=10.0,
                fast_math=True, num_tokens_bound=self._bound_for(buf, tokens),
                shared_l1_weights=_sw[0], shared_l2_weights=_sw[1],
                shared_recipe=(1, 1, 32),
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
        # [T17-PERLAYER] This is the INITIAL placement (`initial_layout`),
        # installed before any traffic has been seen, so there is nothing to
        # plan per layer yet: every layer starts from the same layout. It is
        # broadcast to all 40 rows, which the buffers are now shaped for; the
        # real per-layer plans arrive in `rearrange_now` after warm-up.
        _nlay = int(buf.slot_logical.shape[0])
        plan_2d = [list(plan_list) for _ in range(_nlay)]
        plan, slots_of = sm90.install_eplb_plan(buf, plan_2d)
        # [T17-PERLAYER] slots_of is [layers, experts, MAX_REPLICAS]. The RR
        # gate needs copy counts PER LAYER: the placement differs per layer, so
        # a single max-across-layers count can hand a route a copy index that
        # does not exist for its own layer (that slot holds -1 and the token is
        # dropped). (The ECO arm does not use this path at all -- its copies
        # come from EcoMeta, which IS per-layer.)
        count = (slots_of >= 0).sum(-1).clamp_min(1)          # [layers, E]
        if _eco_enabled():
            # [ECO-SM90] Build the plan role's EcoMeta for this static
            # placement and write it into the buffer. It is host-side and
            # static: copies[E][32], phy2log[P], num_fixed[R].
            from gc_moe.eco.meta_sm90 import (
                install_into_buffer as _install_eco_meta)
            meta = _install_eco_meta(buf, plan_2d, ranks, experts)
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
        #
        # [T17-PERLAYER] `_gidx` walks the SAME global order as the
        # `components` list built below (modules in order, each module's layers
        # in LIFO pop order). That order is what (a) `_LAYER_OF`/the stats rows
        # and (b) `shuffler.move(target_layers[i])` are keyed by, so the plan
        # row, the stats row and the weight move all describe the same layer.
        _gidx = 0
        for m in modules:
            m._eplb = dict(plan=plan, slots=slots_of, per_rank=per_rank,
                           # [T17-PERLAYER] `slot_table` is the WHOLE
                           # [layers, experts, MAX_REPLICAS] stack flattened;
                           # `apply()` offsets into it with the row of the
                           # layer that owns the packed pair (see _LAYER_ROW).
                           slot_table=slots_of.reshape(-1).contiguous(),
                           slot_width=int(slots_of.shape[-1]),
                           slot_experts=int(slots_of.shape[1]),
                           count=count, stats=stats, layout=list(plan_list))
            m._packed = {}
            m._packed_layers = []
            # [T17-PERLAYER] `plan` is [layers, physical] and each layer must
            # gather ITS OWN row. The row MUST be the same index that (a) the
            # stats table attributes this layer's traffic to and (b) the later
            # `shuffler.move(target_layers[i])` uses -- and both of those are
            # driven by the same GLOBAL LIFO pop order (`components[]` order),
            # not by the layer's position in `_pending_eplb`. Using the
            # registration order here (the first port did) makes layer A's
            # weights carry the placement computed from layer B's traffic,
            # i.e. the rearrangement would be silently paired with the wrong
            # layer. `_gidx` reproduces `components[]` order exactly.
            while m._pending_eplb:
                layer, packed = m._pending_eplb.pop()
                _row = plan[_gidx]
                pair = sm90.gather_slot_weights(packed, buf, _row, group)
                _LAYER_ROW[id(pair[0][0])] = int(_gidx)
                _gidx += 1
                m._packed[id(layer)] = pair
                m._packed_layers.append((layer, pair))
                layer.w13_weight._gc_packed = pair
                m._last_packed = pair
                del packed, pair
            m._pending_eplb.clear()

        # WeightShuffler wants one component list per layer, holding the active
        # GEMM tensors for that layer. GC MoE's packed pair is
        #   ((w1 data, w1 sf, w1 secondary), (w2 data, w2 sf, w2 secondary))
        components, layer_keys, _names = [], [], []
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
                _names.append(str(getattr(layer, 'layer_name', '') or ''))
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
        # [T17-PERLAYER] The kernel resolves the EcoMeta row as `epoch % layers`
        # (one dispatch call per MoE layer, in FORWARD order). That only matches
        # this component order if the modules were constructed in forward order,
        # so print the extracted indices: they must be 0,1,2,...,n-1.
        import re as _re
        _idx = [int(m.group(1)) if (m := _re.search(r'layers\.(\d+)', n)) else -1
                for n in _names]
        print(f'[eplb] component order (model layer idx, first 8)={_idx[:8]} '
              f'last={_idx[-1:]} monotone_forward={_idx == sorted(_idx)}',
              flush=True)
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
               os.environ.get('GC_MOE_SM90_STREAM_M_BLOCKS') == '1',
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
                # [ROLE4] Reserve the shared-expert region so role 4 can run.
                num_shared_experts=int(os.environ.get('GC_MOE_SHARED_IN_GC', '0') or 0),
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



def install_role4():
    """[ROLE4] gc_moe's role 4 owns the shared expert; vLLM must not compute it.

    Why not patch `output` / drop `_shared_experts`:
      * `MoERunner.__init__` freezes `_forward_entry = _moe_forward_shared`,
        whose schema returns a TUPLE. Removing the object makes
        `_maybe_combine` return a bare Tensor ->
        "Object of type 'Tensor' is not an instance of 'sequence'".
      * `output` is a read-only property on the class; shadowing it breaks
        `SharedExperts.output` lookups.
    So we keep every vLLM code path intact and neutralise the *computation*:
    `SharedExperts._layer(shared_experts_input)` is replaced by a zero-output
    stub. vLLM still runs its normal bookkeeping (and its `assert`s pass),
    while the real shared term comes from gc_moe's combine kernel, which adds
    `shared_y`. Anything non-zero from vLLM would be added twice.
    """
    if os.environ.get('GC_MOE_SHARED_IN_GC', '0') != '1':
        return False
    try:
        from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
            SharedExperts as _SE)
    except Exception as exc:                     # noqa: BLE001
        print(f'[ROLE4] cannot import SharedExperts: {exc}', flush=True)
        return False
    if getattr(_SE, '_gc_role4_patched', False):
        return True

    class _ZeroShared(torch.nn.Module):
        """Stand-in for the bf16 shared MLP: returns zeros, costs nothing.

        It keeps the real MLP so gc_moe can quantise its weights once.
        """

        def __init__(self, real):
            super().__init__()
            self._real = real

        def forward(self, x):
            return torch.zeros_like(x)

    _orig_init = _SE.__init__

    def _init(self, layer, *a, **kw):
        _orig_init(self, layer, *a, **kw)
        self._gc_real_layer = layer          # real bf16 MLP (weights)
        self._layer = _ZeroShared(layer)     # neutralised compute
        # Append in construction order; gc_moe experts are constructed in the
        # same layer order, so index k here <-> index k in _EXPERT_REGISTRY.
        _SHARED_MODULES.append(self)

    _SE.__init__ = _init

    try:
        from vllm.model_executor.layers.fused_moe.runner import moe_runner as _mr
        _orig_rinit = _mr.MoERunner.__init__

        _r4_n = {'i': -1}

        def _rinit(self, *a, **kw):
            _orig_rinit(self, *a, **kw)
            se = getattr(self, '_shared_experts', None)
            if se is not None:
                _r4_n['i'] += 1
                k = _r4_n['i']
                _SE_RUNNER[id(se)] = self
                # Pair by CONSTRUCTION ORDER: the k-th MoERunner owns the k-th
                # FusedMoEExpertsModular, because `routed_experts` is built
                # before the runner and appended to _EXPERT_REGISTRY there.
                # Preferred pairing: THIS runner's own routed_experts object
                # IS the gc_moe expert (a FusedMoEExpertsModular subclass).
                re_obj = getattr(self, 'routed_experts', None)
                # Compare by identity: nn.Module implements __eq__ and a plain
                # `in` test could match a different layer.
                hit = None
                if re_obj is not None:
                    for exc in _EXPERT_REGISTRY:
                        if exc is re_obj:
                            hit = exc
                            break
                if hit is not None:
                    hit._role4_layer = self
                elif k < len(_EXPERT_REGISTRY):
                    _EXPERT_REGISTRY[k]._role4_layer = self
                print(f'[ROLE4-DBG2] runner#{k} reg={len(_EXPERT_REGISTRY)} '
                      f'identity={hit is not None}', flush=True)

        _mr.MoERunner.__init__ = _rinit
    except Exception as exc:                     # noqa: BLE001
        print(f'[ROLE4] cannot hook MoERunner.__init__: {exc}', flush=True)

    _SE._gc_role4_patched = True
    print('[ROLE4] vLLM shared GEMM neutralised (zero stub); gc_moe role 4 owns it',
          flush=True)
    return True


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

    # [RANDOM-ARM-FIX] `GC_MOE_SLOT_PLACEMENT=random` is the CONTROL arm: it must
    # keep the random static placement for the whole run. `_install_eplb` gives
    # it a random layout, but this function used to re-plan anyway, silently
    # replacing it with a DeepInfer placement after warm-up. Measured on BS32:
    #   random arm  rearranged ... max_copies=83  mean_replicated/layer=46.0
    #   eplb   arm  rearranged ... max_copies=85  mean_replicated/layer=46.5
    # i.e. the two arms were the SAME placement (IR 1.2834 vs 1.2953, a 1%
    # difference entirely from measurement noise), so the control was void.
    # The random arm still needs the EPLB machinery (56 slots, slot routing),
    # it just must not re-plan; returning True keeps it recorded as an
    # EPLB-enabled run with a frozen random placement.
    if _slot_placement() == 'random':
        import torch.distributed as _d
        if _d.get_rank() == 0:
            print('[eplb] rearrangement SKIPPED (random placement is static)',
                  flush=True)
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

    # [T17-DUMP] Save the RAW per-(layer, slot) counters so the "per-layer vs
    # merged placement" question can be answered on the CPU without guessing a
    # traffic model. Purely additive: the plan below is unchanged.
    _dump = os.environ.get('GC_EPLB_STATS_DUMP', '')
    if _dump:
        import numpy as _np
        _arr = stats.detach().to('cpu').numpy()
        _gathered = [None] * ranks
        dist.all_gather_object(_gathered, _arr, group=dist.group.WORLD)
        if buf.rank_idx == 0:
            _np.savez(_dump, stats=_np.stack(_gathered),
                      layout=_np.asarray(layout, _np.int64), per_rank=per_rank,
                      ranks=ranks, experts=experts)
            print(f'[t17-dump] stats saved {_dump} '
                  f'shape={_np.stack(_gathered).shape}', flush=True)

    # [T17-PERLAYER] DeepInfer plans EVERY layer separately. Its API is
    #   rebalance_experts_global(tokens_per_expert[steps, layers, experts], ...)
    #     -> physical_to_logical_map[layers, physical]
    # and everything downstream in that function keeps the layer axis:
    #   weights = tpe.sum(dim=0) / logical_count     -> [layers, experts]
    #   loads / peaks / occupancy = zeros([layers, ...])   <- reset PER LAYER
    #   redundancy_indices = values.sum(0).argmin(-1)      -> [layers]
    # so layer L can give expert e three copies while layer L' gives it one.
    #
    # This port used to collapse the layers first (`stats.sum(0)` ->
    # `hist.view(1, 1, experts)`), which produced ONE placement shared by all
    # 40 layers. Measured on the real counters ([8 ranks, 40 layers, 56 slots]):
    #   merged placement   IR mean 1.2803, worst layer 1.9484
    #   per-layer          IR mean 1.1324, worst layer 1.2283   (+11.5%)
    # i.e. the merged plan left whole layers with their hot experts piled onto
    # one rank -- exactly what a single average cannot avoid.
    per_layer = os.environ.get('GC_MOE_ECOPLB_PER_LAYER', '1') == '1'
    n_layers = int(stats.shape[0])
    if per_layer:
        # [layers, experts], each layer reduced over RANKS only.
        demand = torch.stack([
            demand_from_stats(stats[l:l + 1], layout, buf.rank_idx,
                              per_rank, experts)
            for l in range(n_layers)])
        dist.all_reduce(demand, group=dist.group.WORLD)
        hist = demand.view(1, n_layers, experts)
    else:
        demand = demand_from_stats(stats, layout, buf.rank_idx, per_rank, experts)
        dist.all_reduce(demand, group=dist.group.WORLD)
        hist = demand.view(1, 1, experts)

    # --- 2. plan on rank 0, broadcast -------------------------------------
    hist_layers = hist.shape[1]
    if buf.rank_idx == 0:
        new_plan, _, _ = rebalance_experts_global(hist, buf.physical, per_rank)
        new_plan = new_plan.contiguous()
    else:
        new_plan = torch.empty((hist_layers, buf.physical), dtype=torch.int32,
                               device=buf.buffer.device)
    dist.broadcast(new_plan, src=0, group=dist.group.WORLD)
    # [T17-PERLAYER] [layers, physical]; for the merged arm this is the single
    # row broadcast to every layer so the rest of the code is shared.
    target_layers = new_plan.cpu().tolist()
    target = target_layers[0]

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
        # [T17-PERLAYER] The single-copy-first permutation depends on which
        # experts got copies IN THAT LAYER, so it must be applied per layer.
        for _l in range(len(target_layers)):
            target_layers[_l], _slot_order = _reorder_scf(
                list(target_layers[_l]), ranks)
        _ECO_SLOT_ORDER['order'] = _slot_order
        target = target_layers[0]

    # --- 4. move weights between GPUs -------------------------------------
    # [T17-PERLAYER] Each layer moves to ITS OWN placement; the shuffler takes
    # one target per component (== one per layer), so no new structure is
    # needed -- layer L now serves a different set of experts than layer L'.
    shuffler = m._eplb_shuffler
    n_layers = len(shuffler.components)
    if len(target_layers) == 1:
        move_targets = [list(target_layers[0]) for _ in range(n_layers)]
    elif len(target_layers) == n_layers:
        move_targets = [list(t) for t in target_layers]
    else:
        raise RuntimeError(f'{len(target_layers)} placements but '
                           f'{n_layers} layers')

    shuffler.move([list(layout) for _ in range(n_layers)], move_targets)

    # --- 5. repoint routing ----------------------------------------------
    # [T17-PERLAYER] slot_logical / expert_slots also become per-layer.
    new_t, slots_of = m._gc.sm90_gc.install_eplb_plan(buf, target_layers)
    # [T17-PERLAYER] copy COUNTS are per layer: the placement differs per layer,
    # so a max-across-layers count would let the RR gate pick a copy index that
    # does not exist in that layer (slots_of == -1 there, token dropped).
    count = (slots_of >= 0).sum(-1).clamp_min(1)
    if _eco_enabled():
        from gc_moe.eco.meta_sm90 import (
            install_into_buffer as _install_eco_meta)
        _install_eco_meta(buf, target_layers, ranks, experts)
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
    # [T17-PERLAYER] Republish the WHOLE per-layer table, not layer 0's row:
    # the RR gate indexes it by the layer that owns the packed pair.
    for mod in live:
        mod._eplb['slot_table'].copy_(slots_of.reshape(-1))
        mod._eplb['count'].copy_(count)
        mod._eplb['layout'] = list(target)
        mod._eplb.update(plan=new_t, slots=slots_of)
        with torch.inference_mode():
            mod._eplb['stats'].zero_()
    if buf.rank_idx == 0:
        _tg = torch.as_tensor(target_layers)
        copies = torch.bincount(_tg.reshape(-1), minlength=experts)
        _per = [(torch.bincount(_tg[l], minlength=experts) > 1).sum()
                for l in range(len(target_layers))]
        print(f'[eplb] rearranged: slots={buf.physical} per_rank={per_rank} '
              f'layers={len(target_layers)} '
              f'max_copies(all-layers)={int(copies.max())} '
              f'max_replicated/layer={int(max(_per))} '
              f'mean_replicated/layer={sum(int(x) for x in _per) / len(_per):.1f}',
              flush=True)
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


def irstep_begin() -> None:
    """# [IR-STEP] Mark the start of the measured window.

    The dispatch role records `ir_step[layer][step]` itself; the only thing it
    needs from the host is WHERE the window starts, because `step` is derived
    as `(epoch - epoch0) // layers`. Writing `epoch0 = eco_tl_step` here (a
    plain device store, no sync) hands it that anchor.

    The epoch is already published by the dispatch role for the forward that
    is in flight, so this must run BETWEEN forwards (it does: the driver calls
    it right before the measured loop, with the queue drained).
    """
    import torch

    live = [m for m in _EXPERT_REGISTRY if getattr(m, '_buffer', None) is not None]
    if not live:
        return False
    buf = live[0]._get_buffer()
    mark = getattr(buf, 'ir_step_mark', None)
    if mark is None:
        return False
    torch.cuda.synchronize()
    # `header[0]` is the dispatch epoch, and it is the SAME value the dispatch
    # role passes to `_ir_step_write` -- unlike `eco_tl_step`, which only the
    # ECO arm publishes (the vanilla arm would anchor at 0).
    with torch.inference_mode():
        # [IRSTEP-ANCHOR-FIX] `header[0]` is written by the dispatch KERNEL, so
        # under CUDA-graph replay the HOST copy read here is stale -- it lags by
        # however many calls were captured. Measured on ALIGNEDQ: mark[0] landed
        # on 15001 while `pull_tl` (which reads `eco_tl_step` inside the kernel)
        # covered 15225..15480, i.e. the two tables were keyed in DIFFERENT epoch
        # spaces and could not be joined at all.
        # `eco_tl_step` is republished by dispatch at the TOP OF EVERY CALL, so
        # it names the call that is about to run. Read it BEFORE writing mark[0].
        _step = getattr(buf, 'eco_tl_step', None)
        _epoch_now = (int(_step[0].item()) if _step is not None
                      else int(buf.header[0].item()))
        mark[0] = _epoch_now                    # epoch0 (same space as pull_tl)
        mark[1] = 0
    # [ECO-FINE-FLOOR] Anchor the FINE event log to the measured window too.
    #
    # `fine_ctr` is monotonic and `fine_tl` holds 65536 records, so with
    # `GC_MOE_FINE_MOD=0` the prefill calls alone filled it: the log stopped at
    # epoch 4 (four prefill layers) and the entire 1000-step measured window
    # recorded NOTHING. The plot built from that data showed 12996 "M blocks"
    # per rank where the real value is ~128, and a 6.7 ms "pull done" that was
    # really a prefill call.
    #
    # Publishing the window's first epoch lets the kernel drop everything below
    # it, and clearing the counter hands the whole 2 MB to the measured window.
    # Both are plain device stores done with the queue drained, so this costs
    # nothing measurable (it runs between forwards, never per step).
    floor = getattr(buf, 'fine_floor', None)
    ctr = getattr(buf, 'fine_ctr', None)
    if floor is not None:
        with torch.inference_mode():
            floor[0] = torch.tensor(_epoch_now, dtype=floor.dtype,
                                    device=floor.device)
            if ctr is not None:
                ctr[0] = 0
        # `eco_tl_step` names the call that JUST finished, so the window's first
        # layer is floor+1. Anchor one layer early and let the kernel's own
        # [floor, floor+FINE_EPOCHS) window cover it; FINE_EPOCHS must be >= 2.
        print(f'[fine-floor] anchored at epoch {_epoch_now} '
              f'(window starts ~{_epoch_now + 1})', flush=True)
    return True


def irstep_end() -> dict:
    """# [IR-STEP] Hand back this rank's [layer][step] table.

    One bulk device->host copy at the end of the window; no per-step RPC. The
    reduction to an IR happens on the driver side, across ranks.
    """
    import os as _os
    import torch

    live = [m for m in _EXPERT_REGISTRY if getattr(m, '_buffer', None) is not None]
    if not live:
        return {'ok': False, 'reason': 'no buffer'}
    buf = live[0]._get_buffer()
    tbl = getattr(buf, 'ir_step', None)
    mb_tbl = getattr(buf, 'ir_step_mb', None)
    ae_tbl = getattr(buf, 'ir_step_ae', None)
    ep_tbl = getattr(buf, 'ir_step_epoch', None)
    mark = getattr(buf, 'ir_step_mark', None)
    if tbl is None:
        return {'ok': False, 'reason': 'no ir_step table'}
    torch.cuda.synchronize()
    layers = int(_os.environ.get('GC_MOE_LAYERS', '40'))
    steps = int(_os.environ.get('GC_IRSTEP_STEPS', '1024'))
    with torch.inference_mode():
        rows = tbl[:layers * steps].view(layers, steps).cpu().tolist()
        rows_mb = (mb_tbl[:layers * steps].view(layers, steps).cpu().tolist()
                   if mb_tbl is not None else None)
        rows_ae = (ae_tbl[:layers * steps].view(layers, steps).cpu().tolist()
                   if ae_tbl is not None else None)
        # [IR-STEP-EPOCH] kernel-written epoch per cell; the only exact key for
        # joining this table with pull_tl.
        rows_ep = (ep_tbl[:layers * steps].view(layers, steps).cpu().tolist()
                   if ep_tbl is not None else None)
        e0 = int(mark[0].item()) if mark is not None else 0
    return {'ok': True, 'rank': int(buf.rank_idx), 'layers': layers,
            'steps': steps, 'epoch0': e0, 'rows': rows,
            'rows_mb': rows_mb, 'rows_ae': rows_ae, 'rows_epoch': rows_ep}


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

    if getattr(Worker, '_gc_moe_irstep_begin', None) is None:

        def gc_moe_irstep_begin(self) -> bool:
            """# [IR-STEP] Anchor the (layer, step) table at this window."""
            return irstep_begin()

        Worker._gc_moe_irstep_begin = gc_moe_irstep_begin

    if getattr(Worker, '_gc_moe_irstep_end', None) is None:

        def gc_moe_irstep_end(self) -> dict:
            """# [IR-STEP] Dump this rank's [layer][step] table."""
            return irstep_end()

        Worker._gc_moe_irstep_end = gc_moe_irstep_end

    if getattr(Worker, '_gc_moe_ir_end', None) is None:

        def gc_moe_ir_end(self) -> dict:
            """# [SME-IR] Stop recording and return this rank's imbalance stats."""
            return ir_end()

        Worker._gc_moe_ir_end = gc_moe_ir_end

    if getattr(Worker, '_gc_moe_eco_trace_dump', None) is None:

        def gc_moe_eco_trace_dump(self, include_fine=True) -> dict:
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
            # [SM-REUSE-DIAG] `header[32]` counts role-6 CTA entries; a non-zero
            # value proves the helper kernel was actually launched.
            try:
                torch.cuda.synchronize()
                _h = buf.buffer.view(torch.int32)
                out['role6_entries'] = int(_h[32].item())
                out['role6_l1_ctr_sum'] = int(_h[33].item())
                out['role6_pub_slots_sum'] = int(_h[34].item())
            except Exception:
                out['role6_entries'] = -1
            tr = getattr(buf, 'eco_trace', None)
            if tr is not None:
                torch.cuda.synchronize()
                out['slots'] = [int(v) for v in tr.cpu().tolist()]
            ptl = getattr(buf, 'pull_tl', None)
            if ptl is not None:
                # [RANK-TS] 14 slots per row (was 10); slots 10..13 hold the
                # per-rank dispatch/compute/combine stamps. Derive the width
                # from the buffer so the two can never disagree again.
                _n = int(ptl.numel()) // 256
                out['pull_rows'] = ptl.view(256, _n).cpu().tolist()
            ftl = getattr(buf, 'fine_tl', None)
            if ftl is not None and include_fine:
                # [ABSTRACE] 10 words per record now (ntile / phase added for
                # the absolute-time figure). The capacity matches the layout
                # reserve (GC_MOE_FINE_RECS, default 262144).
                _recs = int(os.environ.get('GC_MOE_FINE_RECS', '262144'))
                out['fine_rows'] = ftl.view(_recs, 10).cpu().tolist()
                out['fine_recs'] = _recs
                # [ABSTRACE] the ring counter tells the analyser how many
                # records were written in total, so it can sort the wrapped
                # rows back into emission order.
                _ctr = getattr(buf, 'fine_ctr', None)
                if _ctr is not None:
                    torch.cuda.synchronize()
                    out['fine_count'] = int(_ctr[0].item())
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

    def profile_buffer():
        buffers = {id(m._buffer): m._buffer for m in _EXPERT_REGISTRY
                   if getattr(m, '_buffer', None) is not None}
        if len(buffers) != 1:
            raise RuntimeError(f'aligned profiling needs one shared buffer, got {len(buffers)}')
        return next(iter(buffers.values()))

    def gc_moe_profile_snapshot(self, first_epoch=None):
        """Read the live workspace after draining this worker's GPU."""
        torch.cuda.synchronize()
        buf = profile_buffer()
        next_epoch = int(buf.header[0].item())
        out = dict(rank=int(buf.rank_idx), next_epoch=next_epoch, topk=int(buf.num_topk))
        if first_epoch is not None:
            advanced = next_epoch - first_epoch
            if not 1 <= advanced <= buf.pull_tl.shape[0] // 32:
                return dict(out, error=f'epoch advance {advanced} outside the trace ring capacity')
            stamps = buf.pull_tl.view(256, 32).cpu().tolist()
            counts = buf.profile_slot_counts.cpu().tolist()
            out['calls'] = []
            for epoch in range(first_epoch, next_epoch):
                slot = epoch % 256
                stamp = stamps[slot]
                if stamp[0] != epoch:
                    out.setdefault('missing_epochs', []).append(epoch)
                    continue
                out['calls'].append(dict(epoch=epoch, stamps=stamp, owned=counts[slot]))
        return out

    Worker._gc_moe_profile_snapshot = gc_moe_profile_snapshot

    def gc_moe_profile_fine_dump(self, epochs):
        """Export just the requested epochs from the armed fine-trace window."""
        torch.cuda.synchronize()
        buf = profile_buffer()
        count = int(buf.fine_ctr[0].item())
        ring = buf.fine_tl.view(-1, 10)
        out = dict(rank=int(buf.rank_idx), count=count, capacity=len(ring),
                   origin_us=int(buf.fine_t0[0].item()) & 0xFFFFFFFF,
                   hidden=int(buf.hidden), intermediate=int(buf.intermediate_hidden))
        if not 0 < count <= len(ring):
            return dict(out, error='fine ring is empty or overwritten')
        rows = ring[:count]
        out['epochs'] = {
            str(epoch): rows[rows[:, 8] == epoch].cpu().tolist() for epoch in epochs
        }
        return out

    Worker._gc_moe_profile_fine_dump = gc_moe_profile_fine_dump

    def gc_moe_profile_park_fine(self):
        """Point the one-epoch write window at an already completed dispatch."""
        buf = profile_buffer()
        with torch.inference_mode():
            buf.fine_floor[0].copy_(buf.eco_tl_step[0])
        torch.cuda.synchronize()
        return int(buf.fine_ctr[0].item())

    Worker._gc_moe_profile_park_fine = gc_moe_profile_park_fine

    if getattr(Worker, '_gc_moe_prof_start', None) is None:

        def gc_moe_prof_start(self, out_dir: str, capture_id=None, fine=False) -> bool:
            """[PROF] Begin a torch profile on THIS worker's stream.

            Under ENFORCE_EAGER=1 gc_moe launches its four roles as separate
            kernels (no child graph), so a per-worker profile sees them
            individually: role1 dispatch (routing/counts + token pull), role4
            shared expert, role2 routed GEMM and role3 combine.

            Each worker writes its own chrome trace, so the analyser can plot
            one decode step with the 8 ranks on the y axis.
            """
            import os
            import torch
            from torch.profiler import profile, ProfilerActivity

            r = 0
            try:
                import torch.distributed as dist
                if dist.is_initialized():
                    r = dist.get_rank()
            except Exception:                       # noqa: BLE001
                pass
            if fine:
                torch.cuda.synchronize()
                buf = profile_buffer()
                # Any completed dispatch supplies a clock origin on this GPU.
                # It need not be the actual pre-step epoch; resolve_epochs
                # independently matches the captured forward.
                epoch = int(buf.eco_tl_step[0].item())
                row = buf.pull_tl.view(256, 32)[epoch % 256]
                if epoch <= 0 or int(row[0].item()) != epoch or int(row[10].item()) == 0:
                    raise RuntimeError(f'rank {r}: no completed dispatch for a fine-trace origin')
                with torch.inference_mode():
                    buf.fine_t0[0].copy_(row[10])
                    buf.fine_t0[1] = 1
                    buf.fine_ctr.zero_()
                    buf.fine_spin.zero_()
                    buf.fine_floor.zero_()
                torch.cuda.synchronize()
            os.makedirs(out_dir, exist_ok=True)
            self._gc_prof_path = os.path.join(out_dir, f'rank{r}.json')
            self._gc_prof = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=False, with_stack=False, profile_memory=False)
            if capture_id is not None:
                import json
                self._gc_prof.preset_metadata_json(
                    'gcmoe_capture', json.dumps(dict(id=capture_id, rank=r)))
            self._gc_prof.__enter__()
            torch.cuda.synchronize()
            return True

        Worker._gc_moe_prof_start = gc_moe_prof_start

    if getattr(Worker, '_gc_moe_prof_stop', None) is None:

        def gc_moe_prof_stop(self) -> dict:
            """[PROF] Stop this worker's profile and export its chrome trace."""
            import torch
            torch.cuda.synchronize()
            out = {'ok': True}
            try:
                p = getattr(self, '_gc_prof', None)
                if p is not None:
                    p.__exit__(None, None, None)
                path = getattr(self, '_gc_prof_path', None)
                if path:
                    p.export_chrome_trace(path)
                    out['path'] = path
            except Exception as exc:                # noqa: BLE001
                out = {'ok': False, 'error': str(exc)[:300]}
            return out

        Worker._gc_moe_prof_stop = gc_moe_prof_stop

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

# [ROLE4] gc_moe computes the shared expert itself (role 4); vLLM must not.
# MUST be at module scope: inside `register()` it only ran when that function
# was called, and the vLLM classes were already imported by then.
install_role4()
