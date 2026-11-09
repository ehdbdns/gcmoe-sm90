"""SM90-only launch contract; no imports from the SM100 validation/runtime path."""
from dataclasses import dataclass
import math
import os
import torch
from ..sm90_gc import SM90GCSymmBuffer


@dataclass(frozen=True)
class ValidatedCall:
    ranks: int
    experts: int
    capacity: int
    topk: int
    hidden: int
    intermediate: int
    tokens: int
    bound: int
    clamp: float
    fast_math: bool
    shared_experts: int = 0
    # EPLB physical expert slots (0 == no replication == experts).
    physical: int = 0
    # [ECO-SM90] online split plan (role 5). `eco_logical_experts > 0` enables it.
    eco_plan: bool = False
    eco_logical_experts: int = 0
    eco_max_copies: int = 4
    eco_wb: int = 0
    eco_wa: int = 0
    eco_rounds: int = 1
    eco_cold_rounds: int = 6
    # [ECO-MEAS] Cost-decomposition gates. Both are COMPILE-TIME constants, so
    # turning them off leaves no branch behind. They exist only to attribute
    # the ECO arm's overhead; neither changes the algorithm when false.
    eco_null_plan: bool = False      # keep wait/static/D_e, skip cold/warm/auction/publish
    eco_no_auction: bool = False     # keep cold/warm, skip the auction rounds
    # [SM-REUSE] Add role 6: a compute helper that lives in the DISPATCH green
    # context and starts claiming GEMM tasks once this call's pull is done.
    sm_reuse: int = 0
    # [SM-REUSE] SM count of the DISPATCH green partition (role 6 inherits it).
    dispatch_partition_sms: int = 0


def validate_call(y, w1, w2, buffer, stats, recipe, activation, clamp,
                        fast_math, num_tokens_bound, shared_l1_weights=None,
                        shared_l2_weights=None, shared_recipe=None):
    if not isinstance(buffer, SM90GCSymmBuffer) or buffer.buffer is None:
        raise ValueError('expected live SM90GCSymmBuffer')
    device = buffer.buffer.device
    if (y.device != device or y.dtype != torch.bfloat16 or y.ndim != 2
            or not y.is_contiguous() or y.data_ptr() % 16 or y.shape[1] != buffer.hidden
            or not 0 <= y.shape[0] <= buffer.num_max_tokens_per_rank
            or torch.cuda.current_device() != device.index):
        raise ValueError('invalid SM90 y device, shape or dtype')
    if tuple(recipe) != (1,1,32) or activation != 'swiglu':
        raise ValueError('SM90 GC requires recipe (1,1,32) and swiglu')
    clamp = math.inf if clamp is None else float(clamp)
    if math.isnan(clamp) or clamp < 0:
        raise ValueError('activation_clamp must be nonnegative')
    bound = 0 if num_tokens_bound is None else int(num_tokens_bound)
    if bound < 0 or bound > buffer.num_max_tokens_per_rank or (bound and bound < y.shape[0]):
        raise ValueError('num_tokens_bound must cover every rank and fit capacity')
    local = buffer.num_experts // buffer.group.size()
    # Under EPLB a rank holds `phys_local` slots, so weight tensors carry that
    # many expert rows instead of `local`.
    phys_local = (buffer.physical or buffer.num_experts) // buffer.group.size()
    for weight, n, k in ((w1, 2 * buffer.intermediate_hidden, buffer.hidden),
                         (w2, buffer.hidden, buffer.intermediate_hidden)):
        shapes = ((phys_local, n // 256, k // 128, 4, 256, 16),
                  (phys_local, n // 256, k // 128, 256, 4), (phys_local,))
        if not isinstance(weight, (tuple, list)) or len(weight) != 3:
            raise ValueError('SM90 weights must contain independent data/scales/secondary')
        for tensor, shape, dtype in zip(weight, shapes, (torch.int8, torch.uint8, torch.float32)):
            if (not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape
                    or tensor.dtype != dtype or tensor.device != device
                    or not tensor.is_contiguous() or tensor.data_ptr() % 16):
                raise ValueError('invalid SM90 weight tile layout, dtype or device')
    if stats is not None and (stats.device != device or stats.dtype != torch.int32
                              or stats.shape != (phys_local,) or not stats.is_contiguous()):
        raise ValueError('stats must be contiguous int32 [local_experts]')
    if (shared_l1_weights is None) != (shared_l2_weights is None):
        raise ValueError('both shared weight pairs must be supplied together')
    shared_experts = 0
    if shared_l1_weights is not None:
        for weight in (shared_l1_weights, shared_l2_weights):
            if (not isinstance(weight, (tuple, list)) or len(weight) != 2
                    or not all(isinstance(tensor, torch.Tensor) for tensor in weight)):
                raise ValueError('shared weights must be two-tensor data/scales pairs')
        if shared_recipe is not None and tuple(shared_recipe) not in ((1, 1, 32), (1, 128, 128)):
            raise ValueError('unsupported shared recipe')
        shared_i = shared_l2_weights[0].shape[1] * 128
        if shared_i <= 0 or shared_i % buffer.intermediate_hidden:
            raise ValueError('shared intermediate must contain whole shared experts')
        shared_experts = shared_i // buffer.intermediate_hidden
        if shared_experts > buffer.num_shared_experts:
            raise ValueError('workspace does not cover shared experts')
        for weight, n, k in ((shared_l1_weights, 2 * shared_i, buffer.hidden),
                             (shared_l2_weights, buffer.hidden, shared_i)):
            for tensor, shape, dtype in zip(weight,
                    ((n // 256, k // 128, 4, 4, 128, 16), (n // 256, k // 128, 4, 64, 4)),
                    (torch.uint8, torch.float32)):
                if (tuple(tensor.shape) != shape or tensor.dtype != dtype or tensor.device != device
                        or not tensor.is_contiguous() or tensor.data_ptr() % 16):
                    raise ValueError('invalid shared FP8 tile/scales layout')
    return ValidatedCall(buffer.group.size(), buffer.num_experts,
                          buffer.num_max_tokens_per_rank, buffer.num_topk,
                          buffer.hidden, buffer.intermediate_hidden,
                          y.shape[0], bound, clamp, bool(fast_math), shared_experts,
                          int(buffer.physical or 0),
                          bool(getattr(buffer, 'eco', False)),
                          int(getattr(buffer, 'eco_logical', 0) or 0),
                          # [ECO-SM90] Must be >= the largest copy count the
                          # placement produces. With 25% redundancy the hot
                          # expert reached 5-6 copies, so the old default of 4
                          # silently dropped copies (their tokens were pulled by
                          # nobody). EcoMeta holds 32 int16 per expert.
                          int(os.environ.get('GC_MOE_ECO_MAX_COPIES', '8')),
                          int(os.environ.get('GC_MOE_ECO_WB', '0')),
                          int(os.environ.get('GC_MOE_ECO_WA', '0')),
                          int(os.environ.get('GC_MOE_ECO_ROUNDS', '1')),
                          int(os.environ.get('GC_MOE_ECO_COLD_ROUNDS', '6')),
                          os.environ.get('GC_MOE_ECO_NULL_PLAN') == '1',
                          os.environ.get('GC_MOE_ECO_NO_AUCTION') == '1',
                          int(os.environ.get('GC_MOE_SM90_SM_REUSE', '0') or 0),
                          int(os.environ.get('GC_MOE_SM90_DISPATCH_PARTITION_SMS', '0') or 0))


def sm90_fp8_fp4_gc_moe(y, w1, w2, buffer, stats, recipe, activation, clamp,
                        fast_math, num_tokens_bound, shared_l1_weights=None,
                        shared_l2_weights=None, shared_recipe=None):
    config = validate_call(y, w1, w2, buffer, stats, recipe, activation, clamp,
                           fast_math, num_tokens_bound, shared_l1_weights,
                           shared_l2_weights, shared_recipe)
    from ..jit_kernels.impls.sm90_fp8_fp4_gc_moe import launch_sm90_gc_moe
    launch_sm90_gc_moe(config, y, w1, w2, buffer, stats,
                       shared_l1_weights, shared_l2_weights)
