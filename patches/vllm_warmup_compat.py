"""Unblock vLLM 0.31 warmup on machines where Triton cannot parse a kernel.

``vllm/model_executor/warmup/kernel_warmup.py`` imports the MiniMax-M3 MSA
warmup module unconditionally. That import reaches
``vllm/models/minimax_m3/common/ops/index_topk.py``, whose ``@triton.heuristics``
decorator makes Triton 3.7.1 fail while reading the source, so *every* model --
DeepSeek-V4.1 included -- dies during engine bring-up with
``AttributeError: 'NoneType' object has no attribute 'start'``.

The MiniMax-M3 warmup itself is a no-op for any other model: it looks for a
``MiniMaxM3SparseAttention`` module in the model and returns when there is
none. So it is safe to make that import optional rather than patching Triton.

Install before the engine boots (``install()`` is idempotent).
"""
import sys

_STUB = '''"""Stand-in for vllm.model_executor.warmup.minimax_m3_msa_warmup.

The real module imports the MiniMax-M3 sparse-attention kernels, which the
installed Triton cannot parse. For non-MiniMax models the warmup is a no-op,
so this stub keeps engine bring-up working on DeepSeek-V4.1.
"""


def minimax_m3_msa_warmup(worker):
    return None
'''

_MODULES = (
    'vllm.models.minimax_m3.common.ops.index_topk',
    'vllm.models.minimax_m3.common.ops',
    'vllm.models.minimax_m3.common.indexer',
    'vllm.models.minimax_m3.nvidia.model',
    'vllm.models.minimax_m3',
    'vllm.model_executor.warmup.minimax_m3_msa_warmup',
)


def install(verbose: bool = False) -> None:
    if getattr(sys, '_vllm_warmup_compat', False):
        return
    stub = type(sys)('vllm.model_executor.warmup.minimax_m3_msa_warmup')
    stub.__doc__ = _STUB
    exec(compile(_STUB, '<minimax_m3_msa_warmup stub>', 'exec'), stub.__dict__)
    stub.__file__ = '<stub>'
    sys.modules['vllm.model_executor.warmup.minimax_m3_msa_warmup'] = stub
    sys._vllm_warmup_compat = True
    if verbose:
        print('[compat] minimax_m3 warmup stubbed')
