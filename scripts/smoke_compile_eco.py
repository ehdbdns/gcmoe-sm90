"""Compile the SM90 role cubins (including the ECO plan role 5) without weights.

This is the fastest possible GPU-side smoke test: it exercises the CuTe DSL
front end and ptxas for every role, but allocates no model memory and runs no
inference. Use it before any real vLLM run.
"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'source'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'integration'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'patches'))

import gc_ptxas_override  # noqa: F401
import torch  # noqa: E402  (initialises the primary CUDA context)
from gc_moe.apis.sm90_gc_moe import ValidatedCall
from gc_moe.layout.sm90_gc_moe import SM90GCBufferLayout
from gc_moe.runtime.sm90_gc_moe import get_context
from gc_moe.jit_kernels.heuristics.sm90_gc_moe import get_sm90_role_config
from gc_moe.jit_kernels.impls.sm90_fp8_fp4_gc_moe import _role_specs, _compile_roles, _compiler_identity


def main():
    eco = os.environ.get('SMOKE_ECO', '1') == '1'
    E, H, I, TOPK, CAP = 384, 5120, 2304, 6, 2048
    ranks = 8
    physical = 424 if eco else 0
    # trace=True so the ECO roles that touch the timeline can be compiled
    layout = SM90GCBufferLayout(ranks, E, CAP, TOPK, H, I, 0, physical, eco, True)
    config = ValidatedCall(ranks=ranks, experts=E, capacity=CAP, topk=TOPK,
                           hidden=H, intermediate=I, tokens=8, bound=CAP,
                           clamp=10.0, fast_math=True, shared_experts=0,
                           physical=physical,
                           eco_logical_experts=(E if eco else 0),
                           eco_max_copies=4,
                           eco_wb=int(os.environ.get('SMOKE_ECO_WB', '64')),
                           eco_wa=0, eco_rounds=1, eco_cold_rounds=6,
                           # [ECO-MEAS] gates, driven from the environment so
                           # one smoke run can prove every variant compiles
                           eco_null_plan=os.environ.get('SMOKE_NULL') == '1',
                           eco_no_auction=os.environ.get('SMOKE_NOAUC') == '1',
                           sm_reuse=int(os.environ.get('GC_MOE_SM90_SM_REUSE', '0') or 0))
    torch.cuda.init()
    torch.cuda.set_device(0)
    torch.zeros(1, device='cuda')
    ctx = get_context()
    schedule = get_sm90_role_config(config, ctx)
    print('schedule', schedule, flush=True)
    specs = _role_specs(config, layout, schedule, False)
    print('roles', [(s[1][0], s[1][1], s[1][2], s[1][3]) for s in specs], flush=True)
    identity = _compiler_identity('sm_90a')
    t0 = time.time()
    binaries = _compile_roles(specs, identity)
    print(f'compiled roles {[b.role for b in binaries]} in {time.time() - t0:.1f}s', flush=True)


if __name__ == '__main__':
    main()
