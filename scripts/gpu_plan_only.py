"""Run ONLY the SM90 ECO plan role (role 5) on the GPU and diff it vs unit_ref.

No model, no dispatch, no compute: this isolates the plan algorithm and is the
GPU counterpart of the CPU V4 test. Takes a few seconds.

Usage: torchrun --nproc_per_node 8 scripts/gpu_plan_only.py
"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ['GC_SHIM'])
sys.path.insert(0, os.path.join(_HERE, '..', 'source'))
sys.path.insert(0, os.path.join(_HERE, '..', 'integration'))

import ctypes
import numpy as np
import torch, torch.distributed as dist
import gc_ptxas_override  # noqa: F401
import gc_moe
import gc_moe.sm90_gc as sm90_gc
from gc_moe.apis.sm90_gc_moe import ValidatedCall
from gc_moe.runtime.sm90_gc_moe import get_context
from gc_moe.jit_kernels.heuristics.sm90_gc_moe import get_sm90_role_config
from gc_moe.jit_kernels.impls.sm90_fp8_fp4_gc_moe import _role_specs, _get_or_compile_binaries, _binary_key, _compiler_identity
from gc_moe.eco.meta_sm90 import install_into_buffer
from gc_moe.eco.unit_ref import UnitPlanner

H, I, E, TOPK, CAP = 5120, 2304, 384, 6, 64
R = 8
REDUNDANCY = float(os.environ.get('EPLB_REDUNDANCY', '0.25'))
WB = int(os.environ.get('GC_MOE_ECO_WB', '64'))
WA = int(os.environ.get('GC_MOE_ECO_WA', '0'))
ROUNDS = int(os.environ.get('GC_MOE_ECO_ROUNDS', '1'))
COLD = int(os.environ.get('GC_MOE_ECO_COLD_ROUNDS', '6'))


def main():
    lr = int(os.environ['LOCAL_RANK'])
    os.environ.setdefault('GC_MOE_LAYERS', '2')
    if (os.getenv('GC_MOE_ECO_COSTFN', '0') != '0'
            or any(os.getenv('GC_MOE_ECO_' + key) == '1' for key in
                   ('NULL_PLAN', 'NO_AUCTION', 'BLOCKS_ONLY', 'TOKENS_ONLY'))):
        raise ValueError('plan-only differential uses the SM100 linear cost and auction')
    torch.cuda.set_device(lr)
    dev = torch.device(f'cuda:{lr}')
    dist.init_process_group('gloo', init_method='env://', rank=lr, world_size=R)
    group = dist.new_group(list(range(R)), backend='nccl')
    torch.cuda.synchronize()

    from gc_moe.layout.eplb_placement import build_eplb_layout
    demand = torch.ones(E, dtype=torch.float32)
    lay = build_eplb_layout(demand, E, R, REDUNDANCY)
    buf = gc_moe.get_symm_buffer_for_gc_moe_sm90(
        group, E, CAP, TOPK, H, I, activation='swiglu',
        physical=lay.physical, eco=True)
    # Exercise an active copy on every rank, and passive duplicates on the
    # same rank. Use two different layouts when the state is layer-specific.
    layouts = []
    for layer in range(buf.layout.n_layers):
        hot = layer % 2 if buf.layout.eco_per_layer else 0
        row = []
        for rank in range(R):
            row.extend(range(rank * (E // R), (rank + 1) * (E // R)))
            extras = lay.physical // R - E // R
            row.extend([hot] * extras if rank < buf.layout.eco_max_copies else [rank * (E // R)] * extras)
        layouts.append(row)
    sm90_gc.install_eplb_plan(buf, layouts)
    install_into_buffer(buf, layouts, R, E)

    config = ValidatedCall(ranks=R, experts=E, capacity=buf.num_max_tokens_per_rank, topk=TOPK,
                           hidden=H, intermediate=I, tokens=CAP, bound=CAP,
                           clamp=10.0, fast_math=True, physical=lay.physical,
                           eco_logical_experts=E, eco_max_copies=buf.layout.eco_max_copies,
                           eco_wb=WB, eco_wa=WA, eco_rounds=ROUNDS, eco_cold_rounds=COLD)
    ctx = get_context()
    schedule = get_sm90_role_config(config, ctx)
    bm = schedule.tile_m
    if buf.rank_idx == 0:
        print(f'[plan-only] tile_m={bm} grids={schedule.grids}', flush=True)

    rng = np.random.default_rng(int(os.environ.get('PLAN_SEED', '3')) + lr)
    rounds_total = int(os.environ.get('PLAN_CALLS', '10'))
    if rounds_total < 1:
        raise ValueError('PLAN_CALLS must be positive')

    specs = _role_specs(config, buf.layout, schedule, False)
    spec = next(s for s in specs if s[1][0] == 5)
    identity = _compiler_identity('sm_90a')
    binaries = _get_or_compile_binaries(_binary_key(identity, (spec,)), identity, (spec,))
    binary = binaries[0]

    from cuda.bindings import driver as cu
    loaded = ctx._get_loaded_kernel(binary)
    kernel = loaded.kernel
    status, = cu.cuKernelSetAttribute(cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                      binary.smem_bytes, kernel, ctx.device)
    if status != cu.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f'plan shared memory setup failed: {status}')
    status, function = cu.cuKernelGetFunction(kernel)
    if status != cu.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f'plan function lookup failed: {status}')
    args = [ctypes.c_void_p(buf.buffer.data_ptr()), ctypes.c_void_p(buf.peer_table.data_ptr())]
    args += [ctypes.c_void_p(0)] * 12
    args += [ctypes.c_uint32(buf.rank_idx), ctypes.c_uint32(CAP),
             ctypes.c_float(10.0)]
    arg_ptrs = [ctypes.addressof(a) for a in args]     # cuLaunchKernel wants pointer-to-arg
    arr = (ctypes.c_void_p * len(arg_ptrs))(*arg_ptrs)
    stream = torch.cuda.current_stream().cuda_stream

    refs = [UnitPlanner(row, R, E, block_m=bm, wb=WB, wa=WA, rounds=ROUNDS, cold_rounds=COLD)
            for row in layouts]
    previous = {}
    tested_layers = min(buf.layout.n_layers, 2) if buf.layout.eco_per_layer else 1
    failures = 0
    for call in range(rounds_total):
        layer = call % tested_layers
        epoch = 1 + layer + (call // tested_layers) * buf.layout.n_layers
        ids = np.stack([rng.choice(E, TOPK, replace=False) for _ in range(CAP)])
        hot = layer % 2
        n_hot = CAP if call // tested_layers % 2 == 0 else CAP // 8
        for routes in ids[:n_hot]:
            if hot not in routes:
                routes[0] = hot
        buf.topk_idx[:CAP].copy_(torch.as_tensor(ids, device=dev))
        local_counts = torch.bincount(buf.topk_idx[:CAP].flatten(), minlength=E)
        dist.all_reduce(local_counts, group=group)
        counts = local_counts.cpu().numpy()
        want = refs[layer].plan(counts, previous.get(layer))
        # Model dispatch would publish these two epoch words on its stream.
        buf.header[0] = epoch
        buf.eco_ready[0] = epoch + 1
        res = cu.cuLaunchKernel(function, 1, 1, 1, binary.threads, 1, 1,
                                binary.smem_bytes, stream, arr, 0)
        if res[0] != cu.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f'plan launch failed: {res[0]}')
        torch.cuda.synchronize()
        got = buf.eco_quota.cpu().numpy()
        starts = buf.eco_quota_start.cpu().numpy()
        prefix = np.zeros(E, np.int64)
        starts_ok = True
        for slot, expert in enumerate(layouts[layer]):
            starts_ok &= int(starts[slot]) == prefix[expert]
            prefix[expert] += int(got[slot])
        ok = np.array_equal(got, want) and starts_ok and np.array_equal(prefix, counts)
        failures += int(not ok)
        if lr == 0:
            print(f'[call {call}] layer={layer} warm={layer in previous} equal={ok}', flush=True)
            np.save(f'/tmp/eco_counts_{call}.npy', counts)
            np.save('/tmp/eco_phy.npy', np.asarray(layouts[layer]))
        if not ok:
            bad = np.flatnonzero(got != want)[:8]
            print(f'rank={lr} epoch={epoch} slots={bad} got={got[bad]} want={want[bad]} '
                  f'prefix_matches={starts_ok} conserved={np.array_equal(prefix, counts)}', flush=True)
        previous[layer] = (want.copy(), counts.copy())

    bad = torch.tensor([failures], device=dev)
    dist.all_reduce(bad, group=group)
    if bad.item():
        raise RuntimeError(f'GPU_PLAN_V4 FAIL: {bad.item()} rank/call mismatches')
    if lr == 0:
        print('GPU_PLAN_V4 OK', flush=True)


if __name__ == '__main__':
    main()
