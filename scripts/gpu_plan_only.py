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
sys.path.insert(0, os.path.join(_HERE, '..', 'eco_ref'))

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
from gc_moe.eco.meta_sm90 import build_eco_meta
from unit_ref import UnitPlanner

H, I, E, TOPK, CAP = 5120, 2304, 384, 6, 64
R = 8
REDUNDANCY = float(os.environ.get('EPLB_REDUNDANCY', '0.25'))
WB = int(os.environ.get('GC_MOE_ECO_WB', '64'))
WA = int(os.environ.get('GC_MOE_ECO_WA', '0'))
ROUNDS = int(os.environ.get('GC_MOE_ECO_ROUNDS', '1'))
COLD = int(os.environ.get('GC_MOE_ECO_COLD_ROUNDS', '6'))


def main():
    lr = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(lr)
    dev = torch.device(f'cuda:{lr}')
    dist.init_process_group('gloo', init_method='env://', rank=lr, world_size=R)
    group = dist.new_group(list(range(R)), backend='nccl')
    torch.cuda.synchronize()

    from gc_moe.layout.eplb_placement import build_eplb_layout
    demand = torch.ones(E, dtype=torch.float32)
    demand[7] = 3.0   # keeps max_copies == 4 (eco_max_copies)
    lay = build_eplb_layout(demand, E, R, REDUNDANCY)
    phy = [int(x) for x in lay.phy2log]

    buf = gc_moe.get_symm_buffer_for_gc_moe_sm90(
        group, E, CAP, TOPK, H, I, activation='swiglu',
        physical=lay.physical, eco=True)
    plan, slots_of = sm90_gc.install_eplb_plan(buf, phy)
    meta = build_eco_meta(phy, R, E)
    buf.eco_meta.copy_(torch.as_tensor(meta, dtype=torch.int32, device=dev))

    config = ValidatedCall(ranks=R, experts=E, capacity=CAP, topk=TOPK,
                           hidden=H, intermediate=I, tokens=32, bound=CAP,
                           clamp=10.0, fast_math=True, shared_experts=0,
                           physical=lay.physical, eco_logical_experts=E,
                           eco_max_copies=4, eco_wb=WB, eco_wa=WA,
                           eco_rounds=ROUNDS, eco_cold_rounds=COLD)
    ctx = get_context()
    schedule = get_sm90_role_config(config, ctx)
    bm = schedule.tile_m
    if buf.rank_idx == 0:
        print(f'[plan-only] tile_m={bm} grids={schedule.grids}', flush=True)

    # --- plan launches: counts published, then role 5 ---------------------
    gen = torch.Generator(device=dev).manual_seed(int(os.environ.get('PLAN_SEED', '3')))
    rounds_total = int(os.environ.get('PLAN_CALLS', '2'))

    def publish_counts(counts_np):
        """eco_counts row r = rank r's own share of each slot's stream."""
        share = np.zeros((R, E), np.int64)
        for e in range(E):
            base, extra = divmod(int(counts_np[e]), R)
            share[:, e] = base
            share[:extra, e] += 1
        row = np.zeros((R, lay.physical), np.int64)
        for r in range(R):
            for g in range(lay.physical):
                row[r, g] = share[r, phy[g]]
        for r in range(R):
            buf.eco_counts[r * lay.physical:(r + 1) * lay.physical].copy_(
                torch.as_tensor(row[r], dtype=torch.int32, device=dev))

    specs = _role_specs(config, buf.layout, schedule, False)
    spec = next(s for s in specs if s[1][0] == 5)
    identity = _compiler_identity('sm_90a')
    binaries = _get_or_compile_binaries(_binary_key(identity, (spec,)), identity, (spec,))
    binary = binaries[0]

    from cuda.bindings import driver as cu
    loaded = ctx._get_loaded_kernel(binary)
    kernel = loaded.kernel
    cu.cuKernelSetAttribute(cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                            binary.smem_bytes, kernel, ctx.device)
    args = [ctypes.c_void_p(buf.buffer.data_ptr()), ctypes.c_void_p(buf.peer_table.data_ptr())]
    args += [ctypes.c_void_p(0)] * 12
    args += [ctypes.c_uint32(buf.rank_idx), ctypes.c_uint32(32), ctypes.c_float(10.0)]
    arg_ptrs = [ctypes.addressof(a) for a in args]     # cuLaunchKernel wants pointer-to-arg
    arr = (ctypes.c_void_p * len(arg_ptrs))(*arg_ptrs)
    stream = torch.cuda.current_stream().cuda_stream

    def run_plan():
        # The plan reads EVERY rank's eco_counts row, so the test must make
        # sure all rows are published first. In a real call the dispatch role
        # does exactly this with the header[24] cross-rank rendezvous.
        torch.cuda.synchronize()
        dist.barrier()
        # The plan waits for eco_ready[0] == epoch + 1 (published by dispatch
        # in a real call); the test publishes it from the host.
        buf.eco_ready[0] = int(buf.header[0].item()) + 1
        res = cu.cuLaunchKernel(kernel, 1, 1, 1, binary.threads, 1, 1,
                                binary.smem_bytes, stream, arr, 0)
        if res[0] != cu.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f'plan launch failed: {res[0]}')
        torch.cuda.synchronize()
        return buf.eco_quota[:lay.physical].to(torch.int64).cpu().numpy()

    ref = UnitPlanner(phy, R, E, block_m=bm, wb=WB, wa=WA, rounds=ROUNDS,
                      cold_rounds=COLD, hop2_rounds=0)
    prev = None
    failures = 0
    for call in range(rounds_total):
        counts = torch.randint(0, 60, (E,), generator=gen, device=dev,
                               dtype=torch.int32)
        if call == 0:
            counts[7] = 900
        counts = counts.cpu().numpy().astype(np.int64)
        if buf.rank_idx == 0:
            np.save(f'/tmp/eco_counts_{call}.npy', counts)
            np.save('/tmp/eco_phy.npy', np.asarray(phy, np.int64))
        publish_counts(counts)
        want = ref.plan(counts, prev if call else None)
        got = run_plan()
        ok = bool(np.array_equal(got, want))
        # conservation + contiguous quota_start inside every expert
        cons, contig = True, True
        start_np = buf.eco_quota_start[:lay.physical].to(torch.int64).cpu().numpy()
        for e in range(E):
            run_off = 0
            for g in range(lay.physical):
                if phy[g] != e:
                    continue
                if start_np[g] != run_off:
                    contig = False
                run_off += int(got[g])
            if run_off != int(counts[e]):
                cons = False
        if os.environ.get('GC_MOE_ECO_DEBUG') == '1' and buf.rank_idx == 0:
            nk = buf.eco_dbg[5 * lay.physical:5 * lay.physical + E].cpu().numpy().astype(np.int64)
            print(f'   [dbg] kernel D_e[:8]={nk[:8].tolist()} host counts[:8]={counts[:8].tolist()} '
                  f'sums={int(nk.sum())}/{int(counts.sum())}', flush=True)
        if buf.rank_idx == 0:
            print(f'[call {call}] warm={call > 0} equal={ok} conserved={cons} '
                  f'quota_start_contiguous={contig}', flush=True)
            if not ok:
                bad = np.nonzero(got != want)[0]
                print(f'   diffs slots={bad[:8].tolist()} '
                      f'got={got[bad[:8]].tolist()} want={want[bad[:8]].tolist()}',
                      flush=True)
        if not (ok and cons and contig):
            failures += 1
        prev = (got, counts)

    if buf.rank_idx == 0:
        print('GPU_PLAN_V4 OK' if failures == 0 else 'GPU_PLAN_V4 FAIL', flush=True)
    bad_t = torch.tensor([0 if failures == 0 else 1], device=dev)
    dist.all_reduce(bad_t, group=group)
    if buf.rank_idx == 0 and int(bad_t) != 0:
        print('GPU_PLAN_V4 ALL_RANKS FAILED', flush=True)


if __name__ == '__main__':
    main()
