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
    rounds_total = int(os.environ.get('PLAN_CALLS', '10'))

    # [ECO-MOVE] Route count handed to the plan: the whole topk_idx table, so
    # every route `publish_counts` writes is counted.
    TOKENS = int(buf.topk_idx.shape[0])

    def publish_counts(counts_np):
        """Make the kernel's OWN count produce exactly `counts_np`.

        [ECO-MOVE] The plan now builds the demand histogram itself from
        `topk_idx` (it used to be built by dispatch and pushed into
        `eco_counts`). So the test has to provide the route table rather than
        the finished panel. Each rank gets an equal share of every expert's
        routes; writing them into `topk_idx` makes the kernel's own count come
        out as `counts_np`, which is exactly what the reference plan is given.
        """
        share = np.zeros((R, E), np.int64)
        for e in range(E):
            base, extra = divmod(int(counts_np[e]), R)
            share[:, e] = base
            share[:extra, e] += 1
        # topk_idx is [capacity, topk]; the plan reads `tokens * topk` entries.
        cap, tk = buf.topk_idx.shape
        tokens = TOKENS
        assert tokens * tk >= int(share[buf.rank_idx].sum()), (
            f'topk_idx holds {tokens * tk} routes but this rank needs '
            f'{share[buf.rank_idx].sum()}')
        # [ECO-MOVE] Each rank writes ONLY ITS OWN share of the routes: the
        # plan counts this rank's `topk_idx` and then sums the per-rank rows
        # across ranks, so writing every rank's share on every rank would
        # multiply D_e by R.
        mine = share[buf.rank_idx]
        flat = np.full(tokens * tk, -1, np.int64)
        pos = 0
        for e in range(E):
            n = int(mine[e])
            if n:
                flat[pos:pos + n] = e
                pos += n
        assert pos == int(mine.sum()), (pos, int(mine.sum()))
        ties = torch.as_tensor(flat, dtype=torch.int64, device=dev)
        buf.topk_idx[:tokens].copy_(ties.view(tokens, tk))

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
    args += [ctypes.c_uint32(buf.rank_idx), ctypes.c_uint32(TOKENS),
             ctypes.c_float(10.0)]
    arg_ptrs = [ctypes.addressof(a) for a in args]     # cuLaunchKernel wants pointer-to-arg
    arr = (ctypes.c_void_p * len(arg_ptrs))(*arg_ptrs)
    stream = torch.cuda.current_stream().cuda_stream

    def run_plan():
        # [ECO-MOVE] The plan publishes AND consumes the counts panel itself
        # now, so there is no host-side panel to prepare: the plan's own
        # header[24] rendezvous is inside the kernel. Only the wake token is
        # supplied from the host, as dispatch would.
        torch.cuda.synchronize()
        dist.barrier()
        # [ECO-MOVE] The plan gates its own count on `eco_ready[2]` having
        # caught up with the PREVIOUS call. That guard only fires when the
        # epoch advances, so the test has to bump `header[0]` the way a real
        # dispatch does; otherwise the warm start reads a half-written panel.
        epoch = int(buf.header[0].item()) + 1
        buf.header[0] = epoch
        buf.eco_ready[0] = epoch + 1
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
        # [ECO-MOVE] The kernel derives the demand from `topk_idx`, so the test
        # must build counts that FIT in it: capacity*topk routes. Scale the
        # per-expert mean so the total stays inside the route table while the
        # distribution still varies between calls.
        routes_cap = int(buf.topk_idx.numel())
        mean_per_expert = max(1, int(routes_cap * 0.85) // E)
        counts = torch.randint(0, max(2, mean_per_expert * 2), (E,), generator=gen,
                               device=dev, dtype=torch.int32)
        if call == 0:
            counts[7] = mean_per_expert * 6
        # trim to fit exactly
        tot = int(counts.sum())
        if tot > routes_cap:
            counts = (counts.to(torch.float64) * (routes_cap / tot)).to(torch.int32)
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
        # [dbg] every rank reports its OWN row, so a rank that never ran the
        # counter stands out instead of being averaged away.
        if call < 2:
            own = int(buf.eco_counts.view(2, R, lay.physical)[0, buf.rank_idx].sum())
            print(f'   [call {call}] [rank {buf.rank_idx}] own_row_sum={own} '
                  f'epoch_now={int(buf.header[0].item())} '
                  f'ready={buf.eco_ready.tolist()}', flush=True)
        if buf.rank_idx == 0:
            if call < 2 or os.environ.get('PLAN_DBG') == '1':
                _p = buf.eco_counts.view(2, R, lay.physical)[0]
                print(f'   [dbg] eco_counts sum(r0..r7)='
                      f'{[int(_p[r].sum()) for r in range(R)]} '
                      f'expected total={int(counts.sum())} '
                      f'quota_sum={int(got.sum())}', flush=True)
                if call == 1:
                    d = np.nonzero(got != want)[0][:12]
                    print(f'   [dbg] call1 diff slots={d.tolist()} '
                          f'got={got[d].tolist()} want={want[d].tolist()} '
                          f'warm_state_q[:8]={buf.eco_state[:8].tolist()} '
                          f'prev_q_fed={None if prev is None else prev[:8].tolist()}',
                          flush=True)
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
