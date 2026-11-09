"""end-to-end correctness: ECO path must reproduce the replica-free reference.

Same shape as verify_eplb_ep8.py but the second call goes through the ECO plan
(role 5) instead of the host round-robin `replica_plan`.  Replicas compute the
same values as the primary slot, so cos must be ~1.0 and rel ~0.

usage: torchrun --nproc_per_node 8 scripts/verify_eco_e2e.py
"""
import os, sys
sys.path.insert(0, os.environ['GC_SHIM'])
import torch, torch.distributed as dist
import gc_ptxas_override  # noqa: F401
import gc_moe
from deep_gemm.utils import per_token_cast_to_fp4, per_token_cast_to_fp8
from gc_moe.eco.meta_sm90 import build_eco_meta

H, I, E, TOPK, CAP, B = 5120, 2304, 384, 6, 64, 32
R = int(os.environ.get('WORLD_SIZE', '8'))
WB = int(os.environ.get('GC_MOE_ECO_WB', '64'))
WA = int(os.environ.get('GC_MOE_ECO_WA', '0'))
COLD = int(os.environ.get('GC_MOE_ECO_COLD_ROUNDS', '6'))
ROUNDS = int(os.environ.get('GC_MOE_ECO_ROUNDS', '1'))
EXTRA = int(os.environ.get('GC_MOE_EPLB_EXTRA_PER_RANK', '8'))
REORDER = os.environ.get('ECO_REORDER', '1') == '1'


def cast_grouped(shape, gen, dev):
    g, n, k = shape
    q = torch.empty((g, n, k // 2), dtype=torch.int8, device=dev)
    s = torch.empty((g, n, k // B), dtype=torch.float32, device=dev)
    for e in range(g):
        raw = torch.randn((n, k), dtype=torch.bfloat16, device=dev, generator=gen) * 0.05
        q[e], s[e] = per_token_cast_to_fp4(raw, use_ue8m0=True, gran_k=B)
    return q, s


def main():
    lr = int(os.environ['LOCAL_RANK']); torch.cuda.set_device(lr)
    dev = torch.device(f'cuda:{lr}')
    # R comes from the launcher: hard-coding 8 while running with
    # --nproc_per_node 2 made both ranks block forever in TCPStore::wait
    # (measured with gdb: both stuck in doWait, no kernel resident).
    assert R == int(os.environ['WORLD_SIZE']), (R, os.environ['WORLD_SIZE'])
    dist.init_process_group('gloo', init_method='env://', rank=lr, world_size=R)
    group = dist.new_group(list(range(R)), backend='nccl')
    t = torch.ones(1, dtype=torch.int32, device=dev); dist.all_reduce(t, group=group)
    torch.cuda.synchronize(); rank = dist.get_rank(group)
    gen = torch.Generator(device=dev).manual_seed(2026)
    localE = E // R
    w1, w2 = gc_moe.transform_weights_for_gc_moe_sm90(
        cast_grouped((localE, 2 * I, H), gen, dev), cast_grouped((localE, H, I), gen, dev))
    T = 16
    raw = torch.randn((T, H), dtype=torch.bfloat16, device=dev, generator=gen) * 0.5
    xq, xsf = per_token_cast_to_fp8(raw, use_ue8m0=False, gran_k=128)
    sc = torch.randn((T, E), dtype=torch.float32, device=dev, generator=gen)
    tw, ti = torch.topk(sc, TOPK, dim=-1, largest=True, sorted=False)
    tw = torch.softmax(tw, -1)

    # ---------- reference: plain path, no replication ----------
    buf = gc_moe.get_symm_buffer_for_gc_moe_sm90(group, E, CAP, TOPK, H, I,
                                                 activation='swiglu')
    buf.x[:T].copy_(xq); buf.x_sf[:T].copy_(xsf)
    buf.topk_idx[:T].copy_(ti.to(torch.int64)); buf.topk_weights[:T].copy_(tw)
    torch.cuda.synchronize(dev)
    out = torch.empty((T, H), dtype=torch.bfloat16, device=dev)
    if rank == 0: print('[ref] launching plain path', flush=True)
    gc_moe.sm90_fp8_fp4_gc_moe(out, w1, w2, buf, recipe=(1, 1, 32), activation='swiglu',
                               activation_clamp=10.0, fast_math=True, num_tokens_bound=CAP)
    torch.cuda.synchronize(); ref = out.clone()
    if rank == 0: print(f'[ref] done, ref_abs_sum={ref.float().abs().sum().item():.4f}', flush=True)
    buf.destroy()

    # ---------- ECO arm ----------
    # deterministic placement: identity + `EXTRA` extra copies of 0..EXTRA-1,
    # exactly what the runtime installs before any rearrangement.
    P = E + EXTRA * R
    PR = P // R
    phy = list(range(E)) + [i % E for i in range(EXTRA * R)]
    if REORDER:
        from gc_moe.eco.meta_sm90 import reorder_single_copy_first
        import numpy as np
        phy = [int(x) for x in reorder_single_copy_first(np.asarray(phy, np.int64), R)[0]]
    max_rep = max(2, max(phy.count(e) for e in range(E)))

    buf2 = gc_moe.get_symm_buffer_for_gc_moe_sm90(group, E, CAP, TOPK, H, I,
                                                  activation='swiglu',
                                                  physical=P, eco=True, trace=False)
    buf2.eco_meta.copy_(torch.as_tensor(build_eco_meta(phy, R, E),
                                        dtype=torch.int32, device=dev))
    # weights: replicate to every slot holding each expert (host gather)
    from gc_moe.sm90_gc import install_eplb_plan, gather_slot_weights
    plan, slots_of = install_eplb_plan(buf2, phy)
    w1b, w2b = gather_slot_weights((w1, w2), buf2, plan, group)

    buf2.x[:T].copy_(xq); buf2.x_sf[:T].copy_(xsf)
    buf2.topk_idx[:T].copy_(ti.to(torch.int64))   # LOGICAL ids: ECO maps them
    buf2.topk_weights[:T].copy_(tw)
    torch.cuda.synchronize(dev)
    if rank == 0: print(f'[eco] P={P} PR={PR} max_rep={max_rep} reorder={int(REORDER)} setup done', flush=True)

    out.zero_()
    # ECO is inferred from the buffer (`eco=True`, `eco_logical`) plus the
    # GC_MOE_ECO_* env vars; there are no eco kwargs on this entry point.
    os.environ['GC_MOE_ECO_MAX_COPIES'] = str(max_rep)
    gc_moe.sm90_fp8_fp4_gc_moe(out, w1b, w2b, buf2, recipe=(1, 1, 32), activation='swiglu',
                               activation_clamp=10.0, fast_math=True, num_tokens_bound=CAP)
    torch.cuda.synchronize()

    finite = bool(torch.isfinite(out.float()).all())
    cos = torch.nn.functional.cosine_similarity(
        ref.float().flatten(), out.float().flatten(), dim=0).item()
    rel = ((out.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-9)).item()
    ok = finite and cos > 0.999 and rel < 1e-2
    if rank == 0:
        print(f'ECO_E2E reorder={int(REORDER)} finite={finite} '
              f'cos={cos:.8f} rel={rel:.3e} {"OK" if ok else "FAIL"}', flush=True)

    # token accounting: every route must land somewhere
    routes = [torch.zeros((T, TOPK), dtype=torch.int64, device=dev) for _ in range(R)]
    dist.all_gather(routes, ti.to(torch.int64), group=group)
    logical = torch.zeros(E, dtype=torch.int64, device=dev)
    for row in routes:
        logical += torch.bincount(row.reshape(-1), minlength=E)
    cnt = buf2.expert_count[:PR].clone()
    rows_all = [torch.zeros(PR, dtype=torch.int32, device=dev) for _ in range(R)]
    dist.all_gather(rows_all, cnt, group=group)
    holder = torch.zeros(E, dtype=torch.int64, device=dev)
    for r, row in enumerate(rows_all):
        for s in range(PR):
            holder[phy[r * PR + s]] += row[s].to(torch.int64)
    bad = torch.nonzero(holder != logical).flatten()
    if rank == 0:
        print(f'ECO_E2E holder_sum_bad={int(bad.numel())}/{E} '
              f'total_routed={int(logical.sum())} total_pulled={int(holder.sum())}',
              flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
