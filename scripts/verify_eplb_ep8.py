"""EP8: EPLB placement (25% redundancy) + plan-based token split.

Placement comes from build_eplb_layout (B-arm / DeepSeek EPLB algorithm); the
plan only decides which replica serves a slot. Weights are replicated onto the
ranks that hold a copy, so every replica computes the same values; therefore a
correct implementation must reproduce the replica-free path.
"""
import os, sys
sys.path.insert(0, os.environ['GC_SHIM'])
import torch, torch.distributed as dist
import gc_ptxas_override  # noqa: F401
import gc_moe
from deep_gemm.utils import per_token_cast_to_fp4, per_token_cast_to_fp8

H, I, E, TOPK, CAP, B = 5120, 2304, 384, 6, 64, 32
REDUNDANCY = float(os.environ.get('EPLB_REDUNDANCY', '0.25'))


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
    dist.init_process_group('gloo', init_method='env://', rank=lr, world_size=8)
    group = dist.new_group(list(range(8)), backend='nccl')
    t = torch.ones(1, dtype=torch.int32, device=dev); dist.all_reduce(t, group=group)
    torch.cuda.synchronize(); rank = dist.get_rank(group)
    gen = torch.Generator(device=dev).manual_seed(2026)
    R = 8
    localE = E // R
    w1, w2 = gc_moe.transform_weights_for_gc_moe_sm90(
        cast_grouped((localE, 2 * I, H), gen, dev), cast_grouped((localE, H, I), gen, dev))
    T = 16
    raw = torch.randn((T, H), dtype=torch.bfloat16, device=dev, generator=gen) * 0.5
    xq, xsf = per_token_cast_to_fp8(raw, use_ue8m0=False, gran_k=128)
    sc = torch.randn((T, E), dtype=torch.float32, device=dev, generator=gen)
    tw, ti = torch.topk(sc, TOPK, dim=-1, largest=True, sorted=False)

    # ---- reference: plain path, no replication -------------------------
    buf = gc_moe.get_symm_buffer_for_gc_moe_sm90(group, E, CAP, TOPK, H, I,
                                                 activation='swiglu', max_replicas=4)
    buf.x[:T].copy_(xq); buf.x_sf[:T].copy_(xsf)
    buf.topk_idx[:T].copy_(ti.to(torch.int64)); buf.topk_weights[:T].copy_(torch.softmax(tw, -1))
    torch.cuda.synchronize(dev)
    out = torch.empty((T, H), dtype=torch.bfloat16, device=dev)
    gc_moe.sm90_fp8_fp4_gc_moe(out, w1, w2, buf, recipe=(1, 1, 32), activation='swiglu',
                               activation_clamp=10.0, fast_math=True, num_tokens_bound=CAP)
    torch.cuda.synchronize(); ref = out.clone()
    buf.destroy()

    # ---- EPLB: same demand on every rank so the layout is deterministic ----
    demand = torch.rand(E, device='cpu', generator=torch.Generator().manual_seed(5)) * 100
    demand[7] = 900.0
    lay = gc_moe.build_eplb_layout(demand, E, R, REDUNDANCY)
    if rank == 0:
        rep = sum(1 for c in lay.logcnt if c > 1)
        print(f'EPLB physical={lay.physical} per_rank={lay.per_rank} '
              f'redundant={lay.redundant} replicated_experts={rep} '
              f'max_copies={max(lay.logcnt)}', flush=True)

    max_rep = max(lay.max_copies, 2)
    buf2 = gc_moe.get_symm_buffer_for_gc_moe_sm90(group, E, CAP, TOPK, H, I,
                                                  activation='swiglu',
                                                  max_replicas=max_rep,
                                                  physical=lay.physical)
    w1b, w2b = gc_moe.replicate_weights_for_eplb(w1, w2, lay, group)
    perm, n_norep = gc_moe.install_eplb_layout(buf2, lay)
    if rank == 0:
        print(f'n_norep={n_norep}/{lay.per_rank} (phase1 overlap = '
              f'{100.0 * n_norep / lay.per_rank:.1f}%)', flush=True)

    # topk_idx: logical expert -> its primary physical slot. The slot id stays
    # global (slot_owner decodes it), so no permutation is applied here -- the
    # permutation reorders only this rank's *local* phase1/phase2 split.
    slots = buf2.expert_primary[ti]
    ti2 = slots
    buf2.x[:T].copy_(xq); buf2.x_sf[:T].copy_(xsf)
    buf2.topk_idx[:T].copy_(ti2.to(torch.int64))
    buf2.topk_weights[:T].copy_(torch.softmax(tw, -1))
    torch.cuda.synchronize(dev)

    out.zero_()
    gc_moe.sm90_fp8_fp4_gc_moe(out, w1b, w2b, buf2, recipe=(1, 1, 32), activation='swiglu',
                               activation_clamp=10.0, fast_math=True, num_tokens_bound=CAP,
                               n_norep=n_norep, max_replicas=max_rep,
                               replica_plan=True)
    torch.cuda.synchronize()
    finite = bool(torch.isfinite(out.float()).all())
    cos = torch.nn.functional.cosine_similarity(
        ref.float().flatten(), out.float().flatten(), dim=0).item()
    rel = ((out.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-9)).item()
    ok = finite and cos > 0.999 and rel < 1e-2
    if rank == 0:
        print(f'EPLB_EP8 finite={finite} cos={cos:.8f} rel={rel:.3e} '
              f'{"OK" if ok else "FAIL"}', flush=True)

    # Accounting: a slot's `expert_count` is the number of rows it pulled.
    # Replicas of one logical expert split that expert's demand, so summing the
    # counts of all slots holding the same expert must reproduce the demand.
    rows = [torch.zeros(lay.per_rank, dtype=torch.int32, device=dev) for _ in range(R)]
    dist.all_gather(rows, buf2.expert_count[:lay.per_rank].clone(), group=group)
    # expert_count is per local slot on each rank; slot_owner_inv maps a local
    # position back to the global slot id.
    got = torch.zeros(lay.physical, dtype=torch.int64, device=dev)
    inv_table = buf2.slot_owner_inv.cpu()
    for r, row in enumerate(rows):
        for s in range(lay.per_rank):
            got[int(inv_table[r, s])] = row[s].to(torch.int64)
    routes = [torch.zeros((T, TOPK), dtype=torch.int64, device=dev) for _ in range(R)]
    dist.all_gather(routes, ti2.to(torch.int64), group=group)
    routed = torch.zeros(lay.physical, dtype=torch.int64, device=dev)
    for row in routes:
        routed += torch.bincount(row.reshape(-1), minlength=lay.physical)
    # Logical demand: every route that reached an expert, wherever it landed.
    logical = torch.zeros(E, dtype=torch.int64, device=dev)
    routes_logical = [torch.zeros((T, TOPK), dtype=torch.int64, device=dev) for _ in range(R)]
    dist.all_gather(routes_logical, ti.to(torch.int64), group=group)
    for row in routes_logical:
        logical += torch.bincount(row.reshape(-1), minlength=E)
    holder_sum = torch.zeros(E, dtype=torch.int64, device=dev)
    for slot in range(lay.physical):
        holder_sum[lay.phy2log[slot]] += got[slot]
    bad_log = torch.nonzero(holder_sum != logical).flatten()
    if rank == 0:
        # Per-slot detail for the first bad expert shows whether one copy took
        # the whole demand instead of half.
        for e in bad_log[:2].tolist():
            detail = [(g, int(got[g])) for g in range(lay.physical)
                      if lay.phy2log[g] == e]
            print(f'  detail expert {e}: demand {int(logical[e])} slots {detail}',
                  flush=True)
        print(f'experts_with_wrong_holder_sum={int(bad_log.numel())}/{E}', flush=True)
        for e in bad_log[:5].tolist():
            print(f'  expert {e}: demand {int(logical[e])} holder_sum {int(holder_sum[e])}', flush=True)
        print('EPLB_ACCOUNTING', 'OK' if not bool(bad_log.numel()) else 'FAIL', flush=True)
    dist.barrier(group=group); buf2.destroy(); dist.destroy_process_group()


if __name__ == '__main__':
    main()
