"""EP8 ECO correctness check: no model needed (random weights, tiny batch).

Verifies, on real GPUs with the real kernel:
  * role 5 compiles and runs without deadlocking,
  * the plan conserves every expert's tokens:
        sum over a logical expert's copies of quota == its routed count,
  * the two copies of an expert split its stream (both get work when the
    expert is hot), and quota_start is contiguous,
  * all ranks compute the SAME plan,
  * with ECO on, the output matches the replica-free reference run.

Usage: torchrun --nproc_per_node 8 scripts/verify_eco_ep8.py
"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ['GC_SHIM'])
sys.path.insert(0, os.path.join(_HERE, '..', 'integration'))
import torch, torch.distributed as dist
import gc_ptxas_override  # noqa: F401
import gc_moe
import gc_moe.sm90_gc as sm90_gc
from deep_gemm.utils import per_token_cast_to_fp4, per_token_cast_to_fp8

H, I, E, TOPK, CAP, BATCH, B = 5120, 2304, 384, 6, 64, 32, 32
REDUNDANCY = float(os.environ.get('EPLB_REDUNDANCY', '0.25'))
ECO_WB = int(os.environ.get('GC_MOE_ECO_WB', '64'))
ECO_WA = int(os.environ.get('GC_MOE_ECO_WA', '0'))


def cast_grouped(shape, gen, dev):
    g, n, k = shape
    q = torch.empty((g, n, k // 2), dtype=torch.int8, device=dev)
    s = torch.empty((g, n, k // B), dtype=torch.float32, device=dev)
    for e in range(g):
        raw = torch.randn((n, k), dtype=torch.bfloat16, device=dev, generator=gen) * 0.05
        q[e], s[e] = per_token_cast_to_fp4(raw, use_ue8m0=True, gran_k=B)
    return q, s


def make_weights(local_rows, gen, dev):
    w1 = cast_grouped((local_rows, 2 * I, H), gen, dev)
    w2 = cast_grouped((local_rows, H, I), gen, dev)
    return gc_moe.transform_weights_for_gc_moe_sm90(w1, w2)


def run(out, w1, w2, buf, xq, xsf, ids, weights):
    n = xq.shape[0]
    buf.x[:n].copy_(xq)
    buf.x_sf[:n].copy_(xsf)
    buf.topk_idx[:n].copy_(ids.to(torch.int64))
    buf.topk_weights[:n].copy_(weights)
    torch.cuda.synchronize()
    gc_moe.sm90_fp8_fp4_gc_moe(out, w1, w2, buf, recipe=(1, 1, 32),
                               activation='swiglu', activation_clamp=10.0,
                               fast_math=True, num_tokens_bound=CAP)
    torch.cuda.synchronize()


def main():
    lr = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(lr)
    dev = torch.device(f'cuda:{lr}')
    dist.init_process_group('gloo', init_method='env://', rank=lr, world_size=8)
    group = dist.new_group(list(range(8)), backend='nccl')
    t = torch.ones(1, dtype=torch.int32, device=dev)
    dist.all_reduce(t, group=group)
    torch.cuda.synchronize()
    rank = dist.get_rank(group)
    R = 8
    localE = E // R
    gen = torch.Generator(device=dev).manual_seed(2026)

    T = BATCH
    raw = torch.randn((T, H), dtype=torch.bfloat16, device=dev, generator=gen) * 0.5
    xq, xsf = per_token_cast_to_fp8(raw, use_ue8m0=False, gran_k=128)
    sc = torch.randn((T, E), dtype=torch.float32, device=dev, generator=gen)
    tw, ti = torch.topk(sc, TOPK, dim=-1, largest=True, sorted=False)
    weights = torch.softmax(tw, -1)
    # Skew: drive one logical expert hard so replication actually splits.
    hot = 7
    ti[0, :] = hot
    ti[1, :] = hot

    # ---- reference: plain path, no replication --------------------------
    w1, w2 = make_weights(localE, gen, dev)
    buf = gc_moe.get_symm_buffer_for_gc_moe_sm90(group, E, CAP, TOPK, H, I,
                                                 activation='swiglu')
    out = torch.empty((T, H), dtype=torch.bfloat16, device=dev)
    run(out, w1, w2, buf, xq, xsf, ti, weights)
    ref = out.clone()
    buf.destroy()
    dist.barrier()

    # ---- ECO: EPLB placement + role-5 plan -----------------------------
    # Uniform-ish demand with one hot expert: replication must stay within the
    # kernel's `eco_max_copies` (4), so the hot expert is only 2x the rest.
    demand = torch.ones(E, dtype=torch.float32)
    demand[hot] = 2.0
    from gc_moe.layout.eplb_placement import build_eplb_layout
    lay = build_eplb_layout(demand, E, R, REDUNDANCY)
    max_rep = max(lay.max_copies, 2)
    if rank == 0:
        copies = torch.bincount(torch.tensor(lay.phy2log), minlength=E)
        print(f'ECO physical={lay.physical} per_rank={lay.per_rank} '
              f'max_copies={int(copies.max())} '
              f'replicated={int((copies > 1).sum())}', flush=True)

    buf2 = gc_moe.get_symm_buffer_for_gc_moe_sm90(group, E, CAP, TOPK, H, I,
                                                  activation='swiglu',
                                                  physical=lay.physical,
                                                  eco=True)
    plan, slots_of = sm90_gc.install_eplb_plan(buf2, list(lay.phy2log))
    from gc_moe.eco.meta_sm90 import build_eco_meta
    meta = build_eco_meta(list(lay.phy2log), R, E)
    assert meta.size == buf2.eco_meta.numel(), (meta.size, buf2.eco_meta.numel())
    buf2.eco_meta.copy_(torch.as_tensor(meta, dtype=torch.int32, device=dev))
    # Weights per physical slot (replicas carry the same expert's weights).
    w1b, w2b = sm90_gc.gather_slot_weights((w1, w2), buf2, plan, group)

    out.zero_()
    run(out, w1b, w2b, buf2, xq, xsf, ti, weights)
    torch.cuda.synchronize()

    finite = bool(torch.isfinite(out.float()).all())
    cos = torch.nn.functional.cosine_similarity(
        ref.float().flatten(), out.float().flatten(), dim=0).item()
    rel = ((out.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-9)).item()

    # ---- plan accounting -------------------------------------------------
    quota = buf2.eco_quota[:lay.physical].to(torch.int64).cpu()
    start = buf2.eco_quota_start[:lay.physical].to(torch.int64).cpu()
    counts = buf2.expert_count[:lay.per_rank].to(torch.int64)
    # routed tokens per logical expert, gathered across ranks
    ids = [torch.zeros((T, TOPK), dtype=torch.int64, device=dev) for _ in range(R)]
    dist.all_gather(ids, ti.to(torch.int64), group=group)
    demand_e = torch.bincount(torch.cat([x.reshape(-1).cpu() for x in ids]),
                              minlength=E)
    # sum of slot quotas per logical expert
    got = torch.zeros(E, dtype=torch.int64)
    for g in range(lay.physical):
        got[int(lay.phy2log[g])] += quota[g]
    cons = bool(torch.equal(got, demand_e))
    # quota_start contiguous inside each expert's copies
    contig = True
    for g in range(lay.physical):
        e = int(lay.phy2log[g])
        holders = (torch.tensor(lay.phy2log) == e).nonzero().flatten().tolist()
        run_off = 0
        for h in holders:
            if int(start[h]) != run_off:
                contig = False
            run_off += int(quota[h])
    # every rank's plan must be identical
    plans = [torch.zeros(lay.physical, dtype=torch.int64, device=dev) for _ in range(R)]
    dist.all_gather(plans, buf2.eco_quota[:lay.physical].to(torch.int64), group=group)
    same = all(torch.equal(plans[0].cpu(), p.cpu()) for p in plans)
    # hot expert's copies must both carry work
    hot_slots = (torch.tensor(lay.phy2log) == hot).nonzero().flatten().tolist()
    split = sum(1 for h in hot_slots if int(quota[h]) > 0)

    # actual pulled rows per logical expert (must equal the routed demand)
    rows_self = buf2.expert_count[:lay.per_rank].to(torch.int64)
    allrows = [torch.zeros(lay.per_rank, dtype=torch.int64, device=dev) for _ in range(R)]
    dist.all_gather(allrows, rows_self, group=group)
    pulled = torch.zeros(E, dtype=torch.int64)
    for r in range(R):
        row = allrows[r].cpu()
        for s in range(lay.per_rank):
            pulled[int(lay.phy2log[r * lay.per_rank + s])] += int(row[s])
    pulled_ok = bool(torch.equal(pulled, demand_e))
    qn = buf2.eco_quota[:lay.physical].to(torch.int64).cpu()
    ec = buf2.expert_count[:lay.per_rank].to(torch.int64)
    rows = [torch.zeros(lay.per_rank, dtype=torch.int64, device=dev) for _ in range(R)]
    dist.all_gather(rows, ec, group=group)
    if os.environ.get('GC_MOE_ECO_DEBUG') == '1':
        tot_d = buf.eco_dbg[6 * lay.physical:6 * lay.physical + lay.physical].cpu().numpy()
        own_d = buf.eco_dbg[7 * lay.physical:7 * lay.physical + lay.physical].cpu().numpy()
        if rank == 0:
            print(f'  [dbg] rows total sum={int(tot_d.sum())} owned sum={int(own_d.sum())}')
            nz = (own_d > 0).sum()
            print(f'  [dbg] slots with owned>0: {nz}/{lay.physical}')
            print(f'  [dbg] total[:12]={tot_d[:12].tolist()}')
            print(f'  [dbg] owned[:12]={own_d[:12].tolist()}')
    if rank == 0:
        print('  slot  phy  nc  quota  pulled  start', flush=True)
        shown = 0
        for g in range(lay.physical):
            r, s_ = g // lay.per_rank, g % lay.per_rank
            e = int(lay.phy2log[g])
            nc = int((torch.tensor(lay.phy2log) == e).sum())
            if int(qn[g]) != int(rows[r][s_]) and shown < 8:
                print(f'  {g:5d} {e:4d} {nc:3d} {int(qn[g]):6d} {int(rows[r][s_]):7d} '
                      f'{int(buf2.eco_quota_start[g].item())}', flush=True)
                shown += 1
        print(f'  quota total={int(qn.sum())} pulled total={int(sum(int(x.sum()) for x in rows))}', flush=True)
        print(f'  pulled total={int(pulled.sum())} demand={int(demand_e.sum())} '
              f'per_expert_equal={pulled_ok}', flush=True)
        if not pulled_ok:
            bad = (pulled != demand_e).nonzero().flatten()[:6].tolist()
            print(f'  mismatch experts={bad} pulled={[int(pulled[b]) for b in bad]} '
                  f'demand={[int(demand_e[b]) for b in bad]}', flush=True)
    # ref vs out per token: how many tokens differ materially
    tok_err = (out.float() - ref.float()).abs().amax(dim=1)
    if rank == 0:
        nbad_tok = int((tok_err > 1e-2).sum().item())
        print(f'  tokens with error>1e-2: {nbad_tok}/{T} '
              f'per-token max err {[round(float(x), 3) for x in tok_err[:8]]}', flush=True)
    ok = (finite and cos > 0.999 and rel < 1e-2 and cons and contig and same
          and split >= 2 and pulled_ok)
    if rank == 0:
        print(f'ECO_EP8 finite={finite} cos={cos:.8f} rel={rel:.3e} '
              f'conserved={cons} quota_start_contiguous={contig} '
              f'plan_identical={same} hot_copies_used={split}/{len(hot_slots)} '
              f'{"OK" if ok else "FAIL"}', flush=True)
        print(f'  totals: routed={int(demand_e.sum())} quota={int(quota.sum())} '
              f'counts_sum={int(counts.sum())}', flush=True)
        print(f'  hot expert {hot} quotas={[int(quota[h]) for h in hot_slots]}', flush=True)
    bad = 0 if ok else 1
    bad_t = torch.tensor([bad], device=dev)
    dist.all_reduce(bad_t, group=group)
    if rank == 0:
        print('ECO_EP8 ALL_RANKS_OK' if int(bad_t) == 0 else 'ECO_EP8 FAILED', flush=True)


if __name__ == '__main__':
    main()
