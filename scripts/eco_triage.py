"""Watch the ECO handshake markers while one ECO call runs (deadlock triage).

Launches a tiny ECO call in a subprocess and samples `header[24..31]` on rank
0's buffer through a side-channel shm file written by the worker. Simpler: the
worker itself dumps the markers to /tmp every 200 ms, so a hung run still
leaves a trace.
"""
import os, sys, time, json, threading
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ['GC_SHIM'])
sys.path.insert(0, os.path.join(_HERE, '..', 'source'))
sys.path.insert(0, os.path.join(_HERE, '..', 'integration'))

import torch, torch.distributed as dist
import numpy as np
import gc_ptxas_override  # noqa: F401
import gc_moe
import gc_moe.sm90_gc as sm90_gc
from deep_gemm.utils import per_token_cast_to_fp8, per_token_cast_to_fp4
from gc_moe.sm90_gc import transform_weights_for_gc_moe_sm90
from gc_moe.layout.eplb_placement import build_eplb_layout

H, I, E, TOPK, CAP, T, B = 5120, 2304, 384, 6, 64, 32, 32


def main():
    lr = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(lr)
    dev = torch.device(f'cuda:{lr}')
    dist.init_process_group('gloo', init_method='env://', rank=lr, world_size=8)
    group = dist.new_group(list(range(8)), backend='nccl')
    dist.all_reduce(torch.ones(1, dtype=torch.int32, device=dev), group=group)
    torch.cuda.synchronize()
    rank = dist.get_rank(group)
    R, gen = 8, torch.Generator(device=dev).manual_seed(2026)

    def cast(shape):
        g, n, k = shape
        q = torch.empty((g, n, k // 2), dtype=torch.int8, device=dev)
        s = torch.empty((g, n, k // B), dtype=torch.float32, device=dev)
        for e in range(g):
            raw = torch.randn((n, k), dtype=torch.bfloat16, device=dev, generator=gen) * 0.05
            q[e], s[e] = per_token_cast_to_fp4(raw, use_ue8m0=True, gran_k=B)
        return q, s

    localE = E // R
    w1, w2 = transform_weights_for_gc_moe_sm90(cast((localE, 2 * I, H)), cast((localE, H, I)))
    raw = torch.randn((T, H), dtype=torch.bfloat16, device=dev, generator=gen) * 0.5
    xq, xsf = per_token_cast_to_fp8(raw, use_ue8m0=False, gran_k=128)
    sc = torch.randn((T, E), dtype=torch.float32, device=dev, generator=gen)
    tw, ti = torch.topk(sc, TOPK, dim=-1, largest=True, sorted=False)

    demand = torch.ones(E)
    demand[7] = 3.0
    lay = build_eplb_layout(demand, E, R, 0.25)
    buf = gc_moe.get_symm_buffer_for_gc_moe_sm90(
        group, E, CAP, TOPK, H, I, activation='swiglu',
        physical=lay.physical, eco=True)
    plan, slots_of = sm90_gc.install_eplb_plan(buf, list(lay.phy2log))
    from gc_moe.eco.meta_sm90 import build_eco_meta
    buf.eco_meta.copy_(torch.as_tensor(
        build_eco_meta(list(lay.phy2log), R, E), dtype=torch.int32, device=dev))
    w1b, w2b = sm90_gc.gather_slot_weights((w1, w2), buf, plan, group)

    if rank == 0:
        buf.header[24] = 777
        buf.header[25] = 777
        buf.header[26] = 777
        buf.header[27] = 777
        buf.header[28] = 777
        buf.header[29] = 777
        buf.header[30] = 777
    out = torch.empty((T, H), dtype=torch.bfloat16, device=dev)
    buf.x[:T].copy_(xq)
    buf.x_sf[:T].copy_(xsf)
    buf.topk_idx[:T].copy_(ti.to(torch.int64))
    buf.topk_weights[:T].copy_(torch.softmax(tw, -1))
    torch.cuda.synchronize()

    stop = threading.Event()

    def sampler():
        path = f'/tmp/eco_triage_{rank}.json'
        while not stop.is_set():
            try:
                h = buf.header.cpu().tolist()
                q = buf.eco_quota[:4].cpu().tolist()
                rd = buf.eco_ready.cpu().tolist()
                with open(path, 'w') as f:
                    json.dump({'header': h, 'quota': q, 'ready': rd}, f)
            except Exception as exc:  # noqa: BLE001
                pass
            time.sleep(0.2)

    th = threading.Thread(target=sampler, daemon=True)
    th.start()
    if rank == 0:
        print('[triage] launching one ECO call', flush=True)
    try:
        gc_moe.sm90_fp8_fp4_gc_moe(out, w1b, w2b, buf, recipe=(1, 1, 32),
                                   activation='swiglu', activation_clamp=10.0,
                                   fast_math=True, num_tokens_bound=CAP)
        torch.cuda.synchronize()
        if rank == 0:
            print('[triage] call returned', flush=True)
    finally:
        stop.set()
        th.join(timeout=1)
    # ---- collective diagnostics (must run on EVERY rank) ---------------
    local_counts = torch.bincount(ti.reshape(-1), minlength=E).to(torch.int64)
    dist.all_reduce(local_counts, group=group)
    phy = list(lay.phy2log)
    ec = buf.expert_count[:lay.per_rank].to(torch.int64).cpu().numpy()
    q = buf.eco_quota[:lay.physical].to(torch.int64).cpu().numpy()
    st = buf.eco_quota_start[:lay.physical].to(torch.int64).cpu().numpy()
    sc = buf.send_count[:E].to(torch.int64).cpu().numpy()

    if rank == 0:
        h = buf.header.cpu().tolist()
        print('[triage] final header', h, flush=True)
        print('[triage] dispatch entered (h23)', h[23], 'markers h24..30', h[24:31], flush=True)
        print(f'[triage] send_count sum={int(sc.sum())} nonzero={int((sc > 0).sum())}', flush=True)
        print(f'[triage] quota sum={int(q.sum())} demand sum={int(local_counts.sum())}', flush=True)
        print(f'[triage] expert_count(all ranks) sum='
              f'{int(sum(int(x) for x in [0]))}', flush=True)

    # per-rank slices so every rank can report its own slots
    g0 = int(buf.rank_idx) * lay.per_rank
    bad = [(s_, phy[g0 + s_], int(q[g0 + s_]), int(ec[s_]),
            int(local_counts[phy[g0 + s_]]), int(sc[phy[g0 + s_]]), int(st[g0 + s_]))
           for s_ in range(lay.per_rank) if int(q[g0 + s_]) != int(ec[s_])]
    if int(buf.rank_idx) == 0:
        rows = [(s_, int(q[g0+s_]), int(ec[s_]), int(st[g0+s_]))
                for s_ in range(lay.per_rank)]
        print('[triage] rank0 (slot,quota,count,start) all:', flush=True)
        for chunk in range(0, lay.per_rank, 10):
            print('   ', rows[chunk:chunk+10], flush=True)
    print(f'[triage] rank={int(buf.rank_idx)} slots={lay.per_rank} '
          f'quota_sum={int(q[g0:g0 + lay.per_rank].sum())} '
          f'count_sum={int(ec.sum())} mismatched={len(bad)} '
          f'(slot,expert,quota,count,demand,send,start) first={bad[:4]}', flush=True)
    # owner-rank view: for each of this rank's slots, where do the copies live?
    copies_view = []
    for s_ in range(min(6, lay.per_rank)):
        e = phy[g0 + s_]
        holders = [g for g in range(lay.physical) if phy[g] == e]
        copies_view.append((s_, e, holders, [int(q[h]) for h in holders],
                            int(local_counts[e])))
    print(f'[triage] rank={int(buf.rank_idx)} copies view={copies_view}', flush=True)
    # per-rank send_count (logical) gathered, so we can see the true D_e
    sc_all = [torch.zeros(E, dtype=torch.int64, device=dev) for _ in range(R)]
    dist.all_gather(sc_all, buf.send_count[:E].to(torch.int64), group=group)
    sc_sum = torch.stack(sc_all).sum(0).cpu().numpy()
    if g0 == 0:
        for s_ in [1, 2, 14, 16, 19, 20]:
            e = phy[s_]
            holders = [g for g in range(lay.physical) if phy[g] == e]
            print(f'[triage] slot {s_} expert {e} demand={int(local_counts[e])} '
                  f'D_e(send sum)={int(sc_sum[e])} per-rank={[int(x[e]) for x in sc_all]} '
                  f'quota={[int(q[h]) for h in holders]} start={[int(st[h]) for h in holders]} '
                  f'count(holder ranks)={[int(ec[h % lay.per_rank]) if (h//lay.per_rank)==0 else -1 for h in holders]}', flush=True)
    if g0 == 0:
        td = buf.eco_dbg[:lay.physical].cpu().numpy().astype(np.int64)
        lo_d = buf.eco_dbg[1*lay.physical:2*lay.physical].cpu().numpy().astype(np.int64)
        hi_d = buf.eco_dbg[2*lay.physical:3*lay.physical].cpu().numpy().astype(np.int64)
        ow_d = buf.eco_dbg[3*lay.physical:4*lay.physical].cpu().numpy().astype(np.int64)

        per_src = buf.eco_dbg[4*lay.physical:4*lay.physical+48].cpu().numpy()
        print(f'[triage] counts per source (slots 0..5) = {per_src.reshape(6, 8).tolist()}', flush=True)
        print(f'[triage] lo[0:24]={lo_d[:24].tolist()}', flush=True)
        print(f'[triage] hi[0:24]={hi_d[:24].tolist()}', flush=True)
        print(f'[triage] stream_total[0:60]={td[:lay.per_rank].tolist()}', flush=True)
        print(f'[triage] owned[0:60]={ow_d[:lay.per_rank].tolist()}', flush=True)
        hi2 = buf.eco_dbg[4*lay.physical:5*lay.physical].cpu().numpy().astype(np.int64)
        tot2 = buf.eco_dbg[5*lay.physical:6*lay.physical].cpu().numpy().astype(np.int64)
        print(f'[triage] hi@count[0:24]={hi2[:24].tolist()}', flush=True)
        print(f'[triage] total@count[0:24]={tot2[:24].tolist()}', flush=True)
        print(f'[triage] expert_count(rank0)[0:24]={ec[:24].tolist()}', flush=True)
        pass
        print(f'[triage] expert_count[0:24]={ec[:24].tolist()}', flush=True)
        print(f'[triage] ids equal? {bool((own_d[:lay.per_rank]==ec).all())}', flush=True)
        print(f'[triage] quota[:12]={q[:12].tolist()}', flush=True)
        print(f'[triage] start[:12]={st[:12].tolist()}', flush=True)
        print(f'[triage] expert_count(rank0)[:12]={ec[:12].tolist()}', flush=True)
        print(f'[triage] send_count[e] for phy[:12]={[int(sc[phy[i]]) for i in range(12)]}', flush=True)


if __name__ == '__main__':
    main()
