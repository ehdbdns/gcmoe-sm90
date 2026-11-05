"""Ground-truth check of the per-layer/per-slot counters (M blocks, active slots).

Runs ONE GC MoE call per arm and reports, from the kernel's own `expert_count`
(per-slot integer token counts of that call) and from the accumulated `stats`
row the window metrics are derived from:

  sum(expert_count)            -> tokens this rank pulled in the call
  #(expert_count > 0)          -> active slots in the call
  sum ceil(cnt / tile_m)       -> M chunks the routed role actually runs

and the same three numbers recomputed the way `eplb_logic.aggregate_load`
does it (per-step average + a block_m), so the two definitions can be compared
directly. `tile_m` is taken from the compiled role schedule, which is the M
chunk the SM90 compute role really uses.
"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ['GC_SHIM'])
sys.path.insert(0, os.path.join(_HERE, '..', 'source'))
sys.path.insert(0, os.path.join(_HERE, '..', 'integration'))

import torch, torch.distributed as dist
import gc_ptxas_override  # noqa: F401
import gc_moe
import gc_moe.sm90_gc as sm90_gc
from deep_gemm.utils import per_token_cast_to_fp8, per_token_cast_to_fp4
from gc_moe.sm90_gc import transform_weights_for_gc_moe_sm90
from gc_moe.apis.sm90_gc_moe import ValidatedCall
from gc_moe.jit_kernels.heuristics.sm90_gc_moe import get_sm90_role_config
from gc_moe.runtime.sm90_gc_moe import get_context
from gc_moe.layout.eplb_placement import build_eplb_layout

H, I, E, TOPK, B, T = 5120, 2304, 384, 6, 32, 32


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
    out = torch.empty((T, H), dtype=torch.bfloat16, device=dev)

    def call(buf, ww1, ww2, ids):
        n = xq.shape[0]
        buf.x[:n].copy_(xq)
        buf.x_sf[:n].copy_(xsf)
        buf.topk_idx[:n].copy_(ids.to(torch.int64))
        buf.topk_weights[:n].copy_(torch.softmax(tw, -1))
        torch.cuda.synchronize()
        gc_moe.sm90_fp8_fp4_gc_moe(out, ww1, ww2, buf, recipe=(1, 1, 32),
                                   activation='swiglu', activation_clamp=10.0,
                                   fast_math=True, num_tokens_bound=64)
        torch.cuda.synchronize()

    def report(tag, buf, per_slot, route_total, reported_bm):
        # tile_m of the compiled routed role = the real M chunk
        cfg = ValidatedCall(ranks=R, experts=E, capacity=64, topk=TOPK, hidden=H,
                            intermediate=I, tokens=T, bound=64, clamp=10.0,
                            fast_math=True, shared_experts=0,
                            physical=(buf.physical if buf.physical != E else 0),
                            eco_logical_experts=(E if getattr(buf, 'eco', False) else 0))
        tile_m = get_sm90_role_config(cfg, get_context()).tile_m
        c = per_slot.to(torch.float64)
        got = float(c.sum())
        active = int((c > 0).sum())
        blocks_true = int(torch.ceil(c / tile_m).sum()) if tile_m else -1
        # what aggregate_load() would report for the same single step
        blocks_reported = int(((c + reported_bm - 1) // reported_bm).sum())
        if rank == 0:
            print(f'[{tag}] rank0 slots={c.numel()} '
                  f'pulled={got:.0f} (routes to this rank={route_total}) '
                  f'active={active} '
                  f'M-blocks(tile_m={tile_m})={blocks_true} '
                  f'| reported: block_m={reported_bm} blocks={blocks_reported}',
                  flush=True)
        vals = [got, active, blocks_true]
        t = torch.tensor(vals, dtype=torch.float64, device=dev)
        dist.all_reduce(t, group=group)
        if rank == 0:
            print(f'[{tag}] cluster-sum pulled={t[0].item():.0f} '
                  f'(expect {T * TOPK}) active={t[1].item():.0f} '
                  f'M-blocks={t[2].item():.0f} '
                  f'(per layer, summed over ranks)', flush=True)
        return tile_m

    # ---- vanilla -------------------------------------------------------
    buf = gc_moe.get_symm_buffer_for_gc_moe_sm90(group, E, 64, TOPK, H, I, activation='swiglu')
    call(buf, w1, w2, ti)
    # routes whose target expert lives on this rank (vanilla: slot == expert)
    host_routes = int(((ti // localE) == rank).sum())
    report('vanilla', buf, buf.expert_count[:buf.physical // R].clone(), host_routes, 192)
    # repeated calls, to see whether expert_count accumulates
    call(buf, w1, w2, ti)
    c2 = buf.expert_count[:buf.physical // R].clone()
    if rank == 0:
        print(f'[vanilla] after 2nd call: pulled={float(c2.sum())} '
              f'(same call twice -> overwritten, not accumulated)', flush=True)
    buf.destroy()
    dist.barrier()

    # ---- eplb + ECO ----------------------------------------------------
    demand = torch.ones(E); demand[7] = 2.0
    lay = build_eplb_layout(demand, E, R, 0.25)
    buf2 = gc_moe.get_symm_buffer_for_gc_moe_sm90(group, E, 64, TOPK, H, I,
                                                  activation='swiglu',
                                                  physical=lay.physical, eco=True)
    plan, _ = sm90_gc.install_eplb_plan(buf2, list(lay.phy2log))
    from gc_moe.eco.meta_sm90 import build_eco_meta
    buf2.eco_meta.copy_(torch.as_tensor(build_eco_meta(list(lay.phy2log), R, E),
                                        dtype=torch.int32, device=dev))
    w1b, w2b = sm90_gc.gather_slot_weights((w1, w2), buf2, plan, group)
    call(buf2, w1b, w2b, ti)
    report('eco', buf2, buf2.expert_count[:lay.per_rank].clone(), -1, 240)
    if rank == 0:
        q = buf2.eco_quota[:lay.physical].cpu()
        print(f'[eco] quota sum={float(q.sum())} (global routes={T * TOPK})', flush=True)


if __name__ == '__main__':
    main()
