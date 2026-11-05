"""Minimal CUDA-graph capture check for the ECO path (no model weights).

vLLM captures a graph around the MoE call. If the kernel's cross-CTA/peer
handshake cannot complete inside a capture, the capture is invalidated and the
error surfaces much later as "operation failed due to a previous error during
capture". This reproduces that in ~20 s:

  run 1..N : eager (warms JIT + the graph)
  then     : torch.cuda.graph() around the same call, replay twice

Usage: torchrun --nproc_per_node 8 scripts/verify_eco_capture.py
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
from gc_moe.layout.eplb_placement import build_eplb_layout

H, I, E, TOPK, B = 5120, 2304, 384, 6, 32
CAP = int(os.environ.get('TL_CAP', '2048'))
NSE = int(os.environ.get('SHARED_EXPERTS', '1'))
T = int(os.environ.get('TL_TOKENS', '32'))


def main():
    lr = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(lr)
    dev = torch.device(f'cuda:{lr}')
    dist.init_process_group('gloo', init_method='env://', rank=lr, world_size=8)
    group = dist.new_group(list(range(8)), backend='nccl')
    torch.cuda.synchronize()
    rank = dist.get_rank(group)
    R, gen = 8, torch.Generator(device=dev).manual_seed(7)

    def cast(shape):
        g, n, k = shape
        q = torch.empty((g, n, k // 2), dtype=torch.int8, device=dev)
        s = torch.empty((g, n, k // B), dtype=torch.float32, device=dev)
        for e in range(g):
            raw = torch.randn((n, k), dtype=torch.bfloat16, device=dev, generator=gen) * 0.05
            q[e], s[e] = per_token_cast_to_fp4(raw, use_ue8m0=True, gran_k=B)
        return q, s

    localE = E // R
    w1, w2 = transform_weights_for_gc_moe_sm90(cast((localE, 2 * I, H)),
                                               cast((localE, H, I)))
    si = I * NSE
    sw1 = (torch.randn((2 * si, H), dtype=torch.float32, device=dev).to(torch.float8_e4m3fn),
           torch.rand((2 * si, H // 32), dtype=torch.float32, device=dev) + 0.5)
    sw2 = (torch.randn((H, si), dtype=torch.float32, device=dev).to(torch.float8_e4m3fn),
           torch.rand((H, si // 32), dtype=torch.float32, device=dev) + 0.5)
    sw1t, sw2t = transform_weights_for_gc_moe_sm90(sw1, sw2)

    raw = torch.randn((T, H), dtype=torch.bfloat16, device=dev, generator=gen) * 0.5
    xq, xsf = per_token_cast_to_fp8(raw, use_ue8m0=False, gran_k=128)
    sc = torch.randn((T, E), dtype=torch.float32, device=dev, generator=gen)
    tw, ti = torch.topk(sc, TOPK, dim=-1, largest=True, sorted=False)
    weights = torch.softmax(tw, -1)

    demand = torch.ones(E, dtype=torch.float32)
    demand[7] = 2.0
    lay = build_eplb_layout(demand, E, R, 0.25)
    buf = gc_moe.get_symm_buffer_for_gc_moe_sm90(
        group, E, CAP, TOPK, H, I, activation='swiglu',
        num_shared_experts=NSE, physical=lay.physical, eco=True)
    plan, _ = sm90_gc.install_eplb_plan(buf, list(lay.phy2log))
    from gc_moe.eco.meta_sm90 import build_eco_meta
    buf.eco_meta.copy_(torch.as_tensor(build_eco_meta(list(lay.phy2log), R, E),
                                       dtype=torch.int32, device=dev))
    w1b, w2b = sm90_gc.gather_slot_weights((w1, w2), buf, plan, group)

    out = torch.empty((T, H), dtype=torch.bfloat16, device=dev)
    buf.x[:T].copy_(xq)
    buf.x_sf[:T].copy_(xsf)
    buf.topk_idx[:T].copy_(ti.to(torch.int64))
    buf.topk_weights[:T].copy_(weights)

    def call():
        gc_moe.sm90_fp8_fp4_gc_moe(
            out, w1b, w2b, buf, recipe=(1, 1, 32), activation='swiglu',
            activation_clamp=10.0, fast_math=True, num_tokens_bound=CAP,
            shared_l1_weights=sw1t, shared_l2_weights=sw2t)

    for _ in range(3):
        call()
    torch.cuda.synchronize()
    if rank == 0:
        print('[cap] eager warm done', flush=True)

    g = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(g):
            call()
        torch.cuda.synchronize()
    except Exception as exc:  # noqa: BLE001
        print(f'[cap] CAPTURE FAILED rank{rank}: {type(exc).__name__}: {exc}', flush=True)
        raise
    for _ in range(2):
        g.replay()
    torch.cuda.synchronize()
    finite = bool(torch.isfinite(out.float()).all())
    if rank == 0:
        q = buf.eco_quota[:lay.physical].to(torch.int64)
        print(f'[cap] replay ok finite={finite} quota_sum={int(q.sum())}', flush=True)
        print('ECO_CAPTURE OK', flush=True)


if __name__ == '__main__':
    main()
