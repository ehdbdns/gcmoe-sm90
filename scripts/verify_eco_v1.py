"""V1: ECO region reserved but E == P (no replicas) => must be bit-identical."""
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

H, I, E, TOPK, CAP, T, B = 5120, 2304, 384, 6, 64, 32, 32


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
    dist.all_reduce(torch.ones(1, dtype=torch.int32, device=dev), group=group)
    torch.cuda.synchronize()
    rank = dist.get_rank(group)
    R = 8
    gen = torch.Generator(device=dev).manual_seed(2026)

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
    raw = torch.randn((T, H), dtype=torch.bfloat16, device=dev, generator=gen) * 0.5
    xq, xsf = per_token_cast_to_fp8(raw, use_ue8m0=False, gran_k=128)
    sc = torch.randn((T, E), dtype=torch.float32, device=dev, generator=gen)
    tw, ti = torch.topk(sc, TOPK, dim=-1, largest=True, sorted=False)
    weights = torch.softmax(tw, -1)
    out = torch.empty((T, H), dtype=torch.bfloat16, device=dev)

    buf = gc_moe.get_symm_buffer_for_gc_moe_sm90(group, E, CAP, TOPK, H, I,
                                                 activation='swiglu')
    run(out, w1, w2, buf, xq, xsf, ti, weights)
    ref = out.clone()
    buf.destroy()
    dist.barrier()
    if rank == 0:
        print('[v1] baseline done', flush=True)

    buf2 = gc_moe.get_symm_buffer_for_gc_moe_sm90(group, E, CAP, TOPK, H, I,
                                                  activation='swiglu', eco=True)
    sm90_gc.install_eplb_plan(buf2, list(range(E)))
    from gc_moe.eco.meta_sm90 import build_eco_meta
    meta = build_eco_meta(list(range(E)), R, E)
    buf2.eco_meta.copy_(torch.as_tensor(meta, dtype=torch.int32, device=dev))
    out.zero_()
    run(out, w1, w2, buf2, xq, xsf, ti, weights)

    same = bool(torch.equal(out, ref))
    dq = (out.float() - ref.float()).abs().max().item()
    if rank == 0:
        print(f'V1 identical={same} maxabs={dq:.3e}', flush=True)
        q = buf2.eco_quota[:E].to(torch.int64).cpu()
        c = buf2.expert_count[:localE].to(torch.int64)
        print(f'V1 quota sum={int(q.sum())} counts sum={int(c.sum())}', flush=True)
    bad = torch.tensor([0 if same else 1], device=dev)
    dist.all_reduce(bad, group=group)
    if rank == 0:
        print('V1 OK' if int(bad) == 0 else 'V1 FAIL', flush=True)


if __name__ == '__main__':
    main()
