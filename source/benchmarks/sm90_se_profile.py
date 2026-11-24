"""Profile the fused SM90 shared-expert benchmark after installing the published EPLB placement."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[2]
PLACEMENT = ROOT / 'docs/trace_executor_20261119/sm90_main24_stable_decode_bs1024_results/delivery/bs1024'
T, H, I, E, K, CAP, P = 128, 5120, 2304, 384, 6, 2048, 448
SAMPLES = 3


def placement():
    from gc_moe.eplb.policy import rebalance_experts_global
    with np.load(PLACEMENT / 'stats-demand.npz') as data:
        history = torch.from_numpy(data['logical'][:, 20:21].copy())
    plan, inverse, _ = rebalance_experts_global(history, P, P // 8)
    if plan[0].tolist() != json.loads((PLACEMENT / 'placement.json').read_text())[20]:
        raise ValueError('DeepInfer placement differs from the published BS1024 layer 20')
    return plan[0].tolist(), inverse


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', required=True, choices=('rr', 'eco'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    from deep_gemm.utils import per_token_cast_to_fp8
    import gc_moe
    import gc_ptxas_override  # noqa: F401
    from fixed_workload import route_ordinals
    from gc_moe.apis.sm90_gc_moe import validate_call
    from gc_moe.eco.meta_sm90 import reorder_single_copy_first, install_into_buffer
    from gc_moe.jit_kernels.impls.sm90_fp8_fp4_gc_moe import prepare_role_binaries, prepared_matrix_price
    from gc_moe.sm90_gc import install_eplb_plan, gather_slot_weights
    from shared_input import stage_shared_input
    from test_gc_moe_se import build_local_weights, build_shared_weights
    from test_sm90_gc_moe import build_tokens, cast_grouped_weights_to_fp4, cast_tokens_to_fp8

    rank = int(os.environ['LOCAL_RANK'])
    device = torch.device('cuda', rank)
    torch.cuda.set_device(device)
    dist.init_process_group('nccl', device_id=device)
    if dist.get_world_size() != 8 or torch.cuda.get_device_capability(device) != (9, 0):
        raise ValueError('expected eight SM90 devices')
    eco = args.arm == 'eco'
    if args.arm != os.environ['GC_MOE_MATRIX_ARM']:
        raise ValueError('inconsistent arm configuration')
    root = args.output / args.arm
    root.mkdir(parents=True, exist_ok=True)
    buf = gc_moe.get_symm_buffer_for_gc_moe_sm90(dist.group.WORLD, E, CAP, K, H, I,
                                                num_shared_experts=1, physical=P, eco=eco, trace=True)
    buf.replay_stream_mode.fill_(1)
    phys, inverse = placement()
    if eco:
        phys = reorder_single_copy_first(phys, 8)[0].tolist()
    plan, slots = install_eplb_plan(buf, phys, None if eco else inverse)
    if eco:
        install_into_buffer(buf, phys, 8, E)
    raw = build_local_weights(E // 8, I, H, rank, device)
    pair = gc_moe.transform_weights_for_gc_moe_sm90(*(cast_grouped_weights_to_fp4(w) for w in raw))
    del raw
    pair = gather_slot_weights(pair, buf, plan, dist.group.WORLD)
    shared = gc_moe.transform_weights_for_gc_moe_sm90(*(
        per_token_cast_to_fp8(w, use_ue8m0=True, gran_k=32)
        for w in build_shared_weights(I, H, 1, device)))
    x, ids, weights = build_tokens(T, H, E, K, rank, device)
    digest = hashlib.sha256()
    for tensor in (x, ids, weights):
        digest.update(tensor.contiguous().view(torch.uint8).cpu().numpy().tobytes())
    dst = ids
    if not eco:
        table = slots[0]
        copy = (rank + 23333 * torch.arange(T, device=device)[:, None]) % (table >= 0).sum(-1)[ids]
        dst = table[ids, copy].long()
    buf.topk_idx[:T].copy_(dst)
    buf.route_ordinal[:T].copy_(route_ordinals(dst[None])[0])
    buf.topk_weights[:T].copy_(weights)
    q, sf = cast_tokens_to_fp8(x)
    buf.x[:T].copy_(q)
    buf.x_sf[:T].copy_(sf)
    stage_shared_input(x, buf)
    y = torch.empty_like(x)
    def call():
        gc_moe.sm90_fp8_fp4_gc_moe(y, *pair, buf, recipe=(1, 1, 32), activation='swiglu',
                                   activation_clamp=10., fast_math=True, num_tokens_bound=CAP,
                                   shared_l1_weights=shared[0], shared_l2_weights=shared[1])
    for _ in range(3):
        call()
    torch.cuda.synchronize()
    cfg = validate_call(y, *pair, buf, None, (1, 1, 32), 'swiglu', 10., True, CAP, *shared)
    binaries, _ = prepare_role_binaries(cfg, y, *pair, buf, shared_l1_weights=shared[0],
                                        shared_l2_weights=shared[1])
    roles = {b.role: [b.grid, b.threads] for b in binaries}
    price = prepared_matrix_price(buf.layout, T) if eco else None
    if eco and price != {'mode': 2, 'slack': 11469}:
        raise ValueError(f'unexpected plan price {price}')
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()
    epoch = int(buf.header[0])
    buf.long_clocks.zero_()
    buf.long_clocks[:, :2].fill_(-1)
    dist.barrier()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as prof:
        dist.barrier()
        for _ in range(SAMPLES):
            graph.replay()
        torch.cuda.synchronize()
    prof.export_chrome_trace(str(root / f'rank{rank}.json'))
    if int(buf.header[0]) != epoch + SAMPLES or not torch.isfinite(y).all():
        raise ValueError('incomplete replay or nonfinite output')
    rows = []
    logical = torch.bincount(ids.flatten(), minlength=E)
    dist.all_reduce(logical)
    local = plan[rank * 56:(rank + 1) * 56].long()
    for tag in range(epoch, epoch + SAMPLES):
        counts = buf.profile_slot_counts[tag % 256].long()
        observed = torch.zeros_like(logical).scatter_add_(0, local, counts)
        dist.all_reduce(observed)
        if not torch.equal(observed, logical):
            raise ValueError('physical counts differ from requested logical routes')
        stamps = buf.pull_tl.reshape(256, -1)[tag % 256].tolist()
        rows.append(dict(epoch=tag, stamps=[v % 2**32 for v in stamps],
                         clocks=buf.long_clocks[tag % len(buf.long_clocks)].tolist(),
                         rank_tokens=int(counts.sum()), m_blocks=int(((counts + 63) // 64).sum())))
    (root / f'rank{rank}-device.json').write_text(json.dumps(dict(
        roles=roles, calls=rows, input_sha256=digest.hexdigest(), placement=phys, price=price)))
    dist.barrier()
    dist.destroy_process_group()
