"""Profile graph replays with the benchmark inputs and the selected layer placements."""
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
H, I, E, K, CAP, P = 5120, 2304, 384, 6, 2048, 448
SAMPLES = 3


def placement(group, arm, layer):
    from gc_moe.eplb.policy import rebalance_experts_global
    installed = json.loads((group / arm / 'installed-layout.json').read_text())[layer]
    if arm == 'vanilla':
        if installed != list(range(E)):
            raise ValueError('vanilla requires identity placement')
        return installed, None
    if arm == 'random':
        return installed, None
    with np.load(group / 'stats-demand.npz') as data:
        history = torch.from_numpy(data['logical'][:, layer:layer+1].copy())
    plan, inverse, _ = rebalance_experts_global(history, P, P // 8)
    if plan[0].tolist() != json.loads((group / 'placement.json').read_text())[layer]:
        raise ValueError('DeepInfer placement differs from the reference layer')
    phys = plan[0].tolist()
    if arm.startswith('eco'):
        from gc_moe.eco.meta_sm90 import reorder_single_copy_first
        phys = reorder_single_copy_first(phys, 8)[0].tolist()
    if phys != installed:
        raise ValueError(f'{arm} benchmark layout differs from the model')
    return phys, inverse


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', required=True, choices=('vanilla', 'random', 'rr', 'eco', 'eco_smr'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bs', type=int, choices=(512, 1024), default=1024)
    parser.add_argument('--group', type=Path, default=PLACEMENT)
    parser.add_argument("--layers", nargs="+", type=int, choices=range(40))
    parser.add_argument("--routes", type=Path, help="verified three-step history directory")
    args = parser.parse_args(argv)
    tokens = args.bs if args.routes else args.bs // 8
    from deep_gemm.utils import per_token_cast_to_fp8
    import gc_moe
    import gc_ptxas_override  # noqa: F401
    from gc_moe.eco.meta_sm90 import install_into_buffer
    from gc_moe.sm90_gc import install_eplb_plan, gather_slot_weights
    from test_gc_moe_se import build_local_weights, build_shared_weights
    from test_sm90_gc_moe import cast_grouped_weights_to_fp4

    rank = int(os.environ['LOCAL_RANK'])
    device = torch.device('cuda', rank)
    torch.cuda.set_device(device)
    dist.init_process_group('nccl', device_id=device)
    if dist.get_world_size() != 8 or torch.cuda.get_device_capability(device) != (9, 0):
        raise ValueError('expected eight SM90 devices')
    eco = args.arm.startswith('eco')
    if args.arm != os.environ['GC_MOE_MATRIX_ARM']:
        raise ValueError('inconsistent arm configuration')
    physical = E if args.arm == 'vanilla' else P
    buf = gc_moe.get_symm_buffer_for_gc_moe_sm90(dist.group.WORLD, E, CAP, K, H, I,
                                                num_shared_experts=1, physical=physical, eco=eco, trace=True)
    buf.replay_stream_mode.fill_(1)
    raw = build_local_weights(E // 8, I, H, rank, device)
    logical_pair = gc_moe.transform_weights_for_gc_moe_sm90(*(cast_grouped_weights_to_fp4(w) for w in raw))
    del raw
    shared = gc_moe.transform_weights_for_gc_moe_sm90(*(
        per_token_cast_to_fp8(w, use_ue8m0=True, gran_k=32)
        for w in build_shared_weights(I, H, 1, device)))
    slot_pair = None
    for layer in args.layers or [20]:
        root = args.output / f'layer{layer}' if args.layers else args.output
        root = root / args.arm
        root.mkdir(parents=True, exist_ok=True)
        phys, inverse = placement(args.group, args.arm, layer)
        plan, slots = install_eplb_plan(buf, phys, None if eco else inverse)
        if eco:
            install_into_buffer(buf, phys, 8, E)
        slot_pair = gather_slot_weights(logical_pair, buf, plan, dist.group.WORLD, out=slot_pair)
        profile_layer(root, layer, tokens, args.bs, rank, device, buf, eco, plan, slots, slot_pair, shared, args.routes)
    dist.barrier()
    dist.destroy_process_group()


def profile_layer(root, layer, tokens, bs, rank, device, buf, eco, plan, slots, pair, shared, routes):
    import gc_moe
    from fixed_workload import route_ordinals
    from gc_moe.apis.sm90_gc_moe import validate_call
    from gc_moe.jit_kernels.impls.sm90_fp8_fp4_gc_moe import prepare_role_binaries, prepared_matrix_price
    from shared_input import stage_shared_input
    from test_sm90_gc_moe import build_tokens, cast_tokens_to_fp8

    seed_rank = rank if layer == 20 else rank + (layer + 1) * 8
    if routes:
        generator = torch.Generator(device=device).manual_seed(42 + seed_rank * 7919)
        x = torch.randn((tokens, H), dtype=torch.bfloat16, device=device, generator=generator) * 0.5
        with np.load(routes / f'layer{layer}-rank{rank}.npz') as tape:
            ids_by_step = torch.from_numpy(tape['ids']).to(device)
            weights_by_step = torch.from_numpy(tape['weights']).to(device)
        if ids_by_step.shape != (SAMPLES, tokens, K) or weights_by_step.shape != ids_by_step.shape:
            raise ValueError('historical route slice shape mismatch')
    else:
        x, ids, weights = build_tokens(tokens, H, E, K, seed_rank, device)
        ids_by_step = ids[None]
        weights_by_step = weights[None]
    digest = hashlib.sha256()
    for tensor in (x, ids_by_step, weights_by_step):
        digest.update(tensor.contiguous().view(torch.uint8).cpu().numpy().tobytes())
    dst = ids_by_step
    if not eco:
        table = slots[0]
        copy = (rank + 23333 * torch.arange(tokens, device=device)[:, None]) % (table >= 0).sum(-1)[dst]
        dst = table[dst, copy].long()
    ordinals = route_ordinals(dst)

    def stage_routes(index):
        buf.topk_idx[:tokens].copy_(dst[index])
        buf.route_ordinal[:tokens].copy_(ordinals[index])
        buf.topk_weights[:tokens].copy_(weights_by_step[index])

    stage_routes(0)
    q, sf = cast_tokens_to_fp8(x)
    buf.x[:tokens].copy_(q)
    buf.x_sf[:tokens].copy_(sf)
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
    price = prepared_matrix_price(buf.layout, tokens) if eco else None
    if eco and price != {'mode': 2, 'slack': 11469}:
        raise ValueError(f'unexpected plan price {price}')
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()
    epoch = int(buf.header[0])
    if eco:
        buf.auction_counts.fill_(-1)
        if routes:
            buf.eco_state.zero_()  # Match the historical post-placement boundary.
    buf.long_clocks.zero_()
    buf.long_clocks[:, :2].fill_(-1)
    dist.barrier()
    replay2_output = torch.empty_like(y)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as prof:
        dist.barrier()
        for sample in range(SAMPLES):
            if routes:
                if sample:
                    dist.barrier()  # All remote reads from the prior sample have finished.
                stage_routes(sample)
                dist.barrier()
            graph.replay()
            if sample == 1:
                replay2_output.copy_(y)
        torch.cuda.synchronize()
    prof.export_chrome_trace(str(root / f'rank{rank}.json'))
    if int(buf.header[0]) != epoch + SAMPLES or not torch.isfinite(replay2_output).all():
        raise ValueError('incomplete replay or nonfinite output')
    rows = []
    per_rank = buf.physical // 8
    local = plan[rank * per_rank:(rank + 1) * per_rank].long()
    for sample, tag in enumerate(range(epoch, epoch + SAMPLES)):
        ids = ids_by_step[sample if routes else 0]
        logical = torch.bincount(ids.flatten(), minlength=E)
        dist.all_reduce(logical)
        counts = buf.profile_slot_counts[tag % 256].long()
        observed = torch.zeros_like(logical).scatter_add_(0, local, counts)
        dist.all_reduce(observed)
        if not torch.equal(observed, logical):
            raise ValueError('physical counts differ from requested logical routes')
        stamps = buf.pull_tl.reshape(256, -1)[tag % 256].tolist()
        auction = buf.auction_counts[tag % len(buf.auction_counts)].tolist() if eco else None
        rows.append(dict(epoch=tag, stamps=[v % 2**32 for v in stamps],
                         clocks=buf.long_clocks[tag % len(buf.long_clocks)].tolist(),
                         slot_counts=counts.tolist(), auction=auction,
                         rank_tokens=int(counts.sum()), m_blocks=int(((counts + 63) // 64).sum())))
    (root / f'rank{rank}-device.json').write_text(json.dumps(dict(
        roles=roles, calls=rows, bs=bs, source_tokens=tokens, historical_routes=bool(routes), layer=layer, input_sha256=digest.hexdigest(), placement=plan.tolist(), price=price,
        output_sha256=hashlib.sha256(replay2_output.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest())))
    dist.barrier()
    graph.reset()
