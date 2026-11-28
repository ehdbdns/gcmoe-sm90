"""Matched GC trace-off/on replay using one archived EPLB placement and logical gate bank."""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
import torch.distributed as dist


def main():
    p = argparse.ArgumentParser()
    p.add_argument("group", type=Path, help="EP8 BS directory containing plan.npz and measure-routes-R.pt")
    p.add_argument("--layer", type=int, default=20)
    p.add_argument("--step", type=int, default=2, help="one-based post-placement step")
    p.add_argument("--level", type=int, choices=(1, 2), default=1)
    p.add_argument("--pairs", type=int, default=24)
    p.add_argument("--replays", type=int, default=16)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if not 0 <= a.layer < 40 or not 1 <= a.step <= 1000 or a.pairs < 2 or a.replays < 1:
        p.error("invalid layer, step, pairs or replays")
    import gc_moe
    from gc_moe.eco.meta_sm100 import build_eco_meta_sm100
    from gc_moe.eco.measurement import snapshot

    sys.path.insert(0, str(Path(gc_moe.__file__).resolve().parents[1] / "tests"))
    from test_gc_moe_se import (build_local_weights, build_shared_weights, build_tokens, cast_tokens_to_fp8,
                                cast_grouped_weights_to_fp4, cast_shared_weight_to_fp8,
                                make_shared_input_sf, stage_inputs)

    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    if dist.get_world_size() != 8 or torch.cuda.get_device_capability(device)[0] != 10:
        raise ValueError("expected one eight-GPU Blackwell group")
    layout = np.load(a.group / "plan.npz")["layout"][a.layer]
    h, inter, logical, topk = 5120, 2304, 384, 6
    fixture = a.group / f"step{a.step:04d}-layer{a.layer:02d}-routes-{rank}.pt"
    if fixture.exists():
        ids, gates = torch.load(fixture, map_location="cpu", weights_only=True)
    else:
        bank = torch.load(a.group / f"measure-routes-{rank}.pt", map_location="cpu", weights_only=True)
        call = (a.step - 1) * 40 + a.layer
        ids, gates = (v[call] for v in bank)
        del bank
    routes, weights = ids.to(device), gates.to(device)
    del ids, gates
    rows = len(routes)
    x, _, _ = build_tokens(rows, h, logical, topk, rank, device)
    x8, xs = cast_tokens_to_fp8(x)
    # Weight and activation values are reproducible operator fixtures. Routing
    # and replica layout come from the actual model call selected above.
    routed = [build_local_weights(1, inter, h, int(expert), device) for expert in layout.reshape(8, -1)[rank]]
    rw = gc_moe.transform_weights_for_gc_moe(*(cast_grouped_weights_to_fp4(torch.cat(col))
                                               for col in zip(*routed)), activation="swiglu")
    del routed
    sw = gc_moe.transform_weights_for_gc_moe(*(cast_shared_weight_to_fp8(v)
        for v in build_shared_weights(inter, h, 1, device)), activation="swiglu")
    meta = torch.from_numpy(build_eco_meta_sm100(layout, 8, logical)).to(device)
    modes = {}
    for level in (0, a.level):
        os.environ["GC_MOE_SM100_TRACE"] = str(level)
        buf = gc_moe.get_symm_buffer_for_gc_moe(dist.group.WORLD, len(layout), rows, topk, h, inter, 1, eco=True)
        bm = int(os.getenv("GC_MOE_ECO_BLOCK_M") or gc_moe.get_block_m_for_gc_moe(
            8, logical, buf.num_max_tokens_per_rank, rows, topk))
        stage_inputs(buf, x8, xs, routes, weights, make_shared_input_sf(xs, torch.zeros_like(buf.shared_l1_acts_sf), bm))
        state = torch.zeros(len(layout) + logical, dtype=torch.int16, device=device)
        y = torch.empty((rows, h), dtype=torch.bfloat16, device=device)
        def invoke():
            gc_moe.fp8_fp4_gc_moe(y, *rw, buf, shared_l1_weights=sw[0], shared_l2_weights=sw[1],
                                  shared_recipe=(1, 128, 128), eco_meta=meta, eco_state=state,
                                  eco_logical_experts=logical)
        invoke()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            invoke()
        modes[level] = (graph, buf, state, y)
    times = {0: [], a.level: []}
    for pair in range(a.pairs):
        order = (0, a.level) if pair % 2 == 0 else (a.level, 0)
        for mode in order:
            graph, buf, state, y = modes[mode]
            state.zero_()
            for _ in range(4):
                graph.replay()
            torch.cuda.synchronize()
            dist.barrier()
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            start.record()
            for _ in range(a.replays):
                graph.replay()
            end.record()
            end.synchronize()
            elapsed = torch.tensor([start.elapsed_time(end)*1000/a.replays], device=device)
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
            times[mode].append(elapsed.item())
        same = torch.equal(modes[0][2], modes[a.level][2]) and torch.equal(modes[0][3], modes[a.level][3])
        snap = snapshot(modes[a.level][1])
        same &= torch.equal(snap["quota"], snap["counts"])
        if a.level == 2:
            same &= int(snap["clocks"][20]) <= len(snap["events"])
        ok = torch.tensor([int(same)], device=device)
        dist.all_reduce(ok, op=dist.ReduceOp.MIN)
        if not ok.item():
            raise ValueError("trace pair differs in outputs, warm state, quotas, or event capacity")
    if rank == 0:
        delta = 100 * (np.asarray(times[a.level]) / times[0] - 1)
        boot = np.random.default_rng(0).choice(delta, (10000, len(delta))).mean(1)
        a.out.write_text(json.dumps(dict(step=a.step, layer=a.layer, block_m=bm, level=a.level,
            rank_max_us=times, paired_change_pct=delta.tolist(), mean_change_pct=float(delta.mean()),
            bootstrap_95pct=np.quantile(boot, [.025, .975]).tolist()), indent=2))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
