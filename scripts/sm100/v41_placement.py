"""Build a shared layerwise placement fixture and audit the real model weight move."""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch


def prepare(history, output):
    from gc_moe.eplb.policy import rebalance_experts_global

    with np.load(history) as data:
        rows = torch.from_numpy(data["logical"][:8].copy())
    assert rows.shape == (8, 40, 384) and (rows >= 0).all()
    layout, inverse, counts = rebalance_experts_global(rows, 448, 56)
    assert layout.unique(dim=0).shape[0] == 40
    for layer in range(40):
        ordered = [row[:n].tolist() for row, n in zip(inverse[layer], counts[layer])]
        flat = [slot for group in ordered for slot in group]
        assert sorted(flat) == list(range(448))
        assert all(layout[layer, slot] == e for e, group in enumerate(ordered) for slot in group)
    with output.open("xb") as f:
        np.savez_compressed(f, layout=layout.numpy(), inverse=inverse.numpy(), counts=counts.numpy())


def samples(weights):
    # Each tensor is the backend's own flattened [local experts, packed bytes] view.
    layers = []
    for components in weights:
        parts = []
        for component in components:
            raw = component.view(torch.uint8).reshape(56, -1)
            offsets = torch.arange(32, device=raw.device) * (raw.shape[1] - 1) // 31
            parts.append(raw.index_select(1, offsets))
        layers.append(torch.cat(parts, dim=1))
    return torch.stack(layers)


@torch.inference_mode()
def install(worker, filename):
    import torch.distributed as dist
    from vllm.distributed.parallel_state import get_ep_group
    from vllm.distributed.eplb.rebalance_execute import rearrange_expert_weights_inplace

    state, = worker.model_runner.eplb_state.model_states.values()
    group = get_ep_group().device_group
    old = state.physical_to_logical_map
    with np.load(filename) as data:
        layout, inverse, counts = [torch.from_numpy(data[key]).to(old.device)
                                   for key in ("layout", "inverse", "counts")]
    assert old.shape == layout.shape == (40, 448) and group.size() == 8
    views = state.model.expert_weights
    before = samples(views)
    peers = [torch.empty_like(before) for _ in range(8)]
    dist.all_gather(peers, before, group=group)
    bank = torch.cat(peers, dim=1)
    first = state.logical_to_physical_map[:, :, 0].long()
    layers = torch.arange(40, device=old.device)[:, None]
    local = layout[:, group.rank()*56:(group.rank()+1)*56].long()
    expected = bank[layers, first.gather(1, local)]
    dst = state.logical_to_physical_map
    assert dst.shape[:2] == inverse.shape[:2] and dst.size(2) >= inverse.size(2)
    moved = (old != layout).sum(dim=1).tolist()
    assert min(moved) > 0
    rearrange_expert_weights_inplace(old, layout, views, state.expert_buffer, group, state.communicator)
    mismatch = (samples(views) != expected).any(dim=-1).sum()
    dist.all_reduce(mismatch, group=group)
    if mismatch.item():
        raise RuntimeError(f"Packed-weight sampled bytes differ in {mismatch.item()} slots")
    old.copy_(layout)
    dst.fill_(-1)
    dst[:, :, :inverse.size(2)].copy_(inverse)
    state.logical_replica_count.copy_(counts)
    torch.cuda.synchronize()
    return dict(changed_slots=moved, sampled_slot_mismatches=0,
                plan_sha256=hashlib.sha256(Path(filename).read_bytes()).hexdigest())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("history", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    prepare(args.history, args.output)
