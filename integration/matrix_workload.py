"""Boundary checks for the five placements at a single batch size."""

import os
from pathlib import Path

import numpy as np

import torch
import torch.distributed as dist

from fixed_workload import digest, publishes_checkpoint
from stream_checkpoint import checkpoint, json_hash, named_caches, runner_boundary, tensor_hash


def source_history(buf, modules, phase):
    tape = buf.fixed_routes
    ids = tape.ids.flatten(1)
    history = torch.zeros((len(ids), buf.num_experts), dtype=torch.int32, device=ids.device)
    for rows, target in zip(ids.split(256), history.split(256)):
        target.scatter_add_(1, rows, torch.ones_like(rows, dtype=torch.int32))
    dist.all_reduce(history, group=buf.group)
    history = history.reshape(tape.steps, tape.layers, -1)
    if not torch.all(history.sum(-1) == buf.group.size() * tape.batch * buf.num_topk):
        raise ValueError("missing routes in the statistics history")
    owner = next((m for m in modules if m._eplb is not None), None)
    if owner is None:
        layout = torch.arange(buf.num_experts, device=ids.device).expand(tape.layers, -1)
    else:
        layout = owner._eplb["plan"]
    rank, width = int(buf.rank_idx), tape.counts.shape[-1]
    physical = tape.counts.reshape(tape.steps, tape.layers, width)
    mapped = torch.zeros_like(history)
    slot_ids = layout[:, rank * width : (rank + 1) * width].long().expand_as(physical)
    mapped.scatter_add_(-1, slot_ids, physical)
    dist.all_reduce(mapped, group=buf.group)
    if not torch.equal(mapped, history):
        raise ValueError(f"{phase}: physical rows differ from source routes by logical expert")
    if phase == "stats" and rank == 0 and publishes_checkpoint():
        path = Path(os.environ["GC_MOE_FIXED_DIR"]) / "stats-demand.npz"
        with path.open("xb") as f:
            np.savez_compressed(f, logical=history.cpu().numpy())
    return digest(history)


def prepare_matrix(worker, buf, modules, ref):
    tape, rank = buf.fixed_routes, int(buf.rank_idx)
    owner = next((m for m in modules if getattr(m, "_eplb_shuffler", None) is not None), None)
    if owner is None:
        from deepep_compare import _LAYER_ROW

        pairs = [(layer, pair) for m in modules for layer, pair in m._packed_layers]
        ordered = sorted(pairs, key=lambda item: _LAYER_ROW[id(item[1][0][0])])
        layers = [[*pair[0], *pair[1]] for _, pair in ordered]
        layout = [list(range(buf.num_experts)) for _ in layers]
    else:
        layers = owner._eplb_shuffler.components
        layout = owner._eplb["plan"].tolist()
    if len(layers) != tape.layers or any(len(parts) != 6 for parts in layers):
        raise ValueError("incomplete packed weights")
    # Compare logical weights across physically different placements. Include all
    # six tensors and verify every additional copy against its logical expert.
    local = []
    for parts, slots in zip(layers, layout):
        width = len(slots) // buf.group.size()
        local.append(
            [
                (e, json_hash([tensor_hash(t[i]) for t in parts]))
                for i, e in enumerate(slots[rank * width : (rank + 1) * width])
            ]
        )
    shards = [None] * buf.group.size()
    dist.all_gather_object(shards, local, group=buf.group)
    canonical = []
    for layer in range(tape.layers):
        experts = {}
        for shard in shards:
            for e, value in shard[layer]:
                if experts.setdefault(e, value) != value:
                    raise ValueError(f"packed replica differs at layer {layer}, expert {e}")
        canonical.append([experts[e] for e in range(buf.num_experts)])
    if rank == 0:
        import json

        path = Path(os.environ["GC_MOE_FIXED_DIR"]) / (os.environ["GC_MOE_MATRIX_ARM"] + "-layout.json")
        with path.open("x") as f:
            json.dump(layout, f)
    controls = runner_boundary(worker.model_runner, ref)
    controls.update(
        next_epoch=int(buf.header[0].item()), logical_weights=json_hash(canonical), demand=buf._matrix_demand
    )
    snap = checkpoint(
        os.environ["GC_MOE_STRICT_STATE_DIR"],
        rank,
        publishes_checkpoint(),
        named_caches(worker.model_runner),
        controls,
    )
    if buf.eco and buf.eco_state.any().item():
        raise ValueError("ECO state was not reset by placement installation")
    # Within-rank slot permutation is allowed for the ECO single-copy prefix.
    by_rank = [
        [sorted(row[i : i + len(row) // len(shards)]) for i in range(0, len(row), len(row) // len(shards))]
        for row in layout
    ]
    return dict(
        snapshot=snap,
        logical_weights=controls["logical_weights"],
        placement=json_hash(by_rank),
        installed_layout=json_hash(layout),
        demand=buf._matrix_demand,
    )
