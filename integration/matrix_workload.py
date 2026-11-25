"""Boundary checks for the placements at a single batch size."""

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


def verify_rr_counts(buf):
    """Check dispatch slot counters against the pinned TileKernels gate outside timing."""
    tape = buf.fixed_routes
    rank = int(buf.rank_idx)
    width = tape.counts.shape[1]
    inverse = buf.expert_slots
    copies = (inverse >= 0).sum(-1)
    token_key = (rank + 23333 * torch.arange(tape.batch, device=inverse.device)).view(1, -1, 1)
    mismatches = torch.zeros((), dtype=torch.long, device=inverse.device)
    for start in range(0, len(tape.ids), 256):
        ids = tape.ids[start : start + 256]
        layer = tape.layer_ids[start : start + len(ids)].long().view(-1, 1, 1)
        copy = token_key % copies[layer, ids]
        slot = inverse[layer, ids, copy].long().flatten(1)
        expected = tape.counts.new_zeros((len(ids), width * buf.group.size()))
        expected.scatter_add_(1, slot, torch.ones_like(slot, dtype=expected.dtype))
        dist.all_reduce(expected, group=buf.group)
        received = tape.counts[start : start + len(ids)]
        mismatches += torch.count_nonzero(expected[:, rank * width : (rank + 1) * width] != received)
    dist.all_reduce(mismatches, group=buf.group)
    if mismatches.item():
        raise ValueError(f"RR differs from the upstream gate at {mismatches.item()} physical slot counters")
    return len(tape.ids)


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
    if len(layers) != tape.layers or any(len(parts) != (4 if hasattr(buf, "mega") else 6) for parts in layers):
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

        path = Path(os.environ["GC_MOE_FIXED_DIR"]) / (os.getenv("SM90_EIGHT_ARM", os.environ["GC_MOE_MATRIX_ARM"]) + "-layout.json")
        with path.open("x") as f:
            json.dump(layout, f)
    plan_price = None
    if buf.eco and os.environ.get("GC_MOE_MATRIX_SIZE") == "24":
        from gc_moe.jit_kernels.impls.sm90_fp8_fp4_gc_moe import prepared_matrix_price

        plan_price = prepared_matrix_price(buf.layout, tape.batch)
        arm = os.environ["GC_MOE_MATRIX_ARM"]
        expected = {"mode": 2, "slack": 11469}
        if arm in ("eco_linear", "eco_linear_wa1"):
            expected = {"mode": 0, "slack": 90 + int(arm == "eco_linear_wa1")}
        if os.getenv("GC_MOE_EP8_LINEAR") == "1" or os.getenv("SM90_TUNED_EIGHT") == "1":
            expected = {"mode": 0, "slack": 92}
        if plan_price != expected:
            raise ValueError(f"prepared ECO price {plan_price} differs from assigned {expected}")
    controls = runner_boundary(worker.model_runner, ref)
    controls.update(
        next_epoch=int(buf.header[0].item()), logical_weights=json_hash(canonical), demand=buf._matrix_demand
    )
    weights_digest = controls["logical_weights"]
    if os.getenv("SM90_EIGHT_ARM"):
        # Packed hashes are comparable within one backend; the runner compares
        # these separately. KV, routing demand and request controls remain strict.
        del controls["logical_weights"]
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
        plan_price=plan_price,
        logical_weights=weights_digest,
        placement=json_hash(by_rank),
        installed_layout=json_hash(layout),
        demand=buf._matrix_demand,
    )
