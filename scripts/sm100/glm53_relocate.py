"""Move GC's already-packed expert weights into their post-EPLB graph slots."""
import numpy as np
import torch
import torch.distributed as dist


def exchange_indices(previous, following, ranks, rank):
    old = np.asarray(previous).reshape(ranks, -1)
    new = np.asarray(following).reshape(ranks, -1)
    if old.shape != new.shape or set(old.ravel()) != set(new.ravel()):
        raise ValueError("EPLB relocation requires the same physical capacity and logical experts")
    copies = {}
    for source, slots in enumerate(old):
        for slot, expert in enumerate(slots.tolist()):
            copies.setdefault(expert, {}).setdefault(source, slot)
    sends = [[] for _ in range(ranks)]
    receives = [[] for _ in range(ranks)]
    for dest, slots in enumerate(new):
        for slot, expert in enumerate(slots.tolist()):
            owners = copies[expert]
            source = dest if dest in owners else next(iter(owners))
            if source == rank:
                sends[dest].append(owners[source])
            if dest == rank:
                receives[source].append(slot)
    return sends, receives


def relocate(weights, previous, following, group):
    rank = dist.get_rank(group)
    ranks = dist.get_world_size(group)
    sends, receives = exchange_indices(previous, following, ranks, rank)
    device = weights[0][0].device
    src = torch.tensor([i for slots in sends for i in slots], device=device, dtype=torch.long)
    dst = torch.tensor([i for slots in receives for i in slots], device=device, dtype=torch.long)
    send_lengths, recv_lengths = list(map(len, sends)), list(map(len, receives))
    for projection in weights:
        for tensor in projection:
            if tensor.shape[0] != len(dst):
                raise ValueError("GC bank capacity differs from the placement")
            # All source rows are gathered before any destination overwrites a
            # captured address. NCCL exchanges contiguous rows; index_copy_
            # writes scale words using the bank's original MN-major strides.
            payload = tensor.index_select(0, src).contiguous()
            moved = torch.empty(tensor.shape, dtype=tensor.dtype, device=device)
            dist.all_to_all_single(moved, payload, recv_lengths, send_lengths, group=group)
            tensor.index_copy_(0, dst, moved)
