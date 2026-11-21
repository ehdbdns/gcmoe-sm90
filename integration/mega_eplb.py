"""Translate a layer's logical gate output to global DeepGEMM physical slots."""
import torch


def mega_rr_slots(logical_ids, ordered_slots, replica_counts, ep_rank):
    """ordered_slots and replica_counts are the inverse/count returned by DeepInfer."""
    if logical_ids.ndim != 2:
        raise ValueError("expected [tokens, topk] gate output")
    ids = logical_ids.clamp_min(0).long()
    token = torch.arange(ids.shape[0], device=ids.device, dtype=torch.int64)[:, None]
    replica = (ep_rank + token * 23333) % replica_counts[ids]
    return ordered_slots[ids, replica].long().masked_fill(logical_ids < 0, -1)


def mega_quota_slots(logical_ids, physical_to_logical, quota, prior_rows, global_rows):
    """Consume physical quotas in stable (source rank, token, top-k) order."""
    experts = global_rows.numel()
    counts = torch.zeros_like(global_rows).scatter_add_(0, physical_to_logical.long(), quota)
    if (quota < 0).any() or not torch.equal(counts, global_rows):
        raise ValueError("physical quotas disagree with logical demand")
    order = physical_to_logical.argsort(stable=True)
    ends = quota[order].cumsum(0)
    ids = logical_ids.flatten()
    valid = ids >= 0
    active = ids[valid].long()
    by_expert = active.argsort(stable=True)
    local = active.bincount(minlength=experts)
    offsets = global_rows.cumsum(0) - global_rows + prior_rows - (local.cumsum(0) - local)
    index = torch.arange(len(active), device=ids.device) + offsets[active[by_expert]]
    assigned = torch.empty_like(active)
    assigned[by_expert] = order[torch.searchsorted(ends, index, right=True)]
    result = torch.full_like(ids, -1)
    result[valid] = assigned.to(ids.dtype)
    return result.view_as(logical_ids)
