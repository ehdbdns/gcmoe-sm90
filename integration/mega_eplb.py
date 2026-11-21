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
