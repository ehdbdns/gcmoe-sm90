"""DeepInfer replica selection for the V4.1 vLLM expert path."""
import torch
import triton
import triton.language as tl


@triton.jit
def _map(ids, inverse, counts, loads, enabled, limit, out, n, experts, width,
         slots, topk, rank, HAS_LIMIT: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    expert = tl.load(ids + i, i < n, other=-1).to(tl.int64)
    valid = (expert >= 0) & (expert < experts)
    safe = tl.where(valid, expert, 0)
    count = tl.load(counts + safe)
    token = i.to(tl.int64) // topk
    copy = (rank + token * 23333) % tl.maximum(count, 1)
    slot = tl.load(inverse + safe * width + copy, valid, other=-1)
    tl.store(out + i, slot, i < n)
    record = (tl.load(enabled) != 0) & (slot >= 0) & (slot < slots)
    if HAS_LIMIT:
        record = record & (token < tl.load(limit))
    tl.atomic_add(loads + tl.where(record, slot, 0), 1, record)


def map_and_record(rank, topk_ids, expert_load_view, logical_to_physical_map,
                   logical_replica_count, record_enabled, num_unpadded_tokens=None):
    ids = topk_ids.contiguous()
    if not ids.numel():
        return ids
    assert expert_load_view.ndim == 1 and expert_load_view.is_contiguous()
    out = torch.empty_like(ids)
    _map[(triton.cdiv(ids.numel(), 256),)](
        ids, logical_to_physical_map.contiguous(), logical_replica_count.contiguous(),
        expert_load_view, record_enabled, num_unpadded_tokens, out, ids.numel(),
        logical_replica_count.numel(), logical_to_physical_map.shape[1],
        expert_load_view.numel(), ids.shape[-1], rank, num_unpadded_tokens is not None, 256)
    return out
