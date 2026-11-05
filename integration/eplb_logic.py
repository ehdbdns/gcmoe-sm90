"""Pure-Python EPLB helpers shared by the vLLM backend.

Kept dependency-free (torch only, no vllm, no CUDA) so the logic can be tested
on CPU without loading a model.
"""
from __future__ import annotations

import torch

from gc_moe.eplb.deepinfer_periodic.shuffle import plan_transfers


def demand_from_stats(stats, layout, rank_idx, per_rank, experts):
    """Turn this rank's per-layer per-slot counters into logical-expert demand.

    Args:
        stats: ``[layers, per_rank]`` token counters for THIS rank.
        layout: ``[physical]`` slot -> logical expert (identical on all ranks).
        rank_idx: this rank's index.
        per_rank: physical slots per rank.
        experts: logical expert count.

    Returns:
        ``[experts]`` float tensor (not reduced across ranks).
    """
    physical = len(layout)
    if stats.shape[1] != per_rank:
        raise ValueError(f'stats has {stats.shape[1]} slots, expected {per_rank}')
    if physical != per_rank * (physical // per_rank):
        raise ValueError('layout is not divisible by per_rank')
    dev = stats.device
    layout_t = torch.as_tensor(layout, dtype=torch.int64, device=dev)
    slot_ids = torch.arange(per_rank, device=dev) + rank_idx * per_rank
    per_slot = torch.zeros(physical, dtype=torch.float32, device=dev)
    # Sum over layers: one static layout must cover the whole traffic mix.
    per_slot.index_add_(0, slot_ids, stats.sum(0).float())
    demand = torch.zeros(experts, dtype=torch.float32, device=dev)
    demand.scatter_add_(0, layout_t, per_slot)
    return demand


def build_layer_index(packed_payloads):
    """Map ``id(payload tensor)`` -> layer row index.

    The caller must keep the tensors alive: python object ids are recycled once
    an object dies, and a recycled id silently routes one layer's counters into
    another layer's row (observed as 17x inflated per-layer token counts).
    """
    out = {}
    for idx, t in enumerate(packed_payloads):
        out[id(t)] = idx
    return out


def verify_shuffle(origin, target, per_rank, world):
    """CPU reference for WeightShuffler.move.

    Returns the per-rank row contents the move should produce, as
    ``world[r][s] = original global slot id whose weight belongs there``.
    """
    if len(origin) != len(target) or len(origin) % per_rank:
        raise ValueError('layout size mismatch')
    tr = plan_transfers(origin, target, per_rank)
    world_rows = [[r * per_rank + s for s in range(per_rank)] for r in range(world)]
    snapshot = [row[:] for row in world_rows]
    for r in range(world):
        lo = origin[r * per_rank:(r + 1) * per_rank]
        lt = target[r * per_rank:(r + 1) * per_rank]
        sbe = {}
        for s, e in enumerate(lo):
            sbe.setdefault(e, (r, s))
        for t in tr:
            if t.target_rank == r:
                sbe[t.expert] = (t.source_rank, t.source_slot)
        for s, e in enumerate(lt):
            if lo[s] == e:
                continue
            sr, ss = sbe[e]
            world_rows[r][s] = snapshot[sr][ss]
    return world_rows




def aggregate_load(stats, block_m, steps, world):
    """Per-step load metrics from a window-accumulated counter table.

    IMPORTANT -- `stats` is the SUM over the window, so it can only give
    *averages* plus counts that are monotone in the window ("did this slot ever
    receive a token"). It cannot give a per-step peak. The distinction matters:

      tokens   per-step average   = window_total / steps            [exact]
      blocks   per-step average   = ceil(per-step tokens / block_m) [exact]
      active   slots that received >=1 token ANYWHERE in the window
               -- NOT the per-step active count, and NOT an average of it.
               With 100 steps almost every slot qualifies, which is why this
               number always equals slots_per_rank; do not read it as occupancy.

    `active_per_step` is left None unless the caller supplies an exact sample
    (see `aggregate_load_with_sample`).

    Args:
        stats: ``[layers, per_rank]``, accumulated over the window, inflated by
            ``world`` (the kernel sums every source rank's send_count, and
            send_count is a global counter replicated on each rank).
        block_m: GEMM M-tile size.
        steps: engine steps the window covered.
        world: EP world size.

    Returns:
        dict of per-layer values plus their medians.
    """
    s = stats.to(torch.float64) / max(1, world)
    steps = max(1, int(steps))
    per_step = s / steps
    tok = per_step.sum(dim=1)
    # [METRIC-FIX] `//` on floats is `torch.floor_divide`: (x + bm - 1) // bm
    # with x < 1 gives 0, so a slot that takes a fraction of a token per step
    # was contributing ZERO blocks. The correct per-slot block count is
    # ceil(x / bm), which is what the kernel's tile loop does
    # (`for start in range(0, count, tile_m)`).
    blk = torch.ceil(per_step / block_m).sum(dim=1)
    ever_active = (stats > 0).sum(dim=1).to(torch.float64)
    mean = float(tok.mean())
    return dict(
        tokens=tok.tolist(), blocks=blk.tolist(),
        active_ever=ever_active.tolist(),
        tokens_median=float(tok.median()), tokens_min=float(tok.min()),
        tokens_max=float(tok.max()),
        blocks_median=float(blk.median()), blocks_min=float(blk.min()),
        blocks_max=float(blk.max()),
        active_ever_median=float(ever_active.median()),
        active_ever_min=float(ever_active.min()),
        active_ever_max=float(ever_active.max()),
        # IR must use the per-step average load, not a single sample.
        ir=(float(tok.max()) / mean) if mean else -1.0,
        notes=('tokens/blocks are per-step averages over the window; '
               'active_ever counts slots that saw at least one token in the '
               'whole window (not per-step occupancy)'),
    )


def aggregate_with_sample(stats, block_m, steps, world, sample=None,
                          active_sample=None):
    """`aggregate_load` plus an exact single-step sample.

    Args:
        sample: ``[per_rank]`` token counts for ONE kernel call (the kernel's
            `expert_count`, which is overwritten each call, not accumulated).
            Dividing it by `world` removes the same inflation as `stats`.
            When given, `active_step` / `blocks_step` describe that one call
            exactly, which is what "per-step occupancy" has to mean.

    Returns:
        `aggregate_load`'s dict plus ``sample_tokens``, ``active_step`` and
        ``blocks_step`` (None without a sample).
    """
    out = aggregate_load(stats, block_m, steps, world)
    if sample is None:
        out.update(sample_tokens=None, active_step=None, blocks_step=None,
                   active_header=active_sample)
        return out
    s = sample.to(torch.float64) / max(1, world)
    out.update(
        sample_tokens=float(s.sum()),
        active_step=int((s > 0).sum()),
        # [METRIC-FIX] same float-floor-division bug as `aggregate_load`
        blocks_step=int(torch.ceil(s / block_m).sum()),
        # `active_header` comes from the kernel's own header counter, which is
        # the authoritative per-call occupancy (it counts slots, not tokens).
        active_header=active_sample,
    )
    return out
