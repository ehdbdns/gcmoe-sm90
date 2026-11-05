"""EPLB replica placement for the SM90 GC MoE buffer.

Placement comes from the same algorithm the B arm (RTP-LLM) uses, i.e.
DeepSeek EPLB: :func:`replicate_experts` hands the hottest experts extra copies,
then :func:`balanced_packing` packs physical experts onto GPUs by load.

The only difference here is the input: B arm feeds it per-layer routing counts,
while the GC MoE buffer needs one static layout up front, so the caller passes
expert demand directly (measured or synthetic).

Sizing
------
With ``experts`` logical experts and a redundancy fraction ``r`` the number of
redundant slots is ``round(experts * r)`` and the physical expert count is
``physical = experts + round(experts * r)``. ``physical`` must stay divisible by
``ranks`` so every rank owns the same number of slots, so it is rounded up.

Redundancy is per *global* expert; each rank's local expert block owns a
disjoint set of logical experts, so a replica always lands on a different rank
(the packer is never allowed to place a physical expert back on its home rank).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class EplbLayout:
    """Static EPLB result, in the shape the SM90 buffer needs."""

    physical: int                      # total physical experts (slots)
    per_rank: int                      # physical slots on each rank
    phy2log: tuple[int, ...]           # [physical] slot -> logical expert id
    phy2rank: tuple[int, ...]          # [physical] slot -> owning rank
    slot_local: tuple[int, ...]
    """``[physical]``: the slot's index inside its owning rank's address space.
    ``phy2rank[g]`` holds ``g`` at local position ``slot_local[g]``."""
    logcnt: tuple[int, ...]            # [experts] replicas of each logical expert
    replica_ranks: tuple[tuple[int, ...], ...]
    """``[ranks][per_rank]``: for rank ``r``'s physical slot ``s``, the ranks
    that hold a copy of the same logical expert (always includes ``r``)."""

    @property
    def redundant(self) -> int:
        return self.physical - len(self.logcnt)

    @property
    def max_copies(self) -> int:
        """Largest number of copies any single expert has."""
        return max(self.logcnt) if self.logcnt else 0


def _redundant_slots(experts: int, redundancy: float, ranks: int) -> int:
    """Redundant slot count, rounded so ``physical`` divides evenly by ``ranks``."""
    if not 0.0 <= redundancy < 1.0:
        raise ValueError('redundancy must be in [0, 1)')
    want = int(round(experts * redundancy))
    physical = experts + want
    physical = int(math.ceil(physical / ranks) * ranks)
    if physical // ranks > experts and ranks > 1:
        # More slots per rank than logical experts per rank is fine (a rank can
        # physically hold more experts than it "owns"), but every rank still has
        # to hold at least one slot.
        raise ValueError('degenerate layout: too many physical slots per rank')
    return physical - experts


def _pack(weights: torch.Tensor, num_packs: int):
    """Balanced packing, identical in spirit to EPLB's ``balanced_packing``."""
    num_groups = weights.shape[-1]
    groups_per_pack = num_groups // num_packs
    if groups_per_pack == 1:
        return (torch.arange(num_groups, dtype=torch.int64, device=weights.device),
                torch.zeros(num_groups, dtype=torch.int64, device=weights.device))
    order = weights.float().sort(-1, descending=True).indices.cpu().tolist()
    pack_index = [-1] * num_groups
    rank_in_pack = [-1] * num_groups
    pack_weights = [0.0] * num_packs
    pack_items = [0] * num_packs
    for group in order:
        pack = min((i for i in range(num_packs) if pack_items[i] < groups_per_pack),
                   key=pack_weights.__getitem__)
        pack_index[group] = pack
        rank_in_pack[group] = pack_items[pack]
        pack_weights[pack] += float(weights[group])
        pack_items[pack] += 1
    return (torch.tensor(pack_index, dtype=torch.int64),
            torch.tensor(rank_in_pack, dtype=torch.int64))


def build_eplb_layout(demand, experts: int, ranks: int, redundancy: float = 0.25,
                      layer_demand=None) -> EplbLayout:
    """Build a static EPLB layout for one SM90 GC MoE buffer.

    Args:
        demand: per logical expert load ``[experts]`` (token counts). Drives
            which experts get replicas and how the packer balances GPU load.
        experts: number of logical experts.
        ranks: EP size.
        redundancy: fraction of ``experts`` to add as copies (0.25 == 25%).
        layer_demand: optional ``[layers, experts]``; when given the replica
            count per expert is the max over layers (one static layout has to
            cover the hottest layer).
    """
    if experts % ranks:
        raise ValueError('experts must be divisible by ranks')
    weight = torch.as_tensor(demand, dtype=torch.float32).reshape(-1)
    if weight.numel() != experts:
        raise ValueError(f'demand must have {experts} entries')
    if layer_demand is not None:
        stacked = torch.as_tensor(layer_demand, dtype=torch.float32)
        if stacked.shape[-1] != experts:
            raise ValueError(f'layer_demand must have {experts} columns')
        weight = stacked.max(dim=0).values
    if float(weight.sum()) <= 0:
        weight = torch.ones(experts, dtype=torch.float32)

    redundant = _redundant_slots(experts, redundancy, ranks)
    physical = experts + redundant

    # --- 1. pick which logical experts get copies (EPLB replicate_experts) ---
    # Repeatedly copy the expert with the highest load-per-current-copy.
    logcnt = [1] * experts
    phy2log = list(range(experts))
    for _ in range(redundant):
        best, best_score = 0, -1.0
        for e in range(experts):
            score = float(weight[e]) / logcnt[e]
            if score > best_score:
                best, best_score = e, score
        phy2log.append(best)
        logcnt[best] += 1
    assert len(phy2log) == physical

    # --- 2. pack physical experts onto ranks by load (EPLB balanced_packing) --
    slot_load = torch.tensor([float(weight[e]) / logcnt[e] for e in phy2log],
                             dtype=torch.float32)
    pack_index, _ = _pack(slot_load, ranks)
    pack_index = pack_index.tolist()
    if any(p < 0 for p in pack_index):
        raise RuntimeError('balanced packing left slots unassigned')

    per_rank = physical // ranks
    # Convert (slot -> pack) into a per-rank ordering: each rank's slot s is the
    # slot with rank_in_pack == s in that pack.
    rows: list[list[int]] = [[] for _ in range(ranks)]
    for slot, pack in enumerate(pack_index):
        rows[pack].append(slot)
    if any(len(row) != per_rank for row in rows):
        raise RuntimeError('balanced packing produced uneven rank occupancy')

    home = [0] * physical
    for r, row in enumerate(rows):
        for s, slot in enumerate(row):
            home[slot] = r
    # Re-index so a rank's slots are contiguous in its own address space.
    ordered = [slot for row in rows for slot in row]
    new_index = {slot: i for i, slot in enumerate(ordered)}
    phy2log = [phy2log[slot] for slot in ordered]
    home = [home[slot] for slot in ordered]

    # --- 3. per-rank replica table -----------------------------------------
    # For rank r's slot s we need every rank holding the same *logical* expert.
    holders: dict[int, list[int]] = {}
    for slot, logical in enumerate(phy2log):
        holders.setdefault(logical, []).append(home[slot])
    replica_ranks = []
    for r in range(ranks):
        row = []
        for s in range(per_rank):
            slot = r * per_rank + s
            row.append(tuple(sorted(set(holders[phy2log[slot]]))))
        replica_ranks.append(tuple(row))

    # Local index of every global slot inside its owning rank.
    slot_local = [0] * physical
    counter = [0] * ranks
    for slot in range(physical):
        r = home[slot]
        slot_local[slot] = counter[r]
        counter[r] += 1
    return EplbLayout(physical=physical, per_rank=per_rank,
                      phy2log=tuple(phy2log), phy2rank=tuple(home),
                      slot_local=tuple(slot_local), logcnt=tuple(logcnt),
                      replica_ranks=tuple(replica_ranks))
