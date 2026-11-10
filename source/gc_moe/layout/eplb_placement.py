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

# [EPLB-FIX] diagnostics for the (rare) infeasible replica request
_PACK_WARNINGS: list[str] = []


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


def _pack(weights: torch.Tensor, num_packs: int, logical_of_slot=None):
    """Balanced packing, EPLB's ``balanced_packing`` WITH the replica rule.

    [EPLB-FIX] A logical expert must never hold two slots on the SAME rank.
    The module docstring has always said so ("the packer is never allowed to
    place a physical expert back on its home rank"), but the implementation
    only balanced load -- measured on the real geometry: 5 of 64 replicated
    experts at 8 slots/rank (and 3 of 40 at 4/rank) ended up with BOTH copies
    on one rank.  Two consequences, both observed:

      * the duplicate is a passive copy (`_eco_is_active` == 0), so it pulls
        nothing and the effective redundancy silently shrinks (64 -> 59);
      * `replica_ranks[logical]` collapses to a single rank, so that expert's
        tokens cannot be spread at all -- the split the plan computes has no
        second rank to move them to.

    The fix keeps the same greedy, load-ordered walk but restricts each slot
    to the packs that do not already hold a copy of ITS logical expert.  Fall
    back to the unconstrained candidate set only if the constraint makes the
    packing infeasible (more copies of one expert than ranks), and say so.
    """
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
    # packs already holding a copy of each logical expert
    used: dict[int, set] = {}
    for group in order:
        roomy = [i for i in range(num_packs) if pack_items[i] < groups_per_pack]
        if not roomy:                       # infeasible; keep old behaviour
            roomy = list(range(num_packs))
        if logical_of_slot is None:
            cands = roomy
        else:
            e = int(logical_of_slot[group])
            taken = used.setdefault(e, set())
            cands = [i for i in roomy if i not in taken]
            if not cands:
                # Every rank already holds a copy of this expert: the replica
                # request exceeds `ranks`, so a duplicate is unavoidable.
                cands = roomy
                _PACK_WARNINGS.append(
                    f'expert {e} has more copies than ranks ({num_packs}); '
                    'a duplicate rank is unavoidable')
        pack = min(cands, key=pack_weights.__getitem__)
        if logical_of_slot is not None:
            used.setdefault(int(logical_of_slot[group]), set()).add(pack)
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
    #
    # [EPLB-FIX] A copy must land on a DIFFERENT rank, so `logcnt[e]` can never
    # exceed `ranks`: with 9 copies of one expert and 8 ranks, two of them are
    # forced onto the same rank no matter how the packer walks. Under a skewed
    # demand the greedy "hottest first" rule hit exactly that (measured:
    # max_copies = 9 with 8 hot experts). Capping the count here keeps the
    # invariant "one copy per rank" satisfiable by construction instead of
    # degrading into passive duplicates the plan can never use.
    logcnt = [1] * experts
    phy2log = list(range(experts))
    for _ in range(redundant):
        best, best_score = -1, -1.0
        for e in range(experts):
            if logcnt[e] >= ranks:        # already on every rank
                continue
            score = float(weight[e]) / logcnt[e]
            if score > best_score:
                best, best_score = e, score
        if best < 0:
            # Every expert already sits on all `ranks` ranks: the requested
            # redundancy cannot be placed without a duplicate rank, so stop
            # short and shrink `physical` instead of building a broken layout.
            _PACK_WARNINGS.append(
                f'redundancy {redundant} unreachable: every expert already has '
                f'{ranks} copies; layout shrinks to {len(phy2log)} slots')
            break
        phy2log.append(best)
        logcnt[best] += 1
    physical = len(phy2log)
    # Re-derive the slot count that this (possibly smaller) layout needs.
    per_rank = physical // ranks
    if per_rank * ranks != physical:
        raise RuntimeError(
            f'EPLB layout {physical} slots is not divisible by {ranks} ranks; '
            'request a redundancy that keeps physical a multiple of ranks')
    assert len(phy2log) == physical

    # --- 2. pack physical experts onto ranks by load (EPLB balanced_packing) --
    slot_load = torch.tensor([float(weight[e]) / logcnt[e] for e in phy2log],
                             dtype=torch.float32)
    # [EPLB-FIX] hand the packer each slot's LOGICAL expert so it can keep the
    # copies of one expert on distinct ranks.
    pack_index, _ = _pack(slot_load, ranks, logical_of_slot=phy2log)
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
