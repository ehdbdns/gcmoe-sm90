"""SM90 EP expert permutation (pi) and closed-form replica water-filling.

Design contract
---------------
* Replica *placement* comes only from the static EPLB layout; this module never
  decides where a replica lives.
* The plan only decides *token dispatch*: which replica serves a given slot.
* Every function here is a deterministic pure function, so all ranks derive the
  same result from the same ``(hist, owners)`` input and nothing has to be sent
  back across ranks (8 KB histogram in, 0 bytes out).

Call graph
----------
1. host: :func:`build_permutation` puts replica-free experts first; ``topk_idx``
   is remapped with the inverse so the kernel keeps using its own ordering.
2. kernel (dispatch role, after push): local ``send_count`` is read from every
   peer to form ``hist[ranks][experts]``.
3. kernel (one CTA): :func:`build_waterfill_table` equivalent computes
   ``(D, m, cumsum)`` per expert; pull resolves a slot in O(1).

Closed form
-----------
For expert ``e`` with demand ``D`` (all slots summed over ranks), ``m`` replicas
and ``cum[r]`` = slots contributed by ranks ``< r``, slot ``s`` of rank ``r`` is
served by

    rep = floor((cum[r] + s) * m / D)

which hands every replica ``floor(D/m)`` or ``ceil(D/m)`` slots -- maximum
deviation 1, fully deterministic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


# --------------------------------------------------------------------------
# 1. expert permutation (pi)
# --------------------------------------------------------------------------
def build_permutation(
    local: int, replica_ranks: Sequence[Sequence[int]]
) -> tuple[tuple[int, ...], int]:
    """Order experts so replica-free ones come first.

    Args:
        local: experts per rank (``experts // ranks``).
        replica_ranks: ``local`` entries; ``replica_ranks[e]`` lists the ranks
            holding a copy of local expert ``e`` (including its home rank).
            An empty or single-entry list means "no replica".

    Returns:
        ``(perm, n_norep)`` where ``perm[new_id] = original_id`` and the first
        ``n_norep`` entries are replica-free.
    """
    if len(replica_ranks) != local:
        raise ValueError("replica_ranks must have exactly `local` entries")
    norep = [e for e in range(local) if len(replica_ranks[e]) <= 1]
    rep = [e for e in range(local) if len(replica_ranks[e]) > 1]
    return tuple(norep + rep), len(norep)


def invert_permutation(perm: Sequence[int]) -> tuple[int, ...]:
    """Inverse of ``perm``: original id -> new id (used to remap ``topk_idx``)."""
    inverse = [0] * len(perm)
    for new_id, original in enumerate(perm):
        inverse[original] = new_id
    return tuple(inverse)


def remap_topk_idx(topk_idx: Sequence[Sequence[int]],
                   inverse: Sequence[int]) -> list[list[int]]:
    """Apply the inverse permutation to a ``[tokens][topk]`` routing table.

    Padding entries (negative) are preserved.
    """
    return [[-1 if e < 0 else inverse[e] for e in row] for row in topk_idx]


def permute_expert_major(tensors: Iterable, perm: Sequence[int]) -> list:
    """Reorder expert-major tensors along dim 0 (host side, once)."""
    index = list(perm)
    return [t[index] for t in tensors]


# --------------------------------------------------------------------------
# 2. histogram
# --------------------------------------------------------------------------
def histogram_from_routes(
    topk_idx_by_rank: Sequence[Sequence[Sequence[int]]],
    ranks: int,
    local: int,
) -> list[list[int]]:
    """Reference histogram ``hist[expert][rank]`` from remapped routing tables.

    Test / host-side cross-check only; the kernel derives this from the
    ``send_count`` broadcast.
    """
    experts = ranks * local
    hist = [[0] * ranks for _ in range(experts)]
    for r in range(ranks):
        for row in topk_idx_by_rank[r]:
            for e in row:
                if 0 <= e < experts:
                    hist[e][r] += 1
    return hist


# --------------------------------------------------------------------------
# 3. water-filling (closed form)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ExpertPlan:
    """Water-filling parameters of one expert, shared by every rank."""

    owners: tuple[int, ...]      # ranks holding a replica, ascending
    demand: int                  # D: total slots across ranks
    replicas: int                # m
    cumsum: tuple[int, ...]      # length ranks + 1

    def replica_of(self, rank: int, slot: int) -> int:
        """Which replica (0-based) serves ``rank``'s ``slot``. O(1)."""
        if self.replicas <= 1 or self.demand <= 0:
            return 0
        rep = ((self.cumsum[rank] + slot) * self.replicas) // self.demand
        if rep < 0:
            return 0
        return self.replicas - 1 if rep >= self.replicas else rep

    def owner_rank_of(self, rank: int, slot: int) -> int:
        return self.owners[self.replica_of(rank, slot)]


def build_waterfill_table(
    hist: Sequence[Sequence[int]],
    replica_ranks: Sequence[Sequence[int]],
    n_norep: int,
) -> list[ExpertPlan | None]:
    """Plan for every replicated expert; ``None`` for replica-free ones.

    ``hist`` is indexed ``[expert][rank]`` in **permuted** expert order, as is
    ``replica_ranks``.
    """
    local = len(replica_ranks)
    ranks = len(hist[0]) if hist else 0
    plans: list[ExpertPlan | None] = [None] * local
    for e in range(n_norep, local):
        owners = tuple(sorted(replica_ranks[e]))
        if len(owners) <= 1:
            continue
        row = hist[e]
        cumsum = [0]
        for r in range(ranks):
            cumsum.append(cumsum[-1] + row[r])
        demand = cumsum[-1]
        if demand <= 0:
            continue
        plans[e] = ExpertPlan(owners=owners, demand=demand,
                              replicas=len(owners), cumsum=tuple(cumsum))
    return plans


def verify_balance(plan: ExpertPlan) -> int:
    """Max deviation of per-replica slot counts from a perfect ``D/m`` split."""
    if plan.replicas <= 1:
        return 0
    counts = [0] * plan.replicas
    for r in range(len(plan.cumsum) - 1):
        span = plan.cumsum[r + 1] - plan.cumsum[r]
        for s in range(span):
            counts[plan.replica_of(r, s)] += 1
    return max(counts) - min(counts)


def replica_balance_report(
    hist: Sequence[Sequence[int]],
    replica_ranks: Sequence[Sequence[int]],
    n_norep: int,
) -> list[tuple[int, int, int, int]]:
    """Diagnostics: ``[(expert, D, m, max_deviation), ...]``."""
    plans = build_waterfill_table(hist, replica_ranks, n_norep)
    return [(e, p.demand, p.replicas, verify_balance(p))
            for e, p in enumerate(plans) if p is not None]
