"""# [ECO-SM90] Host-side EcoMeta builder for the SM90 plan role.

Layout matches `eco/include/eco_meta.cuh`, restricted to what the SM90 plan
actually reads (it decodes the static tables itself, so the SM100
`unit_section` acceleration tables are not needed):

    int32 [0, 16)                 reserved header
    int32 [16, 16 + E*16)         copies[E][32] as int16 pairs
                                  (slot id; 0x7FFF = empty; bit15 = passive)
    int32 [16 + E*16, +P)         phy2log[P]
    int32 [16 + E*16 + P, +R)     num_fixed[R]

Pure Python/numpy so it is unit-testable on CPU (`eco_ref/eco_meta.py` wraps
this module for the reference tests).
"""
from __future__ import annotations

import numpy as np

K_EMPTY = 0x7FFF
K_PASSIVE = 0x8000
COPIES_WORDS_PER_EXPERT = 16      # 32 int16



def reorder_single_copy_first(phy, num_ranks: int):
    """[ECO-WARP] Permute each rank's slots so the single-copy slots form a prefix.

    SM100's EcoMeta contract says ``num_fixed[R]`` is "a prefix of its local
    slots" (eco_meta.cuh L17), and the dispatch cursor relies on it: it walks
    the single-copy tokens first and only crosses into the replica region when
    ``token_idx >= eco_fixed_tokens``.  The EPLB packer does not produce that
    order, so the permutation is applied to the placement itself -- every
    consumer (push's source_index, pull, weights, routing map) then agrees.

    Within each rank the slots are renumbered, so a caller must use the
    RETURNED placement everywhere and never the input one.

    Returns:
        (new_phy, per_rank_order) where ``per_rank_order[r][j]`` is the index
        of the old local slot that now sits at local slot ``j`` of rank ``r``.
    """
    import numpy as np
    phy = np.asarray(phy, np.int64)
    P, R = len(phy), num_ranks
    S = P // R
    # A slot is an ACTIVE copy iff it is the first slot of its expert on its
    # own rank (build_eco_meta's rule: later duplicates on the same rank are
    # passive). `single` then means "this expert has exactly one active copy",
    # which is what SM100's num_fixed prefix is made of.
    per_rank_seen = [set() for _ in range(R)]
    active_slots = []          # (slot, expert) that are active copies
    for s in range(P):
        e = int(phy[s]); r = s // S
        if e in per_rank_seen[r]:
            continue           # passive duplicate
        per_rank_seen[r].add(e)
        active_slots.append((s, e))
    copies_of = {}
    for s, e in active_slots:
        copies_of.setdefault(e, []).append(s)
    is_single_expert = {e: len(v) == 1 for e, v in copies_of.items()}
    # A slot goes in the single-copy prefix iff it is itself an ACTIVE copy of
    # an expert that has exactly one active copy. Passive duplicates of such an
    # expert must NOT join the prefix: they pull nothing (hi = 0), and putting
    # them early would only waste a cursor step before the plan is ready.
    single_slot = {s for s, e in active_slots if is_single_expert.get(e)}
    out = np.empty_like(phy)
    order = []
    for r in range(R):
        lo, hi = r * S, (r + 1) * S
        singles, replicas = [], []
        for s in range(lo, hi):
            (singles if s in single_slot else replicas).append(s)
        seq = singles + replicas
        out[lo:hi] = phy[seq]
        order.append([s - lo for s in seq])
    return out, order

def build_eco_meta(phy, num_ranks: int, num_logical: int) -> np.ndarray:
    """Build the int32 EcoMeta for one static placement.

    Args:
        phy: ``[P]`` physical slot -> logical expert (the EPLB placement).
        num_ranks: EP size.
        num_logical: number of logical experts.

    Raises:
        ValueError: if a logical expert has no active slot (its tokens would
            simply vanish, which the kernel cannot detect).
    """
    phy = np.asarray(phy, np.int64)
    P, R, E = len(phy), num_ranks, num_logical
    S = P // R

    # copies[e]: active copies (one per rank) then passive ones; a copy is
    # passive when a lower copy of the same expert sits on the same rank.
    copies = [[] for _ in range(E)]
    passive = [[] for _ in range(E)]
    seen_rank = [set() for _ in range(E)]
    for s in range(P):
        e = int(phy[s])
        r = s // S
        if r in seen_rank[e]:
            passive[e].append(s)
        else:
            seen_rank[e].add(r)
            copies[e].append(s)
    for e in range(E):
        if not copies[e]:
            raise ValueError(f'logical expert {e} has no active copy; '
                             'its tokens would be dropped')

    words = [0] * 16
    for e in range(E):
        row = list(copies[e]) + [p | K_PASSIVE for p in passive[e]]
        row = row[:32] + [K_EMPTY] * (32 - len(row))
        for k in range(0, 32, 2):
            words.append(((row[k + 1] & 0xFFFF) << 16) | (row[k] & 0xFFFF))
    assert len(words) == 16 + E * COPIES_WORDS_PER_EXPERT

    words += [int(x) for x in phy]
    # [ECO-WARP] num_fixed[R] must be the length of the SINGLE-COPY PREFIX of
    # rank R's local slots (eco_meta.cuh L17). A slot belongs to that prefix
    # only when it is itself an ACTIVE copy of an expert that has exactly one
    # active copy world-wide. The old test `len(copies[phy[s]]) == 1` also
    # matched a PASSIVE duplicate of such an expert (measured: expert 2 has
    # slots 257/258 on the same rank, 258 is passive, yet it was counted),
    # which made num_fixed larger than the real prefix and broke the cursor.
    words += [sum(1 for s in range(r * S, (r + 1) * S)
                  if s in set(copies[int(phy[s])])
                  and len(copies[int(phy[s])]) == 1) for r in range(R)]
    assert len(words) == 16 + E * COPIES_WORDS_PER_EXPERT + P + R
    # The kernel reads these words as int32; a passive entry's bit 15 makes
    # the packed word unsigned, so wrap it into the signed range explicitly.
    return np.asarray([w - (1 << 32) if w >= (1 << 31) else w for w in words],
                      np.int32)


def read_copies(meta: np.ndarray, e: int, max_copies: int = 4):
    """Decode copies[e][0..max_copies) the way the kernel does."""
    base = 16 + e * COPIES_WORDS_PER_EXPERT
    out = []
    for k in range(max_copies):
        word = int(meta[base + k // 2]) & 0xFFFFFFFF
        out.append((word & 0xFFFF) if k % 2 == 0 else (word >> 16))
    return out


def install_into_buffer(buf, physical_to_logical, num_ranks: int,
                        num_logical: int):
    """Write EcoMeta into a live SM90 buffer's `eco_meta` view."""
    import torch
    meta = build_eco_meta(physical_to_logical, num_ranks, num_logical)
    want = buf.eco_meta.numel()
    if len(meta) != want:
        raise ValueError(f'meta has {len(meta)} words, buffer view has {want}')
    buf.eco_meta.copy_(torch.as_tensor(meta, dtype=torch.int32,
                                       device=buf.eco_meta.device))
    # [ECO-FINE] read-back check: a silent no-op copy here would make the plan
    # read a stale (identity) placement, which shows up as every expert
    # looking like it has exactly 2 copies and one slot owning all tokens.
    import os as _os
    if _os.environ.get('GC_MOE_FINE_TRACE') == '1':
        back = buf.eco_meta.cpu().tolist()
        if back != [int(x) for x in meta]:
            bad = sum(1 for a, b in zip(back, meta) if a != int(b))
            print(f'[eco] !! EcoMeta read-back MISMATCH: {bad}/{len(back)} words',
                  flush=True)
        else:
            print(f'[eco] EcoMeta read-back OK ({len(back)} words)', flush=True)
    return meta
