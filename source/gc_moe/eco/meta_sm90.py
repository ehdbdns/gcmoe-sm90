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
    words += [sum(1 for s in range(r * S, (r + 1) * S)
                  if len(copies[int(phy[s])]) == 1) for r in range(R)]
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
    return meta
