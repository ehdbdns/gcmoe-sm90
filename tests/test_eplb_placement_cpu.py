"""CPU regression: an EPLB logical expert must never hold two slots on one rank.

Bug this locks down (found 2026-11-10): `_pack` balanced load only, so with
`physical=448, ranks=8` five of the 64 replicated experts got BOTH copies on
the same rank. Those duplicates are passive (`_eco_is_active == 0`), so they
pull nothing: the effective redundancy silently dropped 64 -> 59 and the
plan had no second rank to move tokens to.
"""
import sys, os
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'source'))

import torch  # noqa: E402
import gc_moe.layout.eplb_placement as P  # noqa: E402

E, R = 384, 8


def _dups(lay):
    bad = []
    for e in range(E):
        ranks = [lay.phy2rank[s] for s, x in enumerate(lay.phy2log) if x == e]
        if len(ranks) != len(set(ranks)):
            bad.append((e, ranks))
    return bad


def _even(lay):
    occ = Counter(lay.phy2rank)
    return len(occ) == R and all(v == lay.per_rank for v in occ.values())


def main():
    cases = [('uniform 8/rank', torch.ones(E), 64 / 384),
             ('uniform 4/rank', torch.ones(E), 40 / 384),
             ('uniform 1/rank', torch.ones(E), 8 / 384),
             ('no redundancy', torch.ones(E), 0.0)]
    hot = torch.ones(E); hot[:8] = 200.0
    cases.append(('skewed hot x8', hot, 64 / 384))
    ultra = torch.ones(E); ultra[0] = 1e6
    cases.append(('single ultra-hot', ultra, 64 / 384))

    fail = 0
    for name, demand, red in cases:
        lay = P.build_eplb_layout(demand, E, R, red)
        d, ev = _dups(lay), _even(lay)
        good = not d and ev and lay.max_copies <= R
        print(f'  {name:18s} physical={lay.physical} per_rank={lay.per_rank} '
              f'max_copies={lay.max_copies} dup_rank={len(d)} even={ev} '
              f'{"OK" if good else "FAIL"}')
        fail += (not good)
        for e, ranks in d[:3]:
            print(f'      expert {e}: ranks {ranks}')
    if fail:
        print(f'EPLB_PLACEMENT_CPU FAIL ({fail} cases)')
        raise SystemExit(1)
    print('EPLB_PLACEMENT_CPU PASS')


if __name__ == '__main__':
    main()
