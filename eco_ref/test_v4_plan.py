"""V4 scaffolding: unit_ref vs EcoMeta-driven plan, on CPU.

The kernel will read EcoMeta (copies/phy2log/num_fixed) instead of taking the
placement as a Python list. This test proves that an EcoMeta-decoded placement
reproduces unit_ref's quotas bit for bit -- i.e. the Meta encoding loses
nothing. It is the reference half of acceptance V4.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'source'))

import numpy as np
from eco_meta import build_eco_meta, read_copies
from unit_ref import UnitPlanner
from gc_moe.eplb.deepinfer_periodic.config import initial_layout

R, E, EXTRA = 8, 384, 40
P = E + EXTRA
BM = 128


def placement_from_meta(meta, max_copies=4):
    """Reconstruct `phy` the way the kernel sees it: via copies + phy2log."""
    phy = np.array(meta[16 + E * 16:16 + E * 16 + P], np.int64)
    for e in range(E):
        for raw in read_copies(meta, e, max_copies):
            if raw == 0x7FFF:
                continue
            s = raw & 0x7FFF
            assert phy[s] == e, f'copies[{e}] says slot {s}, phy2log says {phy[s]}'
    return phy


def main():
    phy = initial_layout(E, EXTRA)
    meta = build_eco_meta(phy, R, E)
    phy2 = placement_from_meta(meta)
    assert np.array_equal(phy, phy2), 'EcoMeta does not round-trip the placement'

    n = np.zeros(E, np.int64)
    n[:20] = 200
    n[20:40] = 50
    planner = UnitPlanner(phy, R, E, block_m=BM)
    q = planner.plan(n)

    # conservation: every expert's copies must carry exactly its tokens
    ok = True
    for e in range(E):
        slots = [s for s in range(P) if phy[s] == e]
        if int(q[slots].sum()) != int(n[e]):
            ok = False
            print(f'  expert {e}: quota {int(q[slots].sum())} != n {int(n[e])}')
    print(f'V4 meta round-trip: {"OK" if ok else "FAIL"}')
    print(f'V4 quota sum={int(q.sum())} expect={int(n.sum())} '
          f'{"OK" if q.sum() == n.sum() else "FAIL"}')
    print(f'V4 active slots={int((q > 0).sum())}/{P}')
    return 0 if ok and q.sum() == n.sum() else 1


if __name__ == '__main__':
    raise SystemExit(main())
