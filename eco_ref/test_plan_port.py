"""Bit-exactness of plan_port.py vs unit_ref.py over many shapes / demands.

This is the reference half of V4: if the Python port (which the CuTe DSL kernel
will be translated from) matches unit_ref everywhere, then only a *translation*
mistake can break the kernel.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'source'))

import numpy as np
from unit_ref import UnitPlanner
from plan_port import PlanPort
from gc_moe.eplb.deepinfer_periodic.config import initial_layout

fails = []


def cmp(name, phy, n, prev=None, *, bm, wb, wa, rounds, cold, hop2, R, E):
    a = UnitPlanner(phy, R, E, block_m=bm, wb=wb, wa=wa, rounds=rounds,
                    cold_rounds=cold, hop2_rounds=hop2)
    b = PlanPort(phy, R, E, block_m=bm, wb=wb, wa=wa, rounds=rounds,
                 cold_rounds=cold, hop2_rounds=hop2)
    if prev is None:
        qa = a.plan(n)
        qb = b._eco_plan_assign(n)
    else:
        qa = a.plan(n, prev=prev)
        qb = b._eco_plan_assign(n, prev=prev)
    eq = np.array_equal(qa, qb)
    # conservation is an independent invariant
    cons = all(int(qa[[s for s in range(len(phy)) if phy[s] == e]].sum()) == int(n[e])
               for e in range(E))
    if not eq:
        fails.append(f'{name}: mismatch')
    if not cons:
        fails.append(f'{name}: conservation broken')
    return eq, cons


def main():
    R, E, EXTRA = 8, 384, 40
    P = E + EXTRA
    phy = initial_layout(E, EXTRA)

    rng = np.random.default_rng(0)
    cases = {
        'uniform': np.full(E, 100),
        'skew4': np.r_[np.full(20, 400), np.full(E - 20, 20)],
        'sparse': np.r_[np.full(50, 60), np.zeros(E - 50)],
        'tiny': np.full(E, 3),
        'random': rng.integers(0, 200, E),
    }
    for bm in (32, 64, 128):
        for wa in (0, 54):
            for name, n in cases.items():
                n = n.astype(np.int64)
                eq, cons = cmp(f'{name}/bm{bm}/wa{wa}', phy, n, bm=bm, wb=64,
                               wa=wa, rounds=1, cold=6, hop2=0, R=R, E=E)
                print(f'  cold {name:8s} bm={bm:3d} wa={wa:3d} equal={eq} cons={cons}')

    # warm start (plan twice)
    for bm in (64, 128):
        n1 = np.r_[np.full(20, 300), np.full(E - 20, 40)].astype(np.int64)
        n2 = n1.copy(); n2[:20] += 11
        a = UnitPlanner(phy, R, E, block_m=bm, wb=64, wa=0)
        q1 = a.plan(n1)
        eq, cons = cmp(f'warm/bm{bm}', phy, n2, prev=(q1, n1), bm=bm, wb=64,
                       wa=0, rounds=1, cold=6, hop2=0, R=R, E=E)
        print(f'  warm bm={bm:3d} equal={eq} cons={cons}')

    # two-hop
    n = np.r_[np.full(20, 400), np.full(E - 20, 20)].astype(np.int64)
    eq, cons = cmp('two-hop', phy, n, bm=64, wb=64, wa=0, rounds=2, cold=6,
                   hop2=1, R=R, E=E)
    print(f'  two-hop equal={eq} cons={cons}')

    print()
    print('PLAN_PORT', 'FAIL: ' + ', '.join(fails) if fails else 'PASS')
    return 1 if fails else 0


if __name__ == '__main__':
    raise SystemExit(main())
