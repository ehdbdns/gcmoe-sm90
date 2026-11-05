"""Simulate the CuTe DSL kernel logic and compare with unit_ref cold.

The kernel differs structurally from unit_ref (one thread per expert, shared
memory, explicit loops), so this rewrites the kernel's exact control flow in
Python and checks it produces unit_ref's cold-plan quotas.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'source'))

import numpy as np
from unit_ref import UnitPlanner
from gc_moe.eplb.deepinfer_periodic.config import initial_layout

NO_PARTIAL = 0xFFFFFFFF
EMPTY = 0x7FFF


def kernel_cold(phy, E, P, R, bm, wb, wa, counts, max_copies=4):
    """Reproduce _eco_do_cold exactly as the DSL writes it."""
    S = P // R
    # copies[e] from the EcoMeta encoding
    copies, seen_rank = [], []
    for e in range(E):
        cs, seen = [], set()
        for s in range(P):
            if phy[s] == e:
                r = s // S
                if r not in seen:
                    seen.add(r); cs.append(s)
        copies.append(cs)
    ncop = [len(c) for c in copies]

    y = np.zeros(P, np.int64)
    part = np.full(E, NO_PARTIAL, np.int64)
    k = np.zeros(E, np.int64)
    rem = np.zeros(E, np.int64)
    for e in range(E):
        n = counts[e]
        k[e] = -(-n // bm)
        rem[e] = 0 if n == 0 else n - (k[e] - 1) * bm

    # single-copy experts fixed up front
    for s in range(P):
        e = phy[s]
        if ncop[e] == 1 and counts[e] > 0:
            y[s] = k[e] - 1
            part[e] = s

    # ---- _eco_do_cold ----
    x = np.zeros(P, np.int64)
    for s in range(P):
        e = phy[s]
        nc = ncop[e]; n = counts[e]
        v = n // nc if nc > 0 else 0
        if copies[e] and copies[e][0] == s and nc > 0:
            v += n - (n // nc) * nc
        x[s] = v
    t0 = np.bincount(np.arange(P) // S, wb * (-(-x // bm)) + wa * (x > 0) + x, R).astype(np.int64)

    for e in range(E):
        n = counts[e]; nc = ncop[e]
        if nc == 0 or n == 0:
            continue
        g = [0] * max_copies
        xs = [0] * max_copies
        order = list(range(max_copies))
        total_g = 0
        for j in range(max_copies):
            v = xv = 0
            if j < nc:
                s = copies[e][j]
                xv = int(x[s]); v = xv // bm
            g[j] = v; xs[j] = xv; total_g += v
        left = int(k[e]) - total_g
        # selection sort by (t0[rank]*bm - (x%bm)*(wb+bm), j)
        key = lambda j: (int(t0[copies[e][j] // S]) * bm
                         - (xs[j] % bm) * (wb + bm), j)
        BIG = (1 << 62, 0)
        for i in range(max_copies):
            # `i` can exceed nc; the kernel guards with `j < nc` and keeps the
            # identity order there, so mirror that exactly.
            if i >= nc:
                continue
            best = i; bk = key(i)
            for j in range(i + 1, nc):
                kj = key(j)
                if kj < bk:
                    best = j; bk = kj
            order[i], order[best] = order[best], order[i]
        for i in range(max_copies):
            if i < left:
                j = order[i]; g[j] += 1
        jp, bv, seen = 0, 0, False
        for j in range(max_copies):
            if j < nc and g[j] > 0:
                val = g[j] * bm - xs[j]
                if not seen or val > bv:
                    bv, jp, seen = val, j, True
        for j in range(max_copies):
            if j < nc:
                y[copies[e][j]] = g[j] - (1 if j == jp else 0)
        part[e] = copies[e][jp]

    # ---- _eco_emit_quota ----
    q = y * bm
    m = part != NO_PARTIAL
    q[part[m]] += rem[m]
    return q


def main():
    R, E, EXTRA = 8, 384, 40
    P = E + EXTRA
    phy = initial_layout(E, EXTRA)
    fails = []
    rng = np.random.default_rng(1)
    for bm in (32, 64, 128):
        for wa in (0, 54):
            for name, n in {
                'uniform': np.full(E, 100),
                'skew': np.r_[np.full(20, 400), np.full(E - 20, 20)],
                'sparse': np.r_[np.full(50, 60), np.zeros(E - 50)],
                'tiny': np.full(E, 3),
                'rand': rng.integers(0, 200, E),
            }.items():
                n = n.astype(np.int64)
                ref = UnitPlanner(phy, R, E, block_m=bm, wb=64, wa=wa,
                                  rounds=0, cold_rounds=0).plan(n)   # cold only
                got = kernel_cold(phy, E, P, R, bm, 64, wa, n)
                eq = np.array_equal(ref, got)
                cons = all(int(got[[s for s in range(P) if phy[s] == e]].sum()) == int(n[e])
                           for e in range(E))
                tag = f'{name}/bm{bm}/wa{wa}'
                if not eq: fails.append(tag + ' mismatch')
                if not cons: fails.append(tag + ' conservation')
                print(f'  {tag:22s} equal={eq} cons={cons}')
    print()
    print('KERNEL_LOGIC', 'FAIL: ' + ', '.join(fails) if fails else 'PASS')
    return 1 if fails else 0


if __name__ == '__main__':
    raise SystemExit(main())
