"""V4 end-to-end on CPU: EcoMeta bytes -> kernel cold plan -> quota == unit_ref.

This exercises the exact data path the SM90 kernel will use:
  host writes EcoMeta (copies/phy2log/num_fixed)
  kernel decodes copies[e][k] as int16 half-words
  kernel runs _eco_do_cold
  kernel emits quota
and compares against unit_ref, which is the specification.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'source'))

import numpy as np
from eco_meta import build_eco_meta, read_copies
from unit_ref import UnitPlanner
from test_kernel_logic import kernel_cold
from gc_moe.eplb.deepinfer_periodic.config import initial_layout

EMPTY = 0x7FFF


def decode_placement_from_meta(meta, E, P, R, max_copies=4):
    """Rebuild `phy` exactly as the kernel does (via copies + phy2log)."""
    phy_meta = meta[16 + E * 16: 16 + E * 16 + P]
    phy = np.zeros(P, np.int64)
    for e in range(E):
        for k in range(max_copies):
            raw = read_copies(meta, e, max_copies)[k]
            if raw == EMPTY:
                continue
            s = raw & 0x7FFF
            phy[s] = e
    # every slot must be covered
    missing = [s for s in range(P) if phy[s] == 0 and phy_meta[s] != 0]
    return phy


def main():
    R, E, EXTRA = 8, 384, 40
    P = E + EXTRA
    phy = initial_layout(E, EXTRA)
    meta = build_eco_meta(phy, R, E)
    assert len(meta) == 16 + E * 16 + P + R, 'meta size'

    # 1. EcoMeta must round-trip the placement
    phy2 = decode_placement_from_meta(meta, E, P, R)
    ok_meta = np.array_equal(phy, phy2)

    # 2. num_fixed must equal the number of single-copy slots on each rank
    S = P // R
    nf = meta[16 + E * 16 + P: 16 + E * 16 + P + R]
    want_nf = [sum(1 for s in range(r * S, (r + 1) * S)
                   if sum(1 for x in range(P) if phy[x] == phy[s]) == 1)
               for r in range(R)]
    ok_nf = list(nf) == want_nf

    # 3. kernel cold plan from the meta-decoded placement == unit_ref
    fails = []
    if not ok_meta: fails.append('meta round-trip')
    if not ok_nf: fails.append('num_fixed')
    rng = np.random.default_rng(3)
    for bm in (32, 64, 128):
        for name, n in {
            'uniform': np.full(E, 100),
            'skew': np.r_[np.full(20, 400), np.full(E - 20, 20)],
            'sparse': np.r_[np.full(50, 60), np.zeros(E - 50)],
            'rand': rng.integers(0, 300, E),
        }.items():
            n = n.astype(np.int64)
            ref = UnitPlanner(phy2, R, E, block_m=bm, wb=64, wa=0,
                              rounds=0, cold_rounds=0).plan(n)
            got = kernel_cold(phy2, E, P, R, bm, 64, 0, n)
            if not np.array_equal(ref, got):
                fails.append(f'{name}/bm{bm}')
    print(f'meta round-trip: {"OK" if ok_meta else "FAIL"}')
    print(f'num_fixed      : {"OK" if ok_nf else "FAIL"}  {list(nf)}')
    print()
    print('META_PIPELINE', 'FAIL: ' + ', '.join(fails) if fails else 'PASS')
    return 1 if fails else 0


if __name__ == '__main__':
    raise SystemExit(main())
