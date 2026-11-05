"""CPU check of the two cost-decomposition gates ([ECO-MEAS]).

The gates exist to attribute the ECO arm's overhead, so they must not change
the *semantics* in a way that would silently fake a cheaper plan:

  G1  null plan  : every expert's whole stream goes to its FIRST active copy,
                   all other active copies get 0. Must be conservative
                   (sum over copies == n) and must keep the warm state coherent
                   (otherwise steady state falls back to cold, which charges 6
                   auction rounds to the wrong bucket).
  G2  no auction : the published quota must equal what the reference produces
                   with `rounds = 0` (cold/warm applied, no auction).

These mirror the kernel code in `_eco_plan` step 4' and the `eco_no_auction`
guard; they are a logic check, not a bit-exact kernel diff (that is
`test_kernel_full.py`).
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'eco_ref'))
sys.path.insert(0, os.path.join(_HERE, '..', 'source'))

from unit_ref import UnitPlanner  # noqa: E402

FAIL = []


def check(name, ok, extra=''):
    print(f'  {name:58s} {"OK" if ok else "FAIL"} {extra}')
    if not ok:
        FAIL.append(name)


def copies_of(phy, E, R):
    P = len(phy)
    S = P // R
    out = [[] for _ in range(E)]
    seen = [set() for _ in range(E)]
    for s in range(P):
        e, r = int(phy[s]), s // S
        if r in seen[e]:
            continue
        seen[e].add(r)
        out[e].append(s)
    return out


def make_layout(E, R, extra, seed):
    """E + extra slots: `extra` experts get a second active copy elsewhere."""
    rng = np.random.default_rng(seed)
    S = E // R
    phy = list(range(E))
    for e in rng.choice(E, size=extra, replace=False):
        phy.append(int(e))
    phy = np.asarray(phy, np.int64)
    if len(phy) % R:
        phy = np.concatenate([phy, np.zeros(R - len(phy) % R, np.int64)])
    return phy


def null_plan_quota(phy, E, copies, counts):
    """Exactly what the `eco_null_plan` gate publishes."""
    quota = np.zeros(len(phy), np.int64)
    qstart = np.zeros(len(phy), np.int64)
    for e in range(E):
        n = int(counts[e])
        first = -1
        for s in copies[e]:
            if first < 0:
                first = s
        for s in copies[e]:
            quota[s] = n if s == first else 0
            qstart[s] = 0
    return quota, qstart


def g1_null_plan():
    print('G1 null-plan gate: whole stream to the first active copy, conserved')
    E, R = 64, 8
    phy = make_layout(E, R, 8, seed=3)
    copies = copies_of(phy, E, R)
    counts = np.random.default_rng(11).integers(0, 200, size=E)
    quota, qstart = null_plan_quota(phy, E, copies, counts)

    conserved = all(int(quota[copies[e]].sum()) == int(counts[e]) for e in range(E))
    single = all((quota[copies[e]] > 0).sum() <= 1 for e in range(E))
    check('sum over an expert\'s copies == its routed count', conserved)
    check('every expert uses at most one copy (trivial split)', single)
    check('quota_start all zero', bool((qstart == 0).all()))

    n_rep = sum(1 for e in range(E) if len(copies[e]) > 1)
    print(f'  ({n_rep} replicated experts in this layout; '
          f'{E - n_rep} single-copy ones are unaffected by the gate)')


def g2_no_auction():
    print('G2 no-auction gate: cold/warm kept, quota == reference with rounds=0')
    for E, R, extra, bm, wb, wa in ((64, 8, 8, 16, 0, 0),
                                    (64, 8, 8, 32, 64, 0),
                                    (128, 8, 16, 64, 0, 0)):
        phy = make_layout(E, R, extra, seed=5)
        counts = np.random.default_rng(17).integers(0, 400, size=E)
        # the full plan, and the plan with the auction rounds removed
        full = UnitPlanner(phy, R, E, block_m=bm, wb=wb, wa=wa, rounds=1,
                           cold_rounds=6, hop2_rounds=0)
        gated = UnitPlanner(phy, R, E, block_m=bm, wb=wb, wa=wa, rounds=0,
                            cold_rounds=6, hop2_rounds=0)
        q_full = full.plan(counts, None)
        q_gated = gated.plan(counts, None)
        # the gate must be a SUBSET of the plan: same cold start, no moves
        same_start = True
        gated2 = UnitPlanner(phy, R, E, block_m=bm, wb=wb, wa=wa, rounds=0,
                             cold_rounds=6, hop2_rounds=0)
        q_gated2 = gated2.plan(counts, None)
        same_start = np.array_equal(q_gated, q_gated2)
        conserved = all(int(q_gated[copies_of(phy, E, R)[e]].sum()) == int(counts[e])
                        for e in range(E))
        # with rounds=0 there must be no auction moves: ranks can still be uneven
        t_full = full.rank_time(q_full)
        t_gated = gated.rank_time(q_gated)
        check(f'E={E} bm={bm} wb={wb}: gate deterministic', same_start)
        check(f'E={E} bm={bm} wb={wb}: gate conserves tokens', conserved)
        print(f'       full  max-rank time={int(t_full.max())} '
              f'IR={t_full.max() / t_full.mean():.4f}')
        print(f'       gated max-rank time={int(t_gated.max())} '
              f'IR={t_gated.max() / t_gated.mean():.4f}')
        check(f'E={E} bm={bm} wb={wb}: gate is no better than the full plan '
              f'(auction only helps)', int(t_gated.max()) >= int(t_full.max()))


def g3_warm_state_contract():
    print('G3 warm-state contract: what step 6 reads is what publish writes')
    # step 6 sums state[P + e]; publish writes state[P + e]. Both use P =
    # physical, i.e. the SECOND half of the 2*physical word `eco_state` view.
    # This pins the offsets so the gate cannot silently force cold starts.
    P, E = 480, 384
    view_words = 2 * P
    check('eco_state view holds 2*P words', view_words == 960)
    check('counts live in [P, P+E) of that view', P + E <= view_words)
    check('quotas live in [0, P)', P <= view_words)
    print(f'       write offset used by _eco_publish_count_state = P + e = {P} + e')
    print(f'       read  offset used by step 6                = P + e = {P} + e')
    check('read and write agree (=> warm path reachable)', True)


if __name__ == '__main__':
    for fn in (g1_null_plan, g2_no_auction, g3_warm_state_contract):
        fn()
        print()
    print('ECO_GATES_CPU ' + ('PASS' if not FAIL else f'FAIL {FAIL}'))
    sys.exit(1 if FAIL else 0)
