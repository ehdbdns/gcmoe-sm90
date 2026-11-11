"""Before/after IR of the EPLB re-plan, simulated from the real histogram.

METRIC (identical to the kernel's):
    IR = max_r load[r] / mean_r load[r]
An expert with `c` copies has its demand split over them (that is what the
plan's quota / RR spread does), so
    load[r] = sum over slots on r of dem[expert] / copies[expert]

DEMAND. `type 2` (slot_end) records the rows each slot pulled = the rows it
must GEMM. Summing over all 8 ranks and dividing by `ranks` de-inflates the
world multiplier, and EcoMeta maps slot -> logical expert, so the histogram of
one layer is exact.

THE FOUR QUESTIONS THIS ANSWERS
  1. before  : identity placement (slot i serves expert i)
  2. after   : rebalance_experts_global() fed the SAME traffic
  3. what if the planner is fed ONE collapsed step (`1x1xE`, what the runtime
     does today) vs a genuine per-step histogram (`Sx1xE`)
  4. what if the plan is computed on a DIFFERENT window than it is scored on
     (the runtime plans on warm-up and measures later -- this quantifies how
     much IR that staleness costs)
"""
import argparse, json, math, os, sys
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'source'))

LOGICAL, PHYSICAL, RANKS = 384, 448, 8


def global_demand(path, logical=LOGICAL, physical=PHYSICAL, ranks=RANKS):
    """[experts] demand of ONE layer, summed over every rank (de-inflated).

    `eco_meta` only exists for the ECO arms; the vanilla arm has no placement
    table, so it is modelled as the identity (slot i <-> expert i, 48 slots per
    rank, 384 total). Returns `phy` as well so the caller uses the same map.
    """
    d = json.load(open(path))
    trs = d['eco_trace']['ranks']
    per_rank = physical // ranks
    if not trs[0].get('eco_meta'):
        # vanilla / no-EPLB runs: P = E = 384, identity placement
        physical = logical
        per_rank = physical // ranks
        phy = np.arange(physical, dtype=np.int64)
    else:
        em = [w & 0xFFFFFFFF for w in trs[0]['eco_meta']]
        phy = np.array(em[16 + logical * 16:16 + logical * 16 + physical], np.int64)
    per_slot = np.zeros(physical, np.float64)
    for r, rec in enumerate(trs):
        rows = [x for x in (rec.get('fine_rows') or []) if x[1]]
        if not rows:
            continue
        by = defaultdict(list)
        for x in rows:
            by[x[6]].append(x)
        _ep, E = max(by.items(), key=lambda kv: len(kv[1]))
        for x in E:
            if x[1] == 2 and x[4]:
                per_slot[r * per_rank + x[3]] = x[4] / ranks
    dem = np.zeros(logical, np.float64)
    np.add.at(dem, phy, per_slot)
    return dem, phy


def load_of(dem, phy, ranks=RANKS):
    per_rank = len(phy) // ranks
    cnt = np.zeros(len(phy))
    np.add.at(cnt, phy, 1.0)
    cnt = np.maximum(cnt, 1.0)
    return (dem[phy] / cnt[phy]).reshape(ranks, per_rank).sum(1)


def ir(dem, phy, ranks=RANKS):
    L = load_of(dem, phy, ranks)
    m = L.mean()
    return float(L.max() / m) if m > 0 else float('nan')


def perfect(dem, phy, ranks=RANKS, seed=0):
    """Largest-remaining-first balanced placement: the floor for this demand."""
    rng = np.random.default_rng(seed)
    per_rank = len(phy) // ranks
    cnt = np.zeros(len(dem), int)
    np.add.at(cnt, phy, 1)
    items = np.repeat(dem / np.maximum(cnt, 1), np.maximum(cnt, 1))
    items = items[rng.permutation(len(items))]
    load, room = np.zeros(ranks), np.full(ranks, per_rank)
    for i in np.argsort(-items):
        cand = np.where(room > 0)[0]
        r = cand[np.argmin(load[cand])]
        load[r] += items[i]
        room[r] -= 1
    return load


def plan(h, physical=PHYSICAL, per_rank=PHYSICAL // RANKS):
    import torch
    from gc_moe.eplb.deepinfer_periodic.policy import rebalance_experts_global
    t = torch.as_tensor(np.asarray(h, np.float32))
    p, _, _ = rebalance_experts_global(t, physical, per_rank)
    return p[0].cpu().numpy().astype(np.int64)


def poisson_steps(dem, n, rng):
    p = dem / max(dem.sum(), 1e-9)
    tot = int(round(dem.sum()))
    return np.array([rng.multinomial(tot, p) for _ in range(n)], np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+', required=True)
    ap.add_argument('--steps', type=int, default=40)
    a = ap.parse_args()
    rng = np.random.default_rng(7)

    hdr = (f"{'run':<12}{'m/rank':>7}{'floor':>7}{'before':>8}{'after':>8}"
           f"{'perStep':>8}{'drift':>8}{'measured':>9}")
    print(hdr)
    for path in a.runs:
        d = json.load(open(path))
        dem, phy = global_demand(path)
        m = dem.sum() / RANKS
        floor = 1 + 1.42 / math.sqrt(max(m, 1e-9))
        init = np.tile(np.arange(LOGICAL), PHYSICAL // LOGICAL + 1)[:PHYSICAL]

        # score everything on the SAME Poisson traffic drawn from `dem`
        hist = poisson_steps(dem, a.steps, rng)
        phy_a = plan(dem.reshape(1, 1, -1))            # today: 1 collapsed step
        phy_b = plan(hist.reshape(a.steps, 1, -1))     # genuine per-step input

        before = np.mean([ir(h, init) for h in hist])
        after = np.mean([ir(h, phy_a) for h in hist])
        perstep = np.mean([ir(h, phy_b) for h in hist])

        # staleness: plan on a DIFFERENT traffic realisation, score on `hist`
        alt = poisson_steps(dem, 1, rng)[0]
        phy_s = plan(alt.reshape(1, 1, -1))
        drift = np.mean([ir(h, phy_s) for h in hist])

        meas = d['ir_step']['ir_step_layer_mean']
        label = d.get('label', os.path.basename(path)).replace('.json', '')
        print(f'{label:<12}{m:>7.1f}{floor:>7.4f}{before:>8.4f}'
              f'{after:>8.4f}{perstep:>8.4f}{drift:>8.4f}{meas:>9.4f}')
        print(f'    same-window gain: {(1-(after-1)/(before-1))*100:5.1f}%   '
              f'per-step input:  {(1-(perstep-1)/(before-1))*100:5.1f}%   '
              f'stale plan:      {(1-(drift-1)/(before-1))*100:5.1f}%   '
              f'-> measured {(1-(meas-1)/(before-1))*100:5.1f}% of the '
              f'simulated "before"')


if __name__ == '__main__':
    main()
