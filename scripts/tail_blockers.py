"""Who blocks the tail: the M blocks that finish last, per run.

For one measured layer of rank 0 the makespan is set by the M blocks that
finish last; while they run, the rest of the compute partition is idle, which
is the parallelism collapse the `par20` figures show. This script prints, for
the last `--tail-pct` of the call:

  expert   logical expert of the slot that owns the block
  mblk/nblk  index of the M block inside the slot, and how many the slot has
  valid    rows in that block (= min(block_m, owned - start))
  owned    tokens the slot pulled (already world-deflated)
  nc       how many ACTIVE copies this expert has (1 = single, >1 = replica)
  pull_end when that slot finished pulling
  finish   when the block's last N-tile finished
  role     2 = compute partition, 6 = the SM dispatch released (SM-reuse)

usage: tail_blockers.py runs/V32_eco_smr.json [--tail-pct 20] [-v]
"""
import argparse, json, os, sys
from collections import defaultdict

ROLE6 = 1000
PREFILL_US = 2000


def pick(rows):
    best = None
    for ep in sorted({x[6] for x in rows}):
        E = [x for x in rows if x[6] == ep]
        n1 = len({x[2] for x in E if x[1] == 1})
        n2 = len({x[2] for x in E if x[1] == 2})
        if n1 and n1 == n2 and any(x[1] == 6 for x in E) \
                and max(x[0] for x in E) - min(x[0] for x in E) < PREFILL_US:
            best = (ep, E)
    if best is None:
        raise SystemExit('no complete decode call')
    return best


def analyze(run, rank=0, tail_pct=20.0, world=8, logical=384, physical=448):
    d = json.load(open(run))
    rec = d['eco_trace']['ranks'][rank]
    rows = [x for x in (rec.get('fine_rows') or []) if x[1]]
    ep, E = pick(rows)
    u = lambda t: t & 0xFFFFFFFF
    t0 = min(u(x[0]) for x in E if x[1] == 1)

    phy = None
    if rec.get('eco_meta'):
        em = [w & 0xFFFFFFFF for w in rec['eco_meta']]
        off = 16 + logical * 16
        phy = em[off:off + physical]

    owned, nc, pull_end = {}, {}, {}
    for x in E:
        if x[1] == 1:
            nc[x[3]] = x[5]              # a2 = number of ACTIVE copies
        elif x[1] == 2:
            # a1 = rows this slot pulled into l1_acts. Every rank pulls ALL
            # peers' rows for its own experts, so this is world x the rank's
            # own share; it is exactly the number the kernel slices into M
            # blocks, hence it must NOT be divided before comparing with bm.
            owned[x[3]] = x[4]
            pull_end[x[3]] = u(x[0]) - t0

    beg, end, cta = defaultdict(list), defaultdict(list), {}
    for x in E:
        if x[1] == 6:
            beg[(x[3], x[4])].append(x[5])
        elif x[1] == 7:
            end[(x[3], x[4])].append(u(x[0]) - t0)
            cta[(x[3], x[4])] = x[2]

    per_slot = defaultdict(list)
    for k in beg:
        per_slot[k[0]].append(k[1])
    # block_m is min(tile_m, rows) of the first block, so the max `valid` seen
    # equals tile_m whenever some slot holds at least one full block. With
    # fewer rows than tile_m the estimate is the largest block instead, which
    # still reproduces nblk correctly (ceil(rows/bm_est) == 1).
    bm = max(v[0] for v in beg.values())
    blocks = []
    for (slot, start), vals in beg.items():
        starts = sorted(per_slot[slot])
        blocks.append(dict(
            slot=slot, start=start,
            expert=(phy[slot] if phy else slot),
            mblk=starts.index(start), nblk=len(starts),
            valid=vals[0], owned=owned.get(slot, 0),
            nc=nc.get(slot, 1), pull_end=pull_end.get(slot, 0),
            finish=max(end[(slot, start)]),
            role=6 if cta.get((slot, start), 0) >= ROLE6 else 2))
    makespan = max(b['finish'] for b in blocks)
    cut = makespan * (1 - tail_pct / 100.0)
    tail = sorted([b for b in blocks if b['finish'] >= cut],
                  key=lambda b: -b['finish'])
    return dict(label=d.get('label', os.path.basename(run)), epoch=ep,
                block_m=bm, nblocks=len(blocks), makespan=makespan,
                cut=cut, tail=tail, tail_pct=tail_pct)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--tail-pct', type=float, default=20.0)
    ap.add_argument('--top', type=int, default=5)
    a = ap.parse_args()
    for run in a.runs:
        r = analyze(run, tail_pct=a.tail_pct)
        print(f"# {r['label']}  epoch{r['epoch']}  block_m={r['block_m']}  "
              f"blocks={r['nblocks']}  makespan={r['makespan']:.0f} us  "
              f"tail(last {a.tail_pct:.0f}% >= {r['cut']:.0f} us) = {len(r['tail'])} blocks")
        print(f"{'expert':>7}{'mblk':>6}{'nblk':>6}{'valid':>7}{'rows':>7}"
              f"{'nc':>4}{'pull_end':>10}{'finish':>8}{'role':>6}")
        for b in r['tail'][:a.top]:
            print(f"{b['expert']:>7}{b['mblk']:>6}{b['nblk']:>6}{b['valid']:>7}"
                  f"{b['owned']:>7}{b['nc']:>4}{b['pull_end']:>10.0f}"
                  f"{b['finish']:>8.0f}{b['role']:>6}")
        print()


if __name__ == '__main__':
    main()
