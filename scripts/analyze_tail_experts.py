"""Which experts / M blocks keep the tail alive (and therefore slow the layer).

For one measured layer of rank 0 this prints, for every M block, the moment it
FINISHES (last `type 7` record of its (slot, start) pair).  The blocks that
finish last are the ones that define the makespan: in the last stretch of the
call only a handful of CTAs still have work, which is the parallelism collapse
the parallelism plots show.

Per M block we report:
  slot    local slot id
  expert  logical expert (EcoMeta phy2log; for non-ECO arms slot == expert)
  mblk    index of the M block inside that slot (0..n-1)
  valid   rows in that block  (= min(block_m, owned - start))
  nblk    how many M blocks this slot has in total
  owned   tokens this slot pulled (world-deflated)
  finish  time from the first slot_begin, in us

`--tail-pct` selects how much of the call counts as "the tail".
"""
import argparse, json, os, sys
from collections import defaultdict

ROLE6_OFFSET = 1000
PREFILL_SPAN_US = 2000


def pick_call(rows):
    best = None
    for ep in sorted({x[6] for x in rows}):
        E = [x for x in rows if x[6] == ep]
        n1 = len({x[2] for x in E if x[1] == 1})
        n2 = len({x[2] for x in E if x[1] == 2})
        if n1 == 0 or n1 != n2 or not any(x[1] == 6 for x in E):
            continue
        if max(x[0] for x in E) - min(x[0] for x in E) < PREFILL_SPAN_US:
            best = (ep, E)
    if best is None:
        raise SystemExit('no complete decode call')
    return best


def phy2log(rec, n_logical, n_physical):
    em = rec.get('eco_meta')
    if not em:
        return None
    em = [w & 0xFFFFFFFF for w in em]
    off = 16 + n_logical * 16
    return em[off:off + n_physical]


def blocks(E, phy, world=8, block_m=None):
    u = lambda t: t & 0xFFFFFFFF
    t0 = min(u(x[0]) for x in E if x[1] == 1)
    owned = {}
    for x in E:
        if x[1] == 2:
            owned[x[3]] = x[4]
    beg, end = defaultdict(list), defaultdict(list)
    for x in E:
        if x[1] == 6:
            beg[(x[3], x[4])].append(x[5])
        elif x[1] == 7:
            end[(x[3], x[4])].append(u(x[0]) - t0)
    out = []
    per_slot = defaultdict(list)
    for k in beg:
        per_slot[k[0]].append(k[1])
    bm = block_m or max(1, max(beg[k][0] for k in beg))
    for (slot, start), vals in beg.items():
        fin = max(end[(slot, start)])
        real = owned.get(slot, 0) // world
        nblk = len(per_slot[slot])
        idx = sorted(per_slot[slot]).index(start)
        out.append(dict(slot=slot, start=start, expert=(phy[slot] if phy else slot),
                        mblk=idx, nblk=nblk, valid=vals[0], rowsv=len(vals),
                        owned=real, finish=fin,
                        roles=sorted({c for c in [k[2] for k in []]})))
    return out, bm, owned


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run')
    ap.add_argument('--tail-pct', type=float, default=20.0)
    ap.add_argument('--rank', type=int, default=0)
    ap.add_argument('--logical', type=int, default=384)
    ap.add_argument('--physical', type=int, default=448)
    ap.add_argument('--world', type=int, default=8)
    a = ap.parse_args()

    d = json.load(open(a.run))
    rec = d['eco_trace']['ranks'][a.rank]
    rows = [x for x in (rec.get('fine_rows') or []) if x[1]]
    ep, E = pick_call(rows)
    phy = phy2log(rec, a.logical, a.physical)
    blk, bm, owned = blocks(E, phy, a.world)
    makespan = max(b['finish'] for b in blk)
    cut = makespan * (1 - a.tail_pct / 100.0)

    # which CTA finished each block, so role 2 vs role 6 is visible
    cta_of = {}
    for x in E:
        if x[1] == 7:
            cta_of[(x[3], x[4])] = x[2]
    for b in blk:
        b['role'] = 6 if cta_of.get((b['slot'], b['start']), 0) >= ROLE6_OFFSET else 2

    tail = sorted([b for b in blk if b['finish'] >= cut],
                  key=lambda b: -b['finish'])
    print(f'# {os.path.basename(a.run)}  rank{a.rank}  epoch{ep}  block_m={bm}')
    print(f'# blocks={len(blk)}  makespan={makespan} us  '
          f'tail = last {a.tail_pct:.0f}% (finish >= {cut:.0f} us) '
          f'-> {len(tail)} blocks')
    print(f"{'slot':>5}{'expert':>8}{'mblk':>6}{'nblk':>6}{'valid':>7}"
          f"{'owned':>7}{'finish':>8}{'role':>6}")
    for b in tail:
        print(f"{b['slot']:>5}{b['expert']:>8}{b['mblk']:>6}{b['nblk']:>6}"
              f"{b['valid']:>7}{b['owned']:>7}{b['finish']:>8}{b['role']:>6}")


if __name__ == '__main__':
    main()
