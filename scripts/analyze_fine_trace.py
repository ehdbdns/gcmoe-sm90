"""[ECO-FINE] Single-layer timeline from the flat `fine_tl` event log.

Record layout (8 int32): [t_us, type, id, a0, a1, a2, epoch, 0]

  type 1  slot_begin   id=gslot  a0=local slot  a1=CTA  a2=ncopies
  type 2  slot_end     id=gslot  a0=local slot  a1=owned  a2=lo
  type 3  wait_plan    id=gslot  a0=0 begin / 1 released
  type 4  spin_edge    id=CTA    a1=wanted slot a2=active_count seen
  type 5  got_work     id=CTA    a1=slot       a2=expert
  type 6  mblock_begin id=CTA    a0=expert a1=start a2=valid
  type 7  mblock_end   id=CTA    a0=expert a1=start a2=valid
  type 12 copies_raw   id=gslot  a0..a2 = copies[expert][0..2]

usage: analyze_fine_trace.py runs/FINEB.json [rank] [epoch]
"""
import json, statistics as st, sys
from collections import defaultdict, Counter

def load(path, rank=0):
    d = json.load(open(path))
    r = d['eco_trace']['ranks'][rank]
    return d, r, [x for x in (r.get('fine_rows') or []) if x[1]]

def pick_epoch(rows, want=None):
    eps = sorted(set(x[6] for x in rows))
    if want is not None:
        return want, [x for x in rows if x[6] == want]
    # the first epoch that carries all seven event kinds
    for ep in eps:
        E = [x for x in rows if x[6] == ep]
        c = Counter(x[1] for x in E)
        if c.get(1) and c.get(2) and c.get(6) and c.get(7):
            return ep, E
    return eps[0], [x for x in rows if x[6] == eps[0]]

def main(path, rank=0, want=None):
    d, r, rows = load(path, rank)
    ep, E = pick_epoch(rows, want)
    # crop anything newer than the last M-block of this call
    t_end = max((x[0] for x in E if x[1] == 6), default=None)
    if t_end:
        E = [x for x in E if x[0] <= t_end]
    t0 = min(x[0] for x in E if x[1] == 1)
    T = lambda x: x[0] - t0

    print(f'== {path} rank{rank} epoch{ep}  '
          f'{d.get("ms_per_step_median", 0):.2f} ms/step  '
          f'{len(E)} events, span {max(T(x) for x in E)} us ==')

    beg, end, wait = {}, {}, defaultdict(dict)
    for x in E:
        if x[1] == 1: beg[x[2]] = x
        elif x[1] == 2: end[x[2]] = x
        elif x[1] == 3: wait[x[2]]['b' if x[3] == 0 else 'e'] = x

    recs = sorted(((g, T(beg[g]), T(end[g]), end[g][4], beg[g][5])
                   for g in beg if g in end), key=lambda z: z[1])
    nz = [z for z in recs if z[3] > 0]
    print(f'\n[1] PULL: {len(recs)} slots, {len(nz)} with data, '
          f'owned total {sum(z[3] for z in nz)}')
    print(f"{'gslot':>6}{'begin':>8}{'end':>8}{'dur':>7}{'owned':>7}{'copies':>7}")
    for z in nz[:20]:
        print(f'{z[0]:>6}{z[1]:>8}{z[2]:>8}{z[2]-z[1]:>7}{z[3]:>7}{z[4]:>7}')
    if len(nz) > 20:
        print(f'   ... ({len(nz)} total)')
    if nz:
        print(f'   first slot done {min(z[2] for z in nz)} us, '
              f'last slot done {max(z[2] for z in nz)} us')

    ws = sorted(((g, T(v['b']), T(v['e'])) for g, v in wait.items()
                 if 'b' in v and 'e' in v), key=lambda z: z[1])
    if ws:
        print(f'\n[2] REPLICA wait-for-plan: {len(ws)} slots, '
              f'median {st.median([z[2]-z[1] for z in ws]):.0f} us, '
              f'max {max(z[2]-z[1] for z in ws)} us')
        print(f'   earliest wait {min(z[1] for z in ws)} us, '
              f'latest release {max(z[2] for z in ws)} us')

    sp = [x for x in E if x[1] == 4]
    ok = [x for x in E if x[1] == 5]
    print(f'\n[3] CONSUMER: {len(sp)} spin edges, {len(ok)} got-work')
    if sp and ok:
        first_spin, first_ok = min(T(x) for x in sp), min(T(x) for x in ok)
        print(f'   first spin {first_spin} us -> first work {first_ok} us '
              f'= starvation {first_ok - first_spin} us')
    # pair each spin edge with the next got-work of the same CTA
    per = defaultdict(list)
    for x in sorted(E, key=lambda y: y[0]):
        if x[1] in (4, 5):
            per[x[2]].append((x[1], T(x)))
    pairs = []
    for cta, ev in per.items():
        pend = None
        for kind, t in ev:
            if kind == 4: pend = t
            elif pend is not None:
                pairs.append((cta, t - pend)); pend = None
    if pairs:
        v = [p[1] for p in pairs]
        print(f'   per-CTA spin episodes: {len(v)}, median {st.median(v):.0f} us, '
              f'max {max(v)} us')

    tb, blk = defaultdict(list), []
    for x in sorted(E, key=lambda y: y[0]):
        if x[1] == 6: tb[(x[2], x[3], x[4])].append(x)
        elif x[1] == 7:
            k = (x[2], x[3], x[4])
            if tb[k]:
                b = tb[k].pop(0)
                blk.append((T(b), T(x), x[2], x[3]))
    blk.sort()
    if blk:
        print(f'\n[4] GEMM: {len(blk)} M blocks, median {st.median([b[1]-b[0] for b in blk]):.0f} us')
        print(f'   first block {blk[0][0]} us (CTA{blk[0][2]} expert{blk[0][3]}), '
              f'last block end {blk[-1][1]} us')
        print(f'   wall span {blk[-1][1]-blk[0][0]} us, '
              f'sum busy {sum(b[1]-b[0] for b in blk)} us '
              f'=> parallelism {sum(b[1]-b[0] for b in blk)/max(1, blk[-1][1]-blk[0][0]):.2f}')
        pc = Counter(b[3] for b in blk)
        print(f'   experts served: {dict(list(pc.items())[:10])}')

if __name__ == '__main__':
    args = sys.argv[1:]
    p = args[0] if args else 'runs/FINEB.json'
    rk = int(args[1]) if len(args) > 1 else 0
    want = int(args[2]) if len(args) > 2 else None
    main(p, rk, want)
