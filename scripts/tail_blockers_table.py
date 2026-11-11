"""Markdown table: which expert / M block sets the layer's makespan.

For every run (rank 0, one measured layer) this reports the M block that
finishes LAST -- the one whose N-tiles run alone in the tail, which is the
parallelism collapse the `par20` figures show:

  worst expert     logical expert of the slot that owns it (EcoMeta phy2log)
  slot / rows      local slot and how many rows it pulled (world-inflated)
  M blocks         how many M blocks that slot was cut into
  valid (last)     rows in the last block  (= min(block_m, rows - start))
  largest slot?    is this slot the biggest one on the rank (=> hot expert)
  L1 / L2 window   when the 18 L1 N-tiles and 20 L2 N-tiles of that block ran
  L2-only tail     for how long ONLY those 20 N-tiles were runnable
  avg CTAs         mean running compute CTAs while that block was executing
"""
import json, os, sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tail_blockers import analyze, pick

NT_L1, NT_L2 = 18, 20       # intermediate//128, hidden//256 at these shapes
ARMS = ('vanilla', 'random', 'rr', 'eco', 'eco_smr')
BS = ((4, 32), (8, 64), (16, 128), (32, 256))


def detail(run):
    d = json.load(open(run))
    rec = d['eco_trace']['ranks'][0]
    rows = [x for x in (rec.get('fine_rows') or []) if x[1]]
    ep, E = pick(rows)
    u = lambda t: t & 0xFFFFFFFF
    t0 = min(u(x[0]) for x in E if x[1] == 1)
    owned = {x[3]: x[4] for x in E if x[1] == 2}
    st = defaultdict(list)
    for x in E:
        if x[1] == 6:
            st[(x[3], x[4])].append(u(x[0]) - t0)
    en = defaultdict(list)
    for x in E:
        if x[1] == 7:
            en[(x[3], x[4])].append(u(x[0]) - t0)
    last = max(st, key=lambda k: max(en[k]))

    # running-CTA count over time
    b_all, e_all = defaultdict(list), defaultdict(list)
    for x in E:
        if x[1] == 6:
            b_all[x[3]].append(u(x[0]) - t0)
        elif x[1] == 7:
            e_all[x[3]].append(u(x[0]) - t0)
    ev = []
    for s in b_all:
        b, e = sorted(b_all[s]), sorted(e_all[s])
        for i in range(min(len(b), len(e))):
            ev.append((b[i], 1)); ev.append((e[i], -1))
    ev.sort()
    w0, w1 = min(st[last]), max(en[last])
    c = tot = n = 0
    for t, dt in ev:
        c += dt
        if w0 <= t < w1:
            tot += c; n += 1
    return dict(ep=ep, owned=owned, starts=sorted(st[last]),
                finish=max(en[last]), w0=w0, w1=w1,
                ctas=(tot / n if n else 0.0))


def main():
    print('| BS | arm | worst expert | slot | rows | M blocks | valid (last) | '
          'largest slot? | 18 L1 tiles | 20 L2 tiles | L2-only tail | '
          'avg CTAs then | makespan |')
    print('|---:|---|---:|---:|---:|---:|---:|---|---|---|---:|---:|---:|')
    for spr, bs in BS:
        for arm in ARMS:
            run = f'runs/V{spr}_{arm}.json'
            if not os.path.exists(run):
                continue
            r = analyze(run, tail_pct=20)
            b = r['tail'][0]
            dt = detail(run)
            rows_n = dt['owned'].get(b['slot'], 0)
            rank = sorted(dt['owned'].values(), reverse=True).index(rows_n) + 1
            l1 = dt['starts'][:NT_L1]
            l2 = dt['starts'][NT_L1:NT_L1 + NT_L2]
            print(f"| {bs} | {arm} | **{b['expert']}** | {b['slot']} | {rows_n} | "
                  f"{b['nblk']} | {b['valid']} | "
                  f"{'**yes**' if rank == 1 else f'no (#{rank})'} | "
                  f"{l1[0]:.0f}–{l1[-1]:.0f} | {l2[0]:.0f}–{l2[-1]:.0f} | "
                  f"{dt['finish'] - l2[0]:.0f} µs | {dt['ctas']:.1f} | "
                  f"{r['makespan']:.0f} µs |")


if __name__ == '__main__':
    main()
