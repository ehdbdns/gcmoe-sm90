"""Why the tail blocker is the tail blocker: pull order == consume order.

`claim_sm90_task(header[3])` is a single monotonic counter and dispatch
publishes `active_ready[slot]` in the order slots finish pulling, so a slot can
only be claimed after every earlier-published slot has been claimed.  This
script measures that ordering directly: the rank correlation between
`pull_end(slot)`, `first_task_start(slot)` and `finish(slot)`.

It also prints, for the slot that finishes pulling LAST (the makespan owner),
the breakdown of its own timeline.
"""
import json, os, statistics, sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tail_blockers import pick


def ranks(vals):
    order = sorted(vals, key=lambda s: vals[s])
    return {s: i for i, s in enumerate(order)}


def corr(a, b, keys):
    xs = [a[k] for k in keys]
    ys = [b[k] for k in keys]
    mx, my = statistics.mean(xs), statistics.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = (sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)) ** 0.5
    return num / den if den else float('nan')


def main(run, rank=0, world=8):
    d = json.load(open(run))
    rec = d['eco_trace']['ranks'][rank]
    rows = [x for x in (rec.get('fine_rows') or []) if x[1]]
    ep, E = pick(rows)
    u = lambda t: t & 0xFFFFFFFF
    t0 = min(u(x[0]) for x in E if x[1] == 1)

    pull, own = {}, {}
    for x in E:
        if x[1] == 2:
            pull[x[3]] = u(x[0]) - t0
            own[x[3]] = x[4]
    first, last = {}, {}
    for x in E:
        if x[1] == 6:
            t = u(x[0]) - t0
            first[x[3]] = min(first.get(x[3], 1e9), t)
        elif x[1] == 7:
            last[x[3]] = max(last.get(x[3], 0), u(x[0]) - t0)

    keys = [s for s in pull if s in first and own.get(s)]
    rp, rf, rl = ranks(pull), ranks(first), ranks(last)
    print(f'# {os.path.basename(run)} epoch{ep}  slots with work = {len(keys)}')
    print(f'  rank-corr(pull_end, first_task_start) = {corr(rp, rf, keys):+.3f}')
    print(f'  rank-corr(pull_end, last_task_end)    = {corr(rp, rl, keys):+.3f}')
    print(f'  rank-corr(rows,     pull_end)         = '
          f'{corr(ranks(own), rp, keys):+.3f}')

    s = max(keys, key=lambda k: pull[k])
    print(f'  last slot to finish pulling: slot{s}  rows={own[s]} '
          f'(rank #{sorted(own.values(), reverse=True).index(own[s]) + 1} of {len(own)})')
    print(f'    pull_end {pull[s]:.0f} us, first task {first[s]:.0f} us '
          f'(+{first[s] - pull[s]:.0f}), last task end {last[s]:.0f} us')
    blocks = sorted({x[4] for x in E if x[1] == 6 and x[3] == s})
    for st in blocks:
        sel = sorted(((u(x[0]) - t0, x[1]) for x in E
                      if x[3] == s and x[4] == st and x[1] in (6, 7)))
        beg = [t for t, k in sel if k == 6]
        end = [t for t, k in sel if k == 7]
        print(f'    block start={st:<4} 18 L1 {min(beg[:18]):.0f}-{max(end[:18]):.0f}'
              f' (end), 20 L2 {min(beg[18:]):.0f}-{max(end[18:]):.0f} (end)')


if __name__ == '__main__':
    for r in sys.argv[1:]:
        main(r)
