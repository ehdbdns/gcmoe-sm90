"""Compute-CTA parallelism over time, for one measured layer (rank 0).

Panel 1  how many compute CTAs are RUNNING at each instant, stacked as
         role 2 (the compute partition) + role 6 (the SMs dispatch vacated).
         A CTA counts as running between the start and the end of one of its
         GEMM tasks (`type 6` -> `type 7` of the same CTA, paired in order).
Panel 2  a Gantt of the same data, one row per CTA, so the tail is visible:
         with 71 role-2 CTAs and only ~48 M blocks per layer, the last stretch
         is carried by a handful of CTAs -- exactly what SM-reuse repairs.

Marks (same as the timeline figures): blue dotted = all slots pulled,
red dashed = role 6 starts its first task.

usage: plot_parallelism.py <run.json> [rank] [out.png]
"""
import json, sys, os
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

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
        span = max(x[0] for x in E) - min(x[0] for x in E)
        if span < PREFILL_SPAN_US:
            best = (ep, E)
    if best is None:
        raise SystemExit('no complete decode call in fine_rows')
    return best


def collect(E):
    """(start, end, cta) per GEMM task, us relative to the first slot_begin."""
    u = lambda t: t & 0xFFFFFFFF
    t0 = min(u(x[0]) for x in E if x[1] == 1)
    beg, end = defaultdict(list), defaultdict(list)
    for x in E:
        if x[1] == 6:
            beg[x[2]].append(u(x[0]) - t0)
        elif x[1] == 7:
            end[x[2]].append(u(x[0]) - t0)
    iv, unpaired = [], 0
    for cta, bs in beg.items():
        bs.sort()
        es = sorted(end.get(cta, []))
        unpaired += abs(len(bs) - len(es))
        for i in range(min(len(bs), len(es))):
            iv.append((bs[i], es[i], cta))
    pull_done = max(u(x[0]) for x in E if x[1] == 2) - t0
    return t0, iv, pull_done, unpaired


def smooth(xs, win):
    """Centred moving average, so the saw-tooth from claim/finish gaps is
    readable while the shape (ramp, plateau, tail collapse) is preserved."""
    if win <= 1:
        return list(xs)
    out, n = [], len(xs)
    h = win // 2
    for i in range(n):
        lo, hi = max(0, i - h), min(n, i + h + 1)
        out.append(sum(xs[lo:hi]) / (hi - lo))
    return out


def series(iv, t_end, step):
    ev = []
    for b, e, cta in iv:
        d = 1 if cta < ROLE6_OFFSET else 0
        ev.append((b, d, 1 - d))
        ev.append((e, -d, -(1 - d)))
    ev.sort(key=lambda z: (z[0], z[1] + z[2]))
    ts, c2, c6, a, b6, i, t = [], [], [], 0, 0, 0, 0.0
    while t <= t_end:
        while i < len(ev) and ev[i][0] <= t:
            a += ev[i][1]
            b6 += ev[i][2]
            i += 1
        ts.append(t); c2.append(a); c6.append(b6)
        t += step
    return ts, c2, c6


def plot(path, rank, out=None):
    d = json.load(open(path))
    rs = (d.get('eco_trace') or {}).get('ranks') or []
    if not rs:
        raise SystemExit(f'{path}: no eco_trace.ranks')
    r = rs[rank] if rank < len(rs) else rs[0]
    rows = [x for x in (r.get('fine_rows') or []) if x[1]]
    if not rows:
        raise SystemExit(f'{path}: empty fine_rows')

    ep, E = pick_call(rows)
    t0, iv, pull_done, unpaired = collect(E)
    t_end = max(e for _, e, _ in iv)
    ts, c2, c6 = series(iv, t_end, max(0.5, t_end / 800))

    r2_ctas = sorted({c for _, _, c in iv if c < ROLE6_OFFSET})
    r6_ctas = sorted({c for _, _, c in iv if c >= ROLE6_OFFSET})
    n2, n6 = len(r2_ctas), len(r6_ctas)
    tot = [a + b for a, b in zip(c2, c6)]
    win = max(3, len(ts) // 120)                 # ~1% of the call
    tot_s = smooth(tot, win)
    c2_s = smooth(c2, win)
    peak = max(tot)
    peak_s = max(tot_s)
    mean_tot = sum(tot) / len(tot)
    # All reported numbers come from the SMOOTHED curves: the raw 1 us series
    # saw-tooths between 20 and 77 CTAs because every task start/finish shows
    # up individually, and a single layer's tail is otherwise 3-4 points of
    # noise.  Smoothing does not change the shape (ramp / plateau / collapse).
    dt = ts[1] - ts[0]
    k = max(1, len(ts) // 5)                       # last 20% of the call
    tail = sum(tot_s[-k:]) / k
    tail2 = sum(c2_s[-k:]) / k
    q = len(ts) // 4
    plateau = sum(tot_s[q:len(ts) - q]) / max(1, len(ts) - 2 * q)   # mid 50%
    cta_us = sum(tot_s) * dt                       # total busy-CTA time
    # share of the call during which fewer than half of the role-2 CTAs are
    # busy: the "parallelism collapse" the tail causes.
    half = 0.5 * max(n2, 1)
    collapse = sum(1 for v in tot_s if v < half) / len(tot_s)
    r6_first = min((b for b, _, c in iv if c >= ROLE6_OFFSET), default=None)

    label = d.get('label', os.path.basename(path).replace('.json', ''))
    bs = d.get('seqs', '?')
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.15, 1], hspace=0.28)
    a1 = fig.add_subplot(gs[0])
    a2 = fig.add_subplot(gs[1], sharex=a1)
    fig.suptitle(
        f'Compute-CTA parallelism over time — {label} rank{rank}   '
        f'[epoch {ep}]\n'
        f'BS={bs}  role2 CTAs={n2}  role6 CTAs={n6}   '
        f'peak={peak}  plateau(mid 50%)={plateau:.1f}  '
        f'last 20%={tail:.1f}   makespan={t_end:.0f} us', fontsize=12)

    a1.fill_between(ts, 0, c2, step='post', color='tab:orange', alpha=.45,
                    linewidth=0, label=f'role 2 running (mean {sum(c2)/len(c2):.1f} of {n2})')
    if n6:
        a1.fill_between(ts, c2, tot, step='post', color='tab:red', alpha=.55,
                        linewidth=0,
                        label=f'role 6 running (mean {sum(c6)/len(c6):.1f} of {n6})')
    # raw (1 us) saw-tooth kept faint, smoothed outline on top
    a1.step(ts, tot, where='post', color='0.55', lw=.6, alpha=.9)
    a1.plot(ts, tot_s, color='0.10', lw=2.0,
            label=f'total, {win * (ts[1]-ts[0]):.1f} us moving avg (peak {peak_s:.0f})')
    a1.plot(ts, c2_s, color='tab:orange', lw=1.6, ls='-', alpha=.95)
    if n6:
        a1.plot(ts, [a + b for a, b in zip(c2_s, smooth(c6, win))],
                color='tab:red', lw=1.6, ls='-', alpha=.95)
    a1.axhline(n2, color='tab:orange', ls=':', lw=1.0, alpha=.8)
    a1.axvline(pull_done, color='tab:blue', ls=':', lw=1.1, alpha=.9)
    a1.annotate(f'all slots pulled @ {pull_done:.0f} us',
                xy=(pull_done, peak * .55),
                xytext=(pull_done + t_end * .03, peak * .55),
                fontsize=9, color='tab:blue',
                arrowprops=dict(arrowstyle='->', color='tab:blue', lw=.9))
    if r6_first is not None:
        a1.axvline(r6_first, color='tab:red', ls='--', lw=1.3, alpha=.9)
        a1.annotate(f'role 6 joins @ {r6_first:.0f} us',
                    xy=(r6_first, peak * .28),
                    xytext=(r6_first + t_end * .03, peak * .28),
                    fontsize=9, color='tab:red',
                    arrowprops=dict(arrowstyle='->', color='tab:red', lw=.9))
    # shade the tail (last 10% of the call) so the collapse is easy to see
    a1.axvspan(ts[-k], t_end, color='0.6', alpha=.10)
    a1.text(t_end * .985, peak * .04,
            f'last 20%: {tail:.1f} CTAs'
            + (f'  (role 2 alone {tail2:.1f})' if n6 else ''),
            fontsize=8.5, color='0.30', ha='right')
    a1.set_ylabel('running compute CTAs')
    a1.set_ylim(0, max(peak, n2) * 1.16)
    a1.set_xlim(0, t_end * 1.02)
    a1.grid(alpha=.25)
    a1.legend(loc='upper right', fontsize=8.5)

    # ---- Gantt ----
    order = r2_ctas + r6_ctas
    y = {c: i for i, c in enumerate(order)}
    for b, e, c in iv:
        a2.plot([b, e], [y[c], y[c]],
                color='tab:red' if c >= ROLE6_OFFSET else 'tab:orange',
                lw=1.5, solid_capstyle='butt')
    if r6_ctas:
        a2.axhline(n2 - .5, color='0.35', lw=.8, ls='--')
        a2.annotate('role 6 = the 6 SM dispatch released',
                    xy=(r6_first if r6_first is not None else pull_done, n2 + n6 / 2.0),
                    xytext=(t_end * .30, n2 + n6 + 2.5),
                    fontsize=9, color='tab:red',
                    arrowprops=dict(arrowstyle='->', color='tab:red', lw=.9))
    a2.axvline(pull_done, color='tab:blue', ls=':', lw=1.1, alpha=.9)
    if r6_first is not None:
        a2.axvline(r6_first, color='tab:red', ls='--', lw=1.3, alpha=.9)
    a2.set_ylabel('CTA index\n(role 2 above / role 6 below)')
    a2.set_xlabel('absolute time (us, relative to the first slot_begin)')
    a2.set_ylim(-1, max(len(order) + 1, 2))
    a2.grid(alpha=.20, axis='x')
    a2.set_xlim(0, t_end * 1.02)
    a2.xaxis.set_major_locator(MaxNLocator(nbins=18))

    if out:
        plt.savefig(out, dpi=140, bbox_inches='tight')
    plt.close(fig)
    return dict(label=label, epoch=ep, bs=bs, role2_ctas=n2, role6_ctas=n6,
                peak_ctas=peak,
                mean_ctas=round(mean_tot, 1),
                mean_role2=round(sum(c2) / len(c2), 1),
                mean_role6=round(sum(c6) / len(c6), 1),
                plateau_ctas=round(plateau, 1),
                tail20_ctas=round(tail, 1), tail20_role2=round(tail2, 1),
                cta_us=round(cta_us, -1),
                collapse_pct=round(100.0 * collapse, 1),
                pull_done_us=pull_done, role6_first_us=r6_first,
                unpaired=unpaired, makespan_us=round(t_end, 1))


if __name__ == '__main__':
    path = sys.argv[1]
    rank = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    out = sys.argv[3] if len(sys.argv) > 3 else '/tmp/par.png'
    print(' '.join(f'{k}={v}' for k, v in plot(path, rank, out).items()))
