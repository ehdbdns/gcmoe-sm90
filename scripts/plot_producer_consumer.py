"""Producer (pull CTA) vs Consumer (compute CTA) timeline for one rank/episode.

Row 1  producer: every time a pull CTA finishes ONE slot, add that slot's
        M-block count; the point is (that finish time, cumulative released).
Row 2  consumer: every finished M block adds 1; the dashed line on the right
        axis is the outstanding backlog (released - completed).

Type codes in `fine_rows` (8 words: t_us, type, id, a0, a1, a2, epoch, _):
  1 slot_begin : id=global slot, a0=local slot, a1=CTA, a2=ncopies
  2 slot_end   : id=global slot, a0=local slot, a1=owned tokens, a2=lo
  6 task_begin : id=CTA, a0=local slot, a1=start row, a2=valid
  7 task_end   : id=CTA, a0=local slot, a1=start row, a2=valid

Non-ECO arms emit no type-6/7 with a *stable* epoch for every block, so the
consumer row is only drawn when the counts are consistent.

usage: plot_producer_consumer.py <run.json> [rank] [out.png]
"""
import json, sys
from collections import Counter, defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def load(path, rank):
    d = json.load(open(path))
    rs = (d.get('eco_trace') or {}).get('ranks') or []
    if not rs:
        raise SystemExit(f'{path}: no eco_trace.ranks (was trace enabled?)')
    r = rs[rank]
    rows = [x for x in (r.get('fine_rows') or []) if x[1]]
    if not rows:
        raise SystemExit(f'{path}: empty fine_rows')
    ep = Counter(x[6] for x in rows).most_common(1)[0][0]
    return d, [x for x in rows if x[6] == ep], ep


def build(E):
    """Return (prod, cons) event lists in us relative to the first slot_begin."""
    u = lambda t: t & 0xFFFFFFFF
    t0 = min(u(x[0]) for x in E if x[1] == 1)
    mblocks = Counter(x[3] for x in E if x[1] == 6)     # local slot -> blocks
    prod = sorted((u(x[0]) - t0, x[3], mblocks.get(x[3], 0), x[4])
                  for x in E if x[1] == 2)
    cons = sorted(u(x[0]) - t0 for x in E if x[1] == 7)
    return t0, mblocks, prod, cons


def plot(path, rank, out):
    d, E, ep = load(path, rank)
    t0, mblocks, prod, cons = build(E)
    total = sum(mblocks.values())
    label = d.get('label', path.split('/')[-1])

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True,
                                 gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.22})
    fig.suptitle(f'Producer (pull CTA) vs Consumer (compute CTA) — {label} rank{rank}\n'
                 f'BS={d.get("seqs", "?")}  slots={len(prod)}  M blocks={total}', fontsize=12)

    # --- row 1: producer (cumulative released) ---
    pt, pv, c = [], [], 0
    for t, slot, n, owned in prod:
        c += n
        pt.append(t); pv.append(c)
    a1.step(pt, pv, where='post', color='tab:blue', lw=1.6, label='released (cumulative)')
    a1.scatter(pt, pv, s=9, color='tab:blue', zorder=3)
    a1.axhline(total, color='tab:blue', ls=':', lw=.8, alpha=.55)
    a1.annotate(f'{total} blocks released by {pt[-1]} µs',
                xy=(pt[-1], total), xytext=(pt[-1] + 8, total * 0.5),
                fontsize=9, color='tab:blue',
                arrowprops=dict(arrowstyle='->', color='tab:blue', lw=.8))
    a1.set_ylabel('producer: released M blocks\n(cumulative)')
    a1.set_ylim(0, total * 1.08)
    a1.grid(alpha=.25)

    # --- row 2: consumer (cumulative completed) + backlog ---
    if cons:
        a2.step(cons, range(1, len(cons) + 1), where='post', color='tab:orange', lw=1.6)
        a2.scatter(cons, range(1, len(cons) + 1), s=2.6, color='tab:orange', alpha=.5, zorder=3)
    a2.set_ylabel('consumer: completed M blocks\n(cumulative)')
    a2.set_ylim(0, total * 1.08)
    a2.grid(alpha=.25)
    a2.set_xlabel('absolute time (µs, relative to the first slot_begin)')

    # backlog on a shared event stream
    ev = sorted([(t, 1, n) for t, _, n, _ in prod] + [(t, 0, 1) for t in cons],
                key=lambda z: (z[0], z[1]))
    ts, back, R, C = [], [], 0, 0
    for t, kind, n in ev:
        if kind: R += n
        else: C += 1
        ts.append(t); back.append(R - C)
    ax2 = a2.twinx()
    ax2.plot(ts, back, color='0.35', lw=1.2, ls='--')
    ax2.fill_between(ts, back, color='0.5', alpha=.10)
    ax2.set_ylabel('outstanding backlog (blocks)', color='0.35')
    ax2.tick_params(axis='y', labelcolor='0.35')
    ax2.set_ylim(0, max(back) * 1.12)
    if back:
        pk = back.index(max(back))
        ax2.annotate(f'peak backlog {max(back)} @ {ts[pk]} µs',
                     xy=(ts[pk], max(back)), xytext=(ts[pk] + 40, max(back) * .70),
                     fontsize=9, color='0.25',
                     arrowprops=dict(arrowstyle='->', color='0.35', lw=.8))

    for a in (a1, a2):
        a.set_xlim(0, max([pt[-1]] + ([cons[-1]] if cons else [0])) * 1.04)
        a.xaxis.set_major_locator(MaxNLocator(nbins=18))
    a1.legend(loc='lower right', fontsize=8)

    plt.savefig(out, dpi=140, bbox_inches='tight')
    plt.close(fig)
    return dict(label=label, slots=len(prod), blocks=total,
                pull_done=pt[-1], gemm_done=cons[-1] if cons else None,
                peak_backlog=max(back) if back else None)


if __name__ == '__main__':
    path = sys.argv[1]
    rank = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    out = sys.argv[3] if len(sys.argv) > 3 else '/tmp/pc.png'
    info = plot(path, rank, out)
    print(' '.join(f'{k}={v}' for k, v in info.items()), '->', out)
