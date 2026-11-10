"""Producer/consumer timeline with the annotations the README needs.

Row 1  producer: cumulative M blocks RELEASED by the pull CTAs.  A slot's
       blocks become claimable when that slot's pull finishes (type-2 record).
       A vertical dashed line marks "ALL slots pulled".
Row 2  consumer: cumulative M blocks COMPLETED, split by who consumed them:
         role 2 (the routed compute kernel)  -- orange
         role 6 (the helper on the SMs dispatch vacates) -- red
       A dotted red line marks the moment role 6 first completes a block
       (only present in the SM-reuse arm).
Right  outstanding backlog = released - completed.

M-BLOCK IDENTITY.  A "task" record is emitted once per (expert, start,
N-tile): one expert with B blocks and T tiles produces B*T rows.  A block is
therefore a distinct (expert, start) PAIR and its completion time is the LAST
row that pair emits (every tile must finish before the block is done).

`fine_rows` layout (8 words): t_us, type, id, a0, a1, a2, epoch, _
  type 1 slot_begin : id=global slot, a0=local slot, a1=CTA, a2=ncopies
  type 2 slot_end   : id=global slot, a0=local slot, a1=owned, a2=lo
  type 6 task_begin : id=CTA (>=1000 => role 6), a0=local slot, a1=start
  type 7 task_end   : same fields

usage: plot_timeline_annot.py <run.json> [rank] [out.png]
"""
import json, sys, os
from collections import Counter, defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ROLE6_OFFSET = 1000


def _calls(rows):
    """Group the fine rows into MoE calls (one epoch per layer call)."""
    by = defaultdict(list)
    for x in rows:
        by[x[6]].append(x)
    return by


PREFILL_SPAN_US = 2000      # a prefill layer runs ~6.8 ms; decode ~0.2 ms


def pick_call(rows):
    """Pick ONE complete DECODE call (a layer of one step).

    A call is complete when every slot that began pulling also finished
    (type-1 count == type-2 count) and it produced GEMM tasks.

    Among the complete calls take the LATEST epoch.  Two reasons:

      * the log now starts at the measured window, but rows recorded earlier
        (prefill / warmup) stay in the buffer: their epochs are the smallest,
        so "latest" is the measured window;
      * prefill calls have exactly the same SHAPE (all slots pull, all M
        blocks run) but take ~100x longer, so a size test cannot separate them
        -- a span test can, and it is applied first.
    """
    ok = []
    for ep, E in _calls(rows).items():
        n1 = len({x[2] for x in E if x[1] == 1})
        n2 = len({x[2] for x in E if x[1] == 2})
        n6 = len([x for x in E if x[1] == 6])
        if n1 == 0 or n1 != n2 or n6 == 0:
            continue
        span = max(x[0] for x in E) - min(x[0] for x in E)
        ok.append((span, ep, E))
    ok = [o for o in ok if o[0] < PREFILL_SPAN_US] or ok
    if not ok:
        raise SystemExit('no complete call in fine_rows')
    ok.sort(key=lambda o: o[1])
    span, ep, E = ok[-1]
    return ep, E


def build(E):
    u = lambda t: t & 0xFFFFFFFF
    t0 = min(u(x[0]) for x in E if x[1] == 1)
    # --- producer: one event per finished slot, weighted by that slot's blocks
    pairs = {(x[3], x[4]) for x in E if x[1] == 6}          # (slot, start)
    per_slot = Counter(slot for slot, _ in pairs)
    prod = sorted((u(x[0]) - t0, x[3], per_slot.get(x[3], 0)) for x in E if x[1] == 2)
    # --- consumer: a block is done when its LAST tile row lands
    last = {}
    for x in E:
        if x[1] != 7:
            continue
        k = (x[3], x[4])
        t = u(x[0]) - t0
        prev = last.get(k)
        if prev is None or t > prev[0]:
            last[k] = (t, x[2])
    done = sorted(last.values())
    cons2 = [t for t, cta in done if cta < ROLE6_OFFSET]
    cons6 = [t for t, cta in done if cta >= ROLE6_OFFSET]
    return t0, per_slot, prod, cons2, cons6


def plot(path, rank, out):
    d = json.load(open(path))
    rs = (d.get('eco_trace') or {}).get('ranks') or []
    if not rs:
        raise SystemExit(f'{path}: no eco_trace.ranks (trace must be enabled)')
    r = rs[rank] if rank < len(rs) else rs[0]
    rows = [x for x in (r.get('fine_rows') or []) if x[1]]
    if not rows:
        raise SystemExit(f'{path}: empty fine_rows')
    ep, E = pick_call(rows)
    t0, per_slot, prod, cons2, cons6 = build(E)
    total = sum(per_slot.values())
    label = d.get('label', os.path.basename(path).replace('.json', ''))
    bs = d.get('seqs', '?')
    pull_done = prod[-1][0] if prod else 0

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(15, 8.4), sharex=True,
                                 gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.26})
    fig.suptitle(f'Producer / consumer timeline - {label} rank{rank}   '
                 f'BS={bs}  slots={len(prod)}  M blocks={total}  epoch={ep}', fontsize=12)

    # ---- row 1: producer ----
    pt, pv, c = [], [], 0
    for t, slot, n in prod:
        c += n
        pt.append(t); pv.append(c)
    a1.step(pt, pv, where='post', color='tab:blue', lw=1.7,
            label='released by pull CTAs (cumulative M blocks)')
    if pt:
        a1.scatter(pt, pv, s=12, color='tab:blue', zorder=3)
        a1.axvline(pull_done, color='tab:blue', ls='--', lw=1.4, alpha=.85)
        a1.annotate(f'ALL {len(prod)} slots pulled\n@ {pull_done} us',
                    xy=(pull_done, total * .55),
                    xytext=(pull_done + max(4, pull_done * .06), total * .55),
                    fontsize=9, color='tab:blue',
                    arrowprops=dict(arrowstyle='->', color='tab:blue', lw=.9))
    a1.set_ylabel('producer\nreleased (cumulative)')
    a1.set_ylim(0, max(total, 1) * 1.12)
    a1.grid(alpha=.25)
    a1.legend(loc='lower right', fontsize=8)

    # ---- row 2: consumers ----
    if cons2:
        a2.step(cons2, range(1, len(cons2) + 1), where='post',
                color='tab:orange', lw=1.6, label=f'role 2 done ({len(cons2)})')
    if cons6:
        a2.step(cons6, range(1, len(cons6) + 1), where='post',
                color='tab:red', lw=1.8, label=f'role 6 done ({len(cons6)})')
        r6 = cons6[0]
        a2.axvline(r6, color='tab:red', ls=':', lw=1.6, alpha=.9)
        a2.annotate(f'role 6 joins @ {r6} us',
                    xy=(r6, max(1, len(cons6)) * .5),
                    xytext=(r6 + max(4, r6 * .03),
                            max(1, len(cons6)) * .5 + total * .10),
                    fontsize=9, color='tab:red',
                    arrowprops=dict(arrowstyle='->', color='tab:red', lw=.9))
    a2.set_ylabel('consumer\ncompleted (cumulative)')
    a2.set_ylim(0, max(total, 1) * 1.12)
    a2.grid(alpha=.25)
    a2.set_xlabel('absolute time (us, relative to the first slot_begin of the call)')
    a2.legend(loc='lower right', fontsize=8)

    # ---- backlog on the right axis ----
    ev = sorted([(t, 1, n) for t, slot, n in prod] +
                [(t, 0, 1) for t in cons2] + [(t, 0, 1) for t in cons6],
                key=lambda z: (z[0], z[1]))
    ts, back, R, C = [], [], 0, 0
    for t, kind, n in ev:
        if kind:
            R += n
        else:
            C += 1
        ts.append(t); back.append(R - C)
    if ts:
        ax2 = a2.twinx()
        ax2.plot(ts, back, color='0.35', lw=1.2, ls='--', label='outstanding backlog')
        ax2.fill_between(ts, back, color='0.5', alpha=.10)
        ax2.set_ylabel('backlog (blocks)', color='0.35')
        ax2.tick_params(axis='y', labelcolor='0.35')
        ax2.set_ylim(0, max(max(back), 1) * 1.12)

    right = max([pull_done] + cons2 + cons6 + [1])
    for a in (a1, a2):
        a.set_xlim(0, right * 1.05)
        a.xaxis.set_major_locator(MaxNLocator(nbins=18))

    plt.savefig(out, dpi=140, bbox_inches='tight')
    plt.close(fig)
    # fraction of the call during which released > completed: if this is ~all
    # of it, compute never starves and pull is not the bottleneck.
    backlog_pos = sum(back[i] > 0 and (ts[i + 1] - ts[i]) or 0
                      for i in range(len(ts) - 1)) if len(ts) > 1 else 0
    span = (ts[-1] - ts[0]) if len(ts) > 1 else 1
    return dict(label=label, epoch=ep, slots=len(prod), total_mblocks=total,
                pull_done_us=pull_done,
                role6_first_us=(cons6[0] if cons6 else None),
                role2_blocks=len(cons2), role6_blocks=len(cons6),
                cons_end_us=(max(ts) if ts else 0),
                cons_first_us=(min(cons2 + cons6) if (cons2 or cons6) else None),
                backlog_pos_pct=round(100.0 * backlog_pos / max(span, 1), 1))


if __name__ == '__main__':
    path = sys.argv[1]
    rank = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    out = sys.argv[3] if len(sys.argv) > 3 else '/tmp/pc.png'
    print(' '.join(f'{k}={v}' for k, v in plot(path, rank, out).items()))
