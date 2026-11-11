"""Producer/consumer timeline — same layout as the earlier `plot_smreuse.py`.

Row 1  producer: every finished slot adds that slot's M-block count; the point
       is (slot_end time, cumulative released).
       Blue dotted line = "ALL slots pulled" (dispatch releases its 6 SMs).
Row 2  consumer: every finished M block adds 1, split by who ran it
         role 2 (block id < 1000)  -- the compute partition      (orange)
         role 6 (block id >= 1000) -- the SMs dispatch vacated   (red)
       Red dashed line = the moment role 6 joins, red shaded band = the
       interval role 6 is active.
Right  outstanding backlog = released - completed.

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
PREFILL_SPAN_US = 2000      # a prefill layer runs ~6.8 ms; decode ~0.2 ms


def _calls(rows):
    by = defaultdict(list)
    for x in rows:
        by[x[6]].append(x)
    return by


def pick_call(rows):
    """Pick ONE complete DECODE call (a layer of one step).

    Complete = every slot that began pulling also finished, and it produced
    GEMM tasks.  Among those take the LATEST epoch: the log starts at the
    measured window, but rows from prefill/warmup stay in the buffer with
    smaller epochs.  Prefill calls have exactly the same SHAPE and ~100x the
    span, so the span test runs first.
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
    return ok[-1][1], ok[-1][2]


def build(E):
    """Four curves, all in us relative to the first slot_begin.

    CONSUMER ROW = "started" (type 6, the first instruction of the GEMM for
    that task), NOT "finished" (type 7).  The point of the row is when the
    compute CTAs begin consuming; using starts is what makes the consumer line
    start a few microseconds into the call, well BEFORE the last slot is
    pulled -- which is exactly the pull/compute overlap this figure exists to
    show.  (`type 7` is kept as a thin dashed line so the gap between the two
    is "claimed but still running".)

    ROLE 6 "joins" = its first type-6 record, i.e. the first task it actually
    starts.  With type 7 this over-reported by ~40 us at BS=256 (169 vs 126),
    because a role-6 CTA has to finish its first task before it can finish.
    """
    u = lambda t: t & 0xFFFFFFFF
    t0 = min(u(x[0]) for x in E if x[1] == 1)
    # tasks a slot publishes = the tasks the consumer sees for that slot
    per_slot = Counter(x[3] for x in E if x[1] == 6)
    prod = sorted((u(x[0]) - t0, x[3], per_slot.get(x[3], 0)) for x in E if x[1] == 2)
    beg = sorted((u(x[0]) - t0, x[2] < ROLE6_OFFSET) for x in E if x[1] == 6)
    end = sorted(u(x[0]) - t0 for x in E if x[1] == 7)
    c2 = [t for t, is2 in beg if is2]
    c6 = [t for t, is2 in beg if not is2]
    return prod, c2, c6, end, sum(per_slot.values())


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
    prod, c2, c6, done, total = build(E)
    pull_done = prod[-1][0] if prod else 0
    r6_first = c6[0] if c6 else None
    label = d.get('label', os.path.basename(path).replace('.json', ''))
    bs = d.get('seqs', '?')

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True,
                                 gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.22})
    kind = ('SM-reuse: ' if c6 else '') + 'producer (pull) vs consumer (compute)'
    fig.suptitle(f'{kind} — {label} rank{rank}\n'
                 f'BS={bs}  slots={len(prod)}  GEMM tasks (slot x N-tile)={total}  '
                 f'role2={len(c2)}, role6={len(c6)}   [epoch {ep}]',
                 fontsize=12)

    # ---- row 1: producer ----
    pt, pv, c = [], [], 0
    for t, slot, n in prod:
        c += n
        pt.append(t); pv.append(c)
    a1.step(pt, pv, where='post', color='tab:blue', lw=1.6,
            label='released (cumulative)')
    a1.scatter(pt, pv, s=9, color='tab:blue', zorder=3)
    a1.axhline(total, color='tab:blue', ls=':', lw=.8, alpha=.55)
    a1.set_ylabel('producer: released GEMM tasks\n(cumulative)')
    a1.set_ylim(0, max(total, 1) * 1.08)
    a1.grid(alpha=.25)

    # ---- row 2: consumers split by role ----
    a2.step(c2, range(1, len(c2) + 1), where='post', color='tab:orange', lw=1.6,
            label=f'role 2 started ({len(c2)} tasks)')
    if c6:
        a2.step(c6, range(1, len(c6) + 1), where='post', color='tab:red', lw=1.8,
                label=f'role 6 started ({len(c6)} tasks)')
    if done:
        a2.step(done, range(1, len(done) + 1), where='post', color='0.45',
                lw=.9, ls='--', label=f'finished ({len(done)} tasks, both roles)')
    a2.set_ylabel('consumer: started GEMM tasks\n(cumulative, per role)')
    a2.set_ylim(0, max(max(len(c2), len(c6)), 1) * 1.10)
    a2.grid(alpha=.25)
    a2.set_xlabel('absolute time (us, relative to first slot_begin)')

    # ---- backlog on the right axis ----
    ev = sorted([(t, 1, n) for t, _, n in prod] + [(t, 0, 1) for t in done],
                key=lambda z: (z[0], z[1]))
    ts, back, R, C = [], [], 0, 0
    for t, kind, n in ev:
        if kind:
            R += n
        else:
            C += 1
        ts.append(t); back.append(R - C)
    ax2 = a2.twinx()
    ax2.plot(ts, back, color='0.35', lw=1.1, ls='--', label='outstanding backlog')
    ax2.fill_between(ts, back, color='0.5', alpha=.10)
    ax2.set_ylabel('outstanding backlog (tasks)', color='0.35')
    ax2.tick_params(axis='y', labelcolor='0.35')
    ax2.set_ylim(0, max(max(back), 1) * 1.12)

    # ---- the two marks: pull done, role 6 joins ----
    last = max(ts) if ts else 1
    for a in (a1, a2):
        a.axvline(pull_done, color='tab:blue', ls=':', lw=1.0, alpha=.8)
        if r6_first is not None:
            a.axvline(r6_first, color='tab:red', ls='--', lw=1.4, alpha=.9)
    if r6_first is not None:
        a2.axvspan(pull_done, last, color='tab:red', alpha=.06)
    a1.text(pull_done + max(1, pull_done * .02), total * .08,
            f'pull done {pull_done} us\n(dispatch releases its 6 SM)',
            fontsize=8.5, color='tab:blue')
    if r6_first is not None:
        a1.text(r6_first + max(1, r6_first * .02), total * .42,
                f'>>> role 6 joins at {r6_first} us\n'
                f'     ({r6_first - pull_done:+d} us vs pull done)',
                fontsize=9, color='tab:red', fontweight='bold')
    a2.legend(loc='upper left', fontsize=8.5)

    for a in (a1, a2):
        a.set_xlim(0, max(last, pull_done, 1) * 1.04)
        a.xaxis.set_major_locator(MaxNLocator(nbins=18))

    plt.savefig(out, dpi=140, bbox_inches='tight')
    plt.close(fig)

    pos = sum((back[i] > 0) * (ts[i + 1] - ts[i]) for i in range(len(ts) - 1)) if len(ts) > 1 else 0
    span = (ts[-1] - ts[0]) if len(ts) > 1 else 1
    return dict(label=label, epoch=ep, slots=len(prod), mblocks=total,
                total_mblocks=total, plot_unit='tasks (slot x N-tile)',
                # A decode layer pulls in ~20-300 us; anything near a
                # millisecond means the plot fell back to a PREFILL call, which
                # would be a silent, 100x-wrong figure. Flag it loudly.
                suspect=bool(pull_done and pull_done > 600),
                pull_done_us=pull_done,
                role6_first_us=r6_first,
                role2_blocks=len(c2), role6_blocks=len(c6),
                cons_first_us=(min(c2 + c6) if (c2 or c6) else None),
                cons_end_us=last,
                backlog_pos_pct=round(100.0 * pos / max(span, 1), 1))


if __name__ == '__main__':
    path = sys.argv[1]
    rank = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    out = sys.argv[3] if len(sys.argv) > 3 else '/tmp/pc.png'
    print(' '.join(f'{k}={v}' for k, v in plot(path, rank, out).items()))
