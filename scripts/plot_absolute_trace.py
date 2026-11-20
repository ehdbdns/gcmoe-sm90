"""Absolute-time trace of ONE measured layer: pull CTAs and compute CTAs on a
shared time axis, with every slice labelled.

Rows (top to bottom)
  compute CTA 0..C-1     one row per routed CTA (role 2)
  ── separator ──
  pull CTA 0..D-1        one row per dispatch CTA (role 1)

Colours
  pull      light blue   span of one slot: type 1 (slot_begin) -> type 2 (slot_end)
                         label: `E<local slot> s<global slot>`
  GEMM      orange       type 6 (task_begin) -> type 7 (task_end)
                         label: `E<e> T<ntile> M<mblock>`
  combine   green        type 8 (push begin) -> type 9 (push end)
                         label: `P<ntile>`

Everything is in microseconds relative to the FIRST slot_begin of the chosen
call. The figure is deliberately large and uses a tiny font only for slice
labels, so the numbers stay readable when the PNG is opened at 100%.

An aligned *.detail.json uses the same-stem counts and the measured dispatch
entry as t=0. It produces the full figure and a CSV for the requested rank.

usage: plot_absolute_trace.py <run.json | *.detail.json> [rank] [out.png]
"""
import json, os, sys
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

PREFILL_SPAN_US = 2000      # a decode layer is far shorter than a prefill call
ROLE6_OFFSET = 1000         # plot_parallelism's convention for role 6 CTA ids



# [ABSTRACE] Fit a label INSIDE its rectangle: centred, and as large as the box
# allows. Two constraints, both in figure inches:
#   horizontal: text_width  = n_chars * fs * CHAR_W   <= box_w * SHRINK
#   vertical  : text_height = fs / 72                 <= bar_h_in * VSHRINK
# `CHAR_W` is the average advance of the DejaVu Sans digits/letters we use
# (~0.60 em), measured empirically; SHRINK leaves a margin so the glyphs never
# touch the border.
CHAR_W = 0.60
H_SHRINK = 0.92
V_SHRINK = 0.86


def fit_font(text, box_w_in, box_h_in, cap, lines=1):
    """Largest font size (pt) whose rendered text fits the box, capped.

    `text` is the LONGEST line; `lines` splits the vertical budget, which is how
    a two-line label buys a much bigger font on a narrow bar (the box is wide
    but short: 10 us -> 38 inch wide, 0.68 inch tall).
    """
    if box_w_in <= 0 or box_h_in <= 0:
        return 0.0
    # [ABSTRACE-BUG] BOTH bounds must be in POINTS. `by_w` used to be computed
    # in inches (box_w_in / char_width_in) and then compared against a point
    # value from `by_h`, so it came out 72x too small and every label fell below
    # the minimum-size guard -- the full-layer figure ended up with NO labels
    # at all. 1 pt = 1/72 inch, hence the * 72 here.
    text_w_in = max(len(text), 1) * CHAR_W * (1.0 / 72.0)   # inches at 1 pt
    by_w = (box_w_in * H_SHRINK) / text_w_in                 # pt
    by_h = box_h_in * V_SHRINK * 72.0 / max(lines, 1)        # pt
    return max(0.0, min(cap, by_w, by_h))

def load(path, rank):
    d = json.load(open(path))
    rs = (d.get('eco_trace') or {}).get('ranks') or []
    if not rs:
        raise SystemExit(f'{path}: no eco_trace.ranks')
    r = rs[rank] if rank < len(rs) else rs[0]
    rows = [x for x in (r.get('fine_rows') or []) if x[1]]
    if not rows:
        raise SystemExit(f'{path}: empty fine_rows')
    # [ABSTRACE] The kernel log is a RING (slot = counter % FINE_RECS), so the
    # array order is NOT emission order once it wraps. `fine_count` says how
    # many records were written; rotate by `count % FINE_RECS` and drop the
    # unwritten tail to recover the true sequence.
    # both fields live on the RANK dict (the dumper writes them per rank)
    total = int(r.get('fine_count') or 0)
    recs = int(r.get('fine_recs') or 0) or len(rows)
    # [ABSTRACE] Do NOT rely on `fine_count` to rebuild order: `irstep_begin`
    # zeroes the counter between windows, so it no longer reflects how many
    # records the ring holds (measured: count=7901 with 467k live records).
    # Timestamps are monotonic WITHIN a call and we only ever analyse one call,
    # so sorting by time recovers the true sequence and is wrap-safe.
    rows.sort(key=lambda x: x[0])
    return d, rows


def pick_call(rows):
    """One complete decode call: has slot_begin, slot_end AND GEMM tasks."""
    best = None
    for ep in sorted({x[8] for x in rows}):
        E = [x for x in rows if x[8] == ep]
        kinds = {x[1] for x in E}
        if not ({1, 2, 6, 7} <= kinds):
            continue
        span = max(x[0] for x in E) - min(x[0] for x in E)
        if span < PREFILL_SPAN_US:
            best = (ep, E)
    if best is None:
        raise SystemExit('no complete decode call in fine_rows')
    return best


def intervals(E, k_begin, k_end, t0, field_cta=lambda x: x[2]):
    """Pair begin/end records per CTA, in order, WITHIN one epoch only.

    The dump is a fixed-size ring: once 262144 records are reached the global
    counter stops advancing, so a read-back contains the FIRST 262144 events
    and no more. Pairing globally by CTA therefore mixes a begin from one layer
    with an end from another and produces absurd spans (measured: +/-5e7 us).
    `E` is already restricted to one call, so pairing here is safe; the caller
    still validates that b <= e and drops anything else.
    """
    beg, end = defaultdict(list), defaultdict(list)
    for x in E:
        if x[1] == k_begin:
            beg[field_cta(x)].append(x)
        elif x[1] == k_end:
            end[field_cta(x)].append(x)
    out, unpaired = [], 0
    for cta, bs in beg.items():
        bs.sort(key=lambda x: x[0])
        es = sorted(end.get(cta, []), key=lambda x: x[0])
        unpaired += abs(len(bs) - len(es))
        for i in range(min(len(bs), len(es))):
            b, e = bs[i], es[i]
            d = e[0] - b[0]
            if d < 0 or d > 100000:      # a GEMM/push slice is << 100 ms
                continue
            out.append((b[0] - t0, e[0] - t0, cta, b))
    return out, unpaired


def aligned_spans(path, rank):
    detail = json.loads(path.read_text())
    counts_path = path.with_name(path.name.removesuffix('.detail.json') + '.counts.json')
    counts = json.loads(counts_path.read_text())
    for key in ('capture_id', 'step', 'layer', 'epoch'):
        if detail[key] != counts[key]:
            raise ValueError(f'{path}: {key} differs from {counts_path}')
    data, totals = ({row['rank']: row for row in doc['ranks']}[rank]
                    for doc in (detail, counts))
    for key in ('tokens', 'routed_rows', 'm_blocks'):
        if data[key] != totals[key]:
            raise ValueError(f'rank {rank}: {key} differs from aligned counts')

    def spans(items, kind):
        result = []
        for row in items:
            begin, end, cta = row['start_us'], row['end_us'], row['cta']
            if kind == 1:
                label = (begin, kind, row['global_slot'], row['slot'], cta, row['copies'])
            else:
                label = (begin, kind, cta, row['slot'], row['row_start'], row['valid'],
                         row['ntile'], row['phase'])
            result.append((begin, end, cta, label))
        return result

    return (detail['epoch'], totals['block_m'], spans(data['pull_slots'], 1),
            spans(data['tiles'], 6), spans(data['l2_pushes'], 8),
            range(data['dispatch_ctas']), range(data['routed_ctas']))


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    path = sys.argv[1]
    rank = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    out = sys.argv[3] if len(sys.argv) > 3 else None
    aligned = path.endswith('.detail.json')
    if aligned:
        ep, bm, pull, gemm, push, pull_ctas, comp_ctas = aligned_spans(Path(path), rank)
        pull_unpaired = gemm_unpaired = push_unpaired = 0
    else:
        d, rows = load(path, rank)
        # [ABSTRACE] normalise the 32-bit microsecond stamps into non-negative
        # values relative to the earliest record we actually have. The kernel
        # stores `now_us - t0` with a t0 latched on the very first record, which can
        # sit in prefill and be filtered out by the epoch floor, so the surviving
        # rows are negative offsets from THAT origin. Rebasing here keeps the plot
        # independent of where the floor landed.
        tmin = min(x[0] for x in rows)
        if tmin != 0:
            for x in rows:
                x[0] = x[0] - tmin
        ep, E = pick_call(rows)

        # ---- time origin: first slot_begin of this call ----
        t0 = min(x[0] for x in E if x[1] == 1)

        # pull: type 1 -> 2, keyed by global slot (x[2])
        paired, pull_unpaired = intervals(E, 1, 2, t0)
        pull = [(b, e, rec[4], rec) for b, e, _, rec in paired]
        # GEMM: type 6 -> 7, keyed by CTA (x[2])
        gemm, gemm_unpaired = intervals(E, 6, 7, t0)
        # combine push: type 8 -> 9, keyed by CTA
        push, push_unpaired = intervals(E, 8, 9, t0)

        pull_ctas = sorted({c for _, _, c, _ in pull})
        comp_ctas = sorted({c for _, _, c, _ in gemm})
        try:
            bm = int((d.get('load_detail') or {}).get('block_m') or 0) or None
        except (TypeError, ValueError):
            bm = None
    n_pull, n_comp = len(pull_ctas), len(comp_ctas)
    # [ABSTRACE] compute rows on TOP (that is what the reader follows first:
    # GEMM is where the time goes), pull rows at the BOTTOM, separated by one
    # blank row. Row 0 is at the top of a matplotlib axis, so the compute block
    # gets indices 0..n_comp-1.
    comp_row = {c: i for i, c in enumerate(comp_ctas)}
    pull_row = {c: n_comp + 1 + i for i, c in enumerate(pull_ctas)}

    t_end = max([e for _, e, _, _ in pull] + [e for _, e, _, _ in gemm] +
                [e for _, e, _, _ in push] + [1.0])
    t_end *= 1.02
    # a slice needs ~1.6% of the axis width to hold "E123 T4 M2"
    label_min_us = t_end * 0.026

    # ---- figure: one row per CTA, generous size ----
    rows_total = n_pull + 1 + n_comp
    # [ABSTRACE] The whole point of this figure is the per-slice numbering, so
    # the canvas is deliberately huge: ~0.42 inch per CTA row and 90 inch wide,
    # which keeps `E123 T4 M2` readable at 100% zoom.
    height = max(18.0, 0.42 * rows_total)
    FIG_W_IN = 150.0
    fig, ax = plt.subplots(figsize=(FIG_W_IN, height * 1.35))
    bar_h = 0.66

    # [ABSTRACE] Data-to-inch mapping, needed to size each label to its own
    # rectangle. `tight_layout` trims ~4% off each side, so scale the usable
    # width/height by AXES_FRAC; being slightly conservative here only makes the
    # labels a hair smaller, never overflowing.
    AXES_FRAC_W, AXES_FRAC_H = 0.94, 0.90
    in_per_us = FIG_W_IN * AXES_FRAC_W / max(t_end, 1e-9)
    in_per_row = (height * 1.35) * AXES_FRAC_H / max(rows_total, 1e-9)
    bar_h_in = bar_h * in_per_row

    # label caps: the box decides, these are only upper bounds
    fs = 0.0        # unused; kept so the older call sites stay readable

    # ---- pull spans (light blue) ----
    for b, e, cta, rec in pull:
        y = pull_row[cta]
        ax.add_patch(Rectangle((b, y - bar_h / 2), max(e - b, 0.05), bar_h,
                               facecolor='#9ec9ef', edgecolor='#4a7fb5',
                               linewidth=0.25, zorder=2))
        txt = f'E{rec[3]} s{rec[2]}'
        _f = fit_font(txt, (e - b) * in_per_us, bar_h_in, cap=44.0)
        if _f >= 6.0:
            ax.text((b + e) / 2.0, y, txt, fontsize=_f, va='center',
                    ha='center', zorder=6, color='#12314d')

    # ---- separator between the two CTA classes ----
    sep_y = n_comp + 0.5
    ax.axhline(sep_y, color='#888888', linewidth=1.8, zorder=3)
    ax.text(t_end * 0.002, sep_y,
            f'  compute CTAs (role 2) ABOVE  |  pull CTAs (role 1) BELOW  '
            f'| pull={n_pull} CTAs, compute={n_comp} CTAs, epoch={ep}',
            fontsize=20, va='center', ha='left', color='#333333', zorder=9,
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=1.5))

    # ---- GEMM spans (orange) + combine push (green) ----
    push_by_cta = defaultdict(list)
    for b, e, cta, rec in push:
        push_by_cta[cta].append((b, e, rec))

    # [ABSTRACE] Where the PUSH phase of a given compute span starts. The L2
    # push is logged separately (type 8), so the orange bar must stop there and
    # a green bar must take over; a type-7 (L1) span has no push at all.
    push_start = {}
    for pb, pe, pcta, prec in push:
        push_start[(pcta, prec[3], prec[4], prec[6], prec[7])] = pb

    # [ABSTRACE] The green bar must end exactly where its GEMM span ends.
    # `globaltimer//1000` is truncated to whole microseconds, so a push that
    # really finished inside the last microsecond of its GEMM reports
    # `push_end = gemm_end - 1`. Measured: 390/1000 pushes land on gemm_end,
    # 610/1000 land 1 us short, and NONE is short by more than 1 us. Drawing
    # the raw value therefore leaves a 1 us orange sliver after the green --
    # at the L2 zoom's ~550 px/us that reads as "compute still running after
    # push", which is wrong. Snap the green to the containing GEMM's end.
    _gemm_by_key = defaultdict(list)
    for _b, _e, _c, _r in gemm:
        _gemm_by_key[(_c, _r[3], _r[4], _r[6], _r[7])].append((_b, _e))

    def push_end_snap(cta, begin, end, rec):
        """End of the green bar: the enclosing GEMM span's end."""
        if aligned:
            return end  # Display the measured push end on the aligned time axis.
        for _b, _e in _gemm_by_key.get((cta, rec[3], rec[4], rec[6], rec[7]), ()):
            if _b <= begin <= _e:
                return max(end, _e)      # only ever extends, never shrinks
        return end

    for b, e, cta, rec in gemm:
        y = comp_row[cta]
        expert, start, valid, ntile, phase = rec[3], rec[4], rec[5], rec[6], rec[7]
        _ps = push_start.get((cta, expert, start, ntile, phase))
        _c_end = min(e, _ps) if (_ps is not None and b <= _ps <= e) else e
        ax.add_patch(Rectangle((b, y - bar_h / 2), max(_c_end - b, 0.05), bar_h,
                               facecolor='#f6a94a', edgecolor='#b06a10',
                               linewidth=0.25, zorder=2))
        mb = (start // bm) if bm else start
        # [ABSTRACE] Label carries the M block's VALID row count: E20T3M40v40
        tag = f'E{expert}T{ntile}M{mb}v{valid}' + ('+' if phase == 2 else '')
        _f = fit_font(tag, (_c_end - b) * in_per_us, bar_h_in, cap=40.0)
        if _f >= 6.0:
            ax.text((b + _c_end) / 2.0, y, tag, fontsize=_f, va='center',
                    ha='center', zorder=6, color='#4a2a00')

    for cta, lst in push_by_cta.items():
        if cta not in comp_row:
            continue
        y = comp_row[cta]
        for b, e, rec in lst:
            # [ABSTRACE] PUSH IS ITS OWN PHASE: a full-height green bar that
            # follows the orange compute bar, not a thin stripe under it. Its
            # duration scales with `valid` (more rows -> longer push), which is
            # exactly what this split exists to show.
            # No text on the green bars: they are 1-3 us wide, so a label
            # either does not fit or (when it does) hides the very length
            # difference the bar is there to show. The `valid` of a push is
            # already printed on its orange compute bar, and the CSV has both.
            e = push_end_snap(cta, b, e, rec)
            ax.add_patch(Rectangle((b, y - bar_h / 2), max(e - b, 0.05), bar_h,
                                   facecolor='#5fbf6a', edgecolor='#2c7a34',
                                   linewidth=0.25, zorder=3))

    # ---- axes cosmetics ----
    ax.set_xlim(0, t_end)
    # [ABSTRACE] invert: row 0 (compute CTA 0) must appear at the TOP, which is
    # what the reader expects ("dispatch at the top, compute below" in the
    # original figure was role ordering, not time ordering). With inverted
    # limits, index 0 renders at the top edge.
    ax.set_ylim(rows_total - 0.2, -0.8)
    origin = 'dispatch entry' if aligned else 'first slot_begin'
    ax.set_xlabel(f'absolute time within one measured layer (us, t=0 at {origin})', fontsize=26)
    ax.set_ylabel('CTA row', fontsize=26)
    pull_done = max(e for _, e, _, _ in pull) if pull else 0.0
    ax.axvline(pull_done, color='#1f5fbf', linestyle=':', linewidth=1.4,
               zorder=4)
    ax.text(pull_done, rows_total - 0.7, f' all slots pulled {pull_done:.1f}us',
            fontsize=18, va='bottom', ha='left', color='#1f5fbf', zorder=8)

    yt, yl = [], []
    for c in pull_ctas:
        yt.append(pull_row[c]); yl.append(f'pull CTA {c}')
    for c in comp_ctas:
        yt.append(comp_row[c]); yl.append(f'comp CTA {c}')
    ax.set_yticks(yt)
    ax.set_yticklabels(yl, fontsize=max(7.0, min(13.0, 460.0 / rows_total)))
    ax.grid(axis='x', color='#dddddd', linewidth=0.4, zorder=0)

    source = '/'.join(Path(path).parts[-3:]) if aligned else os.path.basename(path)
    title = (f'{source}  rank {rank}  epoch {ep}  '
             f'block_m={bm}  '
             f'pull={len(pull)} spans, gemm={len(gemm)} spans, '
             f'push={len(push)} spans')
    ax.set_title(title, fontsize=30, pad=18)

    # legend (proxy artists)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor='#9ec9ef', edgecolor='#4a7fb5',
                             label='pull a slot (E=local slot, s=global slot)'),
                       Patch(facecolor='#f6a94a', edgecolor='#b06a10',
                             label='GEMM of one M block: E=local slot T=ntile '
                                   'M=mblock v=valid rows'),
                       Patch(facecolor='#5fbf6a', edgecolor='#2c7a34',
                             label='L2 combine push')],
              loc='lower center', bbox_to_anchor=(0.5, -0.055),
              ncol=3, fontsize=22, framealpha=0.95)

    # ---- full CSV: every slice with its numbers, so nothing is lost when the
    # figure can only label the wide ones ----
    csv_path = (os.path.splitext(out or path)[0] + f'.abstrace_rank{rank}.csv')
    with open(csv_path, 'w') as fh:
        slot_column = 'local_slot' if aligned else 'expert'
        fh.write(f'kind,cta,begin_us,end_us,dur_us,{slot_column},ntile,mblock,'
                 'start_row,valid,phase,global_slot\n')
        for b, e, cta, rec in sorted(pull, key=lambda z: z[3][0]):
            fh.write(f'pull,{cta},{b:.1f},{e:.1f},{e - b:.1f},'
                     f'{rec[3]},,,,0,,{rec[2]}\n')
        for b, e, cta, rec in sorted(gemm, key=lambda z: (z[0])):
            mb = (rec[4] // bm) if bm else rec[4]
            fh.write(f'gemm,{cta},{b:.1f},{e:.1f},{e-b:.1f},'
                     f'{rec[3]},{rec[6]},{mb},{rec[4]},{rec[5]},{rec[7]},\n')
        for b, e, cta, rec in sorted(push, key=lambda z: (z[0])):
            mb = (rec[4] // bm) if bm else rec[4]
            fh.write(f'push,{cta},{b:.1f},{e:.1f},{e-b:.1f},'
                     f'{rec[3]},{rec[6]},{mb},{rec[4]},{rec[5]},{rec[7]},\n')

    fig.tight_layout()
    out = out or (os.path.splitext(path)[0] + f'.abstrace_rank{rank}.png')
    fig.savefig(out, dpi=80 if aligned else 150, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {csv_path}')
    print(f'wrote {out}')
    print(f'unpaired: pull={pull_unpaired} gemm={gemm_unpaired} '
          f'push={push_unpaired}')
    print(f'spans: pull={len(pull)} ({n_pull} CTAs)  gemm={len(gemm)} '
          f'({n_comp} CTAs)  push={len(push)}')
    print(f'timeline: pull_done={pull_done:.1f}us  end={t_end:.1f}us')

    if aligned:
        return

    # ---- zoomed panels ----
    # The full-call figure necessarily packs ~1900 GEMM slices onto one axis;
    # at that density no labelling scheme stays readable. These panels zoom a
    # narrow window where EVERY slice can carry its full label.
    #
    # Two windows are emitted:
    #   * the opening window (pull overlaps the first L1 GEMMs), and
    #   * an L2 window (the second GEMM pass), because that is the only phase
    #     that ends with a combine push -- the green segment this figure exists
    #     to show. In the opening window there is no push at all, so a reader
    #     looking for green there would wrongly conclude it was never recorded.
    g0 = min([b for b, _e, _c, _r in gemm] or [0.0])
    l2_begins = sorted(b for b, _e, _c, r in gemm if r[7] == 2)
    if os.environ.get('AT_ZOOM2_LO'):
        z2_lo = float(os.environ['AT_ZOOM2_LO'])
    else:
        # Window scan over the L2 pass. The score deliberately rewards the
        # number of DISTINCT `valid` values first (that is the whole point:
        # show that push length tracks valid), then the number of slices.
        z2_lo, best = 0.0, (-1, -1)
        _w = float(os.environ.get('AT_ZOOM2_W', '30') or 30)
        for _s in l2_begins[::4] or [0.0]:
            _vs = {r[5] for _b, _e, _c, r in push
                   if _s <= _b < _s + _w}
            _n = sum(1 for _b in l2_begins if _s <= _b < _s + _w)
            _score = (len(_vs), _n)
            if _score > best:
                best, z2_lo = _score, _s
        z2_lo = max(0.0, z2_lo - 2.0)
    z2_hi = z2_lo + float(os.environ.get('AT_ZOOM2_US', '30') or 30)

    def render_zoom(zoom_lo, zoom_hi, note):
        # [ABSTRACE] The zoom carries the "every label, as large as its box
        # allows" requirement. Bars are ~10-50 us; to make an 11-char label
        # fit, the window must be narrow relative to the canvas.
        zoom_us = zoom_hi - zoom_lo
        ZFIG_W_IN = 110.0
        ZFIG_H_IN = max(30.0, 1.15 * rows_total)
        fig2, ax2 = plt.subplots(figsize=(ZFIG_W_IN, ZFIG_H_IN))
        z_in_per_us = ZFIG_W_IN * 0.94 / max(zoom_us, 1e-9)
        z_in_per_row = ZFIG_H_IN * 0.90 / max(rows_total, 1e-9)
        z_bar_h_in = bar_h * z_in_per_row
        for b, e, cta, rec in pull:
            if b > zoom_hi or e < zoom_lo:
                continue
            y = pull_row[cta]
            _lo, _hi = max(b, zoom_lo), min(e, zoom_hi)
            ax2.add_patch(Rectangle((_lo, y - bar_h / 2),
                                    max(_hi - _lo, 0.02),
                                    bar_h, facecolor='#9ec9ef',
                                    edgecolor='#4a7fb5', linewidth=0.4, zorder=2))
            tag = f'E{rec[3]}s{rec[2]}'
            _f = fit_font(tag, (_hi - _lo) * z_in_per_us, z_bar_h_in, cap=48.0)
            if _f >= 5.0:
                ax2.text((_lo + _hi) / 2.0, y, tag, fontsize=_f, va='center',
                         ha='center', zorder=6, color='#12314d')
        ax2.axhline(sep_y, color='#888888', linewidth=1.8, zorder=3)
        for b, e, cta, rec in gemm:
            if b > zoom_hi or e < zoom_lo:
                continue
            y = comp_row[cta]
            # [ABSTRACE] Same rule as the full figure: the orange bar stops
            # where this M block's combine push begins, and the push owns the
            # remainder as its own phase.
            _ps = push_start.get((cta, rec[3], rec[4], rec[6], rec[7]))
            _ce = min(e, _ps) if (_ps is not None and b <= _ps <= e) else e
            _lo, _hi = max(b, zoom_lo), min(_ce, zoom_hi)
            if _hi <= _lo:
                continue
            ax2.add_patch(Rectangle((_lo, y - bar_h / 2), max(_hi - _lo, 0.02),
                                    bar_h, facecolor='#f6a94a',
                                    edgecolor='#b06a10', linewidth=0.4, zorder=2))
            mb = (rec[4] // bm) if bm else rec[4]
            # ONE compact line (no spaces) so the glyph gets the whole vertical
            # budget: `E15T17M0v42`. `+` marks the L2 (second) GEMM pass.
            tag = f'E{rec[3]}T{rec[6]}M{mb}v{rec[5]}' + ('+' if rec[7] == 2 else '')
            _f = fit_font(tag, (_hi - _lo) * z_in_per_us, z_bar_h_in, cap=48.0)
            if _f >= 5.0:
                ax2.text((_lo + _hi) / 2.0, y, tag, fontsize=_f, va='center',
                         ha='center', zorder=6, color='#4a2a00')
        for cta, lst in push_by_cta.items():
            if cta not in comp_row:
                continue
            y = comp_row[cta]
            for b, e, rec in lst:
                # [ABSTRACE] Snap the green to the enclosing GEMM's end (same
                # rule as the full figure) so no orange sliver survives after
                # the push, then clip to the zoom window.
                e = push_end_snap(cta, b, e, rec)
                if b > zoom_hi or e < zoom_lo:
                    continue
                _lo, _hi = max(b, zoom_lo), min(e, zoom_hi)
                # push as its own FULL-HEIGHT phase (same rule as the full
                # figure), so compute and push read as two consecutive stages.
                # No label: 1-3 us bars cannot hold text at a readable size,
                # and the text would cover the duration difference.
                ax2.add_patch(Rectangle((_lo, y - bar_h / 2),
                                        max(_hi - _lo, 0.02), bar_h,
                                        facecolor='#5fbf6a', edgecolor='#2c7a34',
                                        linewidth=0.4, zorder=3))
        ax2.set_xlim(zoom_lo, zoom_hi)
        ax2.set_ylim(rows_total - 0.2, -0.8)
        ax2.set_xlabel(f'absolute time in the same layer (us, window '
                       f'{zoom_lo:.0f}..{zoom_hi:.0f})', fontsize=26)
        ax2.set_ylabel('CTA row', fontsize=26)
        ax2.set_yticks(yt)
        ax2.set_yticklabels(yl, fontsize=max(7.0, min(14.0, 500.0 / rows_total)))
        ax2.grid(axis='x', color='#dddddd', linewidth=0.4, zorder=0)
        ax2.set_title(f'{os.path.basename(path)}  rank {rank}  epoch {ep}  '
                      f'ZOOM {zoom_lo:.0f}..{zoom_hi:.0f}us  {note}',
                      fontsize=30, pad=18)
        fig2.tight_layout()
        out2 = (os.path.splitext(out)[0] +
                f'.zoom_{int(round(zoom_lo))}_{int(round(zoom_hi))}us.png')
        fig2.savefig(out2, dpi=150, bbox_inches='tight')
        plt.close(fig2)
        print(f'wrote {out2}')

    # opening window: pull still running, first L1 GEMMs landing (no push yet)
    zoom_lo = float(os.environ.get('AT_ZOOM_LO', '0') or 0) or max(0.0, g0 - 5.0)
    zoom_us = float(os.environ.get('AT_ZOOM_US', '0') or 0) or 25.0
    render_zoom(zoom_lo, zoom_lo + zoom_us, '(opening: pull + L1 GEMM, NO push yet)')
    # L2 window: compute bar truncated at push start, then the green push
    render_zoom(z2_lo, z2_hi, '(L2 pass: orange=v{valid} compute, green=push)')


if __name__ == '__main__':
    main()
