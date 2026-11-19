#!/usr/bin/env python3
"""PNG of the single-layer timeline.

Three swimlanes per rank, because three kernels genuinely run in PARALLEL:
  * dispatch lane : count exchange + pull token   (role 1)
  * shared lane   : shared L1+L2 + ECO plan       (role 4 + role 5)
  * compute lane  : wait token + GEMM + combine   (role 2 + role 3)

Drawing all seven phases on ONE row per rank would hide that overlap and make
the labels collide.
"""
import argparse, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

LANES = [
    (['count exchange', 'pull token'], 'dispatch (role 1)'),
    (['shared L1+L2', 'ECO plan'], 'shared (role 4) + plan (role 5)'),
    (['wait token', 'GEMM', 'combine'], 'compute (role 2) + combine (role 3)'),
]
COLOR = {'count exchange': '#2f5fa8', 'pull token': '#7fa8d8',
         'shared L1+L2': '#b0508f', 'ECO plan': '#8e6bbf',
         'wait token': '#cfcfcf', 'GEMM': '#f08a20', 'combine': '#4fa85f'}
ORDER = ['count exchange', 'pull token', 'shared L1+L2', 'ECO plan',
         'wait token', 'GEMM', 'combine']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--segments', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--xmax', type=float, default=None,
                    help='zoom: truncate the x axis at this many us')
    a = ap.parse_args()
    d = json.load(open(a.segments))
    seg = {int(k): v for k, v in d['segments'].items()}
    lo = min(min(s[0] for s in seg[r].values()) for r in seg)
    hi = max(max(s[1] for s in seg[r].values()) for r in seg)
    span = hi - lo
    zoom = a.xmax is not None
    fig, ax = plt.subplots(figsize=(19 if not zoom else 13, 10.5),
                           dpi=150 if not zoom else 170)
    H = 0.26
    for r in range(8):
        base = (7 - r) * 3.0
        for li, (names, lane_label) in enumerate(LANES):
            y = base + (2 - li)
            for name in names:
                s, e = seg[r][name]
                w = max(e - s, 0.05)
                ax.add_patch(plt.Rectangle((s, y - H / 2), w, H,
                                           facecolor=COLOR[name], edgecolor='white',
                                           linewidth=0.6, zorder=3))
                # label only when the bar can hold it; the narrow shared/plan
                # bars are identified by the legend instead of by text
                if w > span * 0.09:
                    ax.text(s + w / 2, y, name, ha='center', va='center',
                            fontsize=8.2,
                            color='#333' if name in ('wait token', 'pull token')
                            else 'white', zorder=4)
        ax.text(lo - span * 0.015, base + 1, f'rank {r}', ha='right', va='center',
                fontsize=10.5, fontweight='bold')
        ax.axhline(base - 0.5, color='#eeeeee', linewidth=0.8, zorder=1)
    ax.set_yticks([])
    ax.set_ylim(-0.9, 8 * 3.0 - 0.4)
    xhi = a.xmax if a.xmax else hi
    ax.set_xlim(lo - (xhi - lo) * 0.09, xhi + (xhi - lo) * 0.01)
    ax.set_xlabel('time within one decode layer (µs; same origin for all ranks)',
                  fontsize=11)
    ax.grid(axis='x', color='#dddddd', linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    zoom_txt = f', zoom 0–{a.xmax:g} µs' if a.xmax else ''
    ax.set_title(f'{d["label"]} — one MoE layer, 8 ranks × 3 concurrent lanes  '
                 f'(call {d["call"]}, rows {d["row_range"][0]}–{d["row_range"][1]}{zoom_txt})',
                 fontsize=13)
    ax.legend(handles=[Patch(facecolor=COLOR[n], label=n) for n in ORDER],
              loc='upper center', bbox_to_anchor=(0.5, -0.075), ncol=7,
              frameon=False, fontsize=9.5)
    fig.subplots_adjust(left=0.06, right=0.995, top=0.95, bottom=0.115)
    fig.savefig(a.out)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
