#!/usr/bin/env python3
"""Tail decomposition for the slowest rank: pull → L1 → gap → L2 → exit."""
import json, sys
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

src, out = sys.argv[1], sys.argv[2]
d = json.load(open(src))
w = max(d['ranks'], key=lambda r: r['routed_exit_us'])
hot = w['hot_slot']
ht = [t for t in w['tiles'] if t['slot'] == hot['slot']]
l1 = [t for t in ht if t['phase'] == 1]; l2 = [t for t in ht if t['phase'] == 2]
l1s, l1e = min(t['start_us'] for t in l1), max(t['end_us'] for t in l1)
l2s, l2e = min(t['start_us'] for t in l2), max(t['end_us'] for t in l2)
other_l1 = max(t['end_us'] for t in w['tiles'] if t['phase'] == 1 and t['slot'] != hot['slot'])
phases = [
    ('pull (hot slot)', hot['start_us'], hot['end_us'], '#4a7fb5'),
    ('wait\nfor L1', hot['end_us'], l1s, '#95a5a6'),
    (f'L1 x{len(l1)}', l1s, l1e, '#f39c12'),
    ('GAP\nL1→L2', l1e, l2s, '#c0392b'),
    (f'L2 x{len(l2)}', l2s, l2e, '#d35400'),
    ('push', l2e, w['routed_exit_us'], '#27ae60'),
]
fig, ax = plt.subplots(figsize=(20, 6.5), dpi=110)
for i, (name, a, b, c) in enumerate(phases):
    ax.add_patch(Rectangle((a, -0.4), max(b-a, 0.5), 0.8, color=c, alpha=0.9))
    mid = (a+b)/2
    ax.text(mid, 0.0, f'{name}\n{b-a:.0f} µs', ha='center', va='center',
            fontsize=9.5, color='white', fontweight='bold')
ax.axhline(-0.55, color='#ddd', lw=0.8)
# 其它槽最后 L1 的时刻
ax.axvline(other_l1, color='#8e44ad', ls=':', lw=1.6)
ax.text(other_l1, 0.62, f'last L1 of any OTHER slot\n{other_l1:.0f} µs', color='#8e44ad',
        fontsize=8.5, ha='center', va='bottom')
ax.set_xlim(hot['start_us']-15, w['routed_exit_us']+30)
ax.set_ylim(-0.75, 1.15)
ax.set_yticks([])
ax.set_xlabel('µs  (rank-local dispatch entry = 0)', fontsize=10)
ax.grid(axis='x', alpha=0.25)
segs = [('pull→L1', l1s-hot['end_us']), ('L1', l1e-l1s), ('GAP', l2s-l1e),
        ('L2', l2e-l2s), ('L2→exit', w['routed_exit_us']-l2e)]
ax.set_title(f'layer {d["layer"]}  epoch {d["epoch"]}  slowest rank {w["rank"]}  '
             f'hot slot {hot["slot"]} ({hot["owned"]} rows) — tail decomposition\n'
             + '   '.join(f'{n} {v:.0f} µs' for n, v in segs)
             + f'   |   total from pull done: {w["routed_exit_us"]-hot["end_us"]:.0f} µs', fontsize=11.5)
fig.tight_layout()
fig.savefig(out, bbox_inches='tight')
print('wrote', out)
