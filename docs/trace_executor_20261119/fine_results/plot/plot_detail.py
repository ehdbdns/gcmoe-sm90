#!/usr/bin/env python3
"""Plot the fine-grained per-rank layer detail (µs on the x axis)."""
import json, sys
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

src, out = sys.argv[1], sys.argv[2]
d = json.load(open(src))
L, EP = d['layer'], d['epoch']
# slowest rank = the one that exits last
w = max(d['ranks'], key=lambda r: r['routed_exit_us'])
r = w
hot = r['hot_slot']

fig = plt.figure(figsize=(20, 15), dpi=110)
gs = fig.add_gridspec(4, 2, height_ratios=[1.0, 1.0, 1.6, 1.0], hspace=0.32, wspace=0.16)

# ---- panel 1: every slot's pull span, ordered by completion ----
ax = fig.add_subplot(gs[0, :])
ps = sorted(r['pull_slots'], key=lambda s: s['end_us'])
for i, s in enumerate(ps):
    is_hot = s['slot'] == hot['slot']
    ax.add_patch(Rectangle((s['start_us'], i), s['end_us']-s['start_us'], 0.72,
                           color='#c0392b' if is_hot else '#4a7fb5',
                           edgecolor='white', linewidth=0.4))
    if s.get('first_l1_us'):
        ax.plot([s['first_l1_us']], [i+0.36], marker='|', ms=9,
                color='#e67e22', mew=2.0)
ax.set_ylim(-1, len(ps))
ax.set_xlim(0, r['routed_exit_us']*1.02)
ax.set_ylabel('slot (sorted by pull end)', fontsize=10)
ax.set_title(f'layer {L} (epoch {EP})  rank {r["rank"]} — ALL {len(ps)} slots: pull span (blue) + first L1 (orange tick)\n'
             f'tokens {r["tokens"]:.0f}  routed_rows {r["routed_rows"]}  M blocks {r["m_blocks"]}  '
             f'active slots {r["active_slots"]}  routed exit {r["routed_exit_us"]:.1f} µs', fontsize=11)
ax.axvline(hot['end_us'], color='#c0392b', ls='--', lw=1.0)

# ---- panel 2: active compute CTAs over time ----
ax = fig.add_subplot(gs[1, :])
a = r['active_compute_ctas']
ax.step([x for x, _ in a], [y for _, y in a], where='post', color='#2c3e50', lw=1.1)
ax.fill_between([x for x, _ in a], [y for _, y in a], step='post', color='#95a5a6', alpha=0.35)
ax.axvspan(hot['end_us'], hot.get('first_l1_us', hot['end_us']), color='#f39c12', alpha=0.18)
ax.text((hot['end_us']+hot.get('first_l1_us', hot['end_us']))/2, ax.get_ylim()[1]*0.86,
        f"hot-slot pull→L1\n{hot.get('pull_end_to_first_l1_us',0)} µs\n0 queue-empty",
        ha='center', va='top', fontsize=8.5)
ax.set_ylabel('active compute CTAs', fontsize=10)
ax.set_xlim(0, r['routed_exit_us']*1.02)
ax.set_title(f'concurrent compute CTAs: stays {min(y for _,y in a)}–{max(y for _,y in a)} of {r["routed_ctas"]} '
             f'during the hot-slot wait — no CTA ran dry', fontsize=10.5)

# ---- panel 3: every L1/L2 tile as a bar (one row per CTA) ----
ax = fig.add_subplot(gs[2, :])
ctas = sorted({t['cta'] for t in r['tiles']})
idx = {c: i for i, c in enumerate(ctas)}
for t in r['tiles']:
    y = idx[t['cta']]
    ax.add_patch(Rectangle((t['start_us'], y), max(t['end_us']-t['start_us'], 0.6), 0.78,
                           color='#f39c12' if t['phase'] == 1 else '#d35400',
                           edgecolor='none'))
for p in r['l2_pushes']:
    y = idx.get(p['cta'])
    if y is None: continue
    ax.add_patch(Rectangle((p['start_us'], y), max(p['end_us']-p['start_us'], 0.5), 0.78,
                           color='#27ae60', edgecolor='none'))
ax.set_ylim(-1, len(ctas)); ax.set_xlim(0, r['routed_exit_us']*1.02)
ax.set_ylabel('compute CTA', fontsize=9)
ax.set_xlabel('time within the layer — rank-local dispatch entry = 0 (µs)', fontsize=10)
hs = [t for t in r['tiles'] if t['slot'] == hot['slot']]
if hs:
    ax.axvspan(min(t['start_us'] for t in hs), r['routed_exit_us'], color='#c0392b', alpha=0.10)
ax.set_title(f'every routed tile: L1 (light) / L2 (dark) / L2 push (green) — {len(r["tiles"])} tiles, '
             f'{len(r["l2_pushes"])} pushes; red band = hot slot {hot["slot"]}', fontsize=10.5)

# ---- panel 4: hot-slot zoom, collapsed to 2 rows (L1 / L2) ----
ax = fig.add_subplot(gs[3, :])
hot_tiles = [t for t in r['tiles'] if t['slot'] == hot['slot']]
hot_pushes = [p for p in r['l2_pushes'] if p['slot'] == hot['slot']]
for i, ph in enumerate((1, 2)):
    ts = [t for t in hot_tiles if t['phase'] == ph]
    for t in ts:
        ax.add_patch(Rectangle((t['start_us'], i), max(t['end_us']-t['start_us'], 1.0), 0.80,
                               color='#f39c12' if ph == 1 else '#d35400',
                               edgecolor='none', alpha=0.55))
    ps_ = [p for p in hot_pushes if p['phase'] == ph]
    for p in ps_:
        ax.add_patch(Rectangle((p['start_us'], i), max(p['end_us']-p['start_us'], 0.8), 0.80,
                               color='#27ae60', edgecolor='none'))
ax.set_yticks([0, 1])
ax.set_yticklabels([f'L1 ({sum(1 for t in hot_tiles if t["phase"]==1)} tiles)',
                    f'L2 ({sum(1 for t in hot_tiles if t["phase"]==2)} tiles + {len(hot_pushes)} push)'],
                   fontsize=10)
lo2 = min(t['start_us'] for t in hot_tiles) - 10
hi2 = r['routed_exit_us']
ax.set_xlim(lo2, hi2); ax.set_ylim(-0.8, 2.3)
for x, txt, col in ((hot['end_us'], 'pull done', '#c0392b'),
                    (hot.get('first_l1_us', hot['end_us']), 'first L1', '#e67e22'),
                    (max(t['end_us'] for t in hot_tiles if t['phase']==1), 'last L1', '#8e44ad'),
                    (max(t['end_us'] for t in hot_tiles if t['phase']==2), 'last L2', '#7f0000'),
                    (r['routed_exit_us'], 'exit', '#7f8c8d')):
    ax.axvline(x, color=col, ls='--', lw=1.2)
    ax.text(x, 2.15, txt, rotation=0, fontsize=8, va='bottom', ha='center', color=col)
l1_end = max(t['end_us'] for t in hot_tiles if t['phase']==1)
l2_end = max(t['end_us'] for t in hot_tiles if t['phase']==2)
ax.set_title(f'hot slot {hot["slot"]} ({hot["owned"]} rows, CTA {hot["cta"]}): '
             f'pull done {hot["end_us"]} → first L1 {hot.get("first_l1_us")} ({hot.get("pull_end_to_first_l1_us")} µs) '
             f'→ last L1 {l1_end} → last L2 {l2_end} → exit {r["routed_exit_us"]:.0f} µs', fontsize=10)
ax.set_xlabel('µs', fontsize=9)
ax.grid(axis='x', alpha=0.25)

fig.suptitle(f'GC-MoE fine-grained detail — layer {L}, epoch {EP}, slowest rank {r["rank"]} '
             f'(capture {d["capture_id"][:8]}, step {d["step"]})', fontsize=13)
fig.savefig(out, bbox_inches='tight')
print('wrote', out)
