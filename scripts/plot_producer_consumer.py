import json, sys
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

PATH = sys.argv[1]
RANK = int(sys.argv[2]) if len(sys.argv) > 2 else 0
OUT  = sys.argv[3] if len(sys.argv) > 3 else '/tmp/backlog.png'
d = json.load(open(PATH))
r = d['eco_trace']['ranks'][RANK]
rows = [x for x in r['fine_rows'] if x[1]]
ep = Counter(x[6] for x in rows).most_common(1)[0][0]
E = [x for x in rows if x[6] == ep]
u = lambda t: t & 0xFFFFFFFF
t0 = min(u(x[0]) for x in E if x[1] == 1)

mb = Counter(x[3] for x in E if x[1] == 6)                  # local slot -> blocks
prod = sorted((u(x[0]) - t0, x[3], mb.get(x[3], 0)) for x in E if x[1] == 2)
cons = sorted(u(x[0]) - t0 for x in E if x[1] == 7)
total = sum(mb.values())

# --- build the two staircase curves on a common event stream -------------
ev = sorted([(t, 1, n, s) for t, s, n in prod] + [(t, 0, 1, -1) for t in cons],
            key=lambda z: (z[0], z[1]))
ts, rel, comp, back = [], [], [], []
R = C = 0
for t, kind, n, s in ev:
    if kind == 1: R += n
    else: C += 1
    ts.append(t); rel.append(R); comp.append(C); back.append(R - C)

fig, (a1, a2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True,
                             gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.22})
fig.suptitle(f'Producer (pull CTA) vs Consumer (compute CTA) — '
             f'{PATH.split("/")[-1]} rank{RANK} epoch{ep}\n'
             f'{len(prod)} slots, {total} M blocks', fontsize=12)

# ---- row 1: producer ----
pt = [p[0] for p in prod]; pv = []; c = 0
for _, _, n in prod: c += n; pv.append(c)
a1.step(pt, pv, where='post', color='tab:blue', lw=1.6, label='released (cumulative)')
a1.scatter(pt, pv, s=10, color='tab:blue', zorder=3)
a1.axhline(total, color='tab:blue', ls=':', lw=.8, alpha=.6)
a1.annotate(f'{total} blocks released by {pt[-1]} us',
            xy=(pt[-1], total), xytext=(pt[-1]+12, total*0.55),
            fontsize=9, color='tab:blue',
            arrowprops=dict(arrowstyle='->', color='tab:blue', lw=.8))
a1.set_ylabel('producer: released M blocks\n(cumulative)')
a1.set_ylim(0, total*1.08)
a1.grid(alpha=.25)

# ---- row 2: consumer ----
a2.step(cons, range(1, len(cons)+1), where='post', color='tab:orange', lw=1.6,
        label='completed (cumulative)')
a2.scatter(cons, range(1, len(cons)+1), s=3, color='tab:orange', alpha=.55, zorder=3)
a2.set_ylabel('consumer: completed M blocks\n(cumulative)')
a2.set_ylim(0, total*1.08)
a2.grid(alpha=.25)
a2.set_xlabel('absolute time (us, relative to first slot_begin)')

ax2 = a2.twinx()
ax2.plot(ts, back, color='0.35', lw=1.2, ls='--', label='outstanding backlog')
ax2.fill_between(ts, back, color='0.5', alpha=.12)
ax2.set_ylabel('outstanding backlog (blocks)', color='0.35')
ax2.tick_params(axis='y', labelcolor='0.35')
ax2.set_ylim(0, max(back)*1.12)

peak_i = back.index(max(back))
ax2.annotate(f'peak backlog {max(back)} @ {ts[peak_i]} us',
             xy=(ts[peak_i], max(back)), xytext=(ts[peak_i]+60, max(back)*0.72),
             fontsize=9, color='0.25',
             arrowprops=dict(arrowstyle='->', color='0.35', lw=.8))
a2.axvline(pt[-1], color='tab:blue', ls=':', lw=.9, alpha=.7)
a2.text(pt[-1]+3, total*0.12, f'pull done\n{pt[-1]} us', fontsize=8, color='tab:blue')
a2.axvline(cons[-1], color='tab:orange', ls=':', lw=.9, alpha=.7)
a2.text(cons[-1]-45, total*0.5, f'gemm done\n{cons[-1]} us', fontsize=8, color='tab:orange')

for a in (a1, a2):
    a.set_xlim(0, max(cons[-1], pt[-1])*1.03)
    a.xaxis.set_major_locator(MaxNLocator(nbins=18))
plt.savefig(OUT, dpi=140, bbox_inches='tight')
print('wrote', OUT)
print(f'slots={len(prod)} blocks={total} pull_done={pt[-1]}us gemm_done={cons[-1]}us peak_backlog={max(back)}')
