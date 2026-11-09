"""Producer/consumer timeline with the role-6 (SM-reuse) helper highlighted.

Row 1  producer: every finished slot adds that slot's M-block count.
Row 2  consumer: every finished M block adds 1, split into
         role 2 (block id < 1000)  -- the original compute partition
         role 6 (block id >= 1000) -- the helper on the SM dispatch vacated
       plus the outstanding backlog on the right axis.

role 6 is a graph dependency of role 1, so it starts once dispatch has
published `header[9]` (all local slots pulled).  That moment is marked.

usage: plot_smreuse.py <run.json> [rank] [out.png]
"""
import json, sys
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

PATH = sys.argv[1]
RANK = int(sys.argv[2]) if len(sys.argv) > 2 else 0
OUT = sys.argv[3] if len(sys.argv) > 3 else '/tmp/smreuse.png'

d = json.load(open(PATH))
r = d['eco_trace']['ranks'][RANK]
rows = [x for x in r['fine_rows'] if x[1]]
ep = Counter(x[6] for x in rows).most_common(1)[0][0]
E = [x for x in rows if x[6] == ep]
u = lambda t: t & 0xFFFFFFFF
t0 = min(u(x[0]) for x in E if x[1] == 1)

mb = Counter(x[3] for x in E if x[1] == 6)                 # per local slot
prod = sorted((u(x[0]) - t0, x[3], mb.get(x[3], 0)) for x in E if x[1] == 2)
cons = sorted((u(x[0]) - t0, x[2] < 1000) for x in E if x[1] == 7)
c2 = [t for t, is2 in cons if is2]
c6 = [t for t, is2 in cons if not is2]
total = sum(mb.values())

# role 6 entry = first block it finished (it has no start record)
r6_first = min(c6) if c6 else None
# pull done = last slot_end
pull_done = max(t for t, _, _ in prod)

fig, (a1, a2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True,
                             gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.22})
bs = d.get('seqs', '?')
fig.suptitle(f'SM-reuse: producer (pull) vs consumer (compute) — {PATH.split("/")[-1]} rank{RANK}\n'
             f'BS={bs}  slots={len(prod)}  M blocks={total}  '
             f'role2={len(c2)} blocks, role6={len(c6)} blocks', fontsize=12)

# ---- row 1: producer ----
pt, pv, c = [], [], 0
for t, slot, n in prod:
    c += n; pt.append(t); pv.append(c)
a1.step(pt, pv, where='post', color='tab:blue', lw=1.6)
a1.scatter(pt, pv, s=9, color='tab:blue', zorder=3)
a1.axhline(total, color='tab:blue', ls=':', lw=.8, alpha=.55)
a1.set_ylabel('producer: released M blocks\n(cumulative)')
a1.set_ylim(0, total * 1.08)
a1.grid(alpha=.25)

# ---- row 2: consumers split by role ----
a2.step(c2, range(1, len(c2) + 1), where='post', color='tab:orange', lw=1.6,
        label=f'role 2 (compute partition, {len(c2)} blocks)')
a2.step(c6, range(1, len(c6) + 1), where='post', color='tab:red', lw=1.8,
        label=f'role 6 (reused dispatch SMs, {len(c6)} blocks)')
a2.set_ylabel('consumer: completed M blocks\n(cumulative, per role)')
a2.set_ylim(0, max(len(c2), len(c6)) * 1.10)
a2.grid(alpha=.25)
a2.set_xlabel('absolute time (µs, relative to first slot_begin)')

# backlog
ev = sorted([(t, 1, n) for t, _, n in prod] + [(t, 0, 1) for t, _ in cons],
            key=lambda z: (z[0], z[1]))
ts, back, R, C = [], [], 0, 0
for t, kind, n in ev:
    if kind: R += n
    else: C += 1
    ts.append(t); back.append(R - C)
ax2 = a2.twinx()
ax2.plot(ts, back, color='0.35', lw=1.1, ls='--')
ax2.fill_between(ts, back, color='0.5', alpha=.10)
ax2.set_ylabel('outstanding backlog (blocks)', color='0.35')
ax2.tick_params(axis='y', labelcolor='0.35')
ax2.set_ylim(0, max(back) * 1.12)

# ---- mark when role 6 joins ----
for a in (a1, a2):
    a.axvline(pull_done, color='tab:blue', ls=':', lw=1.0, alpha=.8)
    if r6_first is not None:
        a.axvline(r6_first, color='tab:red', ls='--', lw=1.4, alpha=.9)
a2.axvspan(pull_done, max(cons[-1][0], r6_first or 0),
           color='tab:red', alpha=.06)
a1.text(pull_done + 2, total * .08,
        f'pull done {pull_done} µs\n(dispatch releases its 6 SM)',
        fontsize=8.5, color='tab:blue')
if r6_first is not None:
    a1.text(r6_first + 2, total * .42,
            f'>>> role 6 joins at {r6_first} µs\n'
            f'     (+{(r6_first - pull_done)/max(1e-9,pull_done)*100:.0f}% after pull)',
            fontsize=9, color='tab:red', fontweight='bold')
a2.legend(loc='upper left', fontsize=8.5)

for a in (a1, a2):
    a.set_xlim(0, max(cons[-1][0], pull_done) * 1.04)
    a.xaxis.set_major_locator(MaxNLocator(nbins=18))
plt.savefig(OUT, dpi=140, bbox_inches='tight')
plt.close(fig)
print(f'{OUT}  pull_done={pull_done} role6_first={r6_first} '
      f'role2={len(c2)} role6={len(c6)} total={total}')
