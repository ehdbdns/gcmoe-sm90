import json, sys
from collections import defaultdict, Counter

PATH = sys.argv[1] if len(sys.argv) > 1 else 'runs/TR3.json'
RANK = int(sys.argv[2]) if len(sys.argv) > 2 else 0
E, R, P = 384, 8, 448
S = P // R
EMPTY = 0x7FFF

d = json.load(open(PATH))
r = d['eco_trace']['ranks'][RANK]
rows = [x for x in r['fine_rows'] if x[1]]
eps = Counter(x[6] for x in rows)
ep = eps.most_common(1)[0][0]
E_ = [x for x in rows if x[6] == ep]

# decode phy2log for this rank's local slots
em = r['eco_meta']
phy = em[16 + E*16: 16 + E*16 + P]

def ncopies(e):
    """active copies of logical expert e from eco_meta copies[]"""
    n = 0
    for k in range(32):
        w = em[16 + e*16 + k//2]
        v = (w & 0xFFFF) if k % 2 == 0 else (w >> 16) & 0xFFFF
        if v == EMPTY: break
        if v & 0x8000: continue      # passive
        n += 1
    return n

def t2u(t):
    return t & 0xFFFFFFFF

# ---- per-slot records -------------------------------------------------
begin = {}      # globalslot -> (t, cta, ncop)
end = {}
wait_b = {}
wait_r = {}
for x in E_:
    k, sid, a0, a1, a2 = x[1], x[2], x[3], x[4], x[5]
    if k == 1:  begin[sid] = (t2u(x[0]), a0, a2)
    elif k == 2: end[sid] = (t2u(x[0]), a0, a1)      # owned, lo
    elif k == 3:
        if a0 == 0: wait_b[sid] = t2u(x[0])
        else: wait_r[sid] = t2u(x[0])

t0 = min(v[0] for v in begin.values())
def rel(t): return t - t0

slots = []
for sid in sorted(begin):
    bt, cta, ncop = begin[sid]
    et, owned, lo = end.get(sid, (None, 0, 0))
    is_rep = ncop > 1
    start = wait_r.get(sid) if is_rep else bt
    wb = wait_b.get(sid)
    slots.append(dict(gs=sid, ls=sid - RANK*S, e=int(phy[sid]), ncop=ncop, rep=is_rep,
                      begin=rel(bt), wait_b=rel(wb) if wb else None,
                      grant=rel(start) if start else None,
                      start=rel(start) if start else None,
                      end=rel(et) if et else None, owned=owned))
slots.sort(key=lambda s: (s['start'], s['ls']))

print(f'# {PATH} rank{RANK} epoch{ep}  base t0={t0}  slots={len(slots)}')
print(f"# plan_end (eco_tl slot 7) = {r['eco_tl']['slots'][7]*1 if isinstance(r.get('eco_tl'),dict) and 'slots' in r.get('eco_tl',{}) else 'n/a'}")
print()
nrep = sum(1 for s in slots if s['rep'])
print(f'单副本槽 {len(slots)-nrep} / 副本槽 {nrep}   (ncopies>1)')
print()
hdr = f"| # | 槽 | **逻辑专家** | 类型 | begin | 等plan开始 | grant | **开始拉** | end | 拉取耗时 | token |"
print(hdr); print('|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|')
for i, s in enumerate(slots, 1):
    typ = '副本' if s['rep'] else '单副本'
    wb = '-' if s['wait_b'] is None else s['wait_b']
    gr = '-' if (not s['rep'] or s['grant'] is None) else s['grant']
    st = s['start'] if s['start'] is not None else '?'
    en = s['end'] if s['end'] is not None else '?'
    dur = (s['end'] - s['start']) if (s['end'] is not None and s['start'] is not None) else '?'
    print(f"| {i} | {s['ls']} | **{s['e']}** | {typ} | {s['begin']} | {wb} | {gr} | **{st}** | {en} | {dur} | {s['owned']} |")
