import json, sys
from collections import Counter

PATH = sys.argv[1] if len(sys.argv) > 1 else 'runs/X16_128_eco.json'
RANK = int(sys.argv[2]) if len(sys.argv) > 2 else 0
d = json.load(open(PATH))
r = d['eco_trace']['ranks'][RANK]
rows = [x for x in r['fine_rows'] if x[1]]
ep = Counter(x[6] for x in rows).most_common(1)[0][0]
E = [x for x in rows if x[6] == ep]
u = lambda t: t & 0xFFFFFFFF
t0 = min(u(x[0]) for x in E if x[1] == 1)

mb = Counter(x[3] for x in E if x[1] == 6)      # local slot -> M blocks
prod = sorted((u(x[0]) - t0, x[3], mb.get(x[3], 0), x[4]) for x in E if x[1] == 2)
cons = sorted(u(x[0]) - t0 for x in E if x[1] == 7)

print(f'# {PATH.split("/")[-1]} rank{RANK} epoch{ep}')
print(f'# slots={len(prod)}  M_blocks={sum(mb.values())}  producers={len(prod)}  consumers={len(cons)}')
print(f'# producer = pull CTA finishing ONE slot; y2 = cumulative M blocks it released')
print(f'# consumer = compute CTA finishing ONE M block; y2 = cumulative completed')
print()

# ---- producer trace (every finished slot) ------------------------------
print('=== 生产者行：每 pull 完一个槽就 +该槽的 M 块数 ===')
print(f'{"t_us":>6}{"slot":>6}{"owned":>7}{"mblk":>6}{"cum_released":>14}')
cum = 0
for t, slot, n, owned in prod:
    cum += n
    print(f'{t:>6}{slot:>6}{owned:>7}{n:>6}{cum:>14}')
print(f'\n合计释放 {cum} 块')

# ---- consumer trace ----------------------------------------------------
print('\n=== 消费者行：每算完一个 M 块就 +1 ===')
print(f'{"t_us":>6}{"cum_done":>10}')
step = max(1, len(cons)//40)
for i in range(0, len(cons), step):
    print(f'{cons[i]:>6}{i+1:>10}')
print(f'{cons[-1]:>6}{len(cons):>10}   <- last')
print(f'\n合计完成 {len(cons)} 块')

# ---- backlog -----------------------------------------------------------
print('\n=== 未完成积压（released - completed）===')
print(f'{"t_us":>6}{"released":>10}{"completed":>11}{"backlog":>9}')
ev = [(t, 'p', n) for t, _, n, _ in prod] + [(t, 'c', 1) for t in cons]
ev.sort(key=lambda z: (z[0], z[1] == 'c'))
rel = comp = 0
marks = []
for t, kind, n in ev:
    if kind == 'p': rel += n
    else: comp += 1
    marks.append((t, rel, comp, rel - comp))
sel = [m for i, m in enumerate(marks) if i % max(1, len(marks)//40) == 0] + [marks[-1]]
for t, rel, comp, bl in sel:
    bar = '#' * min(80, bl // 8)
    print(f'{t:>6}{rel:>10}{comp:>10}{bl:>9}  {bar}')
print(f'\n峰值积压 = {max(m[3] for m in marks)} 块')
print(f'积压归零时刻 = {next((m[0] for m in marks if m[3] == 0), "未归零")} us')
