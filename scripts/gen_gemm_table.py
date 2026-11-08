import json, sys
from collections import Counter, defaultdict
PATH=sys.argv[1] if len(sys.argv)>1 else 'runs/TR3.json'
RANK=int(sys.argv[2]) if len(sys.argv)>2 else 0
E,R,P=384,8,448; S=P//R
d=json.load(open(PATH)); r=d['eco_trace']['ranks'][RANK]
rows=[x for x in r['fine_rows'] if x[1]]
ep=Counter(x[6] for x in rows).most_common(1)[0][0]
E_=sorted([x for x in rows if x[6]==ep], key=lambda x:(x[0]&0xFFFFFFFF))
em=r['eco_meta']; phy=em[16+E*16:16+E*16+P]
u=lambda t: t & 0xFFFFFFFF

tb=defaultdict(list); blocks=[]
for x in E_:
    if x[1]==6: tb[(x[2],x[3],x[4])].append(x)
    elif x[1]==7:
        k=(x[2],x[3],x[4])
        if tb.get(k):
            b=tb[k].pop(0)
            blocks.append(dict(cta=x[2], expert=x[3], ls=x[3]-RANK*S, le=int(phy[x[3]]),
                               start=x[4], valid=x[5], t0=u(b[0]), t1=u(x[0]),
                               dur=u(x[0])-u(b[0])))
blocks.sort(key=lambda z:(z['t0'], z['cta']))
o=blocks[0]['t0']
for z in blocks: z['a']=z['t0']-o; z['b']=z['t1']-o
print(f'# {PATH} rank{RANK} epoch{ep}')
print(f'M 块数 {len(blocks)}  有 GEMM 的本地槽 {len(set(z["expert"] for z in blocks))}/56  CTA {len(set(z["cta"] for z in blocks))}')
print(f'首块开始 {blocks[0]["a"]} us  末块结束 {max(z["b"] for z in blocks)} us  墙钟跨度 {max(z["b"] for z in blocks)-blocks[0]["a"]} us')
ds=sorted(z['dur'] for z in blocks)
print(f'块耗时 min/中位/max = {ds[0]}/{ds[len(ds)//2]}/{ds[-1]} us')
busy=sum(ds); wall=max(z['b'] for z in blocks)-blocks[0]['a']
print(f'各 CTA 忙合计 {busy} us  平均并行度 {busy/wall:.2f} / {len(set(z["cta"] for z in blocks))} CTA')
print()
print('## B1 每个 CTA\n')
print('| CTA | 块数 | 首块开始 | 首块结束 | 末块结束 | 忙合计 | 空闲 | 首块槽 |')
print('|---:|---:|---:|---:|---:|---:|---:|---:|')
ct=defaultdict(list)
for z in blocks: ct[z['cta']].append(z)
for c in sorted(ct):
    v=sorted(ct[c],key=lambda z:z['a']); bs=sum(z['dur'] for z in v)
    print(f"| {c} | {len(v)} | {v[0]['a']} | {v[0]['b']} | {v[-1]['b']} | {bs} | {v[-1]['b']-v[0]['a']-bs} | {v[0]['ls']} |")
print()
print('## B2 按 local slot\n')
print('| local slot | 逻辑专家 | 块数 | 首块开始 | 末块结束 | 忙合计 |')
print('|---:|---:|---:|---:|---:|---:|')
ps=defaultdict(list)
for z in blocks: ps[z['expert']].append(z)
for s in sorted(ps, key=lambda s: min(z['a'] for z in ps[s])):
    v=sorted(ps[s],key=lambda z:z['a'])
    print(f"| {s-RANK*S} | **{int(phy[s])}** | {len(v)} | {v[0]['a']} | {v[-1]['b']} | {sum(z['dur'] for z in v)} |")
print()
print('## B3 全部 M 块（按开始时间）\n')
print('| # | 开始 | 结束 | 耗时 | CTA | local slot | 逻辑专家 | start row | valid |')
print('|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
for i,z in enumerate(blocks,1):
    print(f"| {i} | {z['a']} | {z['b']} | {z['dur']} | {z['cta']} | {z['ls']} | **{z['le']}** | {z['start']} | {z['valid']} |")
