import json, sys
from collections import Counter
PATH=sys.argv[1]; RANK=int(sys.argv[2]); EP=int(sys.argv[3]) if len(sys.argv)>3 else None
E,R,P=384,8,448; S=P//R; EMPTY=0x7FFF
d=json.load(open(PATH)); r=d['eco_trace']['ranks'][RANK]
rows=[x for x in r['fine_rows'] if x[1]]
ep = EP if EP else Counter(x[6] for x in rows).most_common(1)[0][0]
E_=[x for x in rows if x[6]==ep]
u=lambda t:t&0xFFFFFFFF
em=r['eco_meta']; phy=em[16+E*16:16+E*16+P]
def ncopies(e):
    n=0
    for k in range(32):
        w=em[16+e*16+k//2]; v=(w&0xFFFF) if k%2==0 else (w>>16)&0xFFFF
        if v==EMPTY: break
        if v&0x8000: continue
        n+=1
    return n
beg={};end={};wb={};wr={}
for x in E_:
    gs=x[2]
    if x[1]==1: beg[gs]=u(x[0])
    elif x[1]==2: end[gs]=(u(x[0]),x[4])
    elif x[1]==3: (wb if x[3]==0 else wr)[gs]=u(x[0])
# 基准：最早 pull begin（≈ push_end，因为 pull 紧跟 push）
base=min(beg.values())
recs=[]
for gs in beg:
    nc=ncopies(int(phy[gs])); rep=nc>1
    st=wr.get(gs,beg[gs])
    v=end.get(gs)
    recs.append(dict(gs=gs,ls=gs-RANK*S,e=int(phy[gs]),nc=nc,rep=rep,
        begin=beg[gs]-base, wb=(wb[gs]-base) if gs in wb else None,
        st=st-base, end=(v[0]-base) if v else None, owned=v[1] if v else 0))
recs.sort(key=lambda z:(z['st'],z['ls']))
print(f'### 56 槽拉取表 — {PATH.split("/")[-1]} / rank{RANK} / epoch {ep}')
print(f'（所有时刻相对「首个 pull begin」= 0 µs）')
print()
print(f'pull 窗口 0 → {max(z["end"] for z in recs)} µs；'
      f'单副本最早 {min(z["st"] for z in recs if not z["rep"])} µs，'
      f'副本最早 {min(z["st"] for z in recs if z["rep"])} µs')
print()
print('| # | 槽 | **逻辑专家** | 类型 | begin | 等plan开始 | grant | **开始拉** | end | 拉取耗时 | token |')
print('|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|')
for i,z in enumerate(recs,1):
    typ=f"副本(nc={z['nc']})" if z['rep'] else '单副本'
    wbx='-' if z['wb'] is None else z['wb']
    gr='-' if not z['rep'] else z['st']
    print(f"| {i} | {z['ls']} | **{z['e']}** | {typ} | {z['begin']} | {wbx} | {gr} | **{z['st']}** | {z['end']} | {z['end']-z['st']} | {z['owned']} |")
single=[z for z in recs if not z['rep']]; rep=[z for z in recs if z['rep']]
print()
print('| 类别 | 槽数 | 最早开始拉 | 最晚拉完 | 有 token | token 合计 |')
print('|---|---:|---|---|---:|---:|')
print(f"| 单副本槽 | {len(single)} | {min(z['st'] for z in single)} µs | {max(z['end'] for z in single)} µs | {sum(1 for z in single if z['owned']>0)} | {sum(z['owned'] for z in single)} |")
print(f"| 副本槽 | {len(rep)} | {min(z['st'] for z in rep)} µs | {max(z['end'] for z in rep)} µs | {sum(1 for z in rep if z['owned']>0)} | {sum(z['owned'] for z in rep)} |")
print(f"| **合计** | 56 | — | {max(z['end'] for z in recs)} µs | {sum(1 for z in recs if z['owned']>0)} | {sum(z['owned'] for z in recs)} |")
