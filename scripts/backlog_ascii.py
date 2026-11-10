import json, sys
from collections import Counter
PATH=sys.argv[1]; RANK=int(sys.argv[2]) if len(sys.argv)>2 else 0
d=json.load(open(PATH)); r=d['eco_trace']['ranks'][RANK]
rows=[x for x in r['fine_rows'] if x[1]]
ep=Counter(x[6] for x in rows).most_common(1)[0][0]
E=[x for x in rows if x[6]==ep]; u=lambda t:t&0xFFFFFFFF
t0=min(u(x[0]) for x in E if x[1]==1)
mb=Counter(x[3] for x in E if x[1]==6)
prod=[(u(x[0])-t0, x[3], mb.get(x[3],0)) for x in E if x[1]==2]
cons=[u(x[0])-t0 for x in E if x[1]==7]
ev=sorted([(t,'P',n,slot) for t,slot,n in prod]+[(t,'C',1,None) for t in cons],
          key=lambda z:(z[0], z[1]=='C'))
print(f'# {PATH.split("/")[-1]} rank{RANK} epoch{ep}   slots={len(prod)}  M_blocks={sum(mb.values())}')
print()
W=92
print(f'{"t_us":>5} | {"生产者(累计释放块)":<{W//2}} | {"消费者(累计完成块)":<{W//2}} | 积压')
print('-'*(7+W+W+8))
rel=comp=0; last_p=last_c=0; ticks=[]
for t,k,n,slot in ev:
    if k=='P': rel+=n
    else: comp+=1
    ticks.append((t,rel,comp,rel-comp))
# 每 ~1/45 抽一行 + 首末
N=45; stride=max(1,len(ticks)//N)
sel=[ticks[i] for i in range(0,len(ticks),stride)]
if sel[-1]!=ticks[-1]: sel.append(ticks[-1])
peak=max(x[3] for x in ticks)
for t,rel_,comp_,bl in sel:
    bp='#'*int(W//2*rel_/max(1,rel)) if rel else ''
    bc='#'*int(W//2*comp_/max(1,rel)) if comp_ else ''
    print(f'{t:>5} | {bp:<{W//2}} | {bc:<{W//2}} | {bl}')
print()
print(f'峰值积压 {peak} 块；{ticks[-1][0]} us 时归零（最后一块算完）')
print(f'生产者全部释放完毕: {max(p[0] for p in prod)} us')
print(f'消费者全部完成: {cons[-1]} us')
