"""Figure drawn from one_decode_layer_postrearr.perfetto.json.

Fixes vs the first attempt:
  * y labels were duplicating the text drawn inside the bar -> now the label is
    just "rank N", and ranks are grouped with separators (as in the figure);
  * the role name is drawn only inside its bar (the file's slice name).
"""
import json, sys, collections
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

SRC=sys.argv[1] if len(sys.argv)>1 else \
 '/home/admin/workspace/aop_lab/app_source/gcmoe_exp/t19_costfn/prof/prof_eco_role4/one_decode_layer_postrearr.perfetto.json'
OUT=sys.argv[2] if len(sys.argv)>2 else \
 '/home/admin/workspace/aop_lab/app_source/gcmoe_exp/t19_costfn/prof/prof_eco_role4/phases5_role4_from_perfetto.png'

d=json.load(open(SRC)); ev=d['traceEvents']
order={e['pid']: e['args']['sort_index'] for e in ev
       if e.get('ph')=='M' and e.get('name')=='process_sort_index'}
plab={e['pid']: e['args']['name'] for e in ev
      if e.get('ph')=='M' and e.get('name')=='process_name'}
S=collections.defaultdict(list)
for e in ev:
    if e.get('ph')=='X': S[e['pid']].append(e)
pids=sorted(S, key=lambda p: order[p])
print('tracks:',len(pids))

ROLE_COL={'role1':'#2f5fa8','role4':'#b0508f','role5':'#9a9a9a',
          'role2':'#f08a20','role3':'#4fa85f'}
def role_of(p):
    n=S[p][0]['name']
    for k in ('role 1','role 2','role 3','role 4','role 5'):
        if k in n: return k.replace(' ','')
    return 'role1'
def txt(e):
    # "<desc>   <dur> us"  ->  keep desc only; duration is drawn to scale
    return e['name'].rsplit('   ',1)[0].strip()

t0=min(e['ts'] for p in pids for e in S[p])
PX=max(e['ts']+e['dur'] for p in pids for e in S[p])-t0
CH=13*0.60*(PX/(46*72.0))

fig,ax=plt.subplots(figsize=(46,18.2)); H=0.34
ys=[];yl=[];row=0;rank_prev=None;rank_first=True
for p in pids:
    role=role_of(p); col=ROLE_COL[role]
    # rank index = sort_index // 10
    rank=order[p]//10
    if rank!=rank_prev:
        if rank_prev is not None: ax.axhline(row-0.5,color='#9a9a9a',lw=1.0,zorder=1)
        rank_prev=rank; rank_first=True
    for e in S[p]:
        a=e['ts']-t0; b=a+e['dur']; y=row; w=max(b-a,2)
        ax.add_patch(Rectangle((a,y-H),w,2*H,facecolor=col,
                               edgecolor='white',lw=0.8,zorder=2))
        t=txt(e)
        if w>len(t)*CH*1.35:
            ax.text((a+b)/2,y,t,ha='center',va='center',fontsize=13,
                    color='white' if role!='role5' else '#222',
                    fontweight='bold',zorder=6)
        else:
            ax.text(b+3,y,f'{t} {b-a:.0f}',ha='left',va='center',fontsize=10,
                    color='#111',fontweight='bold',zorder=6)
    ys.append(row)
    yl.append(f'rank {rank}' if rank_first else '')
    rank_first=False
    row+=1

ax.set_yticks(ys); ax.set_yticklabels(yl,fontsize=14)
ax.set_xlim(-PX*0.01,PX*1.10); ax.set_ylim(row-0.4,-0.9)
ax.set_xlabel('time within one MoE layer (us)',fontsize=22)
ax.set_title('GC-MoE ECO+EPLB BS256 — one decode layer AFTER EPLB rearrange — shared expert in gc_moe ROLE 4\n'
             'this figure is drawn directly from one_decode_layer_postrearr.perfetto.json',
             fontsize=19,pad=12)
ax.grid(axis='x',color='#e8e8e8',lw=0.7,zorder=0)
lab={}
for p in pids: lab.setdefault(role_of(p), txt(S[p][0]))
ax.legend(handles=[Patch(facecolor=ROLE_COL[k],label=v) for k,v in
                   sorted(lab.items(), key=lambda kv: order[min(
                       q for q in pids if role_of(q)==kv[0])])],
          loc='upper center',bbox_to_anchor=(0.5,-0.03),ncol=5,fontsize=15)
fig.tight_layout(); fig.savefig(OUT,dpi=110,bbox_inches='tight'); plt.close(fig)
print('wrote',OUT)
