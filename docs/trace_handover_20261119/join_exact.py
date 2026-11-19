"""Join trace / pull_tl / ir_step using the KERNEL-WRITTEN epoch."""
import json, sys, os
ROOT='/home/admin/workspace/aop_lab/app_source/gcmoe_exp/t19_costfn'
runp, tracedir = sys.argv[1], sys.argv[2]
d=json.load(open(runp))
t=d['ir_step']['tables']; L=t['layers']; S=t['steps']
eps=t.get('epochs')
print('tables has kernel-written epochs:', eps is not None)
if not eps: sys.exit('no rows_epoch -> cannot join exactly')
# epoch -> (layer, step)
m={}
for li in range(L):
    for si in range(S):
        e=eps[li*S+si][0]
        if e: m[e]=(li,si)
print('epochs recorded in table:', len(m))
print('  range', min(m), '..', max(m))
pulls=[sorted([x for x in r['pull_rows'] if x and x[0]], key=lambda x:x[10])
       for r in d['eco_trace']['ranks']]
ring=[x[0] for x in pulls[0]]
print('pull_tl ring   :', ring[0], '..', ring[-1], 'n', len(ring))
hit=[e for e in ring if e in m]
print('ring epochs present in ir_step table:', len(hit), '/', len(ring))
print('  first/last:', hit[0], hit[-1] if hit else None)
print()
if hit:
    print('VERDICT: EXACT JOIN POSSIBLE')
    print('  e.g. epoch', hit[0], '-> layer', m[hit[0]][0], 'step', m[hit[0]][1])
else:
    print('VERDICT: no overlap')
