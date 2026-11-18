"""Keep the LAST N gc_moe kernel rows of one rank's torch profile.

Each forward call is 5 consecutive rows:
    dispatch(24x256) -> shared(71x256) -> routed(71x384) -> plan(1x512) -> combine(288x384)
`pull_tl` is a 256-deep ring, so only the last 256 calls carry kernel stamps;
this snapshot keeps 3000 rows (600 calls) so the tail can be located safely.

usage: extract_tail.py <rank> [n_rows]
"""
import json, os, sys
import ijson

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, '..', '..', 'prof', 'final_v2', 'traces'))
OUT = '/tmp/rt'
R = int(sys.argv[1])
N = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
def role(g,b):
    if g==24: return 'dispatch'
    if g==71 and b==256: return 'shared'
    if g==71 and b==384: return 'routed'
    if g==1: return 'plan'
    if g==288: return 'combine'
    return None
buf=[]; row=0
with open(f'{SRC}/rank{R}.json','rb') as fh:
    for e in ijson.items(fh,'traceEvents.item',use_float=True):
        if e.get('cat')!='kernel' or 'gc_moe' not in e.get('name',''): continue
        a=e.get('args') or {}
        g=tuple(a.get('grid') or ()); b=tuple(a.get('block') or ())
        if not g: continue
        k=role(g[0], b[0] if b else 0)
        buf.append({'row':row,'role':k,'grid':g[0],'block':b[0] if b else 0,
                    'ts':e['ts'],'dur':e.get('dur',0)})
        if len(buf)>N: buf.pop(0)
        row+=1
os.makedirs(OUT, exist_ok=True)
json.dump({'total': row, 'rows': buf}, open(os.path.join(OUT, f'tail{R}.json'), 'w'))
print(f'rank{R}: total {row} calls {row//5} kept {len(buf)}')
