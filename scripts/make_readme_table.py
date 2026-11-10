"""Build the README main table from the V* run jsons.

Columns: ms/step, vs-natives, IR (new definition), mean M blocks (cross-rank
sum), mean active experts (cross-rank sum), M-block IR.
"""
import json, os

ARMS = (('vanilla', '**原生 gcmoe（无副本）**'),
        ('random', '+ random + rr'),
        ('rr', '+ eplb + rr'),
        ('eco', '+ eplb + eco'),
        ('eco_smr', '+ eplb + eco + **SM-reuse**'))
BS = ((4, 32), (8, 64), (16, 128), (32, 256))
BOLD = 'eco_smr'


def load(spr, arm):
    p = f'runs/V{spr}_{arm}.json'
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    s = d.get('ir_step') or {}
    return dict(ms=d['ms_per_step_median'],
                ir=s.get('ir_step_layer_mean', -1.0),
                mb_ir=s.get('mb_ir_mean', -1.0),
                mb=s.get('mb_total_mean', -1.0),
                ae=s.get('ae_total_mean', -1.0))


def f(x, nd=2, bold=False):
    t = f'{x:.{nd}f}'
    return f'**{t}**' if bold else t


def main():
    print('| BS | 臂 | ms/step | vs 原生 | **IR（新口径）** | '
          'M块数平均 | 激活专家数平均 | **M块 IR** |')
    print('|---:|---|---:|---:|---:|---:|---:|---:|')
    for spr, bs in BS:
        base = load(spr, 'vanilla')
        for i, (arm, name) in enumerate(ARMS):
            r = load(spr, arm)
            head = f'**{bs}**' if i == 0 else ''
            if r is None:
                print(f'| {head} | {name} | (缺) | | | | | |')
                continue
            delta = '—' if (arm == 'vanilla' or base is None) else \
                f'{(r["ms"] / base["ms"] - 1) * 100:+.2f}%'
            b = arm == BOLD
            print(f'| {head} | {name} | {f(r["ms"], 2, b)} | {delta} | '
                  f'{f(r["ir"], 4, b)} | {f(r["mb"], 1, b)} | '
                  f'{f(r["ae"], 1, b)} | {f(r["mb_ir"], 4, b)} |')


if __name__ == '__main__':
    main()
