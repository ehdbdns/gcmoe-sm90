"""Regenerate the README result section from runs/V*_*.json + docs/plots/ir20.

Writes three blocks between HTML markers so the file stays editable by hand:

  <!-- IR20:TABLE:BEGIN -->   ... 6-column main table
  <!-- IR20:TABLE:END -->
  <!-- IR20:PLOTS:BEGIN -->   ... the 20 timelines
  <!-- IR20:PLOTS:END -->

usage: update_readme.py <README.md> [--check]
"""
import json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARMS = (('vanilla', '**原生 gcmoe（无副本）**'),
        ('random', '+ random + rr'),
        ('rr', '+ eplb + rr'),
        ('eco', '+ eplb + eco'),
        ('eco_smr', '+ eplb + eco + **SM-reuse**'))
BS = ((4, 32), (8, 64), (16, 128), (32, 256))
BOLD = 'eco_smr'


def load(spr, arm):
    p = os.path.join(ROOT, f'runs/V{spr}_{arm}.json')
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    s = d.get('ir_step') or {}
    if not s.get('mb_total_mean'):
        return None
    return dict(ms=d['ms_per_step_median'],
                ir=s.get('ir_step_layer_mean', -1.0),
                mb_ir=s.get('mb_ir_mean', -1.0),
                mb=s.get('mb_total_mean', -1.0),
                ae=s.get('ae_total_mean', -1.0))


def fmt(x, nd=2, bold=False):
    t = f'{x:.{nd}f}'
    return f'**{t}**' if bold else t


def table():
    out = ['| BS | 臂 | ms/step | vs 原生 | **IR（新口径）** | M块数平均 | 激活专家数平均 | **M块 IR** |',
           '|---:|---|---:|---:|---:|---:|---:|---:|']
    got = 0
    for spr, bs in BS:
        base = load(spr, 'vanilla')
        for i, (arm, name) in enumerate(ARMS):
            r = load(spr, arm)
            head = f'**{bs}**' if i == 0 else ''
            if r is None:
                out.append(f'| {head} | {name} | (缺) | | | | | |')
                continue
            got += 1
            delta = '—' if (arm == 'vanilla' or base is None) else \
                f'{(r["ms"] / base["ms"] - 1) * 100:+.2f}%'
            b = arm == BOLD
            out.append(f'| {head} | {name} | {fmt(r["ms"], 2, b)} | {delta} | '
                       f'{fmt(r["ir"], 4, b)} | {fmt(r["mb"], 1, b)} | '
                       f'{fmt(r["ae"], 1, b)} | {fmt(r["mb_ir"], 4, b)} |')
    return '\n'.join(out), got


def plots():
    out, missing = [], []
    for spr, bs in BS:
        out.append(f'#### BS = {bs}\n')
        for arm, _ in ARMS:
            png = f'docs/plots/ir20/BS{bs}_{arm}.png'
            if not os.path.exists(os.path.join(ROOT, png)):
                missing.append(png)
                continue
            out.append(f'**BS{bs} · {arm}**\n\n![]({png})\n')
    # the two numbers every figure marks, in one table
    ann = os.path.join(ROOT, 'docs/plots/ir20/annotations.md')
    if os.path.exists(ann):
        body = open(ann).read()
        tbl = body[body.find('| BS'):]
        tbl = tbl.split('\n## Missing')[0].rstrip()
        out.append('\n**每张图标注的两个时刻**（rank 0，测量窗口内某一层）：\n')
        out.append(tbl + '\n')
        out.append('`all slots pulled` = 该层最后一个槽 pull 完成的时刻（蓝虚线）；'
                   '`role 6 joins` = role 6 第一次算完 M 块的时刻（红点线），'
                   '非 SM-reuse 臂为空。\n')
    return '\n'.join(out), missing


def splice(text, begin, end, body):
    i = text.find(begin)
    j = text.find(end)
    if i < 0 or j < 0:
        raise SystemExit(f'marker missing: {begin} / {end}')
    return text[:i + len(begin)] + '\n' + body + '\n' + text[j:]


def main():
    readme = sys.argv[1]
    text = open(readme).read()
    tbl, got = table()
    pl, missing = plots()
    text = splice(text, '<!-- IR20:TABLE:BEGIN -->', '<!-- IR20:TABLE:END -->', tbl)
    if not missing:
        text = splice(text, '<!-- IR20:PLOTS:BEGIN -->', '<!-- IR20:PLOTS:END -->', pl)
    if '--check' in sys.argv:
        print(f'rows={got}/20 plots_missing={len(missing)}')
        return
    open(readme, 'w').write(text)
    print(f'rows={got}/20 plots_missing={len(missing)}')
    for m in missing:
        print('  missing', m)


if __name__ == '__main__':
    main()
