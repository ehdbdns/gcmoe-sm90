"""Render the 20-arm main table (5 arms x BS 32/64/128/256) as markdown.

Reads _iso_runs/M20B{bs}_{arm}.json. Columns: ms, vs vanilla, IR (new
definition), M blocks avg, M block PEAK over 8 ranks, active experts avg,
M-block IR.
"""
import json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARMS = [('vanilla', '**原生 gcmoe（无副本）**'),
        ('random', '+ random + rr'),
        ('rr', '+ eplb + rr'),
        ('eco', '+ eplb + eco'),
        ('eco_smr', '+ eplb + eco + **SM-reuse**')]
BS = (32, 64, 128, 256)


# BS32 was measured twice: the first pass used the old (wrong) ECO weights
# wb=260/wa=111 and the buggy random arm; `W26B32_*` is the corrected rerun.
# Prefer the corrected files when they exist.
_OVERRIDE = {('32', 'eco'): 'W26B32_eco',
             ('32', 'eco_smr'): 'W26B32_eco_smr',
             ('32', 'random'): 'W26B32_random'}


def load(bs, arm):
    name = _OVERRIDE.get((str(bs), arm), f'M20B{bs}_{arm}')
    p = os.path.join(ROOT, f'_iso_runs/{name}.json')
    if not os.path.exists(p):
        return None
    with open(p) as f:
        d = json.load(f)
    ir = d.get('ir_step') or {}
    return dict(ms=d['ms_per_step_median'],
                ir=ir.get('ir_step_layer_mean'),
                mb=ir.get('mb_total_mean'),
                peak=ir.get('mb_peak_mean'),
                ae=ir.get('ae_total_mean'),
                mbir=ir.get('mb_ir_mean'),
                null=ir.get('ir_poisson_null_mean'))


def fmt(v, nd=4):
    return f'{v:.{nd}f}' if isinstance(v, float) else '—'


def main():
    out = []
    hdr = ('| BS | 臂 | ms/step | vs 原生 | **IR（新口径）** | M块数平均 | '
           '**M块峰值** | 激活专家数平均 | **M块 IR** |')
    sep = '|---:|---|---:|---:|---:|---:|---:|---:|---:|'
    out += [hdr, sep]
    for bs in BS:
        base = load(bs, 'vanilla')
        first = True
        for arm, label in ARMS:
            r = load(bs, arm)
            if r is None:
                out.append(f'| {"**%d**" % bs if first else ""} | {label} | '
                           f'_missing_ | | | | | | |')
                first = False
                continue
            vs = ('—' if (base is None or arm == 'vanilla')
                  else f'{(r["ms"] / base["ms"] - 1) * 100:+.2f}%')
            out.append(
                f'| {"**%d**" % bs if first else ""} | {label} | '
                f'{r["ms"]:.2f} | {vs} | {fmt(r["ir"])} | {r["mb"]:.1f} | '
                f'{r["peak"]:.1f} | {r["ae"]:.1f} | {fmt(r["mbir"])} |')
            first = False
    print('\n'.join(out))


if __name__ == '__main__':
    main()
