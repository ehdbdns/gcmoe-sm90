"""How much IR can a placement possibly remove?  (lower-bound study)

IR is `mean over (step,layer) of max_r tok[r] / mean_r tok[r]`.  For one
(step, layer) the per-rank token counts are a multinomial sample over the
experts, so even a PERFECT placement (every rank gets exactly its fair share
of each expert) leaves Poisson noise.  That floor is

    IR_floor  ~=  1 + 1.42 / sqrt(m)          (m = tokens per rank per layer)

and only the part above it is placement quality.  This script reports, for
every arm, how much of the measured IR is (i) irreducible sampling noise,
(ii) removed by the current placement, (iii) left on the table.

It also replays the EPLB decision offline to show WHY so little gets removed:
`rearrange_now()` feeds `rebalance_experts_global` a histogram collapsed with
`demand.view(1, 1, experts)`, i.e. ONE synthetic step, so the greedy placer
cannot see that a hot expert is hot on some steps and cold on others.
"""
import glob, json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RUNS = os.path.join(ROOT, 'runs_ro')


def rows(path):
    d = json.load(open(path))
    s = d.get('ir_step') or {}
    if not s.get('ok'):
        return None
    return dict(label=d.get('label'), ms=d['ms_per_step_median'],
                ir=s['ir_step_layer_mean'], null=s['ir_poisson_null_mean'],
                m=s['routes_per_rank_per_layer'],
                mb=s['mb_total_mean'], mbir=s['mb_ir_mean'])


def main():
    arms = {}
    for p in sorted(glob.glob(os.path.join(RUNS, 'V*_*.json'))):
        r = rows(p)
        if not r:
            continue
        bs = r['label'].split('_')[0][1:]
        spr = {'4': 32, '8': 64, '16': 128, '32': 256}.get(bs)
        arm = r['label'].split('_', 1)[1]
        if spr:
            arms[(spr, arm)] = r

    print(f"{'BS':>4}{'arm':>10}{'IR':>8}{'floor':>8}{'excess':>8}"
          f"{'removed':>9}{'left':>8}")
    base = {}
    for bs in (32, 64, 128, 256):
        v = arms.get((bs, 'vanilla'))
        if not v:
            continue
        base[bs] = v['ir'] - v['null']
        for arm in ('vanilla', 'random', 'rr', 'eco', 'eco_smr'):
            r = arms.get((bs, arm))
            if not r:
                continue
            exc = r['ir'] - r['null']
            rem = 100 * (1 - exc / base[bs]) if base[bs] else 0.0
            print(f"{bs:>4}{arm:>10}{r['ir']:>8.4f}{r['null']:>8.4f}"
                  f"{exc:>8.4f}{rem:>8.0f}%{exc:>8.4f}")
        print()


if __name__ == '__main__':
    main()
