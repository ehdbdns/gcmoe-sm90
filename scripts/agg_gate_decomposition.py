"""Decompose the ECO arm's overhead from the gate experiments.

Arms (all 4 seq/rank, 100 measured steps, 25% redundancy, same 60 slots/rank):
  eplb_rr    EPLB placement + host round-robin, NO plan role  -> routed grid 72
  A_null     ECO with eco_null_plan=1  (plan CTA runs, but skips cold/warm +
             auction + fine publish)                            -> routed grid 71
  B_noauc    ECO with eco_no_auction=1 (cold/warm kept, auction dropped)
  C_full     ECO full

Deltas:
  A_null - eplb_rr     = one fewer SM  +  the ECO dispatch/pull changes
  B_noauc - A_null     = the plan's non-auction cost (cold/warm + publish)
  C_full - B_noauc     = the AUCTION itself
"""
import json
import os
import sys

ARMS = [('eplb_rr', 'runs/eplb_rr_100.json'),
        ('A_null', 'runs/A_null.json'),
        ('B_noauc', 'runs/B_noauc.json'),
        ('C_full', 'runs/C_full.json')]


def main():
    vals = {}
    for name, path in ARMS:
        if not os.path.exists(path):
            print(f'missing {path}')
            continue
        d = json.load(open(path))
        vals[name] = (d['ms_per_step_median'], d.get('ir', -1.0),
                      d.get('tokens_per_rank', -1))
    print(f'{"arm":10s}{"ms/step":>9s}{"IR":>9s}{"tok/rank":>10s}')
    for k, (ms, ir, tp) in vals.items():
        print(f'{k:10s}{ms:9.2f}{ir:9.4f}{tp:10d}')

    print('\n=== decomposition (ms/step) ===')
    def d(a, b):
        return vals[a][0] - vals[b][0] if a in vals and b in vals else float('nan')
    rows = [('A_null - eplb_rr   (1 SM + ECO dispatch/pull)', d('A_null', 'eplb_rr')),
            ('B_noauc - A_null   (plan non-auction)', d('B_noauc', 'A_null')),
            ('C_full - B_noauc   (AUCTION)', d('C_full', 'B_noauc')),
            ('C_full - eplb_rr   (total ECO overhead)', d('C_full', 'eplb_rr'))]
    for label, v in rows:
        print(f'{label:44s}{v:+8.2f}')
    pred = sum(v for k, v in rows[:3])
    print(f'{"sum of the three parts":44s}{pred:+8.2f}'
          f'   (check vs total {rows[3][1]:+.2f})')


if __name__ == '__main__':
    main()
