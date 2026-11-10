"""Render the 20 producer/consumer timelines for the README.

Reads runs/T{spr}_{arm}.json (the trace sweep) and writes
docs/plots/ir20/BS{bs}_{arm}.png, plus docs/plots/ir20/annotations.md listing
the two moments every figure marks:
  * when ALL slots of the layer had finished pulling
  * when role 6 (the SM-reuse helper) first completed an M block
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_timeline_annot import plot

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARMS = ('vanilla', 'random', 'rr', 'eco', 'eco_smr')
BS = ((4, 32), (8, 64), (16, 128), (32, 256))
OUT = os.path.join(ROOT, 'docs', 'plots', 'ir20')


def main():
    os.makedirs(OUT, exist_ok=True)
    rows, missing = [], []
    for spr, bs in BS:
        for arm in ARMS:
            src = os.path.join(ROOT, f'runs/V{spr}_{arm}.json')
            if not os.path.exists(src):
                missing.append(f'BS{bs}/{arm}')
                continue
            png = os.path.join(OUT, f'BS{bs}_{arm}.png')
            try:
                info = plot(src, 0, png)
            except SystemExit as exc:
                missing.append(f'BS{bs}/{arm}: {exc}')
                continue
            info['bs'] = bs
            info['arm'] = arm
            # A decode layer pulls in ~20-300 us; anything near a millisecond
            # means the plot fell back to a PREFILL call (the log was still
            # consumed before the measured window opened), which would be a
            # silent, 100x-wrong figure. Flag it loudly.
            info['suspect'] = bool(info['pull_done_us'] and info['pull_done_us'] > 600)
            rows.append(info)
            print(f"BS{bs:<4} {arm:<8} slots={info['slots']:<3} "
                  f"blocks={info['total_mblocks']:<5} "
                  f"pull_done={info['pull_done_us']:<5}us "
                  f"role6_join={info['role6_first_us']} "
                  f"role2={info['role2_blocks']} role6={info['role6_blocks']} "
                  f"backlog>0={info['backlog_pos_pct']}%"
                  + ('   <<< SUSPECT (prefill-scale pull)' if info['suspect'] else ''))
    with open(os.path.join(OUT, 'annotations.md'), 'w') as f:
        f.write('# Timeline annotations (rank 0, one measured layer)\n\n')
        f.write('| BS | arm | slots | M blocks | all slots pulled (us) | '
                'role 6 joins (us) | role 2 blocks | role 6 blocks | '
                'first consumer block (us) | last block done (us) | backlog>0 |\n')
        f.write('|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n')
        for r in rows:
            f.write(f"| {r['bs']} | {r['arm']} | {r['slots']} | {r['total_mblocks']} | "
                    f"{r['pull_done_us']} | {r['role6_first_us'] or '—'} | "
                    f"{r['role2_blocks']} | {r['role6_blocks']} | "
                    f"{r['cons_first_us'] or '—'} | {r['cons_end_us']} | "
                    f"{r['backlog_pos_pct']}% |\n")
        if missing:
            f.write('\n## Missing\n\n' + '\n'.join(f'- {m}' for m in missing) + '\n')
    print(f'\n{len(rows)} plots -> {OUT}')
    if missing:
        print('MISSING: ' + ', '.join(missing))


if __name__ == '__main__':
    main()
