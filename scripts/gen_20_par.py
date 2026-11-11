"""Render the 20 compute-CTA parallelism figures.

Reads runs/V{spr}_{arm}.json and writes docs/plots/par20/BS{bs}_{arm}.png plus
docs/plots/par20/parallelism.md (the numbers quoted in the README).
"""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_parallelism import plot

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARMS = ('vanilla', 'random', 'rr', 'eco', 'eco_smr')
BS = ((4, 32), (8, 64), (16, 128), (32, 256))
OUT = os.path.join(ROOT, 'docs', 'plots', 'par20')


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
            # The workspace lives on a NAS with a hard group quota that can be
            # full for minutes at a time.  Re-render to /tmp (local overlay)
            # and only copy the finished PNG over if it is missing or if
            # --force is passed, so a full quota can never leave a truncated
            # image in the repo.
            tmp = f'/tmp/par20_{bs}_{arm}.png'
            try:
                info = plot(src, 0, tmp if not os.path.exists(png) or '--force' in sys.argv
                            else None)
                if os.path.exists(tmp) and (not os.path.exists(png) or '--force' in sys.argv):
                    import shutil
                    shutil.copyfile(tmp, png)
            except SystemExit as exc:
                missing.append(f'BS{bs}/{arm}: {exc}')
                continue
            info['bs'], info['arm'] = bs, arm
            rows.append(info)
            print(f"BS{bs:<4} {arm:<8} ctas={info['role2_ctas']}+{info['role6_ctas']} "
                  f"peak={info['peak_ctas']:<3} mean={info['mean_ctas']:<5} "
                  f"plateau={info['plateau_ctas']:<5} "
                  f"last20={info['tail20_ctas']:<5} "
                  f"(role2-only {info['tail20_role2']}) "
                  f"makespan={info['makespan_us']:<6} CTAus={info['cta_us']}")
    with open(os.path.join(OUT, 'parallelism.md'), 'w') as f:
        f.write('# Compute-CTA parallelism, rank 0, one measured layer\n\n')
        f.write('| BS | arm | role2 CTAs | role6 CTAs | peak | plateau '
                '(mid 50%) | last 20% | last 20% (role 2 only) | '
                'CTA\u00b5s | makespan (us) | below 50%-busy |\n')
        f.write('|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n')
        for r in rows:
            f.write(f"| {r['bs']} | {r['arm']} | {r['role2_ctas']} | {r['role6_ctas']} | "
                    f"{r['peak_ctas']} | {r['plateau_ctas']} | {r['tail20_ctas']} | "
                    f"{r['tail20_role2']} | {r['cta_us']:.0f} | "
                    f"{r['makespan_us']:.0f} | {r['collapse_pct']}% |\n")
        if missing:
            f.write('\n## Missing\n\n' + '\n'.join(f'- {m}' for m in missing) + '\n')
    print(f'\n{len(rows)} plots -> {OUT}')
    if missing:
        print('MISSING: ' + ', '.join(missing))


if __name__ == '__main__':
    main()
