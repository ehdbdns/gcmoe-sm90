"""Compare SM90 linear/fine step boundaries from a completed six-arm delivery."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def summarize(directory):
    measured = json.loads((directory / 'run.json').read_text())['measured']
    gpu = {}
    for rank in range(8):
        with np.load(directory / f'rank{rank}.npz') as data:
            clocks = data['clocks'].reshape(1000, 40, 6)
            g, c = clocks[:, :, 0], clocks[:, :, 3]
            gaps = g[1:, 0] - c[:-1, -1]
            metrics = {'between_steps': gaps, 'within_moe': (c-g).sum(1),
                       'between_layers': (g[:, 1:]-c[:, :-1]).sum(1)}
            if 'step_boundary' in data:
                b = data['step_boundary']
                i = np.arange(24, 999, 50)
                points = np.stack([c[i, -1], b[i, 1], b[i, 2], b[i, 3],
                                   b[i+1, 0], g[i+1, 0]], axis=1)
                spans = np.diff(points, axis=1)
                if np.any(spans < 0) or np.any(b[i, 0] > g[i, 0]):
                    raise ValueError(f'{directory} rank {rank}: invalid GPU marker order')
                for col, name in enumerate(('combine_to_forward_return', 'forward_to_sampler',
                                            'sample_tokens', 'sampler_to_next_execute',
                                            'execute_to_first_moe')):
                    metrics[name] = spans[:, col]
                metrics['marked_gaps'] = gaps[i]
                metrics['unmarked_gaps'] = gaps[~np.isin(np.arange(999)%50, [23,24,25])]
            for key, value in metrics.items():
                gpu.setdefault(key, []).append(value)
    series = {k: np.mean(v, axis=0)/1e6 for k,v in gpu.items()}
    series['host_10step'] = np.array(measured['ten_step_ms'])
    if 'host_boundary_ns' in measured:
        host = measured['host_boundary_ns']
        start, returned, collected = np.array(host['steps']).T
        spans = {}
        for name in ('schedule','update_from_output','get_output'):
            rows = np.array(host[name])
            if not np.array_equal(rows[:,0], np.arange(1000)):
                raise ValueError(f'{directory}: unexpected {name} call count/order')
            spans[name] = rows[:,2] - rows[:,1]
        core = spans.pop('get_output')
        spans['other_get_output_including_worker_wait'] = core - sum(spans.values())
        spans['other_engine_step'] = returned - start - core
        spans['collect'] = collected - returned
        if any(np.any(times < 0) for times in spans.values()):
            raise ValueError(f'{directory}: overlapping host spans')
        series.update({k:v/1e6 for k,v in spans.items()})
    return series


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('batch_directory', type=Path)
    args = p.parse_args()
    root = args.batch_directory
    pairs = list(csv.DictReader((root/'pairs.csv').open()))
    if not any({r['arm_a'],r['arm_b']} == {'eco_linear','eco'}
               and r['strict_pair_comparable']=='True' for r in pairs):
        raise ValueError('cost pair failed strict comparison')
    old, new = [summarize(root/s) for s in ('eco_linear','eco')]
    if old.keys() != new.keys():
        raise ValueError('different instrumentation across cost arms')
    print('interval,n,linear_ms,fine_ms,delta_ms,paired_delta_median_ms,fine_slower_count')
    for key in old:
        a,b = old[key],new[key]
        d = b-a
        print(f'{key},{len(d)},{a.mean():.6f},{b.mean():.6f},{d.mean():.6f},'
              f'{np.median(d):.6f},{np.count_nonzero(d>0)}')


if __name__ == '__main__':
    main()
