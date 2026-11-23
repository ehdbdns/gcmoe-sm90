"""Report device critical-rank MoE time summed across 40 layers per step."""
import csv
import json
import sys
from pathlib import Path

import numpy as np


def step_times(directory):
    ranks = []
    for rank in range(8):
        with np.load(directory / f'rank{rank}.npz') as data:
            times = data['clocks'][:, :4].astype(np.int64)
        if times.shape != (40000, 4) or np.any(times[:, 0] <= 0) or np.any(np.diff(times) < 0):
            raise ValueError(f'invalid timestamps for rank {rank}')
        ranks.append(times)
    calls = np.stack(ranks, axis=1)
    critical = np.argmax(calls[:, :, 3] - calls[:, :, 0], axis=1)
    segments = np.diff(calls[np.arange(len(calls)), critical], axis=1)
    return segments.reshape(1000, 40, 3).sum(axis=1) / 1e6


def compare_releases(root):
    def samples(trial):
        rows = {}
        for path in (root / trial / 'perfetto').glob('*.counts.json'):
            record = json.loads(path.read_text())
            for rank in record['ranks']:
                key = record['step'], record['layer'], rank['rank']
                if key in rows:
                    raise ValueError(f'duplicate release sample {key}')
                rows[key] = rank
        return rows

    baseline, candidate = samples('baseline'), samples('source_inbox')
    if baseline.keys() != candidate.keys() or len(candidate) != 5 * 8:
        raise ValueError('incomplete paired release sample')
    with (root / 'first_m_block.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['step', 'layer', 'rank', 'block_kind', 'baseline_graph_to_release_us',
                         'inbox_graph_to_release_us', 'inbox_minus_baseline_us'])
        for key, before in sorted(baseline.items()):
            after = candidate[key]
            if (before['slots_owned'], before['block_m']) != (after['slots_owned'], after['block_m']):
                raise ValueError(f'changed workload at {key}')
            for field, kind in [('first_l1_batch_us', 'any'), ('first_full_l1_batch_us', 'full')]:
                delays = [r[field] for r in (before, after)]
                if delays == [None, None]:
                    writer.writerow([*key, kind, None, None, None])
                    continue
                if None in delays:
                    raise ValueError(f'unpaired release at {key}')
                times = [r['raw_kernels']['dispatch']['ts'] - r['graph_start_us'] + delay
                         for r, delay in zip((before, after), delays)]
                writer.writerow([*key, kind, *times, times[1] - times[0]])


if __name__ == '__main__':
    root = Path(sys.argv[1])
    writer = csv.writer(sys.stdout)
    writer.writerow(['arm', 'steps', 'graph_to_first_ms', 'first_to_last_ms', 'last_to_combine_ms',
                     'moe_ms', 'moe_change_pct_vs_baseline'])
    baseline = {}
    for trial in sys.argv[2:] or ('baseline', 'split', 'dispatch'):
        steps = step_times(Path((root / trial / 'measurement.path').read_text().strip()))
        for sample in (steps[np.arange(len(steps)) != 2], steps):
            means = sample.mean(axis=0)
            total = means.sum()
            if trial == 'baseline':
                baseline[len(sample)] = total
            writer.writerow([trial, len(sample), *means, total, 100 * (total / baseline[len(sample)] - 1)])

    if 'source_inbox' in sys.argv[2:]:
        compare_releases(root)
