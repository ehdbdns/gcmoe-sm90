"""Report device critical-rank MoE time summed across 40 layers per step."""
import csv
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
