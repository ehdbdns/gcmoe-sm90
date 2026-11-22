"""Analyze the existing step-2 Kineto files on CPU, in each rank's local timeline."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np


def activity(events, start, end):
    totals = Counter()
    cursor, idle, largest = start, 0., 0.
    previous, next_name, before = '', '', ''
    for event in events:
        ts = event['ts']
        if ts >= end:
            break
        stop = min(end, ts + event['dur'])
        if stop <= start:
            continue
        left = max(start, ts)
        gap = max(0., left - cursor)
        idle += gap
        if gap > largest:
            largest, before, next_name = gap, previous, event['name']
        totals[event['name']] += stop - left
        if stop > cursor:
            cursor, previous = stop, event['name']
    tail = end - cursor
    if tail > largest:
        largest, before, next_name = tail, previous, 'MoE entry'
    return idle + tail, largest, before, next_name, totals


def analyze(folder, archive, arm, rank):
    trace = json.loads((folder / f'profile-rank{rank}.json').read_text())
    spans = [e for e in trace['traceEvents'] if e.get('ph') == 'X' and 'dur' in e]
    events = sorted((e for e in spans if e.get('cat') in ('kernel', 'gpu_memcpy', 'gpu_memset')),
                    key=lambda e: e['ts'])
    anchors = [e['ts'] + e['dur']/2 for e in events if 'sm100_phase_clock_anchor' in e['name']]
    dev = np.load(folder / f'anchor-rank{rank}.npy').astype(np.int64)
    if len(anchors) != 2 or len(dev) != 2 or dev[1] <= dev[0]:
        raise ValueError(f'{arm} rank {rank}: missing clock calibration')
    slope = (anchors[1] - anchors[0]) / int(dev[1] - dev[0])
    with np.load(archive / arm / f'measure-rank{rank}.npz') as data:
        clocks = data['clocks'][2].astype(np.int64)
    if clocks.shape != (40, 19) or not dev[0] <= clocks[0, 0] < clocks[-1, 3] <= dev[1]:
        raise ValueError('clock/trace window mismatch')
    mapped = anchors[0] + (clocks - dev[0]) * slope
    api = sorted((e for e in spans if e.get('cat') in ('cuda_runtime', 'cuda_driver')
                  and e['name'].startswith('cudaGraphLaunch') and anchors[0] < e['ts'] < anchors[1]),
                 key=lambda e: e['ts'])
    # The worker replays one MoE graph per layer. Reject ambiguous launch attribution.
    launches_ok = len(api) == len(clocks) and all(
        e['ts'] <= mapped[i, 0] + 5 for i, e in enumerate(api))
    rows, kernels = [], []
    for layer in range(1, len(clocks)):
        start, end = mapped[layer-1, 3], mapped[layer, 0]
        if end <= start:
            raise ValueError('invalid inter-layer interval')
        idle, largest, before, after, totals = activity(events, start, end)
        # Compare entry relative to the near-simultaneous post-barrier boundary,
        # as well as absolute calibrated Kineto time across ranks (baseTime can differ).
        row = dict(arm=arm, layer=layer, rank=rank, gap_us=end-start,
                   entry_host_ns=int(trace['baseTimeNanoseconds'])+round(end*1000),
                   device_active_union_us=end-start-idle, uncovered_us=idle, largest_uncovered_us=largest,
                   preceding_activity=before, following_activity=after, launches_seen=len(api),
                   prev_end_to_launch_us='', launch_to_entry_us='')
        if launches_ok:
            row.update(prev_end_to_launch_us=api[layer]['ts']-start, launch_to_entry_us=end-api[layer]['ts'])
        rows.append(row)
        kernels.extend(dict(arm=arm, layer=layer, rank=rank, name=name, summed_us=us)
                       for name, us in totals.items())
    return rows, kernels


def main():
    p = argparse.ArgumentParser()
    for name in ('raw_bs512', 'archive', 'output'):
        p.add_argument(name, type=Path)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows, kernels = [], []
    for arm in ('gc', 'mega'):
        for rank in range(8):
            r, k = analyze(args.raw_bs512 / arm, args.archive, arm, rank)
            rows.extend(r)
            kernels.extend(k)
    for name, data in [('intervals', rows), ('kernels', kernels)]:
        with (args.output / f'{name}.csv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)


if __name__ == '__main__':
    main()
