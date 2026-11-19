#!/usr/bin/env python3
"""Summarize existing Kineto traces before deciding how to join device epochs."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re


def summarize(trace):
    geometry = Counter()
    graph_launches = Counter()
    layer_markers = Counter()
    for event in trace['traceEvents']:
        name = event.get('name', '')
        lower_name = name.lower()
        if event.get('cat') == 'kernel' and 'gc_moe' in lower_name:
            args = event.get('args') or {}
            grid = args.get('grid') or ['?']
            block = args.get('block') or ['?']
            geometry[f'{grid[0]}x{block[0]}'] += 1
        if event.get('cat') in ('cuda_runtime', 'cuda_driver'):
            if 'graphlaunch' in lower_name:
                graph_launches[name] += 1
        if re.fullmatch(r'ProfilerStep#\d+', name):
            layer_markers[name] += 1
    return dict(capture=trace.get('gcmoe_capture'), gcmoe_kernel_count=sum(geometry.values()),
                kernel_geometry=dict(sorted(geometry.items())), graph_launches=dict(graph_launches),
                profiler_steps=dict(layer_markers))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace_dir', type=Path)
    args = parser.parse_args()
    summary = []
    for rank in range(8):
        path = args.trace_dir / f'rank{rank}.json'
        data = summarize(json.loads(path.read_text()))
        summary.append(dict(rank=rank, **data))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
