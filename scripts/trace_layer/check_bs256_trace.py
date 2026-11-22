"""Compare this trace replay with the published BS256 run of the same arm."""
import json
import os
import sys
from pathlib import Path

import numpy as np

old, new, arm = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
load = lambda p: json.loads(p.read_text())
before, after = [load(d / 'run.json') for d in (old, new)]


def same(name, a, b):
    if a != b:
        raise ValueError(f'{arm}: historical mismatch in {name}')


for key in ('prompt_hash', 'output_hash', 'prefill_steps', 'steps', 'layers', 'batch',
            'stream', 'cost', 'rearranged'):
    same(key, before[key], after[key])
trial = os.environ.get('SM90_ECO_TRIAL')
switches = {'single_push': 'GC_MOE_SM90_SINGLE_PUSH', 'source_index': 'GC_MOE_SM90_SOURCE_INDEX_PULL', 'split': 'GC_MOE_SM90_SPLIT_EXCHANGE', 'dispatch': 'GC_MOE_SM90_PLAN_DISPATCH'}
if trial:
    same('known trial', trial in ('baseline', *switches), True)
    for variant, key in switches.items():
        same(key, after['controls'].get(key), '1' if trial == variant else None)
for key in set(before['controls']) | set(after['controls']):
    if trial and key in switches.values():
        continue
    if key not in {'GC_MOE_FIXED_DIR', 'GC_MOE_STRICT_STATE_DIR', 'GC_MOE_EXEC_ROOT',
                   'GC_MOE_SRC', 'GC_MOE_CHECKPOINT_ROOT'} and not key.startswith('GC_MOE_FINE_'):
        same(key, before['controls'].get(key), after['controls'].get(key))
for phase in ('stats', 'measured'):
    for field in ('initial', 'audit'):
        def ordered(run):
            return sorted(run[phase][field], key=lambda r: r['rank'])
        left, right = ordered(before), ordered(after)
        same('ranks', [r['rank'] for r in right], list(range(8)))
        for a, b in zip(left, right):
            for key in a:
                if key != 'device_memory_bytes':
                    same(f'{phase}/{field}/rank{a["rank"]}/{key}', a[key], b[key])
for rank in range(8):
    with np.load(old/f'rank{rank}.npz') as a, np.load(new/f'rank{rank}.npz') as b:
        if trial:
            np.testing.assert_array_equal(a['auction'], b['auction'])
            np.testing.assert_array_equal(a['clocks'][:, 5], b['clocks'][:, 5])
        for key in ('counts', 'epochs', 'native'):
            if not np.array_equal(a[key], b[key]):
                raise ValueError(f'{arm}: rank{rank} {key} differs from historical array')
print(arm, 'historical workload verified for both windows and all eight ranks')

if len(sys.argv) == 5:
    dest = Path(sys.argv[4])
    layers = set()
    for path in sorted(dest.glob('*.counts.json')):
        c = load(path)
        layer = c['layer']
        same('step', c['step'], 2)
        same('alignment', c['alignment'], 'Kineto baseTimeNanoseconds')
        layers.add(layer)
        if trial:
            detail = load(path.with_name(path.name.replace('.counts.json', '.detail.json')))
            if trial == 'single_push':
                sent = sum(span['rows'] for rank in detail['ranks'] for span in rank['singleton_sends'])
                owned = sum(slot['owned'] for rank in detail['ranks']
                            for slot in rank['pull_slots'] if slot['copies'] == 1)
                same('singleton pushed versus owned rows across all ranks', sent, owned)
            for rank in detail['ranks']:
                same('singleton sender CTAs', sorted(s['cta'] for s in rank['singleton_sends']),
                     list(range(8)) if trial == 'single_push' else [])
                if trial == 'source_index' or (trial == 'baseline' and os.getenv('SM90_CHECK_INDEX_TIMING')):
                    same('quota publication count', len(rank['quota_publications']), 1)
                    expected = {s['slot'] for s in rank['pull_slots'] if s['copies'] > 1 and s['owned']}
                    same('replica transfer coverage', {s['slot'] for s in rank['replica_timing']}, expected)
                same('replica exchange spans', len(rank['replica_exchange']), int(trial == 'split'))
        for row in c['ranks']:
            if trial:
                expected = (20, 72, 72, 1) if trial == 'dispatch' else (24, 71, 71, 1)
                same('launch grids', tuple(row['raw_kernels'][k]['grid']
                     for k in ('dispatch', 'shared', 'routed', 'plan')), expected)
            with np.load(old/f'rank{row["rank"]}.npz') as a:
                idx = 2 * 40 + layer
                same('sample epoch', c['epoch'], int(a['epochs'].reshape(-1)[idx]))
                same('sample physical slot rows', row['slots_owned'], a['counts'][idx].tolist())
        assert path.with_name(path.name.replace('.counts.json', '.detail.perfetto.json')).is_file()
    same('sample layers', layers, {0, 8, 20, 32, 39})
