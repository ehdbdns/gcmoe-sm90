"""Compare this trace replay with the published BS256 run of the same arm."""
import json
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
for key in set(before['controls']) | set(after['controls']):
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
        for row in c['ranks']:
            with np.load(old/f'rank{row["rank"]}.npz') as a:
                idx = 2 * 40 + layer
                same('sample epoch', c['epoch'], int(a['epochs'].reshape(-1)[idx]))
                same('sample physical slot rows', row['slots_owned'], a['counts'][idx].tolist())
        assert path.with_name(path.name.replace('.counts.json', '.detail.perfetto.json')).is_file()
    same('sample layers', layers, {0, 8, 20, 32, 39})
