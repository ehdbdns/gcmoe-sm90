"""Build the four-track SM90-style overview from archived SM100 device clocks."""
import json
from pathlib import Path
import sys


def overview(counts, trace):
    anchors = {event['pid']: event['ts'] for event in trace['traceEvents']
               if event.get('name') == 'MoE' and event.get('ph') == 'X'}
    if set(anchors) != set(range(8)) or sorted(row['rank'] for row in counts['ranks']) != list(range(8)):
        raise ValueError('expected eight aligned ranks')
    origin = min(anchors.values())
    events = []
    for row in counts['ranks']:
        r = row['rank']
        c = row['device_clocks_ns']
        args = dict(step=counts['step'], layer=counts['layer'], rank=r, tokens=row['tokens'],
                    m_blocks=row['m_blocks'], block_m=row['block_m'])
        events.append(dict(ph='M', name='process_name', pid=r, tid=0, args=dict(name=f'Rank {r}')))
        # One transition per lane, as in aligned_step.py. Dispatch CTAs
        # overlap around the barrier: split the overview at the first pull.
        tracks = (
            ('dispatch', [('count exchange', c[4], c[6]), ('pull token', c[6], c[7])]),
            ('shared', [('shared L1+L2', min(c[8], c[10]), max(c[9], c[11]))]),
            ('routed', [('wait token', c[0], c[1]), ('GEMM', c[1], c[2])]),
            ('combine', [('combine', c[18], c[3])]),
        )
        for tid, (role, spans) in enumerate(tracks, 1):
            events.append(dict(ph='M', name='thread_name', pid=r, tid=tid, args=dict(name=role)))
            for name, start, end in spans:
                if not c[0] <= start <= end <= c[3]:
                    raise ValueError(f'invalid {name} interval on rank {r}')
                events.append(dict(ph='X', pid=r, tid=tid, cat=role, name=name,
                                   ts=anchors[r] - origin + (start - c[0]) / 1000,
                                   dur=(end - start) / 1000, args=args))
    return dict(traceEvents=events)


def write_overview(folder, prefix, counts, trace):
    target = folder / 'overview'
    target.mkdir(exist_ok=True)
    (target / f'{prefix}.perfetto.json').write_text(json.dumps(overview(counts, trace)) + '\n')


if __name__ == '__main__':
    root = Path(sys.argv[1])
    for arm in ('mega', 'gc'):
        folder = root / arm
        for layer in (0, 8, 20, 32, 39):
            prefix = f'step002_layer{layer:02}'
            def load(suffix):
                return json.loads((folder / 'perfetto' / f'{prefix}.{suffix}.json').read_text())
            write_overview(folder, prefix, load('counts'), load('perfetto'))
