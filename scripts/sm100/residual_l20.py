"""Export the calibrated post-dense-replay L19 exit -> L20 entry window."""
import csv
import json
from pathlib import Path
import sys

import numpy as np


def window(events, start, end):
    cursor = start
    previous = 'L19 MoE exit'
    for e in sorted(events, key=lambda item: item['ts']):
        left, right = max(start, e['ts']), min(end, e['ts'] + e['dur'])
        if right <= left:
            continue
        if left > cursor:
            yield 'gap', cursor, left, previous, e['name'], '', ''
        yield e['cat'], left, right, e['name'], '', e['args'].get('stream'), e['args'].get('correlation')
        if right > cursor:
            cursor, previous = right, e['name']
    if end > cursor:
        yield 'gap', cursor, end, previous, 'L20 MoE entry', '', ''


def main(raw, archive, out):
    with out.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow('arm rank kind start_after_exit_us end_after_exit_us duration_us name_or_previous following stream correlation'.split())
        for arm in ('gc', 'mega'):
            for r in range(8):
                trace = json.loads((raw / arm / f'profile-rank{r}.json').read_text())
                events = [e for e in trace['traceEvents'] if e.get('ph') == 'X' and
                          e.get('cat') in ('kernel', 'gpu_memcpy', 'gpu_memset')]
                marks = sorted(e['ts'] + e['dur']/2 for e in events if 'sm100_phase_clock_anchor' in e['name'] and e['dur'] > 0)
                dev = np.load(raw / arm / f'anchor-rank{r}.npy')
                with np.load(archive / arm / f'measure-rank{r}.npz') as data:
                    c = data['clocks'][1]
                a, b = int(c[19, 3]), int(c[20, 0])
                if len(marks) != 2 or not dev[0] < a < b < dev[1]:
                    raise ValueError(f'invalid step 1 anchors: {arm} rank {r}')
                scale = (marks[1] - marks[0]) / int(dev[1]-dev[0])
                start, end = [marks[0] + (t-int(dev[0]))*scale for t in (a,b)]
                for kind, x, y, name, following, stream, corr in window(events, start, end):
                    writer.writerow((arm,r,kind,x-start,y-start,y-x,name,following,stream,corr))


if __name__ == '__main__':
    main(*map(Path, sys.argv[1:]))
