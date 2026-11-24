"""Join captured graph kernels to their exact role geometry and replay epoch."""
import argparse
import json
from pathlib import Path

NAMES = {1: 'dispatch', 2: 'routed kernel', 3: 'combine', 4: 'shared expert L1',
         8: 'shared expert L2', 5: 'ECO plan', 7: 'dispatch admission', 6: 'reuse kernel'}


def check(value, message):
    if not value:
        raise ValueError(message)


def decode(trace, device, eco, *, reuse=False, tokens=128):
    calls = device['calls']
    geometry = {tuple(v): int(k) for k, v in device['roles'].items()}
    required = {1, 2, 3, 4, 8} | ({5, 7} if eco else set())
    if reuse:
        required.add(6)
    check(set(geometry.values()) == required, 'missing or ambiguous role geometry')
    kernels = {role: [] for role in required}
    for event in trace['traceEvents']:
        if event.get('cat') == 'kernel' and 'gc_moe' in event['name'].lower():
            args = event['args']
            role = geometry[args['grid'][0], args['block'][0]]
            kernels[role].append((event['ts'], event['ts'] + event['dur']))
    for events in kernels.values():
        events.sort()
        check(len(events) == len(calls), 'profiler must expose every child kernel of each replay')
    prev_end = -float('inf')
    for i, row in enumerate(calls):
        spans = {NAMES[r]: times[i] for r, times in kernels.items()}
        c0, c1 = spans['combine']
        check(all(prev_end <= a < b <= c0 + .01 for name, (a, b) in spans.items()
                  if name != 'combine'), 'ambiguous graph replay boundary')
        prev_end = c1
        stamps, clocks = row['stamps'], row['clocks']
        check(stamps[0] == clocks[4] == row['epoch'] and stamps[22:24] == [64, tokens],
              'mismatched device epoch, M or token count')
        check(clocks[0] > 0 and clocks[0] <= clocks[1] < clocks[2] <= clocks[3], 'invalid CTA clocks')
        d0, d1 = spans['dispatch']
        exchange = d0 + ((stamps[3] - stamps[10]) % 2**32)
        check(d0 <= exchange <= d1, 'count exchange outside dispatch')
        pull = d0 + ((stamps[4] - stamps[10]) % 2**32)
        pull_end = d0 + ((stamps[5] - stamps[10]) % 2**32)
        check(exchange <= pull <= pull_end <= d1 + 20, 'pull stamps outside dispatch')
        check(pull <= d1, 'first pull outside dispatch')
        spans['count exchange'] = d0, pull
        spans['local count exchange'] = d0, exchange
        spans['pull token'] = pull, min(pull_end, d1)
        # Use the last combine CTA as a per-call anchor between globaltimer and CUPTI.
        r0, r1 = spans['routed kernel']
        if reuse:
            reuse_start, reuse_end = spans['reuse kernel']
            r0, r1 = min(r0, reuse_start), max(r1, reuse_end)
        first = c1 + (clocks[1] - clocks[3]) / 1000
        last = c1 + (clocks[2] - clocks[3]) / 1000
        check(r0 - 20 <= first < last <= r1 + 20, 'routed CTA stamps disagree with CUPTI')
        start, end = max(first, r0), min(last, r1)
        check(start < end, 'empty CTA span')
        spans['wait token'] = r0, start
        spans['routed GEMM'] = start, end
        yield spans, dict(epoch=row['epoch'], sample=i, rank_tokens=row['rank_tokens'],
                          m_blocks=row['m_blocks'], first_cta_raw_us=first, last_cta_raw_us=last)


def export(root, bs=1024, matrix=False):
    inputs = {}
    layouts = {}
    arms = ('vanilla', 'random', 'rr', 'eco', 'eco_smr') if matrix else ('rr', 'eco')
    for arm in arms:
        events, totals = [], [0] * 3
        traces = [json.loads((root / arm / f'rank{r}.json').read_text()) for r in range(8)]
        bases = [trace['baseTimeNanoseconds'] for trace in traces]
        for rank, trace in enumerate(traces):
            device = json.loads((root / arm / f'rank{rank}-device.json').read_text())
            signature = device['input_sha256']
            check(inputs.setdefault(rank, signature) == signature, 'two arms used different benchmark inputs')
            placement = device['placement']
            physical = 384 if arm == 'vanilla' else 448
            check(len(placement) == physical, 'wrong placement size')
            if arm in ('rr', 'eco', 'eco_smr'):
                order = [sorted(placement[j:j+56]) for j in range(0, 448, 56)]
                check(layouts.setdefault(rank, order) == order, 'EPLB replica ownership differs between arms')
            check(len(device['calls']) == 3, 'expected three post-warmup calls')
            events.append(dict(ph='M', name='process_name', pid=rank, tid=0, args={'name': f'rank {rank}'}))
            lanes = {}
            for spans, meta in decode(trace, device, arm.startswith('eco'), reuse=arm == 'eco_smr', tokens=bs // 8):
                totals[meta['sample']] += meta['rank_tokens']
                if matrix:
                    if meta['sample'] != 1:
                        continue
                    meta.update(bs=bs, layer=device["layer"], arm=arm)
                for name, (start, end) in spans.items():
                    if matrix and name in ('dispatch', 'local count exchange', 'routed kernel',
                                           'dispatch admission', 'reuse kernel'):
                        continue
                    track = 'count exchange + pull token' if matrix and name in ('count exchange', 'pull token') else name
                    if track not in lanes:
                        lanes[track] = len(lanes) + 1
                        events.append(dict(ph='M', name='thread_name', pid=rank, tid=lanes[track],
                                           args={'name': track}))
                    events.append(dict(ph='X', name=name, cat='SM90 MoE', pid=rank, tid=lanes[track],
                                       ts=start + (bases[rank] - min(bases)) / 1000, dur=end-start, args=meta))
        check(totals == [bs*6]*3, f'wrong global token assignments: {totals}')
        origin = min(e['ts'] for e in events if e['ph'] == 'X')
        for event in events:
            if event['ph'] == 'X':
                event['ts'] -= origin
        suffix = 'replay2.perfetto.json' if matrix else 'perfetto.json'
        (root / f'{arm}.{suffix}').write_text(json.dumps(dict(traceEvents=events, displayTimeUnit='us')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--bs', type=int, choices=(512, 1024), default=1024)
    parser.add_argument('--matrix', action='store_true')
    args = parser.parse_args()
    export(args.root, args.bs, args.matrix)
