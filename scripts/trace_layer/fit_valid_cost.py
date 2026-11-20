"""Fit the H20 streaming planner from validated per-tile spans; score a held-out layer."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from statistics import mean

BUCKETS = (8, 16, 32, 40, 48, 64)
N1, N2 = 18, 20
SCALE = 8  # integer planner ticks per CTA-microsecond
OLD_TABLE = (703.2, 746.2, 849.6, 994., 1176.8, 1498.)


def bucket(v):
    if not 1 <= v <= 64:
        raise ValueError(f'unexpected valid={v}')
    return next(i for i, edge in enumerate(BUCKETS) if v <= edge)


def load_trace(path):
    doc = json.loads(path.read_text())
    if path.name.endswith('.absolute.perfetto.json'):
        tag = doc['gcmoe_capture']
        streams = [defaultdict(list) for _ in range(8)]
        for e in doc['traceEvents']:
            if e['ph'] == 'X' and e['cat'] in ('tiles', 'l2_pushes'):
                streams[e['pid']][e['cat']].append(e['args'])
        stem = path.name.removesuffix('.absolute.perfetto.json')
    else:
        tag = doc
        streams = {r['rank']: r for r in doc['ranks']}
        stem = path.name.removesuffix('.detail.json')
    counts = json.loads(path.with_name(stem + '.counts.json').read_text())
    if any(tag[k] != counts[k] for k in ('step', 'layer', 'epoch', 'capture_id')):
        raise ValueError('counts/trace identity mismatch')
    rows = []
    if sorted(r['rank'] for r in counts['ranks']) != list(range(8)):
        raise ValueError('expected all eight ranks')
    for r in counts['ranks']:
        rank = r['rank']
        if r['block_m'] != 64 or r['raw_kernels']['routed']['grid'] != 71:
            raise ValueError('expected the H20 M64 / 71-CTA routed configuration')
        src = streams[rank]
        def key(t):
            return (t['slot'], t['m_block'], t['ntile'], t['cta'], t['valid'])
        pushes = {key(t): t for t in src['l2_pushes']}
        if len(pushes) != len(src['l2_pushes']):
            raise ValueError('duplicate L2 push')
        blocks = defaultdict(lambda: [[], [], []])
        seen = set()
        for t in src['tiles']:
            v, phase, ntile = t['valid'], t['phase'], t['ntile']
            idx = t['slot'], t['m_block']
            ident = (*idx, phase, ntile)
            if (not 0 <= idx[0] < len(r['slots_owned']) or idx[1] < 0 or not 1 <= v <= 64
                    or t['end_us'] < t['start_us'] or ident in seen or phase not in (1, 2) or not 0 <= ntile < (N1 if phase == 1 else N2)
                    or v != min(64, r['slots_owned'][idx[0]] - 64*idx[1])):
                raise ValueError('invalid tile identity or geometry')
            seen.add(ident)
            elapsed = t['end_us'] - t['start_us']
            spans = blocks[(*idx, v)]
            if phase == 2:
                p = pushes.pop(key(t))
                if not t['start_us'] <= p['start_us'] <= p['end_us'] <= t['end_us']:
                    raise ValueError('push outside L2 tile')
                push = p['end_us'] - p['start_us']
                elapsed -= push
                spans[2].append(push)
            spans[phase-1].append(elapsed)
        if pushes or len(blocks) != r['m_blocks']:
            raise ValueError('incomplete trace')
        samples = []
        for (_, _, valid), spans in blocks.items():
            if tuple(map(len, spans)) != (N1, N2, N2):
                raise ValueError('incomplete block')
            samples.append((valid, *map(mean, spans)))
        kernels = r['raw_kernels']
        graph = kernels['combine']['end'] - min(kernels[k]['ts'] for k in ('dispatch','shared','plan'))
        rows.append((rank, samples, graph))
    return (tag['capture_id'], tag['epoch']), rows


def fit_phase(values, quantized):
    # Equal weight per observed valid size. Within-bucket differences identify the
    # common row slope, while the bucket intercept captures the WGMMA shape.
    groups = defaultdict(list)
    for v, time in values.items():
        groups[bucket(v) if quantized else 0].append((v, time))
    if quantized and len(groups) != len(BUCKETS):
        raise ValueError('training trace does not cover every compute bucket')
    var = cov = 0.
    for points in groups.values():
        x, y = map(mean, zip(*points))
        var += sum((v-x)**2 for v,_ in points)
        cov += sum((v-x)*(t-y) for v,t in points)
    slope = max(0., cov/var) if var else 0.
    bases = [mean(t-slope*v for v,t in groups[i]) for i in sorted(groups)]
    return bases if quantized else bases*len(BUCKETS), slope


def fit(rows):
    samples = defaultdict(list)
    for _, blocks, _ in rows:
        for v, *times in blocks:
            samples[v].append(times)
    phases = [fit_phase({v:mean(t[i] for t in times) for v,times in samples.items()}, i != 2)
              for i in range(3)]
    weights = (N1, N2, N2)
    slope = round(SCALE*sum(w*m[1] for w,m in zip(weights, phases)))
    offsets = tuple(round(SCALE*sum(w*m[0][i] for w,m in zip(weights, phases)))
                    for i in range(len(BUCKETS)))
    return offsets, slope, phases


def score(rows, offsets, slope):
    models = {'linear':lambda v:26+v, 'old_table':lambda v:OLD_TABLE[bucket(v)],
              'old_less_nominal_gap':lambda v:OLD_TABLE[bucket(v)] - 37*1.821,
              'valid':lambda v:(offsets[bucket(v)]+slope*v)/SCALE}
    scores = {name:[] for name in models}
    work, graph = [], []
    block_errors = {name: [] for name in models if name != 'linear'}
    for rank, blocks, wall in rows:
        observed = [N1*a + N2*(b+c) for _,a,b,c in blocks]
        work.append(sum(observed))
        graph.append(wall)
        for name, model in models.items():
            pred = [model(block[0]) for block in blocks]
            scores[name].append(sum(pred))
            if name in block_errors:
                block_errors[name].extend(abs(x-y) for x,y in zip(observed,pred))
    def share_mae(a,b):
        return 100*mean(abs(x/sum(a)-y/sum(b)) for x,y in zip(a,b))
    return dict(block_mae_cta_us={name:mean(err) for name,err in block_errors.items()},
                rank_share_mae_pp={name:dict(tile_work=share_mae(s, work), graph=share_mae(s, graph))
                                   for name,s in scores.items()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', type=Path, nargs='+', required=True)
    parser.add_argument('--holdout', type=Path, nargs='+', required=True)
    opts = parser.parse_args()
    used, training = set(), []
    for path in opts.train:
        identity, rows = load_trace(path)
        if identity in used:
            raise ValueError('duplicate training layer')
        used.add(identity)
        training.extend(rows)
    offsets, slope, phases = fit(training)
    print(json.dumps(dict(intercepts_q8=offsets, slope_q8=slope, per_phase_fits=phases,
                          blocks_per_valid=sorted(Counter(v for _,b,_ in training for v,*_ in b).items()))))
    for path in opts.holdout:
        identity, rows = load_trace(path)
        if identity in used:
            raise ValueError('repeated training/holdout layer')
        used.add(identity)
        print(path.name, json.dumps(score(rows, offsets, slope)))
