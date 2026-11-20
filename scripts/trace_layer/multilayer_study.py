"""Package one EP8 measurement, then compare matched baseline/stream repetitions."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean, median

from aligned_step import require


FINE_LAYERS = [*range(0, 40, 4), 39]
STEPS = (2, 5, 8, 11)
ARMS = ('baseline', 'stream')


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    path.write_text(json.dumps(data, separators=(',', ':'), allow_nan=False) + '\n')


def csv_table(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def graph_tracks(trace, counts):
    """Put the full MoE interval on the same Kineto clock as the role/CTA tracks."""
    require(counts['alignment'] == 'Kineto baseTimeNanoseconds', 'missing common GPU time axis')
    events = trace['traceEvents']
    anchors = {e['pid']: e['ts'] for e in events if e['name'] == 'count exchange' and e['ph'] == 'X'}
    require(set(anchors) == set(range(8)), 'missing dispatch anchor')
    for row in counts['ranks']:
        rank = row['rank']
        start = anchors[rank] + (row['graph_start_us'] - row['raw_kernels']['dispatch']['ts'])
        args = {k:row[k] for k in ('graph_root', 'moe_us', 'tokens', 'routed_rows', 'm_blocks',
                                    'plan_cost', 'plan_share', 'actual_share', 'share_error_pp')}
        args.update(rank=rank, step=counts['step'], layer=counts['layer'], epoch=counts['epoch'])
        events.extend([dict(ph='M', name='thread_name', pid=rank, tid=700, args={'name':'MoE graph'}),
                       dict(ph='X', cat='MoE', name='MoE graph', pid=rank, tid=700,
                            ts=start, dur=row['moe_us'], args=args)])
    trace['gcmoe_capture'] = {k: counts[k] for k in ('capture_id', 'step', 'layer', 'epoch')}
    return trace


def collect(raw, dest, mode, step):
    run = read(raw / 'result.json')
    steps = 100 if mode == 'e2e' else 1
    require(run['timing'] == 'device_synced_window' and run['measure_steps'] == steps
            and run['eplb'] and run['seqs'] == 256 and run['window_s'] > 0
            and run['gen_tokens_per_step'][:2] == [256, 256], 'unexpected measured workload')
    record = dict(mode=mode, step=step, execution_sha=(raw/'execution.sha').read_text().strip(),
                  raw_path=str(raw), capture_id=run['capture_id'], prompt_hash=run['prompt_sha256'],
                  output_hash=hashlib.sha256(json.dumps(run['tokens'], sort_keys=True).encode()).hexdigest(),
                  measured_steps=steps, e2e_ms_per_step=1000*run['window_s']/steps)
    if 'eco_costfn' in run:
        record.update(eco_costfn=run['eco_costfn'], stream_m_blocks=run['stream_m_blocks'])
    if mode != 'e2e':
        layers = []
        rank_rows = []
        layer_rows = []
        per_rank = [0.] * 8
        for path in sorted((raw/'deliver').glob('*.counts.json')):
            c = read(path)
            require(c['capture_id'] == run['capture_id'] and c['step'] == step, 'mixed capture or step')
            stem = path.name.removesuffix('.counts.json')
            layers.append(c['layer'])
            if run.get('eco_costfn') == 2:
                from gc_moe.eco.h20_valid_cost import slot_cost
                for rank in c['ranks']:
                    expected = sum(slot_cost(q) for q in rank['slots_owned'])
                    require(rank['plan_cost'] == expected,
                            f"{stem} rank {rank['rank']}: valid score {rank['plan_cost']} != {expected}")
            write(dest/path.name, c)
            for suffix, output in [('perfetto','perfetto')] + (
                    [('detail.perfetto','absolute.perfetto')] if mode == 'fine' else []):
                trace = read(path.with_name(f'{stem}.{suffix}.json'))
                write(dest/f'{stem}.{output}.json', graph_tracks(trace, c))
            slowest = max(c['ranks'], key=lambda r: r['moe_us'])
            worst = max(c['ranks'], key=lambda r: abs(r['share_error_pp']))
            layer_rows.append(dict(step=step, layer=c['layer'], epoch=c['epoch'],
                                   worst_plan_rank=worst['rank'], worst_plan_share=worst['plan_share'],
                                   worst_actual_share=worst['actual_share'], worst_error_pp=worst['share_error_pp'],
                                   slowest_rank=slowest['rank'], max_rank_moe_us=slowest['moe_us'],
                                   mean_rank_moe_us=mean(r['moe_us'] for r in c['ranks']),
                                   **c['graph_plan_accuracy']))
            for r in c['ranks']:
                rank_rows.append(dict(step=step, layer=c['layer'], epoch=c['epoch'],
                                      **{k:r[k] for k in ('rank','tokens','m_blocks','routed_rows',
                                                          'graph_root','graph_start_us','combine_end_us',
                                                          'moe_us','plan_cost','plan_share',
                                                          'actual_share','share_error_pp')}))
                per_rank[r['rank']] += r['moe_us']
        require(layers == (FINE_LAYERS if mode == 'fine' else list(range(40))), 'incomplete layers')
        csv_table(dest/'ranks.csv', rank_rows)
        csv_table(dest/'layers.csv', layer_rows)
        errors = [abs(r['share_error_pp']) for r in rank_rows]
        record.update(plan_mae_pp=mean(errors), plan_max_error_pp=max(errors))
        if mode == 'coarse':
            record.update(moe_sum_ms_by_rank=[v/1000 for v in per_rank],
                          moe_max_rank_sum_ms=max(per_rank)/1000)
    write(dest/'run.json', record)


def compare(root):
    pairs = []
    captures = set()
    paired_layers = []
    marker = root/'comparison.txt'
    valid_cost = marker.exists() and marker.read_text().strip() == 'valid_cost'
    for step in STEPS:
        for mode in ('e2e','coarse','fine'):
            runs = [read(root/f'step{step:02d}'/arm/mode/'run.json') for arm in ARMS]
            base, stream = runs
            if valid_cost:
                require(base['execution_sha'] == stream['execution_sha']
                        and [(r['stream_m_blocks'], r['eco_costfn']) for r in runs] == [(1,0), (1,2)],
                        'valid-cost comparison requires the same streaming kernel revision and cost modes 0/2')
            require(all(r['step'] == step and r['mode'] == mode for r in runs)
                    and base['prompt_hash'] == stream['prompt_hash'], 'unmatched inputs')
            if mode != 'e2e':
                for r in runs:
                    require(r['capture_id'] and r['capture_id'] not in captures, 'duplicate capture')
                    captures.add(r['capture_id'])
                by_arm = []
                for arm in ARMS:
                    with (root/f'step{step:02d}'/arm/mode/'layers.csv').open() as f:
                        by_arm.append({int(r['layer']): r for r in csv.DictReader(f)})
                require(by_arm[0].keys() == by_arm[1].keys(), 'unmatched captured layers')
                for layer in sorted(by_arm[0]):
                    b, s = (table[layer] for table in by_arm)
                    paired_layers.append(dict(step=step, mode=mode, layer=layer,
                                              baseline_max_rank_us=b['max_rank_moe_us'],
                                              stream_max_rank_us=s['max_rank_moe_us'],
                                              delta_us=float(s['max_rank_moe_us'])-float(b['max_rank_moe_us']),
                                              baseline_plan_mae_pp=b['mae_pp'], stream_plan_mae_pp=s['mae_pp']))
            metrics = {'e2e':['e2e_ms_per_step'], 'coarse':['e2e_ms_per_step','moe_max_rank_sum_ms',
                       'plan_mae_pp','plan_max_error_pp'], 'fine':['plan_mae_pp','plan_max_error_pp']}[mode]
            for k in metrics:
                b, s = base[k], stream[k]
                pairs.append(dict(step=step, mode=mode, metric=k, baseline=b, stream=s,
                                  delta=s-b, delta_pct=100*(s-b)/b if b else None,
                                  matched_outputs=base['output_hash']==stream['output_hash']))
    csv_table(root/'paired.csv', pairs)
    csv_table(root/'layer_pairs.csv', paired_layers)
    summary = []
    for mode, metric in dict.fromkeys((p['mode'],p['metric']) for p in pairs):
        subset = [p for p in pairs if (p['mode'],p['metric'])==(mode,metric)]
        summary.append(dict(mode=mode, metric=metric, pairs=len(subset),
                            baseline_mean=mean(p['baseline'] for p in subset),
                            stream_mean=mean(p['stream'] for p in subset),
                            median_paired_delta=median(p['delta'] for p in subset),
                            stream_lower=sum(p['delta']<0 for p in subset),
                            pairs_matching_outputs=sum(p['matched_outputs'] for p in subset)))
    csv_table(root/'summary.csv', summary)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    package = sub.add_parser('collect')
    package.add_argument('raw', type=Path)
    package.add_argument('dest', type=Path)
    package.add_argument('mode', choices=['coarse','e2e','fine'])
    package.add_argument('step', type=int, choices=STEPS)
    sub.add_parser('compare').add_argument('root', type=Path)
    args = parser.parse_args()
    if args.action == 'collect':
        collect(args.raw, args.dest, args.mode, args.step)
    else:
        compare(args.root)
