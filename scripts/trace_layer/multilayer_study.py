"""Package one EP8 measurement, then compare matched baseline/stream repetitions."""
import argparse
import csv
import hashlib
import json
import math
import os
import struct
from pathlib import Path
from statistics import mean, median

from aligned_step import require


FINE_LAYERS = [*range(0, 40, 4), 39]
BASELINE_SHA = '98b6194a56e7468653bb6230bf0b3887c9466dce'
STREAM_SHA = '6617ee976fa86eb77eb3a34b8b6df3e60059a586'
STEPS = (2, 5, 8, 11)  # Archived multi-process matrices.
ARMS = ('baseline', 'stream')


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    path.write_text(json.dumps(data, separators=(',', ':'), allow_nan=False) + '\n')


def reference(previous, dest, prepared=None):
    prior = read(previous/'run.json')
    if prepared is not None and not os.environ.get('GC_MOE_REFERENCE_RESULT'):
        trajectory = read(prepared)
    else:
        checksum, location = (previous/'result.sha256').read_text().strip().split(maxsplit=1)
        path = Path(os.environ.get('GC_MOE_REFERENCE_RESULT') or location)
        payload = path.read_bytes()
        require(hashlib.sha256(payload).hexdigest() == checksum, 'wrong historical baseline result')
        old = json.loads(payload)
        trajectory = dict(prompt_hash=old['prompt_sha256'], tokens=old['tokens'])
    output_hash = hashlib.sha256(json.dumps(trajectory['tokens'], sort_keys=True).encode()).hexdigest()
    require(trajectory['prompt_hash'] == prior['prompt_hash'] and output_hash == prior['output_hash'],
            'historical token reference does not match its receipt')
    write(dest/'tokens.json', trajectory)


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
    if mode == 'paired':
        tail = run['timing_tail']
        require(tail['engine_core'] == 'inproc' and tail['steps'] == 100
                and tail['profiler_active'] is False and tail['fine_writes'] is False
                and tail['ms_per_step'] > 0, 'missing unprofiled 100-step timing')
        require(len(run['tokens']) == run['seqs'] and all(run['tokens'].values()), 'missing output tokens')
        record.update(profiled_step_ms=record['e2e_ms_per_step'], timed_steps=tail['steps'],
                      e2e_ms_per_step=tail['ms_per_step'])
        if 'ten_step_host_ms_per_step' in tail:
            blocks = tail['ten_step_host_ms_per_step']
            require(len(blocks) == 10 and all(math.isfinite(v) and v > 0 for v in blocks)
                    and math.isclose(mean(blocks), tail['ms_per_step']), 'invalid 100-step timing breakdown')
            record['ten_step_host_ms_per_step'] = blocks
    if 'eco_costfn' in run:
        record.update(eco_costfn=run['eco_costfn'], stream_m_blocks=run['stream_m_blocks'])
    if 'fixed_controls' in run:
        record['fixed_controls'] = run['fixed_controls']
        require(run['eco'] and run['eco_weights'] == dict(wb=26, wa=0, rounds=1, cold_rounds=6),
                'unexpected ECO weights or auction rounds')
    if mode != 'e2e':
        layers = []
        slot_counts = [[] for _ in range(8)]
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
            if 'fixed_controls' in run:
                for rank in c['ranks']:
                    require(rank['block_m'] == 64 and rank['plan_cost'] == 26 * rank['m_blocks'] + rank['routed_rows'],
                            f"{stem} rank {rank['rank']}: exported plan differs from the M64 linear score")
            write(dest/path.name, c)
            for suffix, output in [('perfetto','perfetto')] + (
                    [('detail.perfetto','absolute.perfetto')] if mode == 'fine' or (mode == 'paired' and c['layer'] in FINE_LAYERS) else []):
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
                slot_counts[r['rank']].extend(r['slots_owned'])
        require(layers == (FINE_LAYERS if mode == 'fine' else list(range(40))), 'incomplete layers')
        csv_table(dest/'ranks.csv', rank_rows)
        csv_table(dest/'layers.csv', layer_rows)
        errors = [abs(r['share_error_pp']) for r in rank_rows]
        record.update(plan_mae_pp=mean(errors), plan_max_error_pp=max(errors))
        if mode in ('coarse', 'paired'):
            record.update(moe_sum_ms_by_rank=[v/1000 for v in per_rank],
                          moe_max_rank_sum_ms=max(per_rank)/1000)
        if 'fixed_workload' in run:
            audit = run['fixed_workload']
            initial = sorted(audit['initial'], key=lambda r: r['rank'])
            ranks = sorted(audit['ranks'], key=lambda r: r['rank'])
            require([r['rank'] for r in initial] == [r['rank'] for r in ranks] == list(range(8)),
                    'incomplete fixed-workload audit')
            require(len({r['placement'] for r in initial}) == 1, 'ranks disagree on EPLB placement')
            for r in ranks:
                require(r['calls'] == 4200 and len(r['owned_steps']) == 105, 'incomplete route replay')
                require(int(layer_rows[0]['epoch']) == r['first_epoch'] + 40*step,
                        'profiled call and replay step have different epochs')
                values = slot_counts[r['rank']]
                observed = hashlib.sha256(struct.pack(f'<{len(values)}i', *values)).hexdigest()
                require(observed == r['owned_steps'][step], 'profiled step counts differ from replay audit')
            for t in range(105):
                for layer in range(40):
                    require(sum(r['owned_totals'][t][layer] for r in ranks) == 256*6*8,
                            f'incomplete routed rows at step {t} layer {layer}')
            native = [r['native_top1'] for r in ranks if r['native_top1'] is not None]
            require(native and len(native[0]) == 256 and all(v == native[0] for v in native),
                    'missing or inconsistent native top-1 audit')
            write(dest/'workload.json', dict(initial=initial, ranks=ranks))
            record['fixed_workload'] = True
    write(dest/'run.json', record)


def compare_matrix(root):
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


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def checkpoint_controls(root, audits, step=0):
    directory = root/'checkpoint'
    if step:
        directory = directory/'treatment'
    checkpoints, inputs, outputs = [], [], []
    prior_differences = []
    for rank in range(8):
        manifest = read(directory/f'rank{rank}.json')
        initial = [audit['initial'][rank] for audit in audits]
        first = [row['treatment_call' if step else 'first_call'] for row in initial]
        expected = manifest['sha256']
        required = {'requests', 'physical_blocks', 'cache_groups', 'next_epoch', 'eco_step',
                    'placement', 'meta', 'eco_state', 'packed_weights'}
        require(manifest['rank'] == rank and manifest['version'] == 1
                and set(manifest['controls']) == required and len(expected) == 64
                and len(bytes.fromhex(expected)) == 32, f'invalid rank {rank} manifest')
        for audit, row, call in zip(audits, initial, first, strict=True):
            require(row['rank'] == call['rank'] == rank
                    and audit['ranks'][rank]['first_epoch'] + step * 40 == call['epoch'] == manifest['controls']['next_epoch'],
                    f'rank {rank} probe is outside the branch boundary')
            for key in ('quantized', 'scales', 'output'):
                require(len(call[key]) == 64 and len(bytes.fromhex(call[key])) == 32, f'invalid {key} probe')
        snapshots = [dict(row['treatment_checkpoint' if step else 'checkpoint']) for row in initial]
        if step:
            prior = [c.pop('before_restore_sha256') for c in snapshots]
            require(prior[0] == expected and all(len(h) == 64 and len(bytes.fromhex(h)) == 32 for h in prior),
                    'invalid end-of-control KV fingerprints')
            if prior[0] != prior[1]:
                prior_differences.append(rank)
        checkpoints.append(all(snapshot == dict(
            source_sha256=expected, readback_sha256=expected, controls_sha256=json_hash(manifest['controls']),
            layout_sha256=json_hash(manifest['layout']), kv_bytes=sum(manifest['layout']['sizes']),
            kv_views=len(manifest['layout']['views'])) for snapshot in snapshots)
            and min(manifest['layout']['sizes']) > 0 and bool(manifest['layout']['views']))
        inputs.append(all(first[0][k] == first[1][k] for k in ('quantized', 'scales')))
        outputs.append(first[0]['output'] == first[1]['output'])
    prior_info = {'control_end_kv_different_ranks': prior_differences} if step else {}
    return dict(**prior_info, checkpoint_match=all(checkpoints), first_moe_input_match=all(inputs),
                first_moe_output_match=all(outputs),
                first_moe_input_different_ranks=[r for r, ok in enumerate(inputs) if not ok],
                first_moe_output_different_ranks=[r for r, ok in enumerate(outputs) if not ok])


def compare(root):
    folder = root/'step02'
    if not (folder/'baseline'/'paired').exists():
        return compare_matrix(root)
    dirs = [folder/arm/'paired' for arm in ARMS]
    base, candidate = (read(path/'run.json') for path in dirs)
    require(all(run['step'] == 2 and run['mode'] == 'paired' and run['timed_steps'] == 100
                and run['measured_steps'] == 1 and run['capture_id'] for run in (base, candidate))
            and base['capture_id'] != candidate['capture_id'], 'expected two distinct step 2 captures')
    require(base['prompt_hash'] == candidate['prompt_hash'], 'unmatched prompts')
    marker = root/'comparison.txt'
    valid_cost = marker.exists() and marker.read_text().strip() == 'valid_cost'
    protocol = marker.read_text().strip() if marker.exists() else ''
    aa_control = protocol == 'fixed_stream_r7'
    r6 = aa_control or protocol == 'fixed_stream_r6'
    checkpointed = r6 or protocol == 'fixed_stream_r5'
    ordered_replay = checkpointed or protocol == 'fixed_stream_r4'
    fixed_stream = ordered_replay or protocol == 'fixed_stream'
    controls = {}
    flags = [(r['stream_m_blocks'], r['eco_costfn']) for r in (base, candidate)]
    if valid_cost:
        require(base['execution_sha'] == candidate['execution_sha'] == (root/'measurement.sha').read_text().strip()
                and flags == [(1, 0), (1, 2)], 'invalid valid-cost arms')
    elif fixed_stream:
        require(flags == [(0, 0), (1, 0)] and base['execution_sha'] == candidate['execution_sha']
                == (root/'measurement.sha').read_text().strip()
                and all(r.get('fixed_workload') for r in (base, candidate)), 'invalid fixed stream arms')
        audits = [read(path/'workload.json') for path in dirs]
        left, right = audits
        controls['placement_and_initial_state_match'] = all(
            all(b[k] == s[k] for k in ('rank', 'placement', 'eco_state'))
            for b, s in zip(left['initial'], right['initial'], strict=True))
        keys = ('rank', 'batch_order', 'route_ids', 'route_weights', 'layer_order', 'owned_steps')
        controls['routes_and_all_slot_counts_match'] = all(
            all(b[k] == s[k] for k in keys) for b, s in zip(left['ranks'], right['ranks'], strict=True))
        if ordered_replay:
            config = base['fixed_controls']
            required = dict(GC_MOE_ECO='1', GC_MOE_ECO_COSTFN='0', GC_MOE_ECO_WB='26',
                            GC_MOE_ECO_WA='0', GC_MOE_ECO_NO_AUCTION='0', GC_MOE_ECO_NULL_PLAN='0',
                            GC_MOE_ECO_NOSPLIT='0', GC_MOE_ECO_TOKENS_ONLY='0', GC_MOE_ECO_BLOCKS_ONLY='0',
                            GC_MOE_BLOCK_M_FORCE='64', GC_MOE_MB_CLAIM='1', GC_MOE_FIXED_REUSE='1')
            controls['same_run_controls'] = (config == candidate['fixed_controls']
                                             and all(config.get(k) == v for k, v in required.items()))
            controls['same_physical_routes'] = all(
                b['route_ordinals'] == s['route_ordinals'] for b, s in zip(left['ranks'], right['ranks'], strict=True)
            ) and controls['routes_and_all_slot_counts_match'] and controls['placement_and_initial_state_match']
            histories = []
            for arm, audit in enumerate(audits):
                require(all(r['warmup_stream'] == 0 for r in audit['initial'])
                        and all(r['replay_stream'] == arm for r in audit['ranks']),
                        'stream switch has wrong warmup or measurement state')
                history = audit['ranks'][0]['native_history']
                samples = history['replay_top1']
                require(len(samples) == 105 and all(len(row) == 256 for row in samples)
                        and samples[2] == audit['ranks'][0]['native_top1'], 'incomplete native sample history')
                histories.append(history)
            controls['last_warm_input_match'] = all(
                all(b[k] == s[k] for k in ('last_warm_input', 'last_warm_scales'))
                for b, s in zip(left['initial'], right['initial'], strict=True))
            controls['warm_native_match'] = histories[0]['warm_top1_sha256'] == histories[1]['warm_top1_sha256']
            mismatch = [sum(b != s for b, s in zip(bs, ss, strict=True))
                        for bs, ss in zip(*(h['replay_top1'] for h in histories), strict=True)]
            controls['native_differences_by_step'] = mismatch
            controls['first_native_difference_step'] = next((i for i, n in enumerate(mismatch) if n), None)
        if checkpointed:
            controls.update(checkpoint_controls(root, audits))
            controls['same_hardware'] = ((dirs[0]/'hardware.csv').read_text() == (dirs[1]/'hardware.csv').read_text()
                                          and len((dirs[0]/'hardware.csv').read_text().splitlines()) == 8
                                          and (dirs[0]/'runtime.log').read_bytes() == (dirs[1]/'runtime.log').read_bytes())
            controls['same_run_controls'] &= (config.get('GC_MOE_KV_BYTES') == str(11 * 1024**3)
                                              and bool(config.get('GC_MOE_STRICT_STATE_DIR')))
            if r6:
                from first_step import first_step_controls
                controls.update(first_step_controls(root, audits))
                controls['same_run_controls'] &= config.get('GC_MOE_FIRST_STEP_AUDIT') == '1'
            if aa_control:
                from first_step import attention_controls
                controls['same_run_controls'] &= config.get('GC_MOE_AA_CONTROL') == '1'
                controls['treatment_boundary'] = checkpoint_controls(root, audits, step=1)
                controls['treatment_step'] = first_step_controls(root, audits, step=1)
                controls['flashmla'] = attention_controls(root, audits)
                controls['negative_control_match'] = (all(controls[k] for k in (
                    'checkpoint_match', 'first_step_chain_match', 'first_step_finite', 'shared_weights_match',
                    'shared_branch_active', 'same_hardware', 'same_run_controls'))
                    and controls['flashmla']['control_match']
                    and not controls['treatment_boundary']['control_end_kv_different_ranks'])
            blocks = [r['ten_step_host_ms_per_step'] for r in (base, candidate)]
            for values, run in zip(blocks, (base, candidate), strict=True):
                require(len(values) == 10 and all(math.isfinite(v) and v > 0 for v in values)
                        and math.isclose(mean(values), run['e2e_ms_per_step']), 'invalid ten-step measurements')
            controls['ten_step_e2e_delta_ms'] = [s-b for b, s in zip(*blocks, strict=True)]
        native = [next(r['native_top1'] for r in audit['ranks'] if r['native_top1'] is not None)
                  for audit in audits]
        controls['native_top1_different_of_256'] = sum(b != s for b, s in zip(*native, strict=True))
    else:
        require((base['execution_sha'], candidate['execution_sha']) == (BASELINE_SHA, STREAM_SHA)
                and flags == [(0, 0), (1, 0)], 'invalid scheduling arms')
    matched = base['output_hash'] == candidate['output_hash']
    pairs = []
    for metric in ('e2e_ms_per_step', 'moe_max_rank_sum_ms', 'profiled_step_ms', 'plan_mae_pp', 'plan_max_error_pp'):
        b, s = base[metric], candidate[metric]
        pairs.append(dict(step=2, metric=metric, baseline=b, stream=s, delta=s-b,
                          delta_pct=100*(s-b)/b if b else None, matched_outputs=matched))
    csv_table(root/'paired.csv', pairs)
    tables = []
    for path in dirs:
        with (path/'layers.csv').open() as f:
            table = list(csv.DictReader(f))
        require([int(row['layer']) for row in table] == list(range(40)), 'incomplete paired layers')
        tables.append(table)
    layers = []
    for b, s in zip(*tables):
        layers.append(dict(step=2, layer=b['layer'], baseline_max_rank_us=b['max_rank_moe_us'],
                           stream_max_rank_us=s['max_rank_moe_us'],
                           delta_us=float(s['max_rank_moe_us'])-float(b['max_rank_moe_us']),
                           baseline_plan_mae_pp=b['mae_pp'], stream_plan_mae_pp=s['mae_pp']))
    csv_table(root/'layer_pairs.csv', layers)
    faster = {metric: candidate[metric] < base[metric] for metric in ('e2e_ms_per_step', 'moe_max_rank_sum_ms')}
    controlled_faster = matched and all(faster.values())
    if fixed_stream:
        controlled_faster &= controls['placement_and_initial_state_match'] and controls['routes_and_all_slot_counts_match']
        if ordered_replay:
            controlled_faster &= controls['same_run_controls'] and controls['same_physical_routes']
        controls['fixed_workload_faster'] = controlled_faster
    equivalence = True
    if ordered_replay:
        equivalence = not any(controls['native_differences_by_step'])
        if checkpointed:
            equivalence &= all(controls[k] for k in ('checkpoint_match', 'same_hardware', 'first_moe_input_match',
                                                       'first_moe_output_match'))
            if r6:
                equivalence &= all(controls[k] for k in ('first_step_chain_match', 'shared_branch_active',
                                                          'shared_weights_match', 'first_step_finite'))
            if aa_control:
                post = controls['treatment_step']
                equivalence &= (controls['negative_control_match'] and controls['flashmla']['treatment_match']
                                and controls['treatment_boundary']['checkpoint_match']
                                and all(post[k] for k in ('first_step_chain_match', 'shared_branch_active',
                                                           'shared_weights_match', 'first_step_finite')))
            controls['strict_pair_comparable'] = (equivalence and matched and controls['same_run_controls']
                                                  and controls['same_physical_routes'])
        else:
            equivalence &= controls['last_warm_input_match'] and controls['warm_native_match']
    write(root/'comparison.json', dict(study='valid_cost' if valid_cost else 'scheduling',
          matched_outputs=matched, lower=faster, faster_layers=sum(row['delta_us'] < 0 for row in layers),
          total_layers=len(layers), **controls,
          scheduling_gate_passed=not valid_cost and controlled_faster
          and controls.get('native_top1_different_of_256', 0) == 0
          and equivalence))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    package = sub.add_parser('collect')
    package.add_argument('raw', type=Path)
    package.add_argument('dest', type=Path)
    package.add_argument('mode', choices=['coarse','e2e','fine','paired'])
    package.add_argument('step', type=int)
    ref = sub.add_parser('reference')
    ref.add_argument('previous', type=Path)
    ref.add_argument('dest', type=Path)
    ref.add_argument('--prepared', type=Path)
    sub.add_parser('compare').add_argument('root', type=Path)
    args = parser.parse_args()
    if args.action == 'collect':
        collect(args.raw, args.dest, args.mode, args.step)
    elif args.action == 'reference':
        reference(args.previous, args.dest, args.prepared)
    else:
        compare(args.root)
