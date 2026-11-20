"""Check graph boundaries, share normalization, and paired study packaging on CPU."""
import csv
import hashlib
import struct
import shutil
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'source'))

from test_aligned_step_cpu import fixture
from aligned_step import build
from first_step import index_order_controls
from multilayer_study import ARMS, BASELINE_SHA, FINE_LAYERS, STEPS, STREAM_SHA, collect, compare, read, write
from test_pull_gemm_detail_cpu import fine_fixture
from pull_gemm_detail import analyze


def check_index_order(root):
    row_hash = 'a' * 64
    effective = hashlib.sha256(bytes.fromhex(row_hash) * 256).hexdigest()
    audits = []
    for arm in (0, 1):
        initial, ranks = [], []
        for rank in range(8):
            states = dict(compressed_decode_layers=[2, 6])
            for call in ('first_call', 'treatment_call'):
                layers = {str(layer): dict(rows=256, permuted_rows=arm, lengths=row_hash,
                                           native=row_hash if arm else effective, consumed=effective)
                          for layer in (2, 6)}
                states[call] = dict(first_step=dict(index_order=layers, attention=dict(
                    hashes=dict(extra_order=[row_hash] * 256))))
            initial.append(states)
            ranks.append(dict(ordered_decode_modes={str(layer): [0] + [arm] * 104 for layer in (2, 6)}))
        audits.append(dict(initial=initial, ranks=ranks))
    result = index_order_controls(root, audits)
    assert result['control_match'] and result['treatment_match']
    audits[1]['initial'][7]['treatment_call']['first_step']['index_order']['6']['consumed'] = 'c' * 64
    result = index_order_controls(root, audits)
    assert result['control_match'] and not result['treatment_match']
    audits[1]['ranks'][7]['ordered_decode_modes']['6'][104] = 0
    try:
        index_order_controls(root, audits)
    except ValueError as exc:
        assert 'incorrect stream switch' in str(exc)
    else:
        raise AssertionError('accepted an incorrect final-step mode')


def main():
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        check_index_order(root)
        raw = root/'raw'
        raw.mkdir()
        deliver = raw/'deliver'
        deliver.mkdir()
        fixture(raw, layers=40)
        # Give different ranks different roots and combine tails, while leaving
        # the routed kernel and every device timestamp exactly as before.
        f = raw/'rank0.json'
        trace = read(f)
        for event in trace['traceEvents']:
            if event['name'] == 'gc_moe_shared':
                event['ts'] -= 6
            if event['name'] == 'gc_moe_combine':
                event['dur'] += 10
        write(f, trace)
        reports = list(build(raw, range(40), 24))
        c, trace = reports[0]
        rows = c['ranks']
        assert rows[0]['graph_root'] == 'shared' and rows[0]['moe_us'] == 150
        assert rows[1]['graph_root'] == 'dispatch' and rows[1]['moe_us'] == 135
        assert rows[0]['actual_share'] == 150/(150+7*135)
        assert rows[0]['plan_share'] == 80/sum(range(80, 88))
        assert abs(sum(r['share_error_pp'] for r in rows)) < 1e-12

        # A sharded detail must reproduce the single-file analyzer.
        write(deliver/'test.counts.json', c)
        write(deliver/'test.perfetto.json', trace)
        fine = fine_fixture(raw, c)
        original = analyze(raw, deliver)
        write(raw/f'fine-epoch-{c["epoch"]}.json', fine)
        write(raw/'fine.json', dict(id='test', epochs=[c['epoch']], per_epoch_files=True))
        assert analyze(raw, deliver) == original
        # Package all 40 layers and verify the added graph track shares the role clock.
        (deliver/'test.counts.json').unlink()
        for counts, trace in reports:
            stem = f'layer{counts["layer"]:02d}'
            write(deliver/f'{stem}.counts.json', counts)
            write(deliver/f'{stem}.perfetto.json', trace)
        result = dict(timing='device_synced_window', measure_steps=1, eplb=True, seqs=256,
                      window_s=.05, gen_tokens_per_step=[256]*3, capture_id='test',
                      prompt_sha256='fixture', tokens={'req0':[1]})
        write(raw/'result.json', result)
        (raw/'execution.sha').write_text('fixture')
        dest = root/'single'
        dest.mkdir()
        collect(raw, dest, 'coarse', 2)
        assert read(dest/'run.json')['moe_max_rank_sum_ms'] == 6
        events = read(dest/'layer00.perfetto.json')['traceEvents']
        full = next(e for e in events if e['name']=='MoE graph' and e['pid']==0 and e['ph']=='X')
        combine = next(e for e in events if e['name']=='combine' and e['pid']==0 and e['ph']=='X')
        assert full['ts'] + full['dur'] == combine['ts'] + combine['dur']
        for step in STEPS:
            for arm in ARMS:
                for mode in ('e2e','coarse','fine'):
                    out = root/f'step{step:02d}'/arm/mode
                    out.mkdir(parents=True)
                    record = dict(read(dest/'run.json'), mode=mode, step=step, capture_id=str(out))
                    write(out/'run.json', record)
                    if mode != 'e2e':
                        with (dest/'layers.csv').open() as src, (out/'layers.csv').open('w') as dst:
                            records = csv.DictReader(src)
                            writer = csv.DictWriter(dst, fieldnames=records.fieldnames)
                            writer.writeheader()
                            writer.writerows(r for r in records if mode=='coarse' or int(r['layer']) in FINE_LAYERS)
        compare(root)
        assert len((root/'layer_pairs.csv').read_text().splitlines()) == 205

        for counts, _ in reports:
            if counts['layer'] in FINE_LAYERS:
                shard = fine_fixture(raw, counts)
                write(raw/f'fine-epoch-{counts["epoch"]}.json', shard)
        write(raw/'fine.json', dict(id='test', epochs=[reports[i][0]['epoch'] for i in FINE_LAYERS],
                                    per_epoch_files=True))
        for stem, _, absolute in analyze(raw, deliver):
            write(deliver/f'{stem}.detail.perfetto.json', absolute)
        result.update(timing_tail=dict(engine_core='inproc', steps=100, ms_per_step=45,
                                       profiler_active=False, fine_writes=False),
                      eco_costfn=0, stream_m_blocks=0,
                      tokens={f'req{i}':[1, 2] for i in range(256)})
        write(raw/'result.json', result)
        (raw/'execution.sha').write_text(BASELINE_SHA)
        paired = root/'two-runs'
        b_dir, s_dir = (paired/'step02'/arm/'paired' for arm in ARMS)
        b_dir.mkdir(parents=True)
        collect(raw, b_dir, 'paired', 2)
        assert len(list(b_dir.glob('*.absolute.perfetto.json'))) == len(FINE_LAYERS)
        assert len(list(b_dir.glob('*.counts.json'))) == 40
        assert len((b_dir/'ranks.csv').read_text().splitlines()) == 321
        base = read(b_dir/'run.json')
        assert base['e2e_ms_per_step'] == 45 and base['profiled_step_ms'] == 50
        s_dir.mkdir(parents=True)
        shutil.copyfile(b_dir/'layers.csv', s_dir/'layers.csv')
        candidate = dict(base, execution_sha=STREAM_SHA, stream_m_blocks=1, capture_id='other',
                         e2e_ms_per_step=43, moe_max_rank_sum_ms=5.9)
        for changes, gate in [({}, True), ({'output_hash':'different'}, False),
                              ({'moe_max_rank_sum_ms':6}, False), ({'e2e_ms_per_step':46}, False)]:
            write(s_dir/'run.json', dict(candidate, **changes))
            compare(paired)
            assert read(paired/'comparison.json')['scheduling_gate_passed'] == gate
        assert len((paired/'paired.csv').read_text().splitlines()) == 6
        assert len((paired/'layer_pairs.csv').read_text().splitlines()) == 41
        # Exercise fixed replay packaging against the profiler's real per-slot rows.
        audits = []
        for rank in range(8):
            values = [q for counts, _ in reports for q in counts['ranks'][rank]['slots_owned']]
            step_hash = hashlib.sha256(struct.pack(f'<{len(values)}i', *values)).hexdigest()
            audits.append(dict(rank=rank, calls=4200, first_epoch=reports[0][0]['epoch']-80,
                               route_ids='ids', route_weights='weights', batch_order='batch',
                               layer_order='layers', owned_steps=[step_hash]*105,
                               owned_totals=[[1536]*40 for _ in range(105)], native_top1=[2]*256))
        result['fixed_workload'] = dict(
            initial=[dict(rank=r, placement='fixed', eco_state='cold') for r in range(8)], ranks=audits)
        write(raw/'result.json', result)
        (raw/'execution.sha').write_text(STREAM_SHA)
        collect(raw, b_dir, 'paired', 2)
        fixed_base = read(b_dir/'run.json')
        write(s_dir/'run.json', dict(fixed_base, stream_m_blocks=1, capture_id='other',
                                     moe_max_rank_sum_ms=5.9, e2e_ms_per_step=43))
        (paired/'comparison.txt').write_text('fixed_stream')
        (paired/'measurement.sha').write_text(STREAM_SHA)
        fixed_audit = read(b_dir/'workload.json')
        write(s_dir/'workload.json', fixed_audit)
        compare(paired)
        assert read(paired/'comparison.json')['scheduling_gate_passed']
        fixed_audit['ranks'][0]['owned_steps'][104] = 'different slots, same total'
        write(s_dir/'workload.json', fixed_audit)
        compare(paired)
        assert not read(paired/'comparison.json')['fixed_workload_faster']
        # Current protocol also binds the physical row order, actual switch,
        # native samples outside step 2 and the configured objective.
        (paired/'comparison.txt').write_text('fixed_stream_r4')
        controls = dict(GC_MOE_ECO='1', GC_MOE_ECO_COSTFN='0', GC_MOE_ECO_WB='26',
                        GC_MOE_ECO_WA='0', GC_MOE_ECO_NO_AUCTION='0', GC_MOE_ECO_NULL_PLAN='0',
                        GC_MOE_ECO_NOSPLIT='0', GC_MOE_ECO_TOKENS_ONLY='0', GC_MOE_ECO_BLOCKS_ONLY='0',
                        GC_MOE_BLOCK_M_FORCE='64', GC_MOE_MB_CLAIM='1', GC_MOE_FIXED_REUSE='1')
        for arm, directory in enumerate((b_dir, s_dir)):
            run = read(directory/'run.json')
            write(directory/'run.json', dict(run, fixed_controls=controls))
            state = read(b_dir/'workload.json')
            for initial in state['initial']:
                initial.update(warmup_stream=0, last_warm_input='x', last_warm_scales='sf')
            for rank in state['ranks']:
                rank.update(route_ordinals='stable', replay_stream=arm)
            state['ranks'][0]['native_history'] = dict(warm_top1_sha256='warm', replay_top1=[[2]*256 for _ in range(105)])
            write(directory/'workload.json', state)
        compare(paired)
        assert read(paired/'comparison.json')['scheduling_gate_passed']
        state = read(s_dir/'workload.json')
        state['ranks'][0]['native_history']['replay_top1'][104][3] = 9
        write(s_dir/'workload.json', state)
        compare(paired)
        summary = read(paired/'comparison.json')
        assert summary['first_native_difference_step'] == 104 and not summary['scheduling_gate_passed']
        state['ranks'][0]['route_ordinals'] = 'different'
        write(s_dir/'workload.json', state)
        compare(paired)
        assert not read(paired/'comparison.json')['fixed_workload_faster']
        # R5 forks after warmup. Validate the original warm logs are retained
        # and the actual checkpoint, first real call and native outputs gate speed.
        from multilayer_study import json_hash
        (paired/'comparison.txt').write_text('fixed_stream_r5')
        (paired/'checkpoint').mkdir()
        hardware = ''.join(f'{r},gpu-{r}\n' for r in range(8))
        for arm, directory in enumerate((b_dir, s_dir)):
            run = read(directory/'run.json')
            run['fixed_controls'].update(GC_MOE_KV_BYTES=str(11 * 1024**3), GC_MOE_STRICT_STATE_DIR='/common')
            run['ten_step_host_ms_per_step'] = [run['e2e_ms_per_step']]*10
            write(directory/'run.json', run)
            (directory/'hardware.csv').write_text(hardware)
            (directory/'runtime.log').write_text('same runtime')
            state = read(b_dir/'workload.json')
            state['ranks'][0]['native_history']['warm_top1_sha256'] = f'warm{arm}'
            for row, rank_audit in zip(state['initial'], state['ranks']):
                rank = row['rank']
                epoch = rank_audit['first_epoch']
                rank_audit['replay_stream'] = arm
                row['last_warm_input'] = f'x{arm}'
                boundary = dict(requests='r', physical_blocks='b', cache_groups='g', next_epoch=epoch,
                                eco_step=epoch-1, placement='p', meta='m', eco_state='e', packed_weights='w')
                layout = dict(views={'kv0': [0, 'torch.float32', [2], [1], 0]}, sizes=[8])
                h = hashlib.sha256(bytes([rank])).hexdigest()
                write(paired/'checkpoint'/f'rank{rank}.json',
                      dict(version=1, rank=rank, controls=boundary, layout=layout, sha256=h))
                row['checkpoint'] = dict(source_sha256=h, readback_sha256=h, controls_sha256=json_hash(boundary),
                                         layout_sha256=json_hash(layout), kv_bytes=8, kv_views=1)
                row['first_call'] = dict(rank=rank, epoch=epoch, quantized=h, scales=h, output=h)
            write(directory/'workload.json', state)
        compare(paired)
        result = read(paired/'comparison.json')
        assert result['strict_pair_comparable'] and result['scheduling_gate_passed']
        assert not result['last_warm_input_match'] and not result['warm_native_match']
        for group, key in [('checkpoint', 'readback_sha256'), ('first_call', 'quantized'), ('first_call', 'output')]:
            state = read(s_dir/'workload.json')
            old = state['initial'][7][group][key]
            state['initial'][7][group][key] = 'f'*64
            write(s_dir/'workload.json', state)
            compare(paired)
            result = read(paired/'comparison.json')
            assert not result['strict_pair_comparable'] and not result['scheduling_gate_passed']
            state['initial'][7][group][key] = old
            write(s_dir/'workload.json', state)
        state['ranks'][0]['native_history']['replay_top1'][104][5] = 3
        write(s_dir/'workload.json', state)
        compare(paired)
        assert not read(paired/'comparison.json')['scheduling_gate_passed']
        from first_step import FIELDS
        (paired/'comparison.txt').write_text('fixed_stream_r6')
        ids = [f'req{i}' for i in range(256)]
        for arm, directory in enumerate((b_dir, s_dir)):
            run = read(directory/'run.json')
            run['fixed_controls']['GC_MOE_FIRST_STEP_AUDIT'] = '1'
            write(directory/'run.json', run)
            audit = read(directory/'workload.json')
            audit['ranks'][0]['native_history']['replay_top1'][104][5] = 2
            for row in audit['initial']:
                call = row['first_call']
                h = call['quantized']
                sampler = dict(row_requests=ids, hidden=dict.fromkeys(ids, h),
                               logits={req: dict(sha256=h, argmax=2) for req in ids[row['rank']::8]})
                chain = [dict(layer=i, epoch=call['epoch']+i, hashes=dict.fromkeys(FIELDS,h), shared_weights=[h]*4,
                              nonfinite=dict.fromkeys(FIELDS,0), shared_nonzero=dict(shared_input=True, shared_y=True),
                              different_rows={key:dict(count=0,examples=[]) for key in FIELDS} if arm else {})
                         for i in range(40)]
                call['first_step'] = dict(rank=row['rank'], layers=chain, sampler=sampler)
            write(directory/'workload.json', audit)
        compare(paired)
        assert read(paired/'comparison.json')['scheduling_gate_passed']
        chain = audit['initial'][7]['first_call']['first_step']['layers']
        chain[3]['hashes']['routed_slots'] = 'f'*64
        chain[3]['different_rows']['routed_slots'] = dict(count=1,examples=[dict(row=9,request='req9',topk_slot=4)])
        write(s_dir/'workload.json', audit)
        compare(paired)
        verdict = read(paired/'comparison.json')
        assert not verdict['scheduling_gate_passed'] and not verdict['strict_pair_comparable']
        first, = verdict['first_observed_differences']
        assert (first['layer'],first['field'],first['rank']) == (3,'routed_slots',7)
        # Both arms now stream. ECO can give different physical replicas work.
        # The same logical routes, checkpoint and native outputs must still agree.
        (paired/'comparison.txt').write_text('fixed_valid_cost')
        real_counts = Path(__file__).resolve().parents[1]/'docs/trace_executor_20261119/fixed_stream_results_r8/step02/stream/paired'
        from gc_moe.eco.h20_valid_cost import slot_cost
        for arm, directory in enumerate((b_dir, s_dir)):
            run = read(directory/'run.json')
            mode = 2 * arm
            run.update(eco_costfn=mode, stream_m_blocks=1,
                       plan_mae_pp=(.24422080336094087, .27156201792549994)[arm],
                       plan_max_error_pp=(.8916918650862748, 1.017429906675342)[arm])
            run['fixed_controls'].update(GC_MOE_COST_STUDY='1', GC_MOE_ORDER_EXTRA='1',
                                        GC_MOE_STUDY_COSTFN=str(mode), GC_MOE_ECO_COSTFN=str(mode))
            write(directory/'run.json', run)
            audit = read(directory/'workload.json')
            for row, rank in zip(audit['initial'], audit['ranks']):
                rank.update(replay_stream=1, ordered_decode_modes={'2': [1]*105})
                row['compressed_decode_layers'] = [2]
                probe = row['first_call']['first_step']
                probe.update(stream_mode=1, attention=dict(mode=1),
                             index_order={'2': dict(rows=256, consumed='abc', lengths='abc')})
            if arm:
                audit['ranks'][0]['owned_steps'][0] = 'different quotas'
                row = audit['initial'][7]['first_call']['first_step']['layers'][3]
                row['hashes']['routed_slots'] = row['hashes']['output']
                row['different_rows']['routed_slots'] = dict(count=0, examples=[])
            write(directory/'workload.json', audit)
            for path in real_counts.glob('*.counts.json'):
                c = read(path)
                c['capture_id'] = run['capture_id']
                if mode == 2:
                    for r in c['ranks']:
                        r['plan_cost'] = sum(map(slot_cost, r['slots_owned']))
                write(directory/f"layer{c['layer']:02}.counts.json", c)
        compare(paired)
        verdict = read(paired/'comparison.json')
        assert verdict['strict_pair_comparable'] and not verdict['same_physical_routes']
        scores = verdict['same_quota_scores']
        assert scores['linear_policy'] == scores['fine_policy']
        assert abs(scores['fine_policy']['linear']['mae_pp'] - .24422080336) < 1e-9
        audit['ranks'][7]['route_ordinals'] = 'unexpected'
        write(s_dir/'workload.json', audit)
        compare(paired)
        assert not read(paired/'comparison.json')['strict_pair_comparable']
        (paired/'comparison.txt').write_text('valid_cost')
        (paired/'measurement.sha').write_text(STREAM_SHA)
        write(b_dir/'run.json', dict(base, stream_m_blocks=1, execution_sha=STREAM_SHA))
        write(s_dir/'run.json', dict(candidate, eco_costfn=2))
        compare(paired)
        assert read(paired/'comparison.json')['study'] == 'valid_cost'
    print('MULTILAYER_STUDY_CPU PASS')


if __name__ == '__main__':
    main()
