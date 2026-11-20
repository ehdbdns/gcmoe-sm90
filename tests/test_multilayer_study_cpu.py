"""Check graph boundaries, share normalization, and paired study packaging on CPU."""
import csv
import hashlib
import struct
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

from test_aligned_step_cpu import fixture
from aligned_step import build
from multilayer_study import ARMS, BASELINE_SHA, FINE_LAYERS, STEPS, STREAM_SHA, collect, compare, read, write
from test_pull_gemm_detail_cpu import fine_fixture
from pull_gemm_detail import analyze


def main():
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
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
        (paired/'comparison.txt').write_text('valid_cost')
        (paired/'measurement.sha').write_text(STREAM_SHA)
        write(b_dir/'run.json', dict(base, stream_m_blocks=1, execution_sha=STREAM_SHA))
        write(s_dir/'run.json', dict(candidate, eco_costfn=2))
        compare(paired)
        assert read(paired/'comparison.json')['study'] == 'valid_cost'
    print('MULTILAYER_STUDY_CPU PASS')


if __name__ == '__main__':
    main()
