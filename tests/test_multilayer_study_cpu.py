"""Check graph boundaries, share normalization, and paired study packaging on CPU."""
import csv
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
        (paired/'comparison.txt').write_text('valid_cost')
        (paired/'measurement.sha').write_text(STREAM_SHA)
        write(b_dir/'run.json', dict(base, stream_m_blocks=1, execution_sha=STREAM_SHA))
        write(s_dir/'run.json', dict(candidate, eco_costfn=2))
        compare(paired)
        assert read(paired/'comparison.json')['study'] == 'valid_cost'
    print('MULTILAYER_STUDY_CPU PASS')


if __name__ == '__main__':
    main()
