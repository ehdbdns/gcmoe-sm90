"""Check graph boundaries, share normalization, and paired study packaging on CPU."""
import csv
from pathlib import Path
from tempfile import TemporaryDirectory

from test_aligned_step_cpu import fixture
from aligned_step import build
from multilayer_study import ARMS, FINE_LAYERS, STEPS, collect, compare, read, write
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
    print('MULTILAYER_STUDY_CPU PASS')


if __name__ == '__main__':
    main()
