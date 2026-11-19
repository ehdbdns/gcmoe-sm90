"""Exercise step/epoch attribution and Perfetto output without a GPU."""
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'trace_layer'))
from aligned_step import build, resolve_epochs  # noqa: E402


def fixture(path, layers=3):
    workers = []
    for rank in range(8):
        calls, events = [], []
        for layer in range(layers):
            epoch = 255 + layer  # crosses a device ring wrap
            elapsed = layer * 240 + layer * layer
            t = 1000 + elapsed + rank * 0.25
            for role, grid, block, offset, duration in (
                ('dispatch', 24, 256, 0, 100),
                ('shared', 71, 256, 1, 29),
                ('routed', 71, 384, 31, 80),
                ('plan', 1, 512, 8, 42),
                ('combine', 288, 384, 115, 20),
            ):
                events.append(dict(cat='kernel', name=f'gc_moe_{role}', ts=t + offset,
                                   dur=duration, args=dict(grid=[grid, 1, 1], block=[block, 1, 1])))
            stamps = [0] * 32
            device = 50000 + elapsed
            for slot, delta in ((3, 10), (4, 12), (5, 98), (10, 0),
                                (12, 115), (14, 31), (15, 52)):
                stamps[slot] = device + delta
            stamps[0], stamps[22], stamps[23] = epoch, 16, 16
            stamps[24:32] = [80 + r for r in range(8)]
            calls.append(dict(epoch=epoch, stamps=stamps, owned=[20, 12, 0]))
        workers.append(dict(rank=rank, topk=2, next_epoch=255 + layers, calls=calls))
        trace = dict(gcmoe_capture=dict(id='test', rank=rank),
                     baseTimeNanoseconds=1000000,
                     traceEvents=list(reversed(events)))
        (path / f'rank{rank}.json').write_text(json.dumps(trace))
    meta = dict(id='test', gemm_start='first_valid_cta_claim', label='synthetic', step=2, layers=layers, first_epoch=255,
                decode_tokens=16, ranks=workers)
    (path / 'capture.json').write_text(json.dumps(meta))


def check_epoch_resolution(path):
    fixture(path, layers=40)
    original = json.loads((path / 'capture.json').read_text())
    assert resolve_epochs(path, original, 24)['first_epoch'] == 255

    # Simulate a snapshot starting two steps early. Distinct inter-layer
    # intervals make it possible to reject both intervening flush steps.
    original['first_epoch'] = 175
    for worker in original['ranks']:
        older = []
        for group in range(2):
            for layer, row in enumerate(worker['calls']):
                epoch = 175 + group * 40 + layer
                stamps = row['stamps'].copy()
                stamps[0] = epoch
                for slot in (3, 4, 5, 10, 12, 14, 15):
                    stamps[slot] -= (2 - group) * 20000 - layer * 55
                older.append(dict(row, epoch=epoch, stamps=stamps))
        worker['calls'] = older + worker['calls']
    resolved = resolve_epochs(path, original, 24)
    assert resolved['first_epoch'] == 255
    assert resolved['device_snapshot_epochs']['first'] == 175
    assert all(len(worker['calls']) == 40 for worker in resolved['ranks'])

    # When two steps have the same timing shape, the parser must refuse to
    # select one solely because it was the last step in the device snapshot.
    for worker in original['ranks']:
        for old, recent in zip(worker['calls'][40:80], worker['calls'][80:]):
            for slot in (3, 4, 5, 10, 12, 14, 15):
                old['stamps'][slot] = recent['stamps'][slot] - 20000
    try:
        resolve_epochs(path, original, 24)
    except ValueError as exc:
        assert 'found [215, 255]' in str(exc), exc
    else:
        raise AssertionError('ambiguous epoch windows should be rejected')


def main():
    with TemporaryDirectory() as tmp:
        path = Path(tmp)
        check_epoch_resolution(path)
        fixture(path)
        outputs = list(build(path, [0, 2], 24, require_overlap=True))
        assert [report['epoch'] for report, _ in outputs] == [255, 257]
        for report, trace in outputs:
            assert report['input_replication'] == 8
            assert len(report['ranks']) == 8
            assert report['pull_gemm_overlap_ranks'] == list(range(8))
            assert sum(e['ph'] == 'X' for e in trace['traceEvents']) == 56
            for rank in report['ranks']:
                assert rank['tokens'] == 4 and rank['routed_rows'] == 32
                assert rank['m_blocks'] == 3
                assert rank['segment_us']['count exchange'] == 12
                assert rank['segment_us']['pull token'] == 88
                assert rank['segment_us']['wait token'] == 21
                assert rank['segment_us']['GEMM'] == 59
                assert rank['checks']['pull_after_first_claim_us'] == 46
                assert rank['checks']['drawn_pull_gemm_overlap_us'] == 48

        def reject(expected, require_overlap=False):
            try:
                list(build(path, [2], 24, require_overlap=require_overlap))
            except ValueError as exc:
                assert expected in str(exc), exc
            else:
                raise AssertionError(f'capture should be rejected: {expected}')

        capture_file = path / 'capture.json'
        capture = json.loads(capture_file.read_text())
        capture['ranks'][4]['calls'][2]['stamps'][15] = 0xFFFFFFFF
        capture_file.write_text(json.dumps(capture))
        reject('first valid CTA claim')
        fixture(path)
        capture = json.loads(capture_file.read_text())
        capture['ranks'][4]['calls'][2]['stamps'][12] += 190
        capture_file.write_text(json.dumps(capture))
        reject('device and CUPTI disagree')
        fixture(path)
        capture = json.loads(capture_file.read_text())
        del capture['gemm_start']
        capture_file.write_text(json.dumps(capture))
        reject('capture must record device-side first valid CTA claims')
        fixture(path)
        capture = json.loads(capture_file.read_text())
        capture['ranks'][4]['calls'][2]['owned'][0] -= 1
        capture_file.write_text(json.dumps(capture))
        reject('do not conserve top-k routes')
        fixture(path)
        capture = json.loads(capture_file.read_text())
        for worker in capture['ranks']:
            stamps = worker['calls'][2]['stamps']
            stamps[15] = stamps[10] + 103
        capture_file.write_text(json.dumps(capture))
        reject('no rank has both measured and visible pull/GEMM overlap', require_overlap=True)
        fixture(path)
        trace_file = path / 'rank4.json'
        trace = json.loads(trace_file.read_text())
        trace['gcmoe_capture']['id'] = 'stale'
        trace_file.write_text(json.dumps(trace))
        reject('another capture')
    print('ALIGNED_STEP_CPU PASS')


if __name__ == '__main__':
    main()
