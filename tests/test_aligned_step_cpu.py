"""Exercise step/epoch attribution and Perfetto output without a GPU."""
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'trace_layer'))
from aligned_step import build  # noqa: E402


def fixture(path):
    workers = []
    for rank in range(8):
        calls, events = [], []
        for layer in range(3):
            epoch = 255 + layer  # crosses a device ring wrap
            t = 1000 + layer * 240 + rank * 0.25
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
            device = 50000 + layer * 240
            for slot, delta in ((3, 10), (4, 12), (5, 98), (10, 0),
                                (12, 115), (14, 31), (15, 52)):
                stamps[slot] = device + delta
            stamps[0], stamps[22], stamps[23] = epoch, 16, 16
            stamps[24:32] = [80 + r for r in range(8)]
            calls.append(dict(epoch=epoch, stamps=stamps, owned=[20, 12, 0]))
        workers.append(dict(rank=rank, topk=2, next_epoch=258, calls=calls))
        trace = dict(gcmoe_capture=dict(id='test', rank=rank),
                     baseTimeNanoseconds=1000000,
                     traceEvents=list(reversed(events)))
        (path / f'rank{rank}.json').write_text(json.dumps(trace))
    meta = dict(id='test', label='synthetic', step=2, layers=3, first_epoch=255,
                decode_tokens=16, ranks=workers)
    (path / 'capture.json').write_text(json.dumps(meta))


def main():
    with TemporaryDirectory() as tmp:
        path = Path(tmp)
        fixture(path)
        outputs = list(build(path, [0, 2], 24))
        assert [report['epoch'] for report, _ in outputs] == [255, 257]
        for report, trace in outputs:
            assert report['input_replication'] == 8
            assert len(report['ranks']) == 8
            assert sum(e['ph'] == 'X' for e in trace['traceEvents']) == 56
            for rank in report['ranks']:
                assert rank['tokens'] == 4 and rank['routed_rows'] == 32
                assert rank['m_blocks'] == 3
                assert rank['segment_us']['count exchange'] == 12
                assert rank['segment_us']['pull token'] == 88
                assert rank['segment_us']['wait token'] == 21
                assert rank['segment_us']['GEMM'] == 59

        def reject(expected):
            try:
                list(build(path, [2], 24))
            except ValueError as exc:
                assert expected in str(exc), exc
            else:
                raise AssertionError(f'capture should be rejected: {expected}')

        capture_file = path / 'capture.json'
        capture = json.loads(capture_file.read_text())
        capture['ranks'][4]['calls'][2]['stamps'][15] = 0xFFFFFFFF
        capture_file.write_text(json.dumps(capture))
        reject('first valid compute tile')
        fixture(path)
        capture = json.loads(capture_file.read_text())
        capture['ranks'][4]['calls'][2]['stamps'][12] += 190
        capture_file.write_text(json.dumps(capture))
        reject('device and CUPTI disagree')
        fixture(path)
        trace_file = path / 'rank4.json'
        trace = json.loads(trace_file.read_text())
        trace['gcmoe_capture']['id'] = 'stale'
        trace_file.write_text(json.dumps(trace))
        reject('another capture')
    print('ALIGNED_STEP_CPU PASS')


if __name__ == '__main__':
    main()
