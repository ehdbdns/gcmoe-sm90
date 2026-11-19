"""Verify same-epoch fine detail against the aligned trace fixture."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from test_aligned_step_cpu import fixture
from aligned_step import build
from pull_gemm_detail import MASK, analyze, signed_delta


def fine_fixture(path, report, stream=False):
    epoch = report['epoch']
    workers = []
    for rank in report['ranks']:
        origin = MASK - 20  # the reconstructed fine timestamps cross a 32-bit wrap
        dispatch = rank['raw_stamps'][10]
        rows = []

        def emit(ts, kind, ident, *args):
            padded = args + (0,) * (5 - len(args))
            rows.append([(dispatch + ts - origin) & MASK, kind, ident, *padded, epoch, 0])

        batch = 0
        for slot, count in enumerate(rank['slots_owned']):
            begin, end = ((12, 70 if stream else 22), (14, 62), (18, 98))[slot]
            global_slot = rank['rank'] * 3 + slot
            emit(begin, 1, global_slot, slot, slot, 1)
            emit(end, 2, global_slot, slot, count, 0)
            for index, start in enumerate(range(0, count, rank['block_m'])):
                valid = min(rank['block_m'], count - start)
                base = max(end, 55) + index * 9
                if stream:
                    if slot == 0 and index == 0:
                        base = 55
                    emit(base - 1, 14, global_slot, slot, index, valid, batch)
                    emit(base + 6, 15, slot, slot, index, valid, batch)
                batch += 1
                cta = slot * 2 + index
                emit(base, 6, cta, slot, start, valid, 0, 1)
                emit(base + 5, 7, cta, slot, start, valid, 0, 1)
                emit(base + 20, 6, cta, slot, start, valid, 0, 2)
                emit(base + 25, 8, cta, slot, start, valid, 0, 2)
                emit(base + 28, 9, cta, slot, start, valid, 0, 2)
                emit(base + 30, 7, cta, slot, start, valid, 0, 2)
        workers.append(dict(rank=rank['rank'], origin_us=origin, count=len(rows), capacity=1000,
                            hidden=256, intermediate=128, epochs={str(epoch): rows[::-1]}))
    fine = dict(id=report['capture_id'], epochs=[epoch], ranks=workers)
    (path / 'fine.json').write_text(json.dumps(fine))
    return fine


def main():
    assert signed_delta(0, MASK) == 1
    with TemporaryDirectory() as tmp:
        path = Path(tmp)
        fixture(path)
        report, coarse = next(build(path, [0], 24))
        (path / 'test.counts.json').write_text(json.dumps(report))
        (path / 'test.perfetto.json').write_text(json.dumps(coarse))
        fine = fine_fixture(path, report)

        [(_, details, overlay)] = analyze(path, path)
        assert len(details['ranks']) == 8
        for rank in details['ranks']:
            assert rank['active_slots'] == 2
            assert rank['hot_slot']['slot'] == 0 and rank['hot_slot']['first_l1_us'] == 55
            assert rank['last_pull_slots'][0]['slot'] == 2
            assert rank['tile_counts'] == {1: 3, 2: 3}
            assert len(rank['l2_pushes']) == 3
            assert rank['first_claim_us'] == 52
        assert sum(event['ph'] == 'X' for event in overlay['traceEvents']) == 56 + 8 * (3 + 6 + 3)

        def reject(expected):
            try:
                analyze(path, path)
            except ValueError as exc:
                assert expected in str(exc), exc
            else:
                raise AssertionError(f'invalid trace accepted: {expected}')

        fine = fine_fixture(path, report, stream=True)
        [(_, detail, _)] = analyze(path, path)
        rank0 = detail['ranks'][0]
        assert rank0['hot_slot']['first_l1_us'] < rank0['hot_slot']['end_us']
        assert len(rank0['block_releases']) == 6
        rows = fine['ranks'][0]['epochs'][str(report['epoch'])]
        early = next(rec for rec in rows if rec[1] == 15)
        early[0] -= 3
        (path / 'fine.json').write_text(json.dumps(fine))
        reject('L2 released before its L1 tile completed')

        fine = fine_fixture(path, report)
        fine['ranks'][0]['count'] = 2000
        (path / 'fine.json').write_text(json.dumps(fine))
        reject('invalid fine ring')
        fine = fine_fixture(path, report)
        rows = fine['ranks'][0]['epochs'][str(report['epoch'])]
        rows.pop(next(i for i, row in enumerate(rows) if row[1] == 7))
        (path / 'fine.json').write_text(json.dumps(fine))
        reject('unpaired events 6/7')
        fine = fine_fixture(path, report)
        fine['id'] = 'older-capture'
        (path / 'fine.json').write_text(json.dumps(fine))
        reject('fine rows are from another capture')
    print('PULL_GEMM_DETAIL_CPU PASS')


if __name__ == '__main__':
    main()
