"""Separate Kineto captures can have different baseTimeNanoseconds."""
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/sm100'))
from native_phase_perfetto import rank_trace


def main():
    with TemporaryDirectory() as tmp:
        folder = Path(tmp)
        epoch = 1_700_000_000_000_000_000
        for side, reading, origin in [('before', 1000, 0), ('after', 1_001_000, 500_000)]:
            stem = folder / f'clock-{side}-rank0'
            event = dict(ph='X', cat='kernel', name='sm100_phase_clock_anchor',
                         ts=(reading - origin) / 1000 - 0.5, dur=1)
            stem.with_suffix('.json').write_text(json.dumps(dict(
                traceEvents=[event], baseTimeNanoseconds=epoch + origin)))
            np.save(stem.with_suffix('.npy'), [reading])
        # End at milestone 3; all phase starts/ends are inside the invocation.
        progress = np.full(19, 100)
        progress[0], progress[3] = 0, 200
        clocks = np.arange(40)[:, None] * 1000 + 2500 + progress
        np.savez(folder / 'measure-rank0.npz', clocks=np.stack([clocks] * 3),
                 counts=np.zeros((3, 40, 48), dtype=int))
        for layer, events, row in rank_trace(folder, 0, epoch, 2, True):
            moe = next(e for e in events if e['name'] == 'MoE')
            assert moe['ts'] == layer + 2.5 and moe['dur'] == 0.2
            assert row['alignment_drift_us'] == 0
            assert all(e['cat'] == 'device phase envelope' for e in events)
        np.save(folder / 'clock-after-rank0.npy', [501_000])
        try:
            list(rank_trace(folder, 0, epoch, 2, True))
        except ValueError as exc:
            assert 'unstable clock alignment' in str(exc)
        else:
            raise AssertionError('bad two-session calibration accepted')
    print('SM100 external clock anchors PASS')


if __name__ == '__main__':
    main()
