"""Compare graph-to-combine spans on the same decode step, layer and physical allocation."""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from tempfile import TemporaryDirectory

import numpy as np

from audit_se_history import audit
from extract_se_routes import extract
from se_replay_report import ARMS, LAYERS, write_rows

HISTORY = (Path(__file__).resolve().parents[2] / 'docs/trace_executor_20261119' /
           'sm90_bs256_main_five_plan_comm_results/delivery/bs256')
ARMS256 = ARMS[:4]


def read_rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def same_routes(left, right):
    for layer in LAYERS:
        for rank in range(8):
            name = f'layer{layer}-rank{rank}.npz'
            with np.load(left/name) as a, np.load(right/name) as b:
                for key in ('ids', 'weights'):
                    if (a[key].shape != b[key].shape or a[key].dtype != b[key].dtype or
                            a[key].tobytes() != b[key].tobytes()):
                        return False
    return True


def compare(replay, capture, *, archived_replay=False, bank=None):
    if archived_replay:
        # The historical launcher verifies the full bank SHA-256 before extracting routes.
        manifest = json.loads((replay/'routes/manifest.json').read_text())
        hashes = dict(reversed(line.split()) for line in (HISTORY/'workload.sha256').read_text().splitlines())
        verified = {r: s['sha256'] for s in manifest['sources'] for r in s['source_ranks']}
        if verified != {r: hashes[f'measure-rank{r}.pt'] for r in range(8)}:
            raise ValueError('wrong reference bank')
        routes_equal = True
    elif bank:
        try:
            with TemporaryDirectory() as tmp:
                routes = Path(tmp)/'reference-routes'
                extract(HISTORY, bank, routes, 256)
                routes_equal = same_routes(capture/'routes', routes)
        except (FileNotFoundError, ValueError) as error:
            # Still report the independently measured live vanilla reference.
            print(f'Historical bank verification failed: {error}')
            routes_equal = False
    else:
        routes_equal = False

    audit(HISTORY, replay, arms=ARMS256)
    historical = read_rows(replay/'history_comparison.csv')
    current = [] if archived_replay else read_rows(replay/'capture_comparison.csv')
    full = {r['arm']: float(r['moe_ms_per_step'])*1000/40 for r in read_rows(HISTORY/'main_table.csv')}
    table = {r['arm']: float(r['moe_us']) for r in read_rows(replay/'main_table.csv')}
    layers, summary = [], []
    for ref, records, arms in [('archive_pre_split', historical, ARMS256), ('live_split', current, ('vanilla',) if current else ())]:
        for arm in arms:
            sample = []
            for layer in LAYERS:
                group = [r for r in records if r['arm'] == arm and int(r['layer']) == layer]
                if (len(group) != 24 or {(int(r['step_zero_based']), int(r['rank'])) for r in group}
                        != {(s, k) for s in range(3) for k in range(8)}):
                    raise ValueError('incomplete rank/step records')
                slots_equal = all(int(r['different_slots']) == 0 and r['auction_equal'] == 'True' for r in group)
                layout = json.loads((replay/f'layer{layer}'/arm/'rank0-device.json').read_text())['placement']
                if ref == 'archive_pre_split':
                    expected = json.loads((HISTORY/arm/'installed-layout.json').read_text())[layer]
                else:
                    expected = list(range(384))
                layout_equal = layout == expected
                paired = (routes_equal or ref == 'live_split') and layout_equal and slots_equal
                second = [r for r in group if int(r['step_zero_based']) == 1]
                real = max(float(r['history_graph_us']) for r in second)
                measured = max(float(r['replay_graph_us']) for r in second)
                if min(real, measured) <= 0:
                    raise ValueError('invalid graph interval')
                sample.append(dict(reference=ref, arm=arm, layer=layer, real_us=real, replay_us=measured,
                                   route_verified=routes_equal or ref == 'live_split', layout_equal=layout_equal,
                                   slots_and_auction_equal=slots_equal, workload_matched=paired,
                                   replay_minus_ref_us=measured-real if paired else None,
                                   replay_minus_ref_pct=100*(measured/real-1) if paired else None))
            layers.extend(sample)
            real = mean(r['real_us'] for r in sample)
            measured = mean(r['replay_us'] for r in sample)
            if not np.isclose(measured, table[arm], rtol=0, atol=1e-7):
                raise ValueError('replay timing does not reproduce the published metric')
            paired = all(r['workload_matched'] for r in sample)
            summary.append(dict(reference=ref, arm=arm, workload_matched=paired, real_sample_us=real, replay_us=measured,
                                replay_minus_ref_us=measured-real if paired else None,
                                replay_minus_ref_pct=100*(measured/real-1) if paired else None,
                                main_window_us=full[arm], replay_vs_window_pct=100*(measured/full[arm]-1),
                                sample_vs_window_pct=100*(real/full[arm]-1)))
    write_rows(replay/'timing_layers.csv', layers)
    write_rows(replay/'timing_summary.csv', summary)
    print((replay/'timing_summary.csv').read_text(), end='')
    return all(r['workload_matched'] for r in summary)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('replay', type=Path)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--capture', type=Path)
    mode.add_argument('--archived-replay', action='store_true')
    p.add_argument('--bank', type=Path)
    a = p.parse_args()
    if not compare(a.replay, a.capture, archived_replay=a.archived_replay, bank=a.bank):
        raise SystemExit(3)  # Tables are complete; some archive pairs could not be certified.
