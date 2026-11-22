"""Compare single-push sender work and ready replica delays in the BS256 traces."""
import argparse
import csv
import json
from pathlib import Path


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        out = csv.DictWriter(f, fieldnames=rows[0])
        out.writeheader()
        out.writerows(rows)


def analyze(archive, run, output):
    replicas, senders = [], []
    loads = {}
    for name, path in (('old_push', archive / 'single_push'), ('baseline', run / 'baseline'),
                       ('candidate', run / 'single_push')):
        for layer in (0, 8, 20, 32, 39):
            file, = (path / 'perfetto').glob(f'step002_layer{layer:02}_epoch*.detail.json')
            sample = json.loads(file.read_text())
            ranks = sample['ranks']
            if (sample['step'], sample['layer']) != (2, layer) or sorted(r['rank'] for r in ranks) != list(range(8)):
                raise ValueError(f'{file}: invalid trace coverage')
            for rank in ranks:
                slots = rank['pull_slots']
                if sorted(s['slot'] for s in slots) != list(range(56)):
                    raise ValueError(f'{file}: incomplete physical slot trace')
                groups = rank['singleton_sends']
                expected = [] if name == 'baseline' else list(range(8))
                if sorted(s['cta'] for s in groups) != expected:
                    raise ValueError(f'{file}: wrong sender CTA set')
                for sender in groups:
                    senders.append(dict(arm=name, layer=layer, rank=rank['rank'], **sender,
                                        duration_us=sender['end_us'] - sender['start_us']))
                quota = {v['slot']: v for v in rank['replica_timing']}
                for slot in slots:
                    key = layer, rank['rank'], slot['slot']
                    load = slot['owned'], slot['copies']
                    if loads.setdefault(key, load) != load:
                        raise ValueError(f'changed slot workload {key}')
                    if not slot['owned'] or slot['copies'] <= 1:
                        continue
                    stamps = quota[slot['slot']]
                    ready, issue = stamps['quota_us'], stamps['first_issue_us']
                    prev = [s for s in slots if s['cta'] == slot['cta'] and s['copies'] == 1
                            and s['end_us'] <= slot['start_us']]
                    # Slot spans on one CTA do not overlap. Intersect them with
                    # this replica's ready-to-first-issue window on this rank.
                    occupied = sum(max(0, min(s['end_us'], issue) - max(s['start_us'], ready)) for s in prev)
                    replicas.append(dict(arm=name, layer=layer, rank=rank['rank'], cta=slot['cta'],
                                         slot=slot['slot'], rows=slot['owned'], quota_us=ready,
                                         first_issue_us=issue, quota_to_issue_us=issue-ready,
                                         prior_single_slots=len(prev), prior_single_after_quota_us=occupied,
                                         last_ready_us=stamps['last_block_ready_us']))
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / 'replica_stalls.csv', replicas)
    write_csv(output / 'singleton_senders.csv', senders)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('archive', 'run', 'output'):
        parser.add_argument(arg, type=Path)
    args = parser.parse_args()
    analyze(args.archive, args.run, args.output)
