"""Report per-rank load, first-wave timing and inter-tile gaps for the four EP8 runs."""
import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

from aligned_step import require


def measured_backlog(rank):
    """Integrate time with at least one published, unclaimed tile on this rank."""
    changes = Counter()
    claims = {}
    blocks = {r['batch']: (r['slot'], r['m_block']) for r in rank['block_releases']}
    tiles_per_phase = Counter(tile['phase'] for tile in rank['tiles'])
    for marker in rank['block_releases']:
        n = tiles_per_phase[marker['phase']] // rank['m_blocks']
        changes[marker['time_us']] += n
    for event in rank['claim_events']:
        if 'ntile' in event:
            key = (event['cta'], *blocks[event['batch']], event['phase'], event['ntile'])
            require(key not in claims, 'duplicate tile claim')
            claims[key] = event['time_us']
            changes[event['time_us']] -= 1
    require(len(claims) == len(rank['tiles']), 'missing per-tile claim timestamps')
    times = sorted(changes)
    total, area, integral, pending = 0, 0, [], []
    for i, ts in enumerate(times):
        if i and total:
            area += ts - times[i - 1]
        total += changes[ts]
        require(total >= 0, 'more observed claims than publications')
        integral.append(area)
        pending.append(total)
    require(total == 0, 'published tickets left over')

    def ready_time(ts):
        i = bisect_right(times, ts) - 1
        if i < 0:
            return 0
        return integral[i] + (ts - times[i] if pending[i] else 0)

    return claims, ready_time


def rank_metrics(rank, counts, exact):
    by_cta = defaultdict(list)
    for tile in rank['tiles']:
        by_cta[tile['cta']].append(tile)
    first_wave = []
    longest, max_ready, ready_cta_us = (0, None, None, None), (0, None, None, None), 0
    if exact:
        claim_times, ready_time = measured_backlog(rank)
    for cta, tiles in by_cta.items():
        tiles.sort(key=lambda item: item['start_us'])
        first_wave.append(tiles[0]['start_us'])
        for prev, curr in zip(tiles, tiles[1:]):
            begin, end = prev['end_us'], curr['start_us']
            require(end >= begin, f'overlapping compute intervals for CTA {cta}')
            if end - begin > longest[0]:
                longest = end - begin, cta, begin, end
            if exact:
                key = (cta, curr['slot'], curr['m_block'], curr['phase'], curr['ntile'])
                claim = claim_times[key]
                # A loader can preclaim the next tile while math is still busy.
                unclaimed_end = max(begin, min(claim, end))
                ready = ready_time(unclaimed_end) - ready_time(begin)
                ready_cta_us += ready
                if ready > max_ready[0]:
                    max_ready = ready, cta, begin, unclaimed_end
    kernels = counts['raw_kernels']
    dispatch = kernels['dispatch']['ts']
    pulls = [slot for slot in rank['pull_slots'] if slot['owned']]
    first = min(first_wave, default=None)
    last = max(first_wave, default=None)
    return dict(tokens=rank['tokens'], routed_rows=rank['routed_rows'], m_blocks=rank['m_blocks'],
                active_slots=rank['active_slots'], compute_grid=rank['routed_ctas'],
                working_ctas=len(first_wave), shared_end_us=round(kernels['shared']['end'] - dispatch, 3),
                routed_entry_us=round(kernels['routed']['ts'] - dispatch, 3),
                first_nonempty_slot_end_us=min((s['end_us'] for s in pulls), default=None),
                first_claim_us=rank['first_claim_us'], first_tile_us=first,
                last_cta_first_tile_us=last, first_wave_spread_us=last - first if first_wave else None,
                routed_exit_us=round(rank['routed_exit_us'], 3), longest_gap_us=longest[0],
                gap_cta=longest[1], gap_begin_us=longest[2], gap_end_us=longest[3],
                backlog_before_claim_cta_us=ready_cta_us if exact else None,
                max_backlog_in_gap_us=max_ready[0] if exact else None,
                backlog_cta=max_ready[1], backlog_window_begin_us=max_ready[2],
                backlog_window_end_us=max_ready[3])


def matrix_rows(root):
    seen = set()
    for rev in ('before', 'after'):
        for mode in ('baseline', 'stream'):
            directory = root / rev / mode
            runs, layers = set(), []
            for path in sorted(directory.glob('*.counts.json')):
                counts = json.loads(path.read_text())
                detail = json.loads(path.with_name(path.name.replace('.counts.json', '.detail.json')).read_text())
                identity = {key: counts[key] for key in ('capture_id', 'step', 'layer', 'epoch')}
                require(all(detail[k] == v for k, v in identity.items()), f'mismatched capture: {path}')
                require(counts['step'] == 2 and counts['gemm_start'] == 'first_valid_cta_claim',
                        f'unexpected step or GEMM boundary: {path}')
                runs.add(counts['capture_id'])
                layers.append(counts['layer'])
                ranks = {row['rank']: row for row in detail['ranks']}
                totals = {row['rank']: row for row in counts['ranks']}
                require(len(detail['ranks']) == len(counts['ranks']) == 8 and
                        ranks.keys() == totals.keys() == set(range(8)), f'missing ranks: {path}')
                for idx in range(8):
                    rank, total = ranks[idx], totals[idx]
                    require(all(rank[k] == total[k] for k in ('tokens', 'routed_rows', 'm_blocks')),
                            f'load mismatch: {path}, rank {idx}')
                    exact = rev == 'after' and mode == 'stream'
                    yield dict(revision=rev, mode=mode, **identity, rank=idx,
                               **rank_metrics(rank, total, exact))
            require(len(runs) == 1 and not seen.intersection(runs) and sorted(layers) == [8, 20],
                    f'expected two layers of a distinct capture: {directory}')
            seen.update(runs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle', type=Path)
    args = parser.parse_args()
    rows = list(matrix_rows(args.bundle))
    with (args.bundle / 'rank_metrics.csv').open('w', newline='') as out:
        writer = csv.DictWriter(out, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    main()
