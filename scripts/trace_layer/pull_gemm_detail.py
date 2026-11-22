#!/usr/bin/env python3
"""Join SM90 fine events to the corresponding torch/Perfetto layer on all ranks."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from aligned_step import ENTRY_TOLERANCE_US, RANKS, require


MASK = 0xFFFFFFFF


def signed_delta(a, b):
    return ((a - b + (1 << 31)) & MASK) - (1 << 31)


def pair_records(rows, begin, end, ident):
    """Pair by the full event identity. Reject duplicates and missing edges."""
    first, last = {}, {}
    for row in rows:
        if row[1] in (begin, end):
            key = ident(row)
            table = first if row[1] == begin else last
            require(key not in table, f'duplicate event type {row[1]} for {key}')
            table[key] = row
    require(first.keys() == last.keys(),
            f'unpaired events {begin}/{end}: {len(first.keys() ^ last.keys())}')
    return [(first[key], last[key]) for key in first]


def active_ctas(tiles):
    """Number of CTAs executing at least one recorded tile at each boundary."""
    changes = defaultdict(Counter)
    for tile in tiles:
        if tile['end_us'] > tile['start_us']:
            cta = tile['cta']
            changes[tile['start_us']][cta] += 1
            changes[tile['end_us']][cta] -= 1
    active = Counter()
    result = []
    for ts, deltas in sorted(changes.items()):
        active.update(deltas)
        count = sum(v > 0 for v in active.values())
        if not result or count != result[-1][1]:
            result.append((ts, count))
    return result


def rank_detail(row, fine, epoch):
    rank = row['rank']
    require(0 < fine['count'] <= fine['capacity'] and not fine.get('error'),
            f'invalid fine ring on rank {rank}')
    rows = fine['epochs'].get(str(epoch), [])
    require(rows and len(rows) <= fine['count']
            and all(len(rec) == 10 and rec[8] == epoch for rec in rows),
            'missing or mixed-epoch fine rows')
    slots = row['slots_owned']
    block_m = row['block_m']
    dispatch = row['raw_stamps'][10]
    pull_end = signed_delta(row['raw_stamps'][5], dispatch)
    claim = signed_delta(row['raw_stamps'][15], dispatch) if any(slots) else None
    routed_end = row['dispatch_to_routed_end_us']
    origin = fine['origin_us']

    def ts(rec):
        value = signed_delta(origin + rec[0], dispatch)
        require(-ENTRY_TOLERANCE_US <= value <= routed_end + ENTRY_TOLERANCE_US,
                f'fine event outside dispatch/routed window: type {rec[1]}, {value} µs')
        return value

    def span(first, last):
        start, end = ts(first), ts(last)
        require(start <= end, f'negative duration for type {first[1]} and id {first[2]}')
        return dict(start_us=start, end_us=end)

    pulls = {}
    for first, last in pair_records(rows, 1, 2, lambda rec: rec[2]):
        local = first[3]
        require(0 <= local < len(slots) and first[2] == rank * len(slots) + local
                and last[3] == local and last[4] == slots[local],
                f'slot count or identity mismatch: slot {local}')
        require(first[4] == local % row['raw_kernels']['dispatch']['grid'],
                f'wrong dispatch CTA for slot {local}')
        pulls[local] = dict(span(first, last), cta=first[4], slot=local,
                            global_slot=first[2], owned=last[4], copies=first[5])
    require(len(pulls) == len(slots), f'expected {len(slots)} slot pairs, got {len(pulls)}')
    require(abs(max(item['end_us'] for item in pulls.values()) - pull_end) <= ENTRY_TOLERANCE_US,
            'fine pull tail disagrees with coarse pull completion')

    plan_waits = []
    # [FINE-FIELD] A record is [ts, type, id, a0, a1, a2, a3, ...]. The
    # begin/released flag of a type-3 replica-plan wait lives in a0 (index 3);
    # index 4 is a1, which this record never sets (kernel passes a0=0/1 and
    # leaves a1..a3 zero). Reading rec[4] here made every wait look like an
    # unpaired "start", so the parse aborted with
    # 'unpaired replica plan waits' although the trace is correctly paired
    # (a0 histogram on rank 0 / epoch 14729: {0: 12, 1: 12}).
    starts = [rec for rec in rows if rec[1] == 3 and rec[3] == 0]
    end_rows = [rec for rec in rows if rec[1] == 3 and rec[3] == 1]
    ends = {rec[2]: rec for rec in end_rows}
    require(len(starts) == len(end_rows) == len(ends)
            and {rec[2] for rec in starts} == set(ends),
            'unpaired replica plan waits')
    for first in starts:
        local = first[2] - rank * len(slots)
        require(local in pulls, f'plan wait on unknown global slot {first[2]}')
        plan_waits.append(dict(span(first, ends[first[2]]), slot=local, cta=pulls[local]['cta']))

    # The fixed BS256 arm uses the 384-thread routed role (N-factor 1).
    require(row['raw_kernels']['routed']['block'] == 384, 'expected 384-thread routed role')
    n_tiles = {1: fine['intermediate'] // 128, 2: fine['hidden'] // 256}
    expected = {(slot, start, min(block_m, owned - start), ntile, phase)
                for slot, owned in enumerate(slots) for start in range(0, owned, block_m)
                for phase, count in n_tiles.items() for ntile in range(count)}
    block_keys = {(slot, start // block_m): valid for slot, start, valid, _, _ in expected}
    releases = {}
    for rec in rows:
        kind = rec[1]
        if kind in (14, 15):
            slot, mblk, valid, batch = rec[3:7]
            key = (kind, slot, mblk)
            require(block_keys.get((slot, mblk)) == valid and key not in releases,
                    f'invalid or duplicate streamed block release: {key}')
            releases[key] = dict(time_us=ts(rec), phase=kind - 13, slot=slot,
                                 m_block=mblk, valid=valid, batch=batch)
    if releases:
        require(releases.keys() == {(kind, *block) for block in block_keys for kind in (14, 15)},
                'missing streamed block releases')
        l1_batches = {release['batch'] for key, release in releases.items() if key[0] == 14}
        require(l1_batches == set(range(len(block_keys))), 'unexpected L1 batch identities')
        for slot, mblk in block_keys:
            l1, l2 = releases[(14, slot, mblk)], releases[(15, slot, mblk)]
            require(l1['batch'] == l2['batch'] and l1['time_us'] <= l2['time_us'],
                    f'inconsistent release order for slot {slot}, block {mblk}')

    ident = lambda rec: tuple(rec[2:8])
    tiles, observed = {}, set()
    first_l1_by_slot = {}
    for first, last in pair_records(rows, 6, 7, ident):
        cta, slot, start, valid, ntile, phase = ident(first)
        key = (slot, start, valid, ntile, phase)
        require(key in expected and key not in observed, f'invalid or duplicate tile {key}')
        observed.add(key)
        item = dict(span(first, last), cta=cta, slot=slot, row_start=start,
                    m_block=start // block_m, valid=valid, ntile=ntile, phase=phase)
        ready_us = (releases[(13 + phase, slot, start // block_m)]['time_us']
                    if releases else pulls[slot]['end_us'])
        require(item['start_us'] >= ready_us, f'tile started before its input was ready: {key}')
        if releases and phase == 1:
            require(item['end_us'] <= releases[(15, slot, start // block_m)]['time_us'],
                    f'L2 released before its L1 tile completed: {key}')
        if phase == 1:
            first_l1_by_slot[slot] = min(item['start_us'], first_l1_by_slot.get(slot, item['start_us']))
        tiles[ident(first)] = item
    require(observed == expected, f'missing {len(expected - observed)} expected tiles')

    if not releases:
        # Baseline publishes L2 at slot granularity after every L1 (M, N).
        l1_finished = {}
        for item in tiles.values():
            if item['phase'] == 1:
                key = item['slot']
                l1_finished[key] = max(item['end_us'], l1_finished.get(key, item['end_us']))
        for item in tiles.values():
            if item['phase'] == 2:
                require(l1_finished[item['slot']] <= item['start_us'],
                        f"L2 started before all L1 tiles in slot {item['slot']} finished")
    pushes = {}
    for first, last in pair_records(rows, 8, 9, ident):
        key = ident(first)
        require(key[-1] == 2 and key in tiles, f'push has no matching L2 tile: {key}')
        tile = tiles[key]
        interval = span(first, last)
        require(tile['start_us'] <= interval['start_us'] <= interval['end_us'] <= tile['end_us'],
                f'push outside its tile interval: {key}')
        pushes[key] = dict(tile, **interval)
    require(pushes.keys() == {key for key in tiles if key[-1] == 2}, 'missing L2 push intervals')

    for slot, ready in first_l1_by_slot.items():
        pulls[slot]['first_l1_us'] = ready
        pulls[slot]['pull_end_to_first_l1_us'] = ready - pulls[slot]['end_us']
    claim_events = []
    claims = set()
    batch_blocks = {r['batch']: (r['slot'], r['m_block'])
                    for r in releases.values() if r['phase'] == 1}
    for rec in rows:
        kind, cta = rec[1:3]
        if kind in (4, 5):
            claim_events.append(dict(time_us=ts(rec), cta=cta,
                                     kind='queue empty' if kind == 4 else 'ticket seen',
                                     ticket=rec[4], **({'active_seen': rec[5]} if kind == 4 else {})))
        elif kind == 16:
            claim_events.append(dict(time_us=ts(rec), cta=cta, kind='ready scan empty',
                                     scanned_batches=rec[3]))
        elif kind == 17:
            phase, batch, ntile = rec[3:6]
            require(batch in batch_blocks and phase in (1, 2), 'invalid ready batch claim')
            slot, mb = batch_blocks[batch]
            release = releases[(13 + phase, slot, mb)]
            key = (cta, slot, mb * block_m, release['valid'], ntile, phase)
            require(key in tiles and key not in claims, 'unmatched or duplicate ready tile claim')
            require(release['time_us'] <= ts(rec) <= tiles[key]['start_us'],
                    'ready tile claim outside release/start interval')
            claims.add(key)
            claim_events.append(dict(time_us=ts(rec), cta=cta, kind='ticket seen',
                                     ticket=batch, batch=batch, ntile=ntile, phase=phase))
    if claims:
        require(claims == tiles.keys(), 'missing ready tile claims')
    ordered_slots = sorted(pulls.values(), key=lambda item: (item['end_us'], item['owned']), reverse=True)
    tile_rows = list(tiles.values())
    if tile_rows:
        require(min(tile['start_us'] for tile in tile_rows) >= claim,
                'fine tile precedes the first valid CTA claim')
    latest = sorted(tile_rows, key=lambda item: item['end_us'], reverse=True)
    return dict(rank=rank, tokens=row['tokens'], routed_rows=row['routed_rows'],
                m_blocks=row['m_blocks'], active_slots=sum(n > 0 for n in slots),
                dispatch_ctas=row['raw_kernels']['dispatch']['grid'],
                routed_ctas=row['raw_kernels']['routed']['grid'],
                first_claim_us=claim, pull_complete_us=pull_end,
                routed_exit_us=routed_end, fine_events=len(rows),
                hot_slot=max(ordered_slots, key=lambda item: item['owned']),
                last_pull_slots=ordered_slots[:5], last_tiles=latest[:12],
                last_l1=next((tile for tile in latest if tile['phase'] == 1), None),
                last_l2=next((tile for tile in latest if tile['phase'] == 2), None),
                last_push=max(pushes.values(), key=lambda tile: tile['end_us'], default=None),
                tile_counts=dict(Counter(item['phase'] for item in tile_rows)),
                active_compute_ctas=active_ctas(tile_rows),
                replica_exchange=[span(a, b) for a, b in pair_records(rows, 18, 19, lambda rec: rec[2])],
                pull_slots=ordered_slots, replica_plan_waits=plan_waits,
                tiles=tile_rows, l2_pushes=list(pushes.values()), claim_events=claim_events,
                block_releases=list(releases.values()))


def overlay(trace, details):
    events = trace['traceEvents'].copy()
    anchors = {event['args']['rank']: event['ts'] for event in events
               if event.get('ph') == 'X' and event['name'] == 'count exchange'}
    require(set(anchors) == set(range(RANKS)), 'Perfetto missing rank dispatch anchors')
    for rank in details:
        r = rank['rank']
        for title, key, offset, size in (
            ('Pull CTA', 'pull_slots', 10, rank['dispatch_ctas']),
            ('Replica waits for plan · CTA', 'replica_plan_waits', 100, rank['dispatch_ctas']),
            ('Routed CTA', 'tiles', 200, rank['routed_ctas']),
            ('L2 push · CTA', 'l2_pushes', 400, rank['routed_ctas']),
        ):
            for cta in range(size):
                events.append(dict(ph='M', name='thread_name', pid=r, tid=offset + cta,
                                   args={'name': f'{title} {cta}'}))
            for item in rank[key]:
                name = f'slot {item["slot"]}'
                if key == 'tiles':
                    name = f'L{item["phase"]} · ' + name + f' M{item["m_block"]} N{item["ntile"]}'
                events.append(dict(ph='X', name=name, cat=key, pid=r, tid=offset + item['cta'],
                                   ts=anchors[r] + item['start_us'],
                                   dur=item['end_us'] - item['start_us'], args=item))
        for item in rank['replica_exchange']:
            events.append(dict(ph='X', name='replica index exchange incl. peer wait', pid=r, tid=650,
                               ts=anchors[r] + item['start_us'], dur=item['end_us'] - item['start_us']))
        if rank['block_releases']:
            events.append(dict(ph='M', name='thread_name', pid=r, tid=600,
                               args={'name': 'Ready M blocks'}))
            for marker in rank['block_releases']:
                events.append(dict(ph='i', s='t', name=f'L{marker["phase"]} publish',
                                   cat='ready block', pid=r, tid=600,
                                   ts=anchors[r] + marker['time_us'], args=marker))
        for marker in rank['claim_events']:
            events.append(dict(ph='i', s='t', name=marker['kind'], cat='claim queue',
                               pid=r, tid=200 + marker['cta'],
                               ts=anchors[r] + marker['time_us'], args=marker))
    return dict(trace, traceEvents=events)


def analyze(capture_dir, deliver_dir, require_streamed_blocks=False, require_batch_claims=False):
    manifest = json.loads((capture_dir / 'capture.json').read_text())
    fine = json.loads((capture_dir / 'fine.json').read_text())
    require(fine.get('id') == manifest['id'], 'fine rows are from another capture')
    device = {item['rank']: item['calls'] for item in manifest['ranks']}
    outputs = []
    selected = []
    for path in sorted(deliver_dir.glob('*.counts.json')):
        counts = json.loads(path.read_text())
        epoch = counts['epoch']
        if epoch not in fine['epochs']:
            continue
        require(counts['capture_id'] == manifest['id']
                and epoch == manifest['first_epoch'] + counts['layer'], 'counts and manifest disagree')
        part = fine
        if fine.get('per_epoch_files'):
            part = json.loads((capture_dir / f'fine-epoch-{epoch}.json').read_text())
            require(part['id'] == fine['id'] and part['epochs'] == [epoch], 'wrong fine epoch file')
        workers = {item['rank']: item for item in part['ranks']}
        require(len(part['ranks']) == RANKS and set(workers) == set(range(RANKS)), 'expected fine rows for 8 ranks')
        selected.append(epoch)
        require(len(counts['ranks']) == RANKS
                and {row['rank'] for row in counts['ranks']} == set(range(RANKS)),
                'expected counts for 8 ranks')
        details = []
        for row in counts['ranks']:
            rank = row['rank']
            device_call = device[rank][counts['layer']]
            require(device_call['epoch'] == epoch and row['raw_stamps'] == device_call['stamps']
                    and row['slots_owned'] == device_call['owned'],
                    f'epoch {epoch}, rank {rank}: counts differ from the device manifest')
            try:
                detail = rank_detail(row, workers[rank], epoch)
                require(not require_streamed_blocks or len(detail['block_releases']) == 2 * row['m_blocks'],
                        'missing early M-block release instrumentation')
                if require_batch_claims:
                    require(sum('ntile' in claim for claim in detail['claim_events']) == len(detail['tiles']),
                            'missing per-tile claim instrumentation')
                details.append(detail)
            except ValueError as exc:
                raise ValueError(f'epoch {epoch}, rank {rank}: {exc}') from exc
        stem = path.name.removesuffix('.counts.json')
        coarse = json.loads((deliver_dir / f'{stem}.perfetto.json').read_text())
        spans = [event for event in coarse['traceEvents'] if event.get('ph') == 'X']
        by_rank = {row['rank']: row for row in counts['ranks']}
        require(len(spans) == RANKS * (6 if manifest.get("arm") == "rr" else 7)
                and len({(event['args']['rank'], event['name']) for event in spans}) == len(spans)
                and all(event['args']['epoch'] == epoch
                        and event['args']['step'] == counts['step']
                        and event['args']['layer'] == counts['layer']
                        and abs(event['dur'] - by_rank[event['args']['rank']]['segment_us'][event['name']]) < 0.002
                        for event in spans),
                'coarse Perfetto spans differ from the aligned counts')
        payload = dict(capture_id=manifest['id'], step=counts['step'], layer=counts['layer'],
                       epoch=epoch, alignment=counts['alignment'], ranks=details)
        outputs.append((stem, payload, overlay(coarse, details)))
    require(sorted(selected) == sorted(fine['epochs']) and selected, 'missing selected layer reports')
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture_dir', type=Path)
    parser.add_argument('deliver_dir', type=Path)
    parser.add_argument('--require-streamed-blocks', action='store_true')
    parser.add_argument('--require-batch-claims', action='store_true')
    args = parser.parse_args()
    try:
        outputs = analyze(args.capture_dir, args.deliver_dir, args.require_streamed_blocks,
                          args.require_batch_claims)
    except (KeyError, OSError, ValueError) as exc:
        parser.exit(1, f'pull/GEMM trace: {exc}\n')
    for stem, report, trace in outputs:
        for suffix, payload in (('detail.json', report), ('detail.perfetto.json', trace)):
            path = args.deliver_dir / f'{stem}.{suffix}'
            path.write_text(json.dumps(payload, separators=(',', ':')) + '\n')
            print(path)


if __name__ == '__main__':
    main()
