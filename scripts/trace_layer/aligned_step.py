#!/usr/bin/env python3
"""Produce post-EPLB Perfetto tracks and per-rank counts from a single capture."""
import argparse
from bisect import bisect_right
import json
import math
from pathlib import Path
from statistics import mean


# Independent GPU stamps are CTA entry/exit observations. CUPTI reports the
# kernel launch span, so a small difference is normal even with correct pairing.
ENTRY_TOLERANCE_US = 20
UNSET = 0xFFFFFFFF
RANKS = 8
ROLES = ('dispatch', 'shared', 'routed', 'plan', 'combine')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_calls(trace, layers, dispatch_grid, rank, eco=True):
    roles = [r for r in ROLES if eco or r != 'plan']
    kernels = {name: [] for name in roles}
    for index, event in enumerate(trace['traceEvents']):
        name = event.get('name', '').lower()
        if event.get('cat') != 'kernel' or 'gc_moe' not in name:
            continue
        grid, _, _ = event['args']['grid']
        block, _, _ = event['args']['block']
        if (grid, block) == (1, 32) and eco and dispatch_grid == 20:
            role = 'admission'
            if role not in kernels:
                roles.append(role)
                kernels[role] = []
        elif (grid, block) == (1, 512):
            role = 'plan'
        elif block == 256:
            role = 'dispatch' if grid == dispatch_grid else 'shared'
        elif block == 384:
            role = 'routed'  # resolved against combine below
        else:
            raise ValueError(f'rank {rank}: unsupported GC MoE geometry {grid}x{block}')
        ts, duration = event['ts'], event['dur']
        require(math.isfinite(ts) and math.isfinite(duration) and duration > 0,
                f'rank {rank}: invalid timestamp at row {index}')
        kernels[role].append(dict(row=index, ts=ts, dur=duration, end=ts + duration,
                                  grid=grid, block=block, name=event['name']))
    grids = sorted({event['grid'] for event in kernels['routed']})
    require(len(grids) == 2, f'rank {rank}: expected distinct routed/combine grids, got {grids}')
    kernels['combine'] = [event for event in kernels['routed'] if event['grid'] == grids[1]]
    kernels['routed'] = [event for event in kernels['routed'] if event['grid'] == grids[0]]
    require(all(len(events) == layers for events in kernels.values()),
            f'rank {rank}: expected {layers} of each role, got '
            f'{ {role: len(events) for role, events in kernels.items()} }')
    # Combine follows all other kernels on this call. Its end delimits a
    # complete layer. Assign each raw event to that time window, then require
    # precisely one of each role. Kineto's array need not be timestamp sorted.
    combined_ends = sorted(event['end'] for event in kernels['combine'])
    calls = [{} for _ in combined_ends]
    for role, events in kernels.items():
        for event in events:
            index = bisect_right(combined_ends, event['ts'])
            require(index < layers and event['end'] <= combined_ends[index]
                    and role not in calls[index],
                    f'rank {rank}: overlapping or incomplete call at raw row {event["row"]}')
            calls[index][role] = event
    require(all(set(call) == set(roles) for call in calls),
            f'rank {rank}: incomplete layer windows')
    return calls


def read_worker_trace(capture_dir, capture, dispatch_grid, rank):
    trace = json.loads((capture_dir / f'rank{rank}.json').read_text())
    require(trace.get('gcmoe_capture') == {'id': capture['id'], 'rank': rank},
            f'rank {rank}: torch trace is from another capture')
    return read_calls(trace, capture['layers'], dispatch_grid, rank, capture.get('arm') != 'rr'), trace.get('baseTimeNanoseconds')


def clock_residuals(call, dispatch, pull_end, routed, combine):
    ts = call['dispatch']['ts']
    return {
        'routed_entry_us': (routed - dispatch) - (call['routed']['ts'] - ts),
        'combine_entry_us': (combine - dispatch) - (call['combine']['ts'] - ts),
        'last_pull_vs_dispatch_end_us': pull_end - dispatch - call['dispatch']['dur'],
    }


def resolve_epochs(capture_dir, capture, dispatch_grid):
    """Select the unique tagged device window corroborated by all worker traces."""
    layers, lower = capture['layers'], capture['first_epoch']
    workers = capture['ranks']
    require(len(workers) == RANKS and {worker['rank'] for worker in workers} == set(range(RANKS)),
            'expected ranks 0–7')
    windows = {}
    for worker in workers:
        rank = worker['rank']
        calls, _ = read_worker_trace(capture_dir, capture, dispatch_grid, rank)
        rows = {row['epoch']: row for row in worker['calls']
                if len(row['stamps']) == 32 and row['stamps'][0] == row['epoch']}
        eligible = {}
        for start in range(lower, worker['next_epoch'] - layers + 1):
            window = [rows.get(start + offset) for offset in range(layers)]
            if any(row is None for row in window):
                continue
            # Align by relative dispatch times, since the device and CUPTI
            # clocks have different origins. Include the whole step so an
            # earlier flush with similar individual layer shapes cannot pass.
            offsets = []
            for call, row in zip(calls, window):
                dispatch, pull_end, routed, combine = (
                    row['stamps'][slot] & UNSET for slot in (10, 5, 14, 12))
                if any(stamp in (0, UNSET) for stamp in (dispatch, pull_end, routed, combine)):
                    break
                residuals = clock_residuals(call, dispatch, pull_end, routed, combine)
                if any(abs(value) > ENTRY_TOLERANCE_US for value in residuals.values()):
                    break
                offsets.append(call['dispatch']['ts'] - dispatch)
            if len(offsets) == layers and max(offsets) - min(offsets) <= 2 * ENTRY_TOLERANCE_US:
                eligible[start] = window
        windows[rank] = eligible

    common = set.intersection(*(set(items) for items in windows.values()))
    candidates = {rank: sorted(items) for rank, items in windows.items()}
    require(len(common) == 1,
            f'expected one shared device epoch window, found {sorted(common)}; candidates={candidates}')
    first = common.pop()
    aligned = [dict(worker, next_epoch=first + layers, calls=windows[worker['rank']][first])
               for worker in workers]
    return dict(capture, first_epoch=first, ranks=aligned,
                device_snapshot_epochs={
                    'first': lower, 'next_by_rank': {w['rank']: w['next_epoch'] for w in workers}},
                epoch_candidates_by_rank=candidates)


def layer_timing(call, stamps, has_work, context):
    d, s, r, p, c = (call.get(role) for role in ROLES)
    dispatch, push_end, pull, pull_end, routed, first_claim, combine = (
        stamps[i] & UNSET for i in (10, 3, 4, 5, 14, 15, 12))
    ordered = (dispatch, push_end, pull, pull_end)
    require(all(value not in (0, UNSET) for value in (*ordered, routed, combine))
            and tuple(sorted(ordered)) == ordered, f'{context}: invalid dispatch stamps')
    require(c['ts'] >= max(v['end'] for k,v in call.items() if k != 'combine'),
            f'{context}: combine overlaps a producer')
    if stamps[22] >= 32:
        require(r['ts'] >= s['end'], f'{context}: routed starts before shared completes')
    residuals = clock_residuals(call, dispatch, pull_end, routed, combine)
    require(all(abs(value) <= ENTRY_TOLERANCE_US for value in residuals.values()),
            f'{context}: device and CUPTI disagree: {residuals}')
    pull_start = d['ts'] + pull - dispatch
    require(d['ts'] <= pull_start <= d['end'], f'{context}: pull starts outside dispatch')
    if has_work:
        require(first_claim not in (0, UNSET) and pull <= first_claim <= combine,
                f'{context}: missing or invalid first valid CTA claim')
        require(first_claim >= routed - 2, f'{context}: first claim predates routed entry')
        # Anchor each offset to its own CUPTI kernel. Two independently
        # measured entry-to-entry residuals above guard against a wrong join.
        raw_gemm = r['ts'] + (first_claim - routed)
        require(raw_gemm <= r['end'], f'{context}: first claim outside routed kernel')
        gemm_start = max(r['ts'], raw_gemm)
    else:
        require(first_claim == UNSET, f'{context}: claim on rank with no routed work')
        gemm_start = raw_gemm = r['end']
    return {
        'count exchange': (d['ts'], pull_start, 'dispatch'),
        'pull token': (pull_start, d['end'], 'dispatch'),
        'shared L1+L2': (s['ts'], s['end'], 'shared'),
        **({'ECO plan': (p['ts'], p['end'], 'plan')} if p else {}),
        'wait token': (r['ts'], gemm_start, 'routed'),
        'GEMM': (gemm_start, r['end'], 'routed'),
        'combine': (c['ts'], c['end'], 'combine'),
    }, dict(residuals, shared_minus_dispatch_us=s['ts'] - d['ts'],
            routed_minus_shared_end_us=r['ts'] - s['end'],
            combine_minus_routed_end_us=c['ts'] - r['end'],
            gemm_start_clamp_us=gemm_start - raw_gemm,
            first_claim_after_routed_entry_us=first_claim - routed if has_work else None,
            pull_after_first_claim_us=pull_end - first_claim if has_work else None,
            drawn_pull_gemm_overlap_us=max(0.0, min(d['end'], r['end'])
                                           - max(pull_start, gemm_start)))


def measure_plan(ranks):
    """Compare rank shares within one layer and epoch (signed error in percentage points)."""
    for row in ranks:
        call = row['raw_kernels']
        root = min((k for k in ('dispatch', 'shared', 'plan', 'admission') if k in call), key=lambda role: call[role]['ts'])
        start, end = call[root]['ts'], call['combine']['end']
        require(start <= call['routed']['ts'] and end > start, 'invalid MoE graph span')
        row.update(graph_root=root, graph_start_us=start, combine_end_us=end, moe_us=end - start)
    score_sum = sum(r['plan_cost'] for r in ranks)
    time_sum = sum(r['moe_us'] for r in ranks)
    require(score_sum > 0, 'zero plan score sum')
    for row in ranks:
        predicted = row['plan_cost'] / score_sum
        actual = row['moe_us'] / time_sum
        row.update(plan_share=predicted, actual_share=actual,
                   share_error_pp=100 * (predicted - actual))
    errors = [abs(r['share_error_pp']) for r in ranks]
    return dict(mae_pp=mean(errors), max_abs_error_pp=max(errors))


def build(capture_dir, layers, dispatch_grid, require_overlap=False):
    capture = json.loads((capture_dir / 'capture.json').read_text())
    require(capture.get('gemm_start') == 'first_valid_cta_claim',
            'capture must record device-side first valid CTA claims')
    n_layers = capture['layers']
    first = capture['first_epoch']
    require(1 <= n_layers <= 256 and first > 0 and capture['decode_tokens'] > 0,
            'invalid capture metadata')
    require(all(0 <= layer < n_layers for layer in layers), 'requested layer outside capture')
    workers = {worker['rank']: worker for worker in capture['ranks']}
    require(len(capture['ranks']) == RANKS and set(workers) == set(range(RANKS)), 'expected ranks 0–7')
    topk = {worker['topk'] for worker in workers.values()}
    require(len(topk) == 1 and min(topk) > 0, 'ranks disagree on top-k')
    topk = topk.pop()
    traces, bases = {}, {}
    for rank, worker in sorted(workers.items()):
        prefix = f'rank {rank}'
        require(worker['next_epoch'] == first + n_layers
                and [call['epoch'] for call in worker['calls']] == list(range(first, first + n_layers)),
                f'{prefix}: wrong epoch interval')
        traces[rank], bases[rank] = read_worker_trace(capture_dir, capture, dispatch_grid, rank)
        for layer, call in enumerate(worker['calls']):
            require(call['stamps'][0] == first + layer and len(call['stamps']) == 32,
                    f'{prefix}: wrong device tag at layer {layer}')

    absolute = all(isinstance(base, int) and base > 0 for base in bases.values())
    base = min(bases.values()) if absolute else None
    for layer in sorted(set(layers)):
        epoch = first + layer
        rows = [workers[rank]['calls'][layer] for rank in range(RANKS)]
        input_tokens = [row['stamps'][23] for row in rows]
        logical = capture['decode_tokens']
        require(input_tokens == [logical] * RANKS or sum(input_tokens) == logical,
                f'epoch {epoch}: unexpected input token counts {input_tokens}')
        replication = sum(input_tokens) // logical
        require(sum(sum(row['owned']) for row in rows) == sum(input_tokens) * topk,
                f'epoch {epoch}: routed token counts do not conserve top-k routes')
        plan = rows[0]['stamps'][24:32]
        require(capture.get('arm') == 'rr' or all(value >= 0 for value in plan) and sum(plan) > 0
                and all(row['stamps'][24:32] == plan for row in rows),
                f'epoch {epoch}: missing or divergent ECO scores')

        result = dict(capture_id=capture['id'], label=capture['label'], step=capture['step'],
                      layer=layer, epoch=epoch, decode_tokens=logical, topk=topk,
                      input_replication=replication, gemm_start=capture['gemm_start'],
                      alignment='Kineto baseTimeNanoseconds' if absolute else 'rank-local dispatch',
                      kineto_base_time_ns=bases, ranks=[])
        offsets = {rank: (bases[rank] - base) / 1000 if absolute
                   else -traces[rank][layer]['dispatch']['ts'] for rank in range(RANKS)}
        origin = min(min(event['ts'] for event in traces[rank][layer].values()) + offsets[rank]
                     for rank in range(RANKS))
        events = []
        for rank, row in enumerate(rows):
            context = f'rank {rank}, step {capture["step"]}, layer {layer}, epoch {epoch}'
            counts, block_m = row['owned'], row['stamps'][22]
            require(block_m in (8, 16, 32, 64) and counts and all(v >= 0 for v in counts),
                    f'{context}: invalid block M or slot counts')
            call = traces[rank][layer]
            segments, checks = layer_timing(call, row['stamps'], bool(sum(counts)), context)
            metrics = dict(rank=rank, tokens=sum(counts) / replication,
                           routed_rows=sum(counts), m_blocks=sum((q + block_m - 1) // block_m for q in counts),
                           block_m=block_m, plan_cost=None if capture.get('arm') == 'rr' else plan[rank],
                           routed_us=call['routed']['dur'],
                           dispatch_to_routed_end_us=call['routed']['end'] - call['dispatch']['ts'])
            if capture.get('stream_release_stamps') and capture.get('gemm_start') == 'first_valid_cta_claim':
                for col, key in ((8, 'first_l1_batch_us'), (9, 'first_full_l1_batch_us')):
                    stamp = row['stamps'][col] & UNSET
                    expected = bool(sum(counts)) if col == 8 else max(counts) >= block_m
                    present = stamp != UNSET
                    require(present == expected, f'{context}: {key} coverage differs from owned slots')
                    delay = (stamp - row['stamps'][10]) & UNSET
                    require(not present or delay <= call['dispatch']['dur'] + ENTRY_TOLERANCE_US,
                            f'{context}: release timestamp outside dispatch')
                    metrics[key] = delay if present else None
                    if present:
                        events.append(dict(ph='i', s='t', cat='dispatch', name=key, pid=rank, tid=1,
                                           ts=call['dispatch']['ts'] + offsets[rank] - origin + delay))
            if capture.get('dispatch_transport') in ('source_push', 'source_inbox'):
                segments['local route preparation'] = segments.pop('count exchange')
                segments['source push and destination completion'] = segments.pop('pull token')
            result['ranks'].append(dict(metrics, slots_owned=counts, raw_kernels=call,
                                        raw_stamps=row['stamps'], checks=checks,
                                        segment_us={name: end - start
                                                    for name, (start, end, _) in segments.items()}))
            events.append(dict(ph='M', name='process_name', pid=rank, tid=0,
                               args={'name': f'Rank {rank}'}))
            for track, role in enumerate(ROLES, start=1):
                events.append(dict(ph='M', name='thread_name', pid=rank, tid=track,
                                   args={'name': role}))
            if 'admission' in call:
                gate = call['admission']
                events.append(dict(ph='X', name='wait for resident plan CTA', cat='admission',
                                   pid=rank, tid=6, ts=gate['ts'] + offsets[rank] - origin,
                                   dur=gate['dur']))
            for name, (start, end, role) in segments.items():
                # [PERFETTO-US] Perfetto's Chrome-JSON importer parses ts/dur as
                # MICROSECONDS by default. `displayTimeUnit` only has an effect
                # for the literal "ms" (which multiplies by 1000); any other
                # value, including "ns", is ignored and the numbers are still
                # read as microseconds. So: keep the raw microsecond values and
                # emit no displayTimeUnit at all. Writing ns-scaled numbers here
                # made a 604 us GEMM show up as 604 ms.
                events.append(dict(ph='X', name=name, cat=role, pid=rank, tid=ROLES.index(role) + 1,
                                   ts=start + offsets[rank] - origin, dur=end - start,
                                   args=dict(metrics, step=capture['step'], layer=layer, epoch=epoch,
                                             trace_row=call[role]['row'])))
        result['pull_gemm_overlap_ranks'] = [
            rank['rank'] for rank in result['ranks']
            if (rank['checks']['pull_after_first_claim_us'] or 0) > 0
            and rank['checks']['drawn_pull_gemm_overlap_us'] > 0]
        require(not require_overlap or result['pull_gemm_overlap_ranks'],
                f'epoch {epoch}: no rank has both measured and visible pull/GEMM overlap')
        if capture.get('arm') != 'rr':
            result['graph_plan_accuracy'] = measure_plan(result['ranks'])
        yield result, dict(traceEvents=events)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture_dir', type=Path)
    parser.add_argument('output_dir', type=Path)
    parser.add_argument('--layers', nargs='+', type=int, required=True)
    parser.add_argument('--dispatch-grid', type=int, default=24)
    parser.add_argument('--require-pull-overlap', action='store_true',
                        help='require measured and visible pull/GEMM overlap on at least one rank per layer')
    args = parser.parse_args()
    try:
        outputs = list(build(args.capture_dir, args.layers, args.dispatch_grid, args.require_pull_overlap))
    except (KeyError, ValueError, OSError) as exc:
        parser.exit(1, f'aligned trace: {exc}\n')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for report, perfetto in outputs:
        stem = f'step{report["step"]:03d}_layer{report["layer"]:02d}_epoch{report["epoch"]}'
        for suffix, payload in (('counts', report), ('perfetto', perfetto)):
            path = args.output_dir / f'{stem}.{suffix}.json'
            path.write_text(json.dumps(payload, indent=2) + '\n')
            print(path)


if __name__ == '__main__':
    main()
