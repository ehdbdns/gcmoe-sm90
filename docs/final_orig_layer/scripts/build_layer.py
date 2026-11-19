#!/usr/bin/env python3
"""Build an 8-rank, 7-segment single-layer timeline from ONE run.

Trusted inputs, all from run label FINAL_V2:
  * prof/final_v2/traces/rank{r}.json  torch-profiler kernel rows.
      - `ts` and `dur` are BOTH microseconds (verified: dispatch.dur~250us
        while ts steps by ~1700us per call).
  * runs/FINAL_V2.json  eco_trace.pull_rows[r][epoch] -- globaltimer us.

Call pairing is done by TRACE ROW NUMBER: one forward call owns 5 consecutive
gc_moe rows (dispatch, shared, routed, plan, combine). Indexing each role's own
list with the same integer is the mistake that invalidated the earlier work.

`pull_tl` is a 256-entry ring, so only the last 256 calls carry kernel stamps.
The default call is therefore 255 (the newest one).

Segment definition (all times relative to the earliest dispatch start of the
call across the 8 ranks, so one shared x-axis):

  count exchange : dispatch.start            -> dispatch.start + ([3]-[10])
  pull token     : dispatch.start + ([3]-[10]) -> dispatch.start + ([5]-[10])
  shared L1+L2   : shared.start -> shared.end
  ECO plan       : plan.start   -> plan.end
  wait token     : routed.start -> min(pull_done_abs, routed.end)
  GEMM           : that instant  -> routed.end
  combine        : combine.start -> combine.end

with pull_done_abs = dispatch.start + ([5]-[10]).
"""
import argparse, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
ROLES = ('dispatch', 'shared', 'routed', 'plan', 'combine')
TAIL = '/tmp/rt'


def load_pulls(run_path):
    """Return, per rank, a dict epoch -> pull_tl row.

    The ring row index is `epoch % 256`, so the only reliable key is the epoch
    TAG written in column 0. Positional indexing breaks as soon as the ring has
    wrapped (which it always has by the time a run ends).
    """
    d = json.load(open(run_path))
    out = []
    for r in d['eco_trace']['ranks']:
        rows = [x for x in (r.get('pull_rows') or []) if x and x[0]]
        out.append({x[0]: x for x in rows})
    meta = {'label': d['label'], 'ms': d['ms_per_step_median'],
            'ir_step': d['ir_step'], 'moe_real': d.get('moe_real'),
            'source': os.path.abspath(run_path),
            'trace_dir': os.path.join(ROOT, 'prof', 'final_v2', 'traces')}
    return out, meta


def call_rows(rank, k):
    """The 5 kernel rows of call `k` on `rank`, from the tail snapshot."""
    snap = json.load(open(os.path.join(TAIL, f'tail{rank}.json')))
    rows, total = snap['rows'], snap['total']
    first = (total - len(rows)) // 5
    rr = rows[(k - first) * 5:(k - first) * 5 + 5]
    assert [x['role'] for x in rr] == list(ROLES), (rank, k, [x['role'] for x in rr])
    return rr


def build(rows_by_rank, pulls, epoch, t0):
    seg, ker, checks = {}, {}, []
    for r in range(8):
        D, S, R, P, C = rows_by_rank[r]
        st = lambda x: x['ts'] - t0
        en = lambda x: x['ts'] + x['dur'] - t0
        p = pulls[r][epoch]
        d0 = st(D)
        d_end = en(D)                       # dispatch kernel's own end
        count_ex = p[3] - p[10]             # routing + count exchange
        # Two DIFFERENT "token readiness" instants, and only one of them is the
        # GEMM start:
        #   [15] = first compute CTA actually HOLDS a task  -> GEMM starts here
        #   [5]  = LAST local slot finished pulling         -> dispatch is done
        # Between them, CTAs that already have work are computing while later
        # slots are still landing. Using [5] as the GEMM start (an earlier
        # version did) invents a wait that does not exist.
        first_task = d0 + (p[15] - p[10])
        pull_done = min(d0 + (p[5] - p[10]), d_end)
        rout_start, rout_end = st(R), en(R)
        # NOTE: `first_task` and `rout_start` come from different clock domains
        # (kernel globaltimer vs CUPTI), so their difference scatters by ~±15 us
        # (measured: -14..+16 over 256 calls, median +3.8). A negative value
        # means "the first task is ready no later than the kernel starts", i.e.
        # there is nothing to wait for. Report the raw value in `checks` and
        # clamp only for drawing.
        gemm_start = min(max(first_task, rout_start), rout_end)
        seg[r] = {
            'count exchange': (d0, d0 + count_ex),
            'pull token': (d0 + count_ex, pull_done),
            'shared L1+L2': (st(S), en(S)),
            'ECO plan': (st(P), en(P)),
            'wait token': (rout_start, gemm_start),
            'GEMM': (gemm_start, rout_end),
            'combine': (st(C), en(C)),
        }
        ker[r] = {'dispatch': (d0, en(D)), 'shared': (st(S), en(S)),
                  'routed': (rout_start, rout_end), 'plan': (st(P), en(P)),
                  'combine': (st(C), en(C))}
        w, g = seg[r]['wait token'], seg[r]['GEMM']
        checks.append({
            'rank': r,
            'shared_start_minus_dispatch': round(st(S) - d0, 3),
            'routed_start_minus_shared_end': round(st(R) - en(S), 3),
            'combine_start_minus_routed_end': round(st(C) - en(R), 3),
            'count_exchange_end_equals_pull_start': round(
                seg[r]['count exchange'][1] - seg[r]['pull token'][0], 9),
            'wait_plus_gemm_minus_routed': round(
                (w[1] - w[0]) + (g[1] - g[0]) - (rout_end - rout_start), 6),
            'first_task_delay_us': round(p[15] - p[10], 1),
            'all_slots_done_us': round(p[5] - p[10], 1),
            'gemm_starts_before_all_slots_done': bool(p[15] < p[5]),
            'gemm_start_minus_routed_start_raw_us': round(first_task - rout_start, 1),
            'dispatch_dur': round(D['dur'], 1),
            'pull_overshoot_clamped_us': round(max(0.0, d0 + (p[5] - p[10]) - d_end), 1),
        })
    return seg, ker, checks


def to_perfetto(seg, k, label, t0):
    ev = []
    for r in range(8):
        tid = 100 + r
        ev.append({'ph': 'M', 'name': 'thread_name', 'pid': 1, 'tid': tid,
                   'args': {'name': f'rank {r}'}})
        for name, (a, b) in seg[r].items():
            ev.append({'ph': 'X', 'pid': 1, 'tid': tid, 'name': name,
                       'ts': round(a, 3), 'dur': round(max(b - a, 0.001), 3),
                       'args': {'rank': r, 'call': k}})
    ev.append({'ph': 'M', 'name': 'process_name', 'pid': 1, 'tid': 0,
               'args': {'name': f'{label} one layer (call {k})'}})
    return {'traceEvents': ev, 'displayTimeUnit': 'us'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default=os.path.join(ROOT, 'runs', 'FINAL_V2.json'))
    ap.add_argument('--outdir', default=os.path.join(ROOT, 'docs', 'final_orig_layer'))
    ap.add_argument('--call', type=int, default=-1)
    ap.add_argument('--name', default=None)
    a = ap.parse_args()
    pulls, meta = load_pulls(a.run)
    if a.call < 0:
        snap = json.load(open(os.path.join(TAIL, 'tail0.json')))
        a.call = (snap['total'] - len(snap['rows'])) // 5 + len(snap['rows']) // 5 - 1
    rows = [call_rows(r, a.call) for r in range(8)]
    t0 = min(x[0]['ts'] for x in rows)
    # The pull_tl ring holds the LAST 256 calls. Its chronological order is the
    # call order, so the ring position of `a.call` is `a.call - first_call`.
    snap0 = json.load(open(os.path.join(TAIL, 'tail0.json')))
    total_calls = snap0['total'] // 5
    tail_calls = len(snap0['rows']) // 5
    first_call = total_calls - tail_calls
    chrono = sorted(pulls[0].values(), key=lambda x: x[10])
    # the ring's LAST 256 calls == the trace's last 256 calls
    pos = len(chrono) - (total_calls - a.call)
    assert 0 <= pos < len(chrono), (pos, len(chrono))
    epoch = chrono[pos][0]                  # the authoritative call id
    print(f'call {a.call} -> ring position {pos} -> epoch {epoch}')
    seg, ker, checks = build(rows, pulls, epoch, t0)
    name = a.name or f'one_layer_call{a.call}'
    os.makedirs(a.outdir, exist_ok=True)
    payload = {'label': meta['label'], 'call': a.call, 'epoch': epoch, 't0_us': t0,
               'row_range': [rows[0][0]['row'], rows[0][-1]['row']],
               'rows_per_rank': [[x['row'] for x in rr] for rr in rows],
               'segments': {str(k): v for k, v in seg.items()},
               'kernels': {str(k): v for k, v in ker.items()},
               'checks': checks, 'source': meta['source'],
               'trace_dir': meta['trace_dir'], 'ms_per_step': meta['ms']}
    json.dump(payload, open(os.path.join(a.outdir, name + '.segments.json'), 'w'), indent=1)
    json.dump(to_perfetto(seg, epoch, meta['label'], t0),
              open(os.path.join(a.outdir, name + '.perfetto.json'), 'w'))
    for c in checks:
        print(c)
    print('row range', payload['row_range'])


if __name__ == '__main__':
    main()
