"""Print the ECO plan cost breakdown from a run json's `eco_trace` block.

All arithmetic is host-side and on ONE rank at a time: `globaltimer` is a
device-wide counter, but subtracting two values from different SMs is only
meaningful for coarse gaps, so every number below is (same rank, same call).

    usage: report_eco_stages.py runs/<label>.json
"""
import json
import statistics
import sys

# 30-slot timeline (see scripts/dump_eco_trace.py for who writes what)
SLOT = {
    0: 'role1_start', 1: 'role1_barrier_done', 2: 'role1_counts_ready',
    3: 'plan_start', 4: 'plan_counts_seen', 5: 'plan_published',
    6: 'plan_wait_first', 7: 'plan_seen_last', 8: 'role1_end',
    17: 'plan_static_done', 18: 'plan_coldwarm_done', 19: 'plan_auction_done',
    27: 'replica_wait_plan_acc',
}

# per-step rows: index -> name (see ECO_TL_SLOTS in the impl)
TL = ['step_tag', 'plan_wake', 'plan_start', 'counts_seen', 'static_done',
      'coldwarm_done', 'auction_done', 'plan_end', 'push_begin', 'push_end',
      'pull_norep_begin', 'pull_norep_end', 'pull_rep_begin', 'pull_rep_end',
      'norep_wait_plan', 'role2_first_task', 'starve_first', 'first_rep_pull']


def us(a, b):
    """Same-rank difference in microseconds.

    The rows store `globaltimer_ns / 1000` in an int32, so past ~36 minutes of
    device uptime the value wraps negative. Two stamps from the SAME call wrap
    by the same multiple of 2**32, so their plain difference is still exact as
    long as the gap is small (always true here: microseconds to ~ms).
    -1 means "one side never wrote" (0 is a valid wrapped timestamp, so the
    sentinel is the explicit 0-from-an-unwritten-slot ... which we detect by
    requiring the whole row to be non-trivial instead).
    """
    if a == 0 or b == 0:
        return -1.0
    d = b - a
    return d / 1e3 if d >= 0 else -1.0


def main(path):
    d = json.load(open(path))
    tr = d.get('eco_trace') or {}
    ranks = tr.get('ranks') or []
    print(f'== {d.get("label")} : {d.get("ms_per_step_median")} ms/step  '
          f'IR={d.get("ir")}  tokens/rank={d.get("tokens_per_rank")}')
    if not ranks:
        print('   no eco_trace in this run (GC_MOE_ECO_TRACE!=1?)')
        return

    # ── 30-slot absolute timeline (last completed call, per rank) ──────────
    print('\n-- 30-slot timeline (us, same rank, last call) --')
    hdr = f'{"rank":>4} ' + ' '.join(f'{n[:12]:>12}' for n in
                                     ('counts', 'static', 'coldwarm', 'auction',
                                      'publish', 'plan_tot', 'wait_acc_us'))
    print(hdr)
    for r in ranks:
        s = r.get('slots') or []
        g = lambda i: (s[i] if len(s) > i else 0)
        row = [us(g(3), g(4)), us(g(4), g(17)), us(g(17), g(18)),
               us(g(18), g(19)), us(g(19), g(5)), us(g(3), g(5)),
               g(27) / 1e3 if len(s) > 27 else -1]
        print(f'{r["rank"]:>4} ' + ' '.join(f'{v:>12.1f}' for v in row))

    # ── per-step rows: average over steps, per rank ────────────────────────
    print('\n-- per-step rows (us, mean over steps) --')
    derived = ['plan_exec', 'plan_wait_counts', 'plan_static', 'plan_coldwarm',
               'plan_auction', 'plan_publish', 'push', 'pull_norep', 'pull_rep',
               'plan_hidden_vs_push', 'norep_wait_plan', 'starve']
    per_rank = {}
    for r in ranks:
        rows = r.get('tl_rows') or []
        cur = r.get('tl_step', -1) % 256
        # keep only rows written by completed calls (non-zero timestamps)
        # A row is "written" when any timestamp slot is non-zero. Testing
        # `row[2] > 0` would drop every row once the int32 microsecond clock
        # wraps negative (~36 min of uptime).
        good = [row for row in rows if len(row) >= 18 and any(row[2:])]
        if not good:
            continue
        acc = {k: [] for k in derived}
        for row in good:
            v = dict(zip(TL, row))
            acc['plan_exec'].append(us(v['plan_start'], v['plan_end']))
            acc['plan_wait_counts'].append(us(v['plan_start'], v['counts_seen']))
            acc['plan_static'].append(us(v['counts_seen'], v['static_done']))
            acc['plan_coldwarm'].append(us(v['static_done'], v['coldwarm_done']))
            acc['plan_auction'].append(us(v['coldwarm_done'], v['auction_done']))
            acc['plan_publish'].append(us(v['auction_done'], v['plan_end']))
            acc['push'].append(us(v['push_begin'], v['push_end']))
            acc['pull_norep'].append(us(v['pull_norep_begin'], v['pull_norep_end']))
            acc['pull_rep'].append(us(v['pull_rep_begin'], v['pull_rep_end']))
            acc['plan_hidden_vs_push'].append(us(v['push_begin'], v['plan_end']))
            acc['norep_wait_plan'].append(v['norep_wait_plan'])
            acc['starve'].append(us(v['pull_norep_end'], v['pull_rep_begin']))
        per_rank[r['rank']] = {k: (statistics.mean(x) if x else -1.0)
                               for k, x in acc.items()}
        print(f'rank {r["rank"]:>2} (rows={len(good):>3}, cur={cur}): ' +
              ' '.join(f'{k}={v:.1f}' for k, v in per_rank[r['rank']].items()))

    if per_rank:
        print('\n-- mean across ranks --')
        for k in derived:
            vals = [v[k] for v in per_rank.values() if v[k] >= 0]
            print(f'   {k:<22} {statistics.mean(vals):>9.1f} us' if vals else
                  f'   {k:<22}      n/a')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'runs/eplb_eco_100.json')
