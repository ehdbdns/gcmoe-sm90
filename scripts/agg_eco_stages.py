"""Average the per-step x per-rank ECO timeline, rejecting mixed rows.

Reads `eco_trace.ranks[]` from a run json and prints the plan cost breakdown.

Why the validation: the timeline row is keyed by the caller's epoch, and each
epoch is one MoE-layer call. 40 layers are 40 rows per engine step. A row is
only trusted when its own timestamps are self-consistent --
    plan_wake <= plan_start <= counts_seen <= static_done <= coldwarm_done
              <= auction_done <= plan_end
-- which also rejects a row that two calls wrote into.

usage: agg_eco_stages.py runs/<label>.json [more.json ...]
"""
import json
import statistics
import sys

TL = ['step_tag', 'plan_wake', 'plan_start', 'counts_seen', 'static_done',
      'coldwarm_done', 'auction_done', 'plan_end', 'push_begin', 'push_end',
      'pull_norep_begin', 'pull_norep_end', 'pull_rep_begin', 'pull_rep_end',
      'norep_wait_plan', 'role2_first_task', 'starve_first', 'first_rep_pull']
I = {k: i for i, k in enumerate(TL)}
# `plan_start` may precede `plan_wake`: the plan CTA is resident and spins
# into call N just before dispatch publishes the wake token, so that pair is
# deliberately NOT required to be ordered.
PLAN_PATH = ('plan_start', 'counts_seen', 'static_done', 'coldwarm_done',
             'auction_done', 'plan_end')


def us(a, b):
    """Same-call difference in MICROSECONDS; None if unusable.

    `_eco_tl_write` already stores `globaltimer_ns // 1000`, i.e. microseconds,
    so the raw difference IS the duration in us -- no further scaling.
    The value is an int32, so past ~36 minutes of uptime it wraps negative; two
    stamps of the SAME call wrap together, so the plain difference stays exact.
    `g < 0` therefore means one side belongs to a different call, and
    `g >= 20000` us is longer than any single MoE call (a mix-up too).
    """
    if a == 0 or b == 0:
        return None
    g = b - a
    return float(g) if 0 <= g < 20_000 else None


def ok_row(row):
    if len(row) < len(TL) or not any(row[1:]):
        return False
    prev = None
    for name in PLAN_PATH:
        v = row[I[name]]
        if prev is not None and (v - prev) < 0:      # same-call wrap is fine only upward
            return False
        prev = v
    return True


def metrics(rows):
    def mean(f):
        v = [x for x in (f(r) for r in rows) if x is not None]
        return statistics.mean(v) if v else None

    g = lambda r, k: r[I[k]]
    return {
        'plan_exec': mean(lambda r: us(g(r, 'plan_start'), g(r, 'plan_end'))),
        'plan_wait_counts': mean(lambda r: us(g(r, 'plan_start'), g(r, 'counts_seen'))),
        'plan_static': mean(lambda r: us(g(r, 'counts_seen'), g(r, 'static_done'))),
        'plan_coldwarm': mean(lambda r: us(g(r, 'static_done'), g(r, 'coldwarm_done'))),
        'plan_auction': mean(lambda r: us(g(r, 'coldwarm_done'), g(r, 'auction_done'))),
        'plan_publish': mean(lambda r: us(g(r, 'auction_done'), g(r, 'plan_end'))),
        'push': mean(lambda r: us(g(r, 'push_begin'), g(r, 'push_end'))),
        'pull_norep': mean(lambda r: us(g(r, 'pull_norep_begin'), g(r, 'pull_norep_end'))),
        'pull_rep': mean(lambda r: us(g(r, 'pull_rep_begin'), g(r, 'pull_rep_end'))),
        # negative => the plan finished while the push was still running
        'plan_minus_push': mean(lambda r: us(g(r, 'push_begin'), g(r, 'plan_end'))),
        'norep_then_plan_wait': mean(lambda r: us(g(r, 'pull_norep_end'), g(r, 'pull_rep_begin'))),
        # slot 14 is itself a DURATION (us), not a timestamp -- no subtraction.
        'norep_done_wait_plan': mean(lambda r: float(g(r, 'norep_wait_plan'))
                                     if 0 < g(r, 'norep_wait_plan') < 20_000 else None),
        # first consumer spin -> first replica expert pulled  (the starvation)
        'starve': mean(lambda r: us(g(r, 'starve_first'), g(r, 'first_rep_pull'))),
    }


def main(paths):
    for path in paths:
        d = json.load(open(path))
        print(f'\n== {d.get("label")}  {d.get("ms_per_step_median"):.2f} ms/step  '
              f'IR={d.get("ir"):.4f}  tokens/rank={d.get("tokens_per_rank")}')
        ranks = (d.get('eco_trace') or {}).get('ranks') or []
        if not ranks:
            print('   no eco_trace (GC_MOE_ECO_TRACE=1?)')
            continue
        per_rank = {}
        for r in ranks:
            rows = r.get('tl_rows') or []
            good = [x for x in rows if ok_row(x)]
            per_rank[r['rank']] = (len(rows), len(good), metrics(good))
        print(f'{"rank":>4} {"rows":>5} {"ok":>5} ' + ' '.join(f'{k[:9]:>9}' for k in per_rank[0][2]))
        for rk, (tot, good, m) in sorted(per_rank.items()):
            cells = ' '.join(f'{(v if v is not None else float("nan")):>9.1f}' for v in m.values())
            print(f'{rk:>4} {tot:>5} {good:>5} {cells}')
        print(f'{"mean":>4} {"":>5} {"":>5} ' + ' '.join(
            f'{(statistics.mean([mm[k] for _, _, mm in per_rank.values() if mm[k] is not None]) if any(mm[k] is not None for _,_,mm in per_rank.values()) else float("nan")):>9.1f}'
            for k in per_rank[0][2]))
        print('   (all values in microseconds, per MoE-layer call; -1/nan = not observed)')


if __name__ == '__main__':
    main(sys.argv[1:] or ['runs/E_eco_v2.json'])
