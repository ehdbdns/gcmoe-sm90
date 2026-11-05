"""Aggregate the raw timeline CSV into the three tables the task asks for.

Reads `eco_tl.csv` (one row per (rank, step), absolute microseconds) and prints:
  1. per-rank means        -- shows WHICH rank is slow
  2. global means          -- the headline numbers
  3. distribution P50/P90/P99 -- the tail

Rows whose step_tag is <= WARMUP are dropped (warm-up is discarded host-side,
never by shortening the measurement window in the kernel).

Key derived checks:
  plan covered by push   : plan_end   - push_end     (negative => plan hid inside the push)
  starvation             : pull_rep_begin - pull_norep_end
  no single-copy starvation: plan_ready_after_norep should be <= 0
"""
import argparse
import csv
import statistics as st


def load(path, warmup):
    rows = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            row = {k: int(v) for k, v in r.items()}
            if row['step_tag'] <= warmup:
                continue
            rows.append(row)
    return rows


def derive(r):
    def d(a, b):
        return r[a] - r[b] if (r[a] and r[b]) else 0
    return dict(
        rank=r['rank'], step=r['step'],
        # NOTE: `plan_start` is "the plan CTA entered _eco_plan", which for a
        # resident CTA happens right after the PREVIOUS call finished, i.e.
        # BEFORE this call's wake. So the honest "how long did this call's plan
        # wait for counts" is measured from the WAKE, not from plan_start.
        # plan_exec must also start at the wake: `plan_start` belongs to the
        # previous call for a resident plan CTA (see below).
        plan_exec=d('plan_end', 'plan_wake'),
        plan_wait_counts=d('plan_counts_seen', 'plan_wake'),
        plan_static=d('plan_static_done', 'plan_counts_seen'),
        plan_coldwarm=d('plan_coldwarm_done', 'plan_static_done'),
        plan_auction=d('plan_auction_done', 'plan_coldwarm_done'),
        plan_publish=d('plan_end', 'plan_auction_done'),
        # negative: the resident plan CTA entered the function before this
        # call's wake (i.e. right after the previous call finished)
        plan_start_minus_wake=d('plan_start', 'plan_wake'),
        push=d('push_end', 'push_begin'),
        pull_norep=d('pull_norep_end', 'pull_norep_begin'),
        pull_rep=d('pull_rep_end', 'pull_rep_begin'),
        plan_vs_push_end=d('plan_end', 'push_end'),
        plan_wake_vs_push_end=d('plan_wake', 'push_end'),
        starvation=d('pull_rep_begin', 'pull_norep_end'),
        norep_ready_after=d('plan_ready_after_norep', 'pull_norep_end'),
        role2_first=d('role2_first_task', 'push_end'),
        # ★ THE starvation window the task asks for:
        #   from the first consumer spin to the first replica expert done
        starvation_window=d('first_replica_pull_done', 'role2_first_spin'),
        role2_first_spin_rel=d('role2_first_spin', 'plan_wake'),
        first_rep_done_rel=d('first_replica_pull_done', 'plan_wake'),
    )


def table(rows, key):
    by = {}
    for r in rows:
        by.setdefault(r[key], []).append(r)
    return by


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('csv')
    ap.add_argument('--warmup', type=int, default=0)
    a = ap.parse_args()
    raw = load(a.csv, a.warmup)
    if not raw:
        raise SystemExit('no rows after warmup filter')
    der = [derive(r) for r in raw]
    keys = [k for k in der[0] if k not in ('rank', 'step')]

    print(f'rows={len(der)}  ranks={sorted({d["rank"] for d in der})}  '
          f'steps={min(d["step"] for d in der)}..{max(d["step"] for d in der)}  '
          f'(us; warmup<= {a.warmup} dropped)')

    print('\n=== 1) per-rank mean (us) ===')
    by = table(der, 'rank')
    hdr = f'{"metric":26s}' + ''.join(f'{r:>9}' for r in sorted(by))
    print(hdr)
    for k in keys:
        line = f'{k:26s}'
        for r in sorted(by):
            line += f'{st.mean([d[k] for d in by[r]]):9.1f}'
        print(line)

    print('\n=== 2) global mean (us) ===')
    for k in keys:
        print(f'{k:26s}{st.mean([d[k] for d in der]):9.1f}')

    print('\n=== 3) distribution (us) ===')
    print(f'{"metric":26s}{"P50":>9}{"P90":>9}{"P99":>9}{"max":>9}')
    for k in keys:
        vs = sorted(d[k] for d in der)
        def q(p):
            i = min(len(vs) - 1, int(round(p * (len(vs) - 1))))
            return vs[i]
        print(f'{k:26s}{q(.5):9.1f}{q(.9):9.1f}{q(.99):9.1f}{max(vs):9.1f}')

    wake_ok = sum(1 for d in der if d['plan_wake_vs_push_end'] < 0)
    print(f'\n=== verdict ===')
    print(f'plan_wake before push_end : {wake_ok}/{len(der)} rows '
          f'({"OK" if wake_ok == len(der) else "FAIL"})')
    print(f'plan_end before push_end  : '
          f'{sum(1 for d in der if d["plan_vs_push_end"] < 0)}/{len(der)}')
    print(f'starvation (rep_begin - norep_end) mean '
          f'{st.mean([d["starvation"] for d in der]):.1f} us')


if __name__ == '__main__':
    main()
