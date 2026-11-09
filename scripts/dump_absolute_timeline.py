"""[ECO-FINE] Dump one call's events ordered by ABSOLUTE globaltimer.

Every record already carries `globaltimer_us`, so sorting the whole array by
that field merges events from every CTA/slot into one true timeline -- no
per-type reconstruction needed.

usage: dump_absolute_timeline.py runs/FINED.json [rank] [epoch] [--csv out]
"""
import json, sys
from collections import defaultdict

# type -> (name, arg names for a0,a1,a2)
KINDS = {
    1:  ('slot_begin',   ('local_slot', 'cta',    'ncopies')),
    2:  ('slot_end',     ('local_slot', 'owned',  'lo')),
    3:  ('wait_plan',    ('phase',      '-',      '-')),
    4:  ('spin_edge',    ('want_slot',  'seen_active', '-')),
    5:  ('got_work',     ('slot',       'expert', '-')),
    6:  ('mblock_begin', ('expert',     'start',  'valid')),
    7:  ('mblock_end',   ('expert',     'start',  'valid')),
    12: ('copies_raw',   ('copies0',    'copies1','copies2')),
    13: ('copies_raw2',  ('copies3',    'copies4','copies5')),
}

def rows_of(path, rank):
    d = json.load(open(path))
    r = d['eco_trace']['ranks'][rank]
    return d, [x for x in (r.get('fine_rows') or []) if x[1]]

def pick(rows, want=None):
    eps = sorted(set(x[6] for x in rows))
    if want is not None:
        return want, [x for x in rows if x[6] == want]
    return eps[0], [x for x in rows if x[6] == eps[0]]

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else 'runs/FINED.json'
    rank = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    want = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3].isdigit() else None
    d, rows = rows_of(path, rank)
    ep, E = pick(rows, want)
    # keep only events belonging to this call (<= last m-block end)
    t_end = max((x[0] for x in E if x[1] == 7), default=None)
    if t_end:
        E = [x for x in E if x[0] <= t_end]
    E.sort(key=lambda x: (x[0], x[1], x[2]))     # absolute time, then type
    t0 = E[0][0]
    print(f'# {path} rank{rank} epoch{ep}  {len(E)} events  '
          f'absolute window {t0}..{E[-1][0]} us ({E[-1][0]-t0} us)')
    print(f"{'abs_us':>16}{'rel_us':>8}  {'event':<13}{'id':>6}  args")
    for x in E:
        name, an = KINDS.get(x[1], (f'type{x[1]}', ('a0', 'a1', 'a2')))
        args = ' '.join(f'{n}={v}' for n, v in zip(an, x[3:6]) if n != '-')
        print(f'{x[0]:>16}{x[0]-t0:>8}  {name:<13}{x[2]:>6}  {args}')

if __name__ == '__main__':
    main()
