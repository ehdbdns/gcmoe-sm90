"""[ECO-FINE] One rank, ONE MoE call: every trace slot on a common 0-us axis.

The three tables use different epoch counters (fine_tl tags each event with
`header[0]`, eco_tl keeps only the last 256 rows keyed by `epoch % 256`, and
pull_tl likewise), so they are joined by ABSOLUTE TIME, not by epoch:
a row belongs to this call when its own stamps fall inside
[origin, origin + call_span].

  eco_tl    21 slots -- per-call stage stamps (min/max/store)
  pull_tl   10 slots -- per-call pull/combine stamps
  fine_tl   8 words  -- event log (slot_begin/end, spin, m-block, ...)
  eco_trace 40 words -- lifetime accumulators (not timestamps)

usage: dump_all_slots.py runs/FINED.json [rank] [epoch]
"""
import json, sys

ECO_TL = [
 (0,'step_tag','epoch id (NOT a time)'),
 (1,'plan_wake','role1 publishes eco_ready[0]: plan may start'),
 (2,'plan_start','plan CTA enters _eco_plan'),
 (3,'counts_seen','plan finished waiting for the eco_counts row'),
 (4,'static_done','static tables + D_e + k/rem + single-copy fixing done'),
 (5,'coldwarm_done','cold or warm start finished'),
 (6,'auction_done','auction rounds finished'),
 (7,'plan_end','quota/quota_start/warm state published'),
 (8,'push_begin','role1 starts the push loop'),
 (9,'push_end','push grid rendezvous done'),
 (10,'pull_norep_begin','FIRST single-copy slot starts pulling'),
 (11,'pull_norep_end','ALL single-copy slots finished'),
 (12,'pull_rep_begin','FIRST replica slot obtained the plan'),
 (13,'pull_rep_end','ALL replica slots finished'),
 (14,'norep_wait_plan','DURATION us: single-copy done -> plan ready'),
 (15,'role2_first_task','first role-2 CTA got work'),
 (16,'starve_first','first role-2 CTA finds no published work'),
 (17,'first_rep_pull','first REPLICA expert published expert_ready'),
 (18,'first_norep_pull','first SINGLE-COPY expert published expert_ready'),
 (19,'role2_first_slot','role-2 saw active_ready for its first slot'),
 (20,'role2_first_data','expert data readable: first GEMM can start'),
]
PULL_TL = [
 (0,'epoch_tag','epoch id (NOT a time)'),
 (1,'wake','dispatch published the plan wake token'),
 (2,'push_start','push loop entry'),
 (3,'push_end','push grid rendezvous done'),
 (4,'pull_start','all CTAs past the cross-rank wait'),
 (5,'pull_end','LAST local slot finished'),
 (6,'n_single','count of single-copy slots (NOT a time)'),
 (7,'n_rep','count of replica slots (NOT a time)'),
 (8,'combine_start','first role-3 CTA enters combine'),
 (9,'combine_end','last role-3 CTA bumped the epoch'),
]
FINE = {
 1:('slot_begin',   'CTA enters a slot\'s pull',   ('gslot','local_slot','cta','ncopies')),
 2:('slot_end',     'slot finished pulling',       ('gslot','local_slot','cta','owned','lo')),
 3:('wait_plan',    'replica slot waits on plan',  ('gslot','phase 0=begin 1=released')),
 4:('spin_edge',    'consumer STARTS a spin streak',('cta','want_slot','seen_active')),
 5:('got_work',     'consumer got a task',         ('cta','slot','expert')),
 6:('mblock_begin', 'GEMM starts one M block',     ('cta','expert','start','valid')),
 7:('mblock_end',   'GEMM finished one M block',   ('cta','expert','start','valid')),
 12:('copies_raw',  'copies[expert][0..2]',        ('gslot','c0','c1','c2')),
 13:('copies_raw2', 'copies[expert][3..5]',        ('gslot','c3','c4','c5')),
}

def timed_rows(rows, idxs):
    """Rows whose timestamp columns fall inside [lo, hi]."""
    return [x for x in rows if any(x[i] for i in idxs)]

def main():
    path = sys.argv[1] if len(sys.argv)>1 else 'runs/FINED.json'
    rank = int(sys.argv[2]) if len(sys.argv)>2 else 0
    want = int(sys.argv[3]) if len(sys.argv)>3 and sys.argv[3].isdigit() else None
    d = json.load(open(path))
    r = d['eco_trace']['ranks'][rank]

    fr = [x for x in (r.get('fine_rows') or []) if x[1]]
    eps = sorted(set(x[6] for x in fr))
    ep = want if want is not None else eps[0]
    E = [x for x in fr if x[6]==ep]
    t_end = max((x[0] for x in E if x[1]==7), default=None)
    if t_end: E = [x for x in E if x[0] <= t_end]
    t0, t1 = min(x[0] for x in E), max(x[0] for x in E)
    lo, hi = t0 - 200, t1 + 200          # tolerate stamps just outside the span

    # join eco_tl / pull_tl rows by ABSOLUTE TIME
    TS_ECO = (1,2,3,4,5,6,7,8,9,10,11,12,13,15,16,17,18,19,20)
    TS_PULL = (1,2,3,4,5,8,9)
    tl_rows = [x for x in (r.get('tl_rows') or [])
               if any(x) and any(lo <= x[i] <= hi for i in TS_ECO if i < len(x))]
    pr_rows = [x for x in (r.get('pull_rows') or [])
               if any(x) and any(lo <= x[i] <= hi for i in TS_PULL if i < len(x))]

    print(f'# {path}  rank {rank}  fine-epoch {ep}   '
          f'{d.get("ms_per_step_median",0):.2f} ms/step')
    print(f'# origin t0 = {t0} us (globaltimer), call span {t1-t0} us')
    print(f'# joined by time: eco_tl rows={len(tl_rows)}  pull_tl rows={len(pr_rows)}')
    print()

    for tag, defs, rows in (('A. eco_tl -- per-call stage stamps', ECO_TL, tl_rows),
                            ('B. pull_tl -- per-call pull/combine stamps', PULL_TL, pr_rows)):
        print('='*106); print(tag); print('='*106)
        print(f"{'slot':>4} {'name':<20}{'abs_us':>16}{'rel_us':>9}   meaning")
        if not rows:
            print('   (no row whose stamps land inside this call)')
        for row in rows:
            print(f'   -- row for the call at abs {min(v for i,v in enumerate(row) if i in TS_ECO+TS_PULL and v)} --')
            for i,name,meaning in defs:
                v = row[i] if i < len(row) else 0
                if i in (0,14,6,7):
                    print(f'{i:>4} {name:<20}{"":>16}{v:>9}   [value] {meaning}')
                elif v:
                    print(f'{i:>4} {name:<20}{v:>16}{v-t0:>9}   {meaning}')
                else:
                    print(f'{i:>4} {name:<20}{"-":>16}{"-":>9}   (not observed)')
        print()

    print('='*106); print('C. fine_tl -- every event in ABSOLUTE order (0 us = first event)'); print('='*106)
    E.sort(key=lambda x:(x[0], x[1], x[2]))
    print(f"{'rel_us':>8}{'abs_us':>16}  {'event':<13}{'id':>6}  args")
    for x in E:
        name, meaning, argc = FINE.get(x[1], (f'type{x[1]}', '', ()))
        a = ' '.join(f'{n}={v}' for n,v in zip(argc, x[3:6]))
        print(f'{x[0]-t0:>8}{x[0]:>16}  {name:<13}{x[2]:>6}  {a}')

    print()
    print('='*106); print('D. eco_trace -- lifetime accumulators (40 words, NOT timestamps)'); print('='*106)
    sl = r.get('slots') or []
    if not sl: print('   (region not allocated: ECO off?)')
    else:
        for i,v in enumerate(sl):
            if v: print(f'  [{i:>2}] {v}')

if __name__ == '__main__':
    main()
