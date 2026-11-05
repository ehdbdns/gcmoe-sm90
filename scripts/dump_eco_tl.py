"""Dump the per-step x per-rank RAW timeline (`eco_tl`) to CSV.

Unlike the 30-slot view (`dump_eco_trace.py`), nothing here is reduced: every
call leaves its own row, every rank keeps its own buffer, so the analysis can
still tell WHICH rank was slow and average over steps afterwards.

Layout: `eco_tl[step % 256][slot]`, int32 microseconds since boot.

Slot map
  0  step_tag            epoch (ordering / validation)
  1  plan_wake           role 1 released eco_ready[0]
  2  plan_start          plan CTA entered
  3  plan_counts_seen    wait_epoch(counts) returned
  4  plan_static_done    static tables + D_e + k/rem + single-copy units
  5  plan_coldwarm_done  cold or warm plan done
  6  plan_auction_done   auction rounds done
  7  plan_end            quota / quota_start / warm state published
  8  push_begin          push loop starts
  9  push_end            push grid rendezvous passed
 10  pull_norep_begin    first single-copy slot starts pulling
 11  pull_norep_end      last single-copy slot published
 12  pull_rep_begin      first replica slot obtained the plan
 13  pull_rep_end        last replica slot published
 14  plan_ready_after_norep  plan became ready after the single-copy phase
 15  role2_first_task    first compute CTA actually got work
 16  role2_first_spin    ★ first consumer spin (no work published)
 17  first_replica_pull_done ★ first replica expert finished pulling

All values are absolute `globaltimer` microseconds, so within one row every
delta is meaningful; `globaltimer` is device-wide, so cross-rank comparison of
absolute values is also valid.
"""
import csv
import os

SLOTS = ['step_tag', 'plan_wake', 'plan_start', 'plan_counts_seen',
         'plan_static_done', 'plan_coldwarm_done', 'plan_auction_done',
         'plan_end', 'push_begin', 'push_end', 'pull_norep_begin',
         'pull_norep_end', 'pull_rep_begin', 'pull_rep_end',
         'plan_ready_after_norep', 'role2_first_task',
         'role2_first_spin', 'first_replica_pull_done']


def dump(buf, path, rank, header=None):
    tl = getattr(buf, 'eco_tl', None)
    if tl is None:
        raise RuntimeError('buffer has no eco_tl view (GC_MOE_ECO=1 required)')
    rows = tl.view(256, 18).cpu().tolist()
    with open(path, 'a', newline='') as fh:
        w = csv.writer(fh)
        if header:
            w.writerow(['rank', 'step'] + SLOTS)
        for step, row in enumerate(rows):
            if not any(row):
                continue
            w.writerow([rank, step] + [int(x) for x in row])
    return path
