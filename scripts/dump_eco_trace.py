"""Decode the ECO timeline trace (30 slots, globaltimer ns) from a live buffer.

Slot map -- WHO writes / WHAT event / WHY that mode
====================================================

Group 1: role 1 (dispatch)
  0   role1_start          every role-1 CTA      dispatch entry          min    earliest CTA
  1   role1_barrier_done   role-1 block 0        local rendezvous #1     store  single writer
  2   role1_counts_ready   role-1 block 0        counts row released     store  single writer
  6   plan_wait_first      replica-slot CTA      starts waiting          min    earliest waiter
  7   plan_seen_last       replica-slot CTA      got the plan            max    the tail
  8   role1_end            every role-1 CTA      CTA exit                max    last to leave

Group 2: role 5 (plan) -- one CTA, all `store`
  3   plan_start           plan entry (before wait_epoch)
  4   plan_counts_seen     wait_epoch(counts) returned
  17  plan_start_done      static tables + D_e + k/rem done
  18  plan_rounds_done     cold/warm + auction rounds done
  19  plan_index_done      quota / quota_start / index published
  5   plan_published       release plan_ready

Group 3: role 2 (routed compute)
  9   role2_sched_start    role-2 CTA enters its loop               min
  10  role2_counts_seen    first poll of the producer counters       min
  11  role2_block_first    first block on a producer counter         min
  12  role2_block_last     last CTA that got work                    max
  13  role2_sched_end      leaves the loop                           max
  16  role2_end            leaves the kernel                         max

Group 4: role 3 (combine)
  14  role3_start          enters combine                            min
  15  role3_end            combine done (epoch bump)                 max

SM90 caveat: role 2 never waits for the *plan*. It consumes
`active_ready`/`expert_ready`, which the dispatch publishes after its pull, and
the dispatch pull is what waits for the plan. So slots 11/12 measure "first/last
block on a producer counter", not "wait for plan"; the plan-facing wait is
slot 27 (accumulated, role 1).
"""
import json


def _check(buf):
    tr = getattr(buf, 'eco_trace', None)
    if tr is None:
        raise RuntimeError('buffer has no eco_trace view '
                           '(created before the layout change? GC_MOE_ECO=1?)')
    n = tr.numel()
    if n < 32:
        raise RuntimeError(f'eco_trace view has {n} words, expected 32')
    return tr


def read_trace(buf):
    t = _check(buf).cpu().tolist()
    return {i: int(v) for i, v in enumerate(t)}


def report(buf, calls, steps=1, label=''):
    s = read_trace(buf)
    if not any(s.get(k) for k in (0, 3, 9, 14)):
        print(f'[eco-trace] {label}: empty (GC_MOE_ECO_TRACE!=1 at compile time?)',
              flush=True)
        return None

    def d(a, b):
        return (s[b] - s[a]) if (s.get(a) and s.get(b) and s[b] >= s[a]) else -1

    def us(x):
        return f'{x/1e3:.1f}' if x >= 0 else 'n/a'

    out = dict(label=label,
               role1=dict(start=s.get(0), rendezvous=d(0, 1), counts_ready=d(1, 2),
                          plan_wait_first=s.get(6), plan_seen_last=s.get(7),
                          tail=d(6, 7), end=d(0, 8)),
               # kernel slots: 17 = static+D_e done, 18 = cold/warm done,
               # 19 = auction done, 5 = published. So:
               #   17->18 = cold/warm    18->19 = AUCTION    19->5 = publish
               plan=dict(start=s.get(3), counts_seen=d(3, 4),
                         static_de=d(4, 17), cold_warm=d(17, 18),
                         auction=d(18, 19), publish=d(19, 5),
                         rounds=d(17, 19), total=d(3, 5)),
               role2=dict(start=s.get(9), counts_seen=d(9, 10),
                          block_first=d(10, 11), block_last=d(9, 12),
                          sched=d(9, 13), teardown=d(13, 16), end=d(9, 16)),
               role3=dict(start=s.get(14), end=d(14, 15)),
               wait_total_ns=s.get(27, 0),
               wait_per_call_us=(s.get(27, 0) / calls / 1e3) if calls else -1,
               warm=bool(s.get(30)), rounds=s.get(31),
               calls=calls, steps=steps)

    print(f'[trace] {label}  warm={out["warm"]} rounds={out["rounds"]}', flush=True)
    print(f'   role1(dispatch): start->end {us(out["role1"]["end"])} us | '
          f'rndv#1 {us(out["role1"]["rendezvous"])} counts {us(out["role1"]["counts_ready"])} '
          f'| plan-wait tail {us(out["role1"]["tail"])} us', flush=True)
    print(f'   role5(plan)    : total {us(out["plan"]["total"])} us | '
          f'wait-counts {us(out["plan"]["counts_seen"])} static+D_e {us(out["plan"]["static_de"])} '
          f'cold/warm {us(out["plan"]["cold_warm"])} AUCTION {us(out["plan"]["auction"])} '
          f'publish {us(out["plan"]["publish"])}', flush=True)
    print(f'   role2(compute) : sched {us(out["role2"]["sched"])} us | '
          f'first-poll {us(out["role2"]["counts_seen"])} first-block {us(out["role2"]["block_first"])} '
          f'last-work {us(out["role2"]["block_last"])} teardown {us(out["role2"]["teardown"])}', flush=True)
    print(f'   role3(combine) : {us(out["role3"]["end"])} us', flush=True)
    print(f'   replica-wait   : {out["wait_per_call_us"]:.1f} us/call '
          f'({out["wait_total_ns"]/1e6:.3f} ms / {calls} calls)', flush=True)
    return out
