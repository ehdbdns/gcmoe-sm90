"""What does the L2 claim ORDER cost?  Discrete-event replay of one layer.

SCHEDULING UNIT (this is what the kernel really hands out):

    ticket = (slot, ntile, phase)          phase in {L1, L2}
    claiming it runs  `for start in range(0, rows, block_m)`  -> nblocks tasks
    (nblocks is 1 or 2 here), sequentially, on the same CTA.

Ticket durations come from the trace (type 6 -> type 7 of the same CTA), so a
replay only changes the ORDER; no tile gets faster.

Policies (L1 is always FIFO over the publication order -- that part is fixed
by `active_ready`):

  fifo           today: one monotonic counter, L2 tickets follow pull order.
                 A ticket whose slot is not L1-complete BLOCKS in
                 `wait_count(l1_done, 18)`, and the CTA holds it meanwhile.
  largest-ready  the abstract "big slots first" queue: among slots whose L1 is
                 complete, hand out the one with the most rows. Never blocks.
  reverse-pull   LIFO over the pull order. Stable (decided at publication),
                 one array + one subtract, and since the last-pulled slot is
                 usually the biggest it approximates `largest-ready`.
"""
import json, os, sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tail_blockers import pick

NT_L1, NT_L2 = 18, 20


def load(run, rank=0, block_m_hint=None):
    d = json.load(open(run))
    rec = d['eco_trace']['ranks'][rank]
    rows = [x for x in (rec.get('fine_rows') or []) if x[1]]
    ep, E = pick(rows)
    u = lambda t: t & 0xFFFFFFFF
    t0 = min(u(x[0]) for x in E if x[1] == 1)
    pull, nrows = {}, {}
    for x in E:
        if x[1] == 2:
            pull[x[3]] = u(x[0]) - t0
            nrows[x[3]] = x[4]

    # rebuild each task's (begin, end) by pairing type6/type7 per CTA in order
    tb = defaultdict(list)                 # (slot,start,cta) -> [begin]
    for x in E:
        if x[1] == 6:
            tb[(x[3], x[4], x[2])].append(u(x[0]) - t0)
    te = defaultdict(list)
    for x in E:
        if x[1] == 7:
            te[(x[3], x[4], x[2])].append(u(x[0]) - t0)

    # a "ticket" is (slot, ntile, phase). its tasks are the nblocks blocks of
    # that ntile, executed back to back. Reconstruct them from the trace:
    # per (slot, phase) the tasks are ordered by ntile, and each ntile appears
    # nblocks times (once per block start).
    tasks = defaultdict(list)              # (slot, start) -> [(beg, dur)]
    for (s, st, c), bs in tb.items():
        b0 = min(bs)
        cand = [e for e in te[(s, st, c)] if e >= b0]
        if not cand:
            continue
        tasks[(s, st)].append((b0, min(cand) - b0, c))

    # split each (slot,start) group into L1 (first NT_L1 begins) and L2
    per_slot_start = {}
    for k, v in tasks.items():
        v.sort()
        per_slot_start[k] = v
    tickets = {}                           # (slot, ntile, phase) -> duration
    for s in pull:
        if nrows.get(s, 0) == 0:
            continue
        blk_starts = sorted({st for (ss, st) in per_slot_start if ss == s})
        for st in blk_starts:
            v = per_slot_start[(s, st)]
            n_l1 = min(NT_L1, len(v))
            for i, (b, d, c) in enumerate(v):
                nt = i if i < n_l1 else i - n_l1
                ph = 'L1' if i < n_l1 else 'L2'
                tickets[(s, nt, ph)] = tickets.get((s, nt, ph), 0.0) + d
    # L1 completeness = all L1 tickets of the slot issued AND finished
    return ep, pull, nrows, tickets


def simulate(pull, nrows, tickets, policy, n_cta):
    slots = sorted({s for (s, nt, ph) in tickets}, key=lambda s: pull[s])
    l1_t = {s: sorted([k for k in tickets if k[0] == s and k[2] == 'L1'],
                      key=lambda k: k[1]) for s in slots}
    l2_t = {s: sorted([k for k in tickets if k[0] == s and k[2] == 'L2'],
                      key=lambda k: k[1]) for s in slots}
    l1_order = [k for s in slots for k in l1_t[s]]
    l2_order = [k for s in slots for k in l2_t[s]]          # = fifo
    l1_next = 0
    l2_ptr = defaultdict(int)
    l1_left = {s: len(l1_t[s]) for s in slots}
    l1_done = {}
    free = [0.0] * n_cta
    phase = ['L1'] * n_cta
    l2_issued = 0
    done = 0
    total = len(l1_order) + len(l2_order)
    guard = 0
    while done < total and guard < 2_000_000:
        guard += 1
        i = min(range(n_cta), key=lambda k: free[k])
        t = free[i]
        if phase[i] == 'L1':
            if l1_next >= len(l1_order):
                phase[i] = 'L2'
                continue
            k = l1_order[l1_next]
            if pull[k[0]] > t:
                free[i] = pull[k[0]]
                continue
            l1_next += 1
            free[i] = t + tickets[k]
            done += 1
            l1_left[k[0]] -= 1
            if l1_left[k[0]] == 0:
                l1_done[k[0]] = free[i]

        else:
            k = None
            if policy == 'largest':
                best = None
                for s in slots:
                    if l1_done.get(s, 1e18) > t or l2_ptr[s] >= len(l2_t[s]):
                        continue
                    if best is None or nrows[s] > nrows[best]:
                        best = s
                if best is not None:
                    k = l2_t[best][l2_ptr[best]]
            elif policy == 'reverse':
                while l2_issued < len(l2_order):
                    cand = l2_order[len(l2_order) - 1 - l2_issued]
                    if l2_ptr[cand[0]] != cand[1]:
                        l2_issued += 1
                        continue
                    k = cand
                    l2_issued += 1
                    break
            else:                                       # fifo
                while l2_issued < len(l2_order):
                    cand = l2_order[l2_issued]
                    if l2_ptr[cand[0]] != cand[1]:
                        l2_issued += 1
                        continue
                    k = cand
                    l2_issued += 1
                    break
            if k is None:
                if l2_issued >= len(l2_order):
                    break
                # no claimable ticket; wait for some L1 to complete
                pend = [v for v in l1_done.values()]
                nxt = [v for s, v in l1_done.items()
                       if l2_ptr[s] < len(l2_t[s]) and v > t]
                free[i] = min(nxt) if nxt else t + 1
                continue
            ready = l1_done.get(k[0], 1e18)
            if ready > t:                     # blocks on wait_count
                free[i] = ready
                if policy != 'largest':
                    if policy == 'fifo':
                        l2_issued -= 1
                    else:
                        l2_issued -= 1
                continue
            l2_ptr[k[0]] += 1
            free[i] = t + tickets[k]
            done += 1
    return max(free)


def main():
    n_cta = 76
    for run in sys.argv[1:]:
        ep, pull, nrows, tickets = load(run)
        base = simulate(pull, nrows, tickets, 'fifo', n_cta)
        print(f'# {os.path.basename(run)} epoch{ep}  slots={len(nrows)}  '
              f'tickets={len(tickets)}  CTAs={n_cta}')
        for p in ('fifo', 'reverse', 'largest'):
            e = simulate(pull, nrows, tickets, p, n_cta)
            print(f'    {p:<9} {e:7.1f} us  {100*(e/base-1):+6.2f}%')


if __name__ == '__main__':
    main()
