#!/usr/bin/env python3
"""Extract ONE forward call of the 5 gc_moe kernels, by TRACE ROW NUMBER.

Why row numbers: one forward call emits exactly 5 consecutive gc_moe kernel
rows, in the fixed order

    dispatch(24x256) -> shared(71x256) -> routed(71x384) -> plan(1x512) -> combine(288x384)

so a call occupies rows [5k, 5k+4]. Taking "the k-th item of each role's own
list" is WRONG: the 5 roles' k-th entries are different calls.

Verified facts about this trace (prof/final_v2/traces):
  * `ts` and `dur` are BOTH microseconds (device clock, same origin as
    `pull_tl`'s globaltimer//1000 up to a constant offset).
  * 39680 calls = 1000 measured steps x 40 layers (minus 320 prefill calls).
  * `pull_tl` is a 256-deep RING; it therefore holds the LAST 256 calls only.

usage: extract_one_call.py <rank> <call_index> [--last]
       call_index is absolute (0-based among gc_moe calls).
"""
import argparse
import json
import os
import sys

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   '..', '..', 'prof', 'final_v2', 'traces')
ROLES = ('dispatch', 'shared', 'routed', 'plan', 'combine')


def role_of(grid, block):
    if grid == 24:
        return 'dispatch'
    if grid == 71 and block == 256:
        return 'shared'
    if grid == 71 and block == 384:
        return 'routed'
    if grid == 1:
        return 'plan'
    if grid == 288:
        return 'combine'
    return None


def scan(rank, wanted_call, keep_last):
    """Return (rows_of_that_call, total_calls) with a single sequential pass."""
    keep = {}
    row = 0
    tail = []
    with open(os.path.join(SRC, f'rank{rank}.json'), 'rb') as fh:
        import ijson
        for e in ijson.items(fh, 'traceEvents.item', use_float=True):
            if e.get('cat') != 'kernel':
                continue
            if 'gc_moe' not in e.get('name', ''):
                continue
            a = e.get('args') or {}
            g = tuple(a.get('grid') or ())
            b = tuple(a.get('block') or ())
            if not g:
                continue
            k = role_of(g[0], b[0] if b else 0)
            if k is None:
                continue
            rec = {'row': row, 'role': k, 'grid': g[0],
                   'block': b[0] if b else 0,
                   'ts': e['ts'], 'dur': e.get('dur', 0)}
            if keep_last:
                tail.append(rec)
                if len(tail) > 5 * 256:
                    tail.pop(0)
            elif wanted_call is not None and wanted_call * 5 <= row < wanted_call * 5 + 5:
                keep[row] = rec
            row += 1
    total = row
    if keep_last:
        return tail, total
    return [keep[i] for i in sorted(keep)], total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('rank', type=int)
    ap.add_argument('call', type=int)
    a = ap.parse_args()
    rows, total = scan(a.rank, a.call, False)
    assert [r['role'] for r in rows] == list(ROLES), [r['role'] for r in rows]
    print(json.dumps({'rank': a.rank, 'call': a.call,
                      'total_calls': total // 5, 'rows': rows}, indent=1))


if __name__ == '__main__':
    main()
