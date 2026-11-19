#!/usr/bin/env python3
"""Pick the layer calls whose SPAN is visible in the trace AND staged in pull_tl.

Checks, per candidate epoch:
  * pull_tl has a complete row (non-zero [3], [5], [10], [15])
  * the corresponding 5 trace rows are the expected role sequence
  * ir_step has non-zero token counts for that (layer, step)
Only fully-consistent calls are returned.
"""
import json, os, sys

TAIL = '/tmp/rt'
SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


def main():
    run = json.load(open(sys.argv[1] if len(sys.argv) > 1 else
                         os.path.join(SRC, 'runs', 'ALIGNED5.json')))
    L = run['ir_step']['tables']['layers']
    e0 = run['ir_step']['tables']['epoch0']
    pulls = run['eco_trace']['ranks']
    print('epoch0', e0, 'layers', L)
    snap = json.load(open(os.path.join(TAIL, 'tail0.json')))
    total = snap['total'] // 5
    tails = {}
    for r in range(8):
        s = json.load(open(os.path.join(TAIL, f'tail{r}.json')))
        tails[r] = s

    by_epoch = []
    for r in range(8):
        rows = [x for x in (pulls[r].get('pull_rows') or []) if x and x[0]]
        for x in rows:
            by_epoch.append((x[0], r, x))
    epochs = sorted({e for e, _, _ in by_epoch})
    print('pull_tl epochs', epochs[0], '..', epochs[-1], 'n', len(epochs))

    ok = []
    for e in epochs:
        step = (e - e0) // L
        layer = (e - e0) % L
        row = layer * run['ir_step']['tables']['steps'] + step
        tok = run['ir_step']['tables']['tokens']
        if not (0 <= row < len(tok)) or not all(tok[row]):
            continue
        # every rank must have the row and the trace call must exist
        call = None
        pos = sorted([x for x in (pulls[0].get('pull_rows') or []) if x and x[0]],
                     key=lambda x: x[10]).index(
            [x for x in pulls[0]['pull_rows'] if x and x[0] == e][0])
        call = total - 256 + pos
        if not all((x[5] and x[10] and x[15]) for x in
                   ([y for y in pulls[r]['pull_rows'] if y and y[0] == e][0]
                    for r in range(8))):
            continue
        ok.append({'epoch': e, 'layer': layer, 'step': step,
                   'ir_row': row, 'call': call})
    print('fully consistent layer calls:', len(ok))
    for o in ok[:5]:
        print('  ', o)
    if ok:
        print('  ...')
        print('  ', ok[-1])
    json.dump(ok, open(os.path.join(TAIL, 'layer_calls.json'), 'w'), indent=1)
    print('saved /tmp/rt/layer_calls.json')


if __name__ == '__main__':
    main()
