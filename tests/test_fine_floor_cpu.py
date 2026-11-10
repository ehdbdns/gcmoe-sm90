"""CPU check of the [ECO-FINE-FLOOR] gating and of the timeline plot.

Two things are verified without a GPU:

1. The gating predicate. `_fine_body` keeps a record iff
       floor == 0                    (no window armed -> record everything)
       or  floor <= epoch < floor + FINE_EPOCHS
   which is reproduced here over a synthetic epoch stream shaped like a real
   run: 15 prefill layer calls, then 1000 measured steps x 40 layers.

2. `scripts/plot_timeline_annot.py` on ONE synthetic decode layer:
   14 active slots, each with 1 M block, each block split into 38 N tiles
   (18 L1 + 20 L2).  The kernel emits ONE record per (CTA, expert, start,
   tile) and the N-tile index is NOT part of it, so the 38 rows of a block are
   indistinguishable -- which is exactly why the plot identifies a block by the
   distinct (expert, start) pair.  532 rows per type must collapse to 14.

The point of (1): before the floor existed, `fine_ctr` (monotonic, capacity
65536) was consumed by prefill, so the measured window recorded ZERO rows.
"""
import os, sys, json, subprocess, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def gate(epoch, floor, fine_epochs):
    if floor == 0:
        return True
    return floor <= epoch < floor + fine_epochs


def test_gate():
    prefill = list(range(1, 16))                 # 15 layer calls
    floor = 16
    measured = list(range(floor, floor + 5 * 40))
    kept = [e for e in prefill + measured if gate(e, floor, 1)]
    assert kept == [16], kept
    print(f'gate: dropped {len(prefill)} prefill calls, kept exactly 1 layer {kept}')


def synth_layer(rank, epoch, n_slots=14, tiles=38, role6_cta=None):
    """One decode layer of fine records, in emission order."""
    rows, t = [], 1_000_000
    # every slot begins pulling at t, ends a bit later
    for s in range(n_slots):
        rows.append([t + s, 1, rank * 56 + s, s, s % 24, 1, epoch, 0])
    # block index b belongs to slot b, its tiles finish spread over 40 us
    for b in range(n_slots):
        end = 40 + b
        rows.append([t + end, 2, rank * 56 + b, b, 512, 0, epoch, 0])
        for k in range(tiles):
            # all N tiles of a block share (expert, start): the record layout
            # has no room for the tile index, so they are indistinguishable.
            cta = role6_cta if (role6_cta and b % 4 == 0) else 7
            rows.append([t + end + k, 6, cta, b, 0, 64, epoch, 0])
            rows.append([t + end + k + 1, 7, cta, b, 0, 64, epoch, 0])
    rows.sort(key=lambda r: (r[0], r[1]))
    return rows


def test_plot():
    tmp = tempfile.mkdtemp()
    payload = {'label': 'SYNTH', 'seqs': 32, 'eco_trace': {'ranks': [
        {'ok': True, 'rank': 0, 'fine_rows': synth_layer(0, 16)}]}}
    src = os.path.join(tmp, 'synth.json')
    json.dump(payload, open(src, 'w'))
    png = os.path.join(tmp, 'synth.png')
    py = sys.executable
    out = subprocess.run([py, os.path.join(ROOT, 'scripts', 'plot_timeline_annot.py'),
                          src, '0', png], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    print(out.stdout.strip())
    assert 'total_mblocks=14' in out.stdout, out.stdout
    assert 'role6_blocks=0' in out.stdout, out.stdout
    assert os.path.exists(png)

    # same layer, but 4 of the 14 blocks are consumed by the SM-reuse helper
    payload['eco_trace']['ranks'][0]['fine_rows'] = synth_layer(0, 16, role6_cta=1000)
    json.dump(payload, open(src, 'w'))
    out = subprocess.run([py, os.path.join(ROOT, 'scripts', 'plot_timeline_annot.py'),
                          src, '0', png], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    print(out.stdout.strip())
    assert 'role6_blocks=4' in out.stdout, out.stdout
    assert 'role2_blocks=10' in out.stdout, out.stdout


if __name__ == '__main__':
    test_gate()
    test_plot()
    print('ALL CPU CHECKS PASSED')
