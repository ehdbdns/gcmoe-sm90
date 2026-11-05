"""CPU regression for the per-layer / per-slot load metrics.

Three separate defects were found in the reported load metrics; this file pins
each one down so they cannot come back:

  T1  block count  : per-slot blocks must be ceil(tokens / tile_m). The old
                     `(x + bm - 1) // bm` on FLOATS is floor-division, so a
                     slot taking a fraction of a token per step counted 0.
  T2  block_m      : must be the M chunk the SM90 routed role runs
                     (`schedule.tile_m`), not the SM100 `get_block_m_for_gc_moe`
                     selector (that returned 192/240 vs the real 64/16).
  T3  active slots : `header[7]` is zeroed by role 3 at the start of every
                     call, so reading it after the window yields 0. The
                     per-call value must be snapshotted after each launch.

The load is built from an exact synthetic kernel call, so the expected numbers
are computable by hand.
"""
import math
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, '..')
sys.path.insert(0, os.path.join(_ROOT, 'source'))
sys.path.insert(0, os.path.join(_ROOT, 'integration'))

from eplb_logic import aggregate_load, aggregate_with_sample  # noqa: E402

FAIL = []


def check(name, got, want, tol=1e-9):
    ok = (abs(got - want) <= tol) if isinstance(want, float) else (got == want)
    print(f'  {name:52s} got={got!r:>18} want={want!r:>18} {"OK" if ok else "FAIL"}')
    if not ok:
        FAIL.append(name)


def t1_blocks_ceil():
    """T1: ceil(x / bm) per slot, including slots below one token per step."""
    print('T1 per-slot block count uses ceil, not float floor-division')
    world, steps, bm, slots = 8, 100, 16, 48
    per_step = {0: 0.5, 1: 3.0, 2: 16.0, 3: 16.4, 4: 0.0}
    stats = torch.zeros((1, slots), dtype=torch.int32)
    for s, v in per_step.items():
        stats[0, s] = int(round(v * steps * world))
    out = aggregate_load(stats, bm, steps, world)

    want_total = sum(per_step.values())
    want_blocks = sum(math.ceil(v / bm) for v in per_step.values())
    old_blocks = sum(float((torch.tensor([v]) + bm - 1) // bm) for v in per_step.values())

    check('tokens / layer', float(out['tokens'][0]), float(want_total))
    check('blocks / layer (fixed)', float(out['blocks'][0]), float(want_blocks))
    print(f'  {"  old formula would have given":52s} {old_blocks:>18.1f}'
          f' {"" if old_blocks != want_blocks else "(same here)"}')
    if old_blocks != want_blocks:
        print(f'  -> the old formula under-counted by '
              f'{want_blocks - old_blocks:.0f} block(s) on this input')
    print(f'  per-slot blocks: '
          f'{[math.ceil(v / bm) for v in per_step.values()]} '
          f'(x={list(per_step.values())}, bm={bm})')


def t2_block_m_identity():
    """T2: the divisor must be the kernel's own M chunk.

    Important nuance, measured on the real run: with ~0.4 token per slot per
    step (24 routes over 60 slots) EVERY slot needs exactly one chunk, so
    blocks == active slots and the divisor is immaterial. It only becomes
    material when a slot carries more than one chunk's worth of tokens, which
    is the case this test pins down (and what `tile_m` must be fed for).
    """
    print('T2 block_m must be the routed role tile_m (divisor matters when a '
          'slot exceeds one chunk)')
    world, steps, slots = 8, 100, 48
    per_step = [0.4] * 47 + [100.0]       # one hot slot: 100 tokens/step
    stats = torch.zeros((1, slots), dtype=torch.int32)
    for s, v in enumerate(per_step):
        stats[0, s] = int(round(v * steps * world))
    b_true = float(aggregate_load(stats, 64, steps, world)['blocks'][0])
    b_sm100 = float(aggregate_load(stats, 240, steps, world)['blocks'][0])
    print(f'  blocks with tile_m=64  (kernel, correct): {b_true:.0f}'
          f'  = 47 quiet slots x 1 + ceil(100/64)=2')
    print(f'  blocks with tile_m=240 (SM100 selector):  {b_sm100:.0f}'
          f'  = 47 x 1 + ceil(100/240)=1')
    check('divisor is material for a hot slot', int(b_true - b_sm100), 1)

    print('  -- and for the load actually measured in this experiment '
          '(≈0.4 token/slot/step):')
    flat = torch.zeros((1, slots), dtype=torch.int32)
    flat[0, :] = int(round(0.4 * steps * world))
    print(f'     blocks(tile_m=64)={float(aggregate_load(flat, 64, steps, world)["blocks"][0]):.0f}'
          f'  blocks(tile_m=240)={float(aggregate_load(flat, 240, steps, world)["blocks"][0]):.0f}'
          f'  active={int(aggregate_load(flat, 64, steps, world)["active_ever"][0])}')
    print('     -> blocks == active slots there, so the DIVISOR was not the '
          'error; the FLOOR-DIVISION was (see T1).')


def t3_active_snapshot():
    """T3: the occupancy must come from a per-call snapshot, not a late read."""
    print('T3 per-call occupancy: sampled value vs a post-window read of 0')
    world, steps, bm, slots = 8, 100, 16, 48
    stats = torch.zeros((1, slots), dtype=torch.int32)
    stats[0, :20] = 8 * steps          # 20 slots, 1 token/step each
    sample = torch.zeros(slots, dtype=torch.int32)
    sample[:20] = world                # one call: those 20 slots, 1 token each

    good = aggregate_with_sample(stats, bm, steps, world, sample=sample,
                                 active_sample=20)
    late = aggregate_with_sample(stats, bm, steps, world, sample=sample,
                                 active_sample=0)   # what a late read gives
    check('active (snapshot)', good['active_header'], 20)
    check('active (late read of header[7])', late['active_header'], 0)
    check('active_step from the one-call sample', good['active_step'], 20)
    check('tokens of that call', good['sample_tokens'], 20.0)
    check('blocks of that call', good['blocks_step'], 20)
    print('  -> a late read silently reports 0 active slots; the snapshot does not')


def t4_conservation():
    """Sanity: the window metrics still conserve tokens."""
    print('T4 window totals conserve the routed token count')
    world, steps, bm, slots = 8, 100, 16, 48
    stats = torch.zeros((1, slots), dtype=torch.int32)
    stats[0, :24] = 8 * steps
    out = aggregate_load(stats, bm, steps, world)
    check('tokens / layer', float(out['tokens'][0]), 24.0)
    check('blocks / layer', float(out['blocks'][0]), 24.0)
    check('active_ever (window-wide)', int(out['active_ever'][0]), 24)


if __name__ == '__main__':
    for fn in (t1_blocks_ceil, t2_block_m_identity, t3_active_snapshot,
               t4_conservation):
        fn()
        print()
    print('METRICS_CPU ' + ('PASS' if not FAIL else f'FAIL {FAIL}'))
    sys.exit(1 if FAIL else 0)
