"""Execute the SM90 planner's cold and warm quota code against UnitPlanner."""
import inspect
import sys
from pathlib import Path
from types import FunctionType, SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'eco_ref'), str(ROOT / 'source')]
from unit_ref import UnitPlanner
from gc_moe.impls.sm90_fp8_fp4_gc_moe import SM90GCMoERoleKernel


def scalar_method(name):
    # One host lane visits every slot at each device barrier. Use the actual
    # undecorated CuTe method and substitute only scalar/register primitives.
    fn = inspect.unwrap(getattr(SM90GCMoERoleKernel, name))
    arch = SimpleNamespace(thread_idx=lambda: (0, 0, 0), sync_threads=lambda: None)
    globals_cpu = dict(fn.__globals__, Int32=int, Uint32=int,
                       cutlass=SimpleNamespace(range=range, range_constexpr=range, const_expr=bool),
                       cute=SimpleNamespace(arch=arch, make_rmem_tensor=lambda shape, dtype: [0] * shape[0]))
    return FunctionType(fn.__code__, globals_cpu, name)


class ScalarPlan:
    threads = 1
    max_m = 64
    eco_wb, eco_wa, eco_tok_w = 26, 0, 1
    eco_costfn = eco_valid_cost = eco_tokens_only = False
    _eco_do_cold = scalar_method('_eco_do_cold')
    _eco_do_warm = scalar_method('_eco_do_warm')
    _eco_cold_key = scalar_method('_eco_cold_key')
    _eco_rank_time = scalar_method('_eco_rank_time')
    _eco_emit_quota = scalar_method('_eco_emit_quota')

    def __init__(self, ref, capacity):
        self.physical, self.eco_logical, self.ranks = ref.P, ref.E, ref.R
        self.eco_max_copies = capacity
        self.copies = [np.flatnonzero((ref.phy == e) & ref.active).tolist() for e in range(ref.E)]
        for field in ('phy', 'act', 'n', 'k', 'rem', 'x', 'T0', 'g', 'pn', 'u', 'y', 'part', 'quota'):
            setattr(self, f'eco_{field}_off', field)

    def _eco_slot_of(self, base, expert, copy):
        return self.copies[expert][copy]

    def _eco_ncopies(self, base, expert):
        return len(self.copies[expert])

    def _eco_state_ptr(self, previous, epoch):
        return previous

    def _eco_s(self, slab, field, i):
        return slab[field][i]

    def _eco_set(self, slab, field, i, value):
        slab[field][i] = value

    def _eco_inc(self, slab, field, i, value):
        slab[field][i] += value

    def quotas(self, ref, counts, previous):
        slab = {name: [0] * self.physical for name in ('T0', 'x', 'u', 'g', 'pn', 'y', 'quota')}
        units = [(n + 63) // 64 for n in counts]
        slab.update(phy=ref.phy.tolist(), act=ref.active.tolist(), n=counts, k=units,
                    rem=[(n - 1) % 64 + 1 if n else 0 for n in counts], part=[0xffffffff] * ref.E)
        for e, slots in enumerate(self.copies):
            if len(slots) == 1 and counts[e]:
                slab['y'][slots[0]] = units[e] - 1
                slab['part'][e] = slots[0]
        if previous is None:
            self._eco_do_cold(None, slab)
        else:
            self._eco_do_warm(previous[0] + previous[1], slab, 1)
        self._eco_emit_quota(slab)
        return slab['quota']


def test_replica_quota():
    for capacity in (4, 8):
        for copies in range(1, capacity + 1):
            # One replicated hot expert, its passive duplicate on each rank,
            # and a single-copy expert supplying independent rank load.
            phy = [e for rank in range(copies) for e in (0, 0, rank + 1)]
            ref = UnitPlanner(phy, copies, copies + 1, wb=26, wa=0, rounds=0, cold_rounds=0)
            device = ScalarPlan(ref, capacity)
            for hot in (0, 1, 63, 64, 65, 255, 513, 1024, 2047):
                prev = None
                for n in (hot, 1025, 65, 0, hot):
                    counts = [n] + [(j * 137 + n % 5) % 257 for j in range(copies)]
                    expected = ref.plan(counts, prev).tolist()
                    actual = device.quotas(ref, counts, prev)
                    assert actual == expected, (capacity, copies, n, actual, expected)
                    assert sum(actual) == sum(counts)
                    prev = (actual, counts)


if __name__ == '__main__':
    test_replica_quota()
    print('ECO_COPIES_CPU PASS')
