"""Execute the SM90 quota, auction and publication code against the SM100 reference."""
import inspect
import sys
from pathlib import Path
from types import FunctionType, SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'source'))
from gc_moe.eco.unit_ref import UnitPlanner
from gc_moe.impls.sm90_fp8_fp4_gc_moe import SM90GCMoERoleKernel


def scalar_method(name):
    # One host lane visits every slot at each device barrier. Use the actual
    # undecorated CuTe method and substitute only scalar/register primitives.
    fn = inspect.unwrap(getattr(SM90GCMoERoleKernel, name))
    arch = SimpleNamespace(thread_idx=lambda: (0, 0, 0), sync_threads=lambda: None)
    globals_cpu = dict(fn.__globals__, Int32=int, Uint32=int, Uint64=int,
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
    _eco_do_auction_round = scalar_method('_eco_do_auction_round')
    _eco_costC = scalar_method('_eco_costC')
    _eco_hp = scalar_method('_eco_hp')
    _eco_in = scalar_method('_eco_in')
    _eco_out = scalar_method('_eco_out')
    _eco_move = scalar_method('_eco_move')
    _eco_of = scalar_method('_eco_of')
    _eco_setof = scalar_method('_eco_setof')
    _eco_publish_slot = scalar_method('_eco_publish_slot')
    _eco_publish_count_state = scalar_method('_eco_publish_count_state')

    def __init__(self, ref, capacity):
        self.physical, self.eco_logical, self.ranks = ref.P, ref.E, ref.R
        self.eco_max_copies = capacity
        self.max_m, self.eco_wb, self.eco_wa, self.eco_slack = ref.bm, ref.wb, ref.wa, ref.slack
        self.copies = [np.flatnonzero((ref.phy == e) & ref.active).tolist() for e in range(ref.E)]
        for field in ('phy','act','n','k','rem','x','T0','g','pn','u','y','part','quota',
                      'T','tot','nof','nacc','cks','ckv','cki','ckh','ckp','hit',
                      'grp','of','gcnt','gof','acc'):
            setattr(self, f'eco_{field}_off', field)

    def _eco_slot_of(self, base, expert, copy):
        return self.copies[expert][copy]

    def _eco_ncopies(self, base, expert):
        return len(self.copies[expert])

    def _eco_state_ptr(self, previous, epoch):
        return previous['state'] if isinstance(previous, dict) else previous

    def _eco_s(self, slab, field, i):
        return slab[field][i]

    def _eco_set(self, slab, field, i, value):
        slab[field][i] = value

    def _eco_inc(self, slab, field, i, value):
        previous = slab[field][i]
        slab[field][i] += value
        return previous

    def _eco_key_set(self, slab, s, value):
        slab['key'][s] = value & ((1 << 64) - 1)

    def _eco_key_get(self, slab, s):
        return slab['key'][s]

    def _field(self, base, field, dtype):
        return base[field]

    def quotas(self, ref, counts, previous):
        slab = {name: [0] * self.physical for name in (
            'T0','x','u','g','pn','y','quota','T','tot','nof','nacc','cks','ckv','cki','ckh',
            'ckp','hit','grp','gcnt','acc','key')}
        slab.update(gof=[0] * (ref.R + 1), of=[0] * (8 * ref.P))
        units = [(n + ref.bm - 1) // ref.bm for n in counts]
        slab.update(phy=ref.phy.tolist(), act=ref.active.tolist(), n=counts, k=units,
                    rem=[(n - 1) % ref.bm + 1 if n else 0 for n in counts], part=[0xffffffff] * ref.E)
        for e, slots in enumerate(self.copies):
            if len(slots) == 1 and counts[e]:
                slab['y'][slots[0]] = units[e] - 1
                slab['part'][e] = slots[0]
        warm = previous is not None and sum(previous[1]) > 0
        if not warm:
            self._eco_do_cold(None, slab)
        else:
            self._eco_do_warm(previous[0] + previous[1], slab, 1)
        self._eco_emit_quota(slab)
        for _ in range(ref.nr if warm else ref.nr_cold):
            self._eco_rank_time(slab, 'quota', 'T')
            if self._eco_do_auction_round(None, slab) == 0:
                break
            self._eco_emit_quota(slab)
        published = dict(eco_quota=[0] * ref.P, eco_quota_start=[0] * ref.P,
                         state=[0] * (ref.P + ref.E))
        prefix = [0] * ref.E
        for s, e in enumerate(ref.phy):
            self._eco_publish_slot(published, slab, s, 1)
            assert published['eco_quota_start'][s] == prefix[e]
            prefix[e] += published['eco_quota'][s]
        for e in range(ref.E):
            self._eco_publish_count_state(published, slab, e, 1)
        assert prefix == counts
        assert published['state'] == published['eco_quota'] + counts
        return published['eco_quota']


def test_auctions():
    rng = np.random.default_rng(1908)
    for wb in (0, 26, 23968):
        for case in range(24):
            ranks, experts = 8, 12
            phy = np.concatenate([np.arange(experts), np.zeros(8, int),
                                  rng.integers(0, experts, 28)])
            rng.shuffle(phy)
            ref = UnitPlanner(phy, ranks, experts, block_m=(16,32,64)[case % 3],
                              wb=wb, wa=case % 4, rounds=2, cold_rounds=6)
            device = ScalarPlan(ref, ranks)
            prev = None
            for _ in range(4):
                counts = rng.integers(0, 2049, size=experts).tolist()
                want = ref.plan(counts, prev).tolist()
                got = device.quotas(ref, counts, prev)
                assert got == want, (wb, case, counts, got, want)
                prev = (got, counts)


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
    test_auctions()
    print('ECO_COPIES_CPU PASS')
