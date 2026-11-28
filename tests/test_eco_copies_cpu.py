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
    matrix = True
    fine_auction = False
    auction_granularity_audit = True
    _eco_transfer_cost = scalar_method('_eco_transfer_cost')
    _eco_small_bid = scalar_method('_eco_small_bid')
    _eco_commit_bid = scalar_method('_eco_commit_bid')
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
    _eco_block_cost = scalar_method('_eco_block_cost')
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
                      'grp','of','gcnt','gof','acc','candidates'):
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
        slab.update(candidates=[0], gof=[0] * (ref.R + 1), of=[0] * (8 * ref.P))
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
        rounds = ref.nr if warm else ref.nr_cold
        self.small_moves = 0
        for i in range(rounds):
            self._eco_rank_time(slab, 'quota', 'T')
            fine = self.fine_auction and warm and i + 1 == rounds
            packed = self._eco_do_auction_round(None, slab, int(fine))
            moves = packed & 65535
            self.small_moves += packed >> 16
            counts_word = slab['candidates'][0]
            assert 0 <= moves <= slab['nof'][0] <= counts_word >> 16 <= (counts_word & 65535) <= ref.P
            if moves == 0:
                break
            if not fine:
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
        self.warm_quota = published['state'][:ref.P]
        if not self.fine_auction:
            assert self.warm_quota == published['eco_quota']
        for e, slots in enumerate(self.copies):
            assert sum(self.warm_quota[s] for s in slots) == counts[e]
            assert sum(self.warm_quota[s] % ref.bm != 0 for s in slots) <= 1
            assert sum((published['eco_quota'][s] + ref.bm - 1) // ref.bm for s in slots) == units[e]
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


def test_fine_auction_prices():
    from gc_moe.eco.h20_valid_cost import FULL, INTERCEPTS, ROW_COST, block_cost, slot_cost
    planner = ScalarPlan(UnitPlanner([0, 0], 2, 1, wb=26, wa=0), 2)
    planner.eco_valid_cost = planner.eco_costfn = True
    planner.cf_full, planner.cf_row = FULL, ROW_COST
    for k, price in enumerate(INTERCEPTS):
        setattr(planner, f'cf_tab{k}', price)
    for valid in range(1, 65):
        slab = dict(phy=[0,0],rem=[valid],y=[2,1],part=[0],T0=[100000,200000])
        shed, partial = planner._eco_out(slab, 0)
        assert partial == 1 and shed == block_cost(valid)
        assert planner._eco_in(slab, 1, partial) == slot_cost(64+valid)-slot_cost(64)
        assert planner._eco_out(slab, 1)[0] == planner._eco_in(slab, 0, 0) == FULL
        assert planner._eco_cold_key(None, slab, 0, [64+valid,0], 0) == 100000 - block_cost(valid%64)


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


def test_small_token_auction():
    # Unit 64 cannot land on rank 1. A bid of 16 balances the pair without
    # allocating a third block. The next warm call must receive canonical quotas.
    ref = UnitPlanner([0, 1, 0, 2], 2, 3, block_m=64, wb=26, wa=2, rounds=1)
    device = ScalarPlan(ref, 2)
    device.fine_auction = True
    counts = [104, 32, 20]
    canonical = [64, 32, 40, 20]
    assert device.quotas(ref, counts, (canonical, counts)) == [48, 32, 56, 20]
    assert device.small_moves == 1 and device.warm_quota == canonical
    assert device.quotas(ref, counts, (device.warm_quota, counts)) == [48, 32, 56, 20]

    rng = np.random.default_rng(20261128)
    phy = np.tile(np.arange(12), 8)
    rng.shuffle(phy)
    ref = UnitPlanner(phy, 8, 12, block_m=64, wb=26, wa=2, rounds=1)
    device = ScalarPlan(ref, 8)
    device.fine_auction = True
    prev, accepted = None, 0
    for i in range(100):
        counts = rng.integers(0, 2049, size=12).tolist()
        counts[i % 12] = (0, 1, 63, 64, 65)[i % 5]
        device.quotas(ref, counts, prev)
        prev = device.warm_quota, counts
        accepted += device.small_moves
    assert accepted > 0

    # At M8 a one-token bid must survive the legacy full/tail kind encoding.
    for m in (8, 16, 32, 64):
        planner = UnitPlanner([0, 1, 0, 2], 2, 3, block_m=m, wb=26, wa=1, rounds=1)
        worker = ScalarPlan(planner, 2)
        worker.fine_auction = True
        previous, small = None, 0
        for _ in range(100):
            demand = rng.integers(1, 3 * m, size=3).tolist()
            worker.quotas(planner, demand, previous)
            small += worker.small_moves
            previous = worker.warm_quota, demand
        assert small > 0, m


if __name__ == '__main__':
    test_small_token_auction()
    test_fine_auction_prices()
    test_replica_quota()
    test_auctions()
    print('ECO_COPIES_CPU PASS')
