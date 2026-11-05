"""Simulate the SM90 kernel's plan control flow and diff it against unit_ref.

The simulator below copies the *kernel* code path (one thread per expert, one
thread per slot, registers for the unit arrays, single-writer shared arrays)
rather than calling plan_port, so a divergence between the DSL and the
reference shows up here on CPU instead of on the GPU.

Covered: cold start, warm start (congruent inverse + delta), one-hop auction,
accept prefix with both breaks, quota + quota_start.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from unit_ref import UnitPlanner

NO_PARTIAL = 0xFFFFFFFF


def copies_of(phy, E, R):
    """EcoMeta's active copy lists: first copy per rank, ascending slot."""
    P = len(phy)
    S = P // R
    out = [[] for _ in range(E)]
    seen = [set() for _ in range(E)]
    for s in range(P):
        e = int(phy[s]); r = s // S
        if r in seen[e]:
            continue
        seen[e].add(r)
        out[e].append(s)
    return out


class KernelSim:
    def __init__(self, phy, R, E, bm, wb, wa, rounds, cold_rounds):
        self.phy = list(phy); self.P = len(phy); self.R = R; self.E = E
        self.S = self.P // R
        self.bm, self.wb, self.wa = bm, wb, wa
        self.slack = wb + bm + wa
        self.rounds, self.cold_rounds = rounds, cold_rounds
        self.copies = copies_of(self.phy, E, R)
        self.T0 = [0] * R
        self.T = [0] * R
        self.quota = [0] * self.P
        self.active = [False] * self.P
        for e in range(E):
            for s in self.copies[e]:
                self.active[s] = True

    # -- plan ---------------------------------------------------------------
    def plan(self, counts, prev=None):
        P, E, bm = self.P, self.E, self.bm
        self.y = [0] * P
        self.part = [NO_PARTIAL] * E
        self.n = list(counts)
        self.k = [-(-n // bm) for n in counts]
        self.rem = [0 if n == 0 else n - (self.k[e] - 1) * bm for e, n in enumerate(counts)]
        # single-copy experts fixed up front
        for s in range(P):
            e = self.phy[s]
            if self.active[s] and len(self.copies[e]) == 1 and self.n[e] > 0:
                self.y[s] = self.k[e] - 1
                self.part[e] = s
        warm = prev is not None and int(np.sum(prev[1])) > 0
        if warm:
            self.do_warm(prev[0], prev[1])
            nr = self.rounds
        else:
            self.do_cold()
            nr = self.cold_rounds
        self.emit_quota()
        for _ in range(nr):
            self.rank_time(self.quota, self.T)
            if self.auction() == 0:
                break
            self.emit_quota()
        return np.asarray(self.quota, np.int64), self.quota_start()

    # -- counters -----------------------------------------------------------
    def rank_time(self, q, out):
        t = [0] * self.R
        for s in range(self.P):
            blocks = -(-q[s] // self.bm)
            t[s // self.S] += self.wb * blocks + (self.wa if q[s] > 0 else 0) + q[s]
        for r in range(self.R):
            out[r] = t[r]

    def emit_quota(self):
        self.quota = [0] * self.P
        for s in range(self.P):
            e = self.phy[s]
            q = self.y[s] * self.bm
            if self.part[e] == s:
                q += self.rem[e]
            self.quota[s] = q

    def quota_start(self):
        out = [0] * self.P
        for s in range(self.P):
            e = self.phy[s]
            start = 0
            for c in self.copies[e]:
                if c < s:
                    start += self.quota[c]
            out[s] = start
        return out

    def costC(self, s, y, hp):
        blk = y + hp
        return (self.wb * blk + (self.wa if blk > 0 else 0)
                + y * self.bm + hp * self.rem[self.phy[s]])

    def hp(self, s):
        return 1 if self.part[self.phy[s]] == s else 0

    def has_unit(self, s):
        return self.y[s] > 0 or self.hp(s) == 1

    def out(self, s):
        y, hp = self.y[s], self.hp(s)
        if hp:
            return self.costC(s, y, 1) - self.costC(s, y, 0), 1
        return self.costC(s, y, 0) - self.costC(s, y - 1, 0), 0

    def inn(self, s, ispart):
        y, hp = self.y[s], self.hp(s)
        y2 = y if ispart else y + 1
        return self.costC(s, y2, hp + ispart) - self.costC(s, y, hp)

    def move(self, a, b, ispart):
        e = self.phy[a]
        if ispart:
            assert self.part[e] == a, (a, b, e, self.part[e])
            self.part[e] = b
        else:
            assert self.y[a] > 0
            self.y[a] -= 1
            self.y[b] += 1

    # -- cold ---------------------------------------------------------------
    def do_cold(self):
        bm = self.bm
        x = [0] * self.P
        for s in range(self.P):
            e = self.phy[s]
            nc = len(self.copies[e])
            if self.active[s] and nc > 0:
                x[s] = self.n[e] // nc
                if self.copies[e][0] == s:
                    x[s] += self.n[e] - (self.n[e] // nc) * nc
        self.rank_time(x, self.T0)
        for e in range(self.E):
            nc = len(self.copies[e])
            if nc <= 1 or self.n[e] == 0:
                continue
            g = [x[s] // bm for s in self.copies[e]]
            left = max(0, self.k[e] - sum(g))
            order = sorted(range(nc), key=lambda j: (
                self.T0[self.copies[e][j] // self.S] * bm
                - (x[self.copies[e][j]] % bm) * (self.wb + bm), j))
            for i in range(left):
                g[order[i]] += 1
            jp = max((j for j in range(nc) if g[j] > 0),
                     key=lambda j: (g[j] * bm - x[self.copies[e][j]], -j))
            for j, s in enumerate(self.copies[e]):
                self.y[s] = g[j] - (1 if j == jp else 0)
            self.part[e] = self.copies[e][jp]

    # -- warm ---------------------------------------------------------------
    def do_warm(self, qp, npv):
        bm, unit, S = self.bm, self.wb + self.bm, self.S
        y0 = list(self.y)
        part0 = list(self.part)
        u_all = {}
        own_all = {}
        for e in range(self.E):
            nc = len(self.copies[e])
            if nc <= 1 or self.n[e] == 0:
                continue
            kp = -(-int(npv[e]) // bm)
            own, remp = NO_PARTIAL, 0
            if kp > 0:
                remp = int(npv[e]) - (kp - 1) * bm
                for j, s in enumerate(self.copies[e]):
                    qs = int(qp[s])
                    if own == NO_PARTIAL and qs >= remp and (qs - remp) % bm == 0:
                        own = j
            u = [0] * nc
            if own != NO_PARTIAL:
                for j, s in enumerate(self.copies[e]):
                    qs = int(qp[s])
                    u[j] = (qs - (remp if j == own else 0)) // bm + (1 if j == own else 0)
                    y0[s] = u[j] - (1 if j == own else 0)
                part0[e] = self.copies[e][own]
            u_all[e], own_all[e] = u, own
        q0 = [0] * self.P
        for s in range(self.P):
            e = self.phy[s]
            q0[s] = y0[s] * bm + (self.rem[e] if part0[e] == s else 0)
        self.rank_time(q0, self.T0)
        for e in range(self.E):
            nc = len(self.copies[e])
            if nc <= 1 or self.n[e] == 0:
                continue
            u = list(u_all[e]); own = own_all[e]
            t0 = [self.T0[s // S] for s in self.copies[e]]
            add = [0] * nc
            delta = self.k[e] - sum(u)
            while delta > 0:
                j = min(range(nc), key=lambda j: (t0[j] + add[j] * unit, j))
                u[j] += 1; add[j] += 1; delta -= 1
            while delta < 0:
                j = max((j for j in range(nc) if u[j] > 0),
                        key=lambda j: (t0[j] + add[j] * unit, -j))
                u[j] -= 1; add[j] -= 1; delta += 1
            if own == NO_PARTIAL or u[own] == 0:
                own = max((j for j in range(nc) if u[j] > 0),
                          key=lambda j: (t0[j], -j))
            for j, s in enumerate(self.copies[e]):
                self.y[s] = u[j] - (1 if j == own else 0)
            self.part[e] = self.copies[e][own]

    # -- auction ------------------------------------------------------------
    def auction(self):
        P, E, R, S = self.P, self.E, self.R, self.S
        self.rank_time(self.quota, self.T)
        tot = sum(self.T)
        if tot == 0:
            return 0
        # 1. per-slot best destination
        hit, ckv, cks, cki, ckh, ckp = {}, {}, {}, {}, {}, {}
        for s0 in range(P):
            e = self.phy[s0]
            nc = len(self.copies[e])
            r0 = s0 // S
            tmax = self.T[r0]
            hit[s0] = 0
            if nc > 1 and self.active[s0] and tmax * R > tot and self.has_unit(s0):
                shed, ip = self.out(s0)
                bv, bs, bi, found = 0, 0, 0, 0
                for k in range(nc):
                    s1 = self.copies[e][k]
                    if s1 == s0:
                        continue
                    in1 = self.inn(s1, ip)
                    v = self.T[s1 // S] + in1
                    if v < tmax:
                        if not found or v < bv or (v == bv and s1 < bs):
                            found, bv, bs, bi = 1, v, s1, in1
                if found:
                    hit[s0] = 1
                    ckv[s0], cks[s0], cki[s0], ckh[s0], ckp[s0] = bv, bs, bi, shed, ip
        # 2. offers per source rank
        offers = []
        base = {}
        for r0 in range(R):
            base[r0] = len(offers)
            tmax = self.T[r0]
            if tmax * R <= tot:
                continue
            excess = tmax * R - tot
            shed = 0
            for _step in range(S):
                best, found = None, False
                for j in range(S):
                    s0 = r0 * S + j
                    if not hit[s0]:
                        continue
                    v = ckv[s0]
                    after = tmax - ckh[s0]
                    key = max(after, v)
                    sl = cks[s0]
                    cur = (key, v, sl, s0)
                    if not found or cur < best:
                        found, best = True, cur
                if not found or shed * R >= excess:
                    break
                key, v, sl, s0 = best
                hit[s0] = 0
                offers.append(dict(rcv=sl // S, negt=-tmax, v=v, s0=s0,
                                   lim=tmax, s1=sl, inc=cki[s0], isp=ckp[s0]))
                shed += ckh[s0]
        if not offers:
            return 0
        # 3./4. per receiver: sort and accept a prefix
        acc = 0
        for rcv in range(R):
            grp = [o for o in offers if o['rcv'] == rcv]
            grp.sort(key=lambda o: (o['negt'], o['v'], o['s0']))
            cur = self.T[rcv]
            take = []
            for o in grp:
                nxt = cur + o['inc']
                if nxt >= o['lim']:
                    break
                if (nxt - self.slack) * R > tot:
                    break
                cur = nxt
                take.append(o)
            for o in take:
                self.move(o['s0'], o['s1'], o['isp'])
            acc += len(take)
        return acc


# ---------------------------------------------------------------------------
def make_phy(kind, E, R, rng):
    S = E // R
    if kind == 'identity':
        return np.arange(E)
    if kind == 'one_replica':          # one expert duplicated on another rank
        phy = list(range(E)) + [0]
        return np.asarray(phy[:E - 1] + [phy[E - 1]], np.int64)
    return np.asarray(sorted(range(E)), np.int64)


def random_layout(E, R, extra, rng):
    """E + extra slots: `extra` experts get a second copy on another rank."""
    S = E // R
    phy = list(range(E))
    hot = rng.choice(E, size=extra, replace=False)
    for e in hot:
        home = e // S
        other = rng.choice([r for r in range(R) if r != home])
        phy.append(int(e))
    phy = np.asarray(phy, np.int64)
    # reorder so each rank holds exactly S+extra/R slots, contiguous
    per = len(phy) // R
    if len(phy) % R:
        phy = np.concatenate([phy, np.zeros(R - len(phy) % R, np.int64)])
        per = len(phy) // R
    return phy, per


def eco_meta_copies(phy, E, R):
    return copies_of(phy, E, R)


def run_case(E, R, extra, counts, bm, wb, wa, prev_counts, prev_quotas,
             rounds=1, cold_rounds=6):
    rng = np.random.default_rng(0)
    phy, per = random_layout(E, R, extra, rng)
    P = len(phy)
    sim = KernelSim(phy, R, E, bm, wb, wa, rounds, cold_rounds)
    ref = UnitPlanner(phy, R, E, block_m=bm, wb=wb, wa=wa, rounds=rounds,
                      cold_rounds=cold_rounds, hop2_rounds=0)
    prev = None
    if prev_counts is not None:
        prev = (np.asarray(prev_quotas, np.int64), np.asarray(prev_counts, np.int64))
    got_q, got_start = sim.plan(counts, prev)
    want_q = ref.plan(counts, None if prev is None else prev)
    ok = np.array_equal(got_q, want_q)
    # conservation
    for e in range(E):
        tot = sum(got_q[s] for s in eco_meta_copies(phy, E, R)[e])
        if tot != counts[e]:
            ok = False
    return ok, got_q, want_q, phy, sim


def main():
    failures = 0
    failures += orig_cases()
    failures += big_cases()
    failures += small_fuzz()
    print(f'total failures {failures}')
    return failures


def big_cases():
    """EP8 / 25% redundancy shape: P = 320 slots, E = 256, bm = 64."""
    failures = 0
    rng = np.random.default_rng(7)
    E, R, extra = 256, 8, 64
    for bm in (16, 32, 64):
        for wb, wa in ((0, 0), (64, 0)):
            counts0 = rng.integers(0, 120, size=E).tolist()
            counts1 = rng.integers(0, 120, size=E).tolist()
            ok, got, want, phy, sim = run_case(E=E, R=R, extra=extra, counts=counts0,
                                               bm=bm, wb=wb, wa=wa,
                                               prev_counts=None, prev_quotas=None)
            if not ok:
                failures += 1
                print(f'FAIL big cold bm={bm} w={wb}/{wa}')
                continue
            sim2 = KernelSim(phy, R, E, bm, wb, wa, 1, 6)
            ref2 = UnitPlanner(phy, R, E, block_m=bm, wb=wb, wa=wa,
                               rounds=1, cold_rounds=6, hop2_rounds=0)
            prev = (got, np.asarray(counts0, np.int64))
            got2, start2 = sim2.plan(counts1, prev)
            want2 = ref2.plan(counts1, prev)
            if not np.array_equal(got2, want2):
                failures += 1
                print(f'FAIL big warm bm={bm} w={wb}/{wa}')
            for e in range(E):
                cs = eco_meta_copies(phy, E, R)[e]
                run = 0
                for s in cs:
                    if start2[s] != run:
                        failures += 1
                        print(f'FAIL start expert {e}: {start2[s]} != {run}')
                        break
                    run += got2[s]
                if run != counts1[e]:
                    failures += 1
                    print(f'FAIL conserve expert {e}: {run} != {counts1[e]}')
    return failures


def small_fuzz():
    """Random small layouts with different copy counts per expert."""
    failures = 0
    rng = np.random.default_rng(11)
    for trial in range(40):
        E = int(rng.choice([16, 32]))
        R = int(rng.choice([2, 4, 8]))
        if E % R:
            continue
        extra = int(rng.choice([0, R, 2 * R]))
        bm = int(rng.choice([16, 32, 64]))
        wb, wa = int(rng.choice([0, 8, 260])), int(rng.choice([0, 111]))
        counts = rng.integers(0, 400, size=E).tolist()
        counts2 = rng.integers(0, 400, size=E).tolist()
        ok, got, want, phy, sim = run_case(E=E, R=R, extra=extra, counts=counts,
                                           bm=bm, wb=wb, wa=wa,
                                           prev_counts=None, prev_quotas=None)
        if not ok:
            failures += 1
            print(f'FAIL fuzz cold E={E} R={R} x={extra} bm={bm} w={wb}/{wa}')
            continue
        sim2 = KernelSim(phy, R, E, bm, wb, wa, 1, 6)
        ref2 = UnitPlanner(phy, R, E, block_m=bm, wb=wb, wa=wa, rounds=1,
                           cold_rounds=6, hop2_rounds=0)
        prev = (got, np.asarray(counts, np.int64))
        got2, _ = sim2.plan(counts2, prev)
        want2 = ref2.plan(counts2, prev)
        if not np.array_equal(got2, want2):
            failures += 1
            print(f'FAIL fuzz warm E={E} R={R} x={extra} bm={bm} w={wb}/{wa}')
            diff = np.nonzero(got2 != want2)[0]
            print('   slots', diff[:8].tolist(), got2[diff[:8]].tolist(), want2[diff[:8]].tolist())
    return failures


def orig_cases():
    E, R = 32, 4
    cases = []
    rng = np.random.default_rng(1)

    # 1) cold plans under several loads and weights
    for bm in (16, 32, 64):
        for wb, wa in ((0, 0), (8, 0), (260, 111)):
            for trial in range(3):
                counts = rng.integers(0, 600, size=E).tolist()
                cases.append(dict(E=E, R=R, extra=4, counts=counts, bm=bm,
                                  wb=wb, wa=wa, prev_counts=None, prev_quotas=None,
                                  tag=f'cold bm={bm} w={wb}/{wa} t{trial}'))
    # 2) warm after a valid previous call
    for bm in (16, 32, 64):
        for wb, wa in ((0, 0), (8, 0), (260, 111)):
            counts0 = rng.integers(0, 600, size=E).tolist()
            counts1 = rng.integers(0, 600, size=E).tolist()
            cases.append(dict(E=E, R=R, extra=4, counts=counts0, bm=bm, wb=wb, wa=wa,
                              prev_counts=None, prev_quotas=None, tag=f'warmbase bm={bm} w={wb}/{wa}'))
            cases[-1]['chain'] = counts1

    failures = 0
    for c in cases:
        ok, got, want, phy, sim = run_case(**{k: v for k, v in c.items() if k != 'tag' and k != 'chain'})
        if not ok:
            failures += 1
            print(f'FAIL {c["tag"]}')
            print('  got ', got[:16].tolist())
            print('  want', want[:16].tolist())
        elif 'chain' in c:
            # second (warm) call on the state the first one produced
            E2, R2 = c['E'], c['R']
            phy_c = phy
            sim2 = KernelSim(phy_c, R2, E2, c['bm'], c['wb'], c['wa'], 1, 6)
            ref2 = UnitPlanner(phy_c, R2, E2, block_m=c['bm'], wb=c['wb'], wa=c['wa'],
                               rounds=1, cold_rounds=6, hop2_rounds=0)
            prev = (got, np.asarray(c['counts'], np.int64))
            got2, start2 = sim2.plan(c['chain'], prev)
            want2 = ref2.plan(c['chain'], prev)
            if not np.array_equal(got2, want2):
                failures += 1
                print(f'FAIL(warm) {c["tag"]}')
                print('  got ', got2[:16].tolist())
                print('  want', want2[:16].tolist())
            # quota_start must be contiguous inside each expert's copies
            for e in range(E2):
                cs = eco_meta_copies(phy_c, E2, R2)[e]
                run = 0
                for s in cs:
                    if start2[s] != run:
                        failures += 1
                        print(f'FAIL(start) expert {e} slot {s}: {start2[s]} != {run}')
                        break
                    run += got2[s]
    print(f'{len(cases)} original cases, {failures} failures')
    return failures


if __name__ == '__main__':
    sys.exit(1 if main() else 0)
