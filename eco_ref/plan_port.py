"""Line-by-line Python port of the ECO plan (unit_ref.py semantics).

Purpose: this is the *execution spec* for the SM90 CuTe DSL kernel. Every
function here maps 1:1 to a kernel function, and every one is unit-tested
against unit_ref.py. Writing it first means the kernel is a translation of
code that is already proven, not a fresh implementation.

Naming follows the task spec's function table:
    _eco_rank_time  _eco_costC  _eco_out  _eco_in  _eco_move
    _eco_emit_quota _eco_plan_assign _eco_do_cold _eco_do_warm
    _eco_do_auction_round _eco_do_accept
"""
from __future__ import annotations

import numpy as np


class PlanPort:
    """Mirror of UnitPlanner, written so the kernel can follow it line by line."""

    def __init__(self, phy, num_ranks, num_logical, block_m=64, wb=320, wa=54,
                 rounds=1, cold_rounds=6, hop2_rounds=0):
        phy = np.asarray(phy, np.int64)
        self.phy, self.P, self.R, self.E = phy, len(phy), num_ranks, num_logical
        self.S = self.P // num_ranks
        self.rank = np.arange(self.P) // self.S
        self.bm, self.wb, self.wa = block_m, wb, wa
        self.nr, self.nr_cold, self.hop2_rounds = rounds, cold_rounds, hop2_rounds
        self.slack = wb + block_m + wa
        # active copy: first copy of an expert on its rank; later ones are passive
        self.active = np.ones(self.P, bool)
        self.slot_of = np.full(self.E, -1, np.int64)
        copies = {}
        for e in range(self.E):
            seen, act = set(), []
            for s in np.nonzero(phy == e)[0]:
                r = int(self.rank[s])
                if r in seen:
                    self.active[s] = False
                    continue
                seen.add(r)
                act.append(int(s))
            copies[e] = act
            self.slot_of[e] = act[0] if act else -1
        self.copies = {e: cs for e, cs in copies.items() if len(cs) > 1}

    # ---- costs (unit_ref L49-78) -------------------------------------------
    def _eco_rank_time(self, q):
        q = np.asarray(q, np.int64)
        return np.bincount(self.rank,
                           self.wb * (-(-q // self.bm)) + self.wa * (q > 0) + q,
                           self.R).astype(np.int64)

    def _eco_costC(self, s, y, hp):
        blk = y + hp
        return (self.wb * blk + (self.wa if blk > 0 else 0)
                + y * self.bm + hp * self.rem[self.phy[s]])

    def _eco_out(self, s):
        y, hp = int(self.y[s]), int(self.part[self.phy[s]] == s)
        if hp:
            return self._eco_costC(s, y, 0) - self._eco_costC(s, y, 1), 1
        return self._eco_costC(s, y - 1, 0) - self._eco_costC(s, y, 0), 0

    def _eco_in(self, s, ispart):
        y, hp = int(self.y[s]), int(self.part[self.phy[s]] == s)
        return (self._eco_costC(s, y + (0 if ispart else 1), hp + ispart)
                - self._eco_costC(s, y, hp))

    def _eco_move(self, a, b, ispart):
        e = self.phy[a]
        if ispart:
            assert self.part[e] == a
            self.part[e] = b
        else:
            assert self.y[a] > 0
            self.y[a] -= 1
            self.y[b] += 1

    def _eco_emit_quota(self):
        q = self.y * self.bm
        m = self.part >= 0
        q[self.part[m]] += self.rem[m]
        return q

    def _eco_has_unit(self, s):
        return self.y[s] > 0 or self.part[self.phy[s]] == s

    # ---- plan (unit_ref L87-105) -------------------------------------------
    def _eco_plan_assign(self, n, prev=None):
        n = np.asarray(n, np.int64)
        self.n = n
        self.k = -(-n // self.bm)
        self.rem = n - (self.k - 1) * self.bm
        self.y = np.zeros(self.P, np.int64)
        self.part = np.full(self.E, -1, np.int64)
        # single-copy experts: their units are fixed up front
        for e in np.nonzero(n > 0)[0]:
            if int(e) not in self.copies:
                s0 = int(self.slot_of[e])
                self.y[s0] = self.k[e] - 1
                self.part[e] = s0
        if prev is None or int(np.asarray(prev[1]).sum()) == 0:
            self._eco_do_cold(n)
            return self._eco_rounds(self.nr_cold)
        self._eco_do_warm(n, np.asarray(prev[0], np.int64),
                          np.asarray(prev[1], np.int64))
        return self._eco_rounds(self.nr)

    # ---- cold (unit_ref L107-129) ------------------------------------------
    def _eco_do_cold(self, n):
        bm = self.bm
        x = np.zeros(self.P, np.int64)
        for e in range(self.E):
            if e not in self.copies and self.slot_of[e] >= 0:
                x[self.slot_of[e]] = n[e]
        for e, cs in self.copies.items():
            tot, k = int(n[e]), len(cs)
            x[cs] = tot // k
            x[cs[0]] += tot - (tot // k) * k
        t0 = self._eco_rank_time(x)
        for e, cs in self.copies.items():
            if n[e] == 0:
                continue
            g = [int(x[s]) // bm for s in cs]
            left = int(self.k[e]) - sum(g)
            order = sorted(range(len(cs)),
                           key=lambda j: (int(t0[self.rank[cs[j]]]) * bm
                                          - (int(x[cs[j]]) % bm) * (self.wb + bm), j))
            for i in range(left):
                g[order[i]] += 1
            jp = max((j for j in range(len(cs)) if g[j] > 0),
                     key=lambda j: (g[j] * bm - int(x[cs[j]]), -j))
            for j, s in enumerate(cs):
                self.y[s] = g[j] - (j == jp)
            self.part[e] = cs[jp]

    # ---- warm (unit_ref L131-173) ------------------------------------------
    def _eco_do_warm(self, n, qp, npv):
        bm, unit = self.bm, self.wb + self.bm
        keep = {}
        for e, cs in self.copies.items():
            if n[e] == 0:
                continue
            kp = -(-int(npv[e]) // bm)
            u, own = [0] * len(cs), -1
            if kp > 0:
                remp = int(npv[e]) - (kp - 1) * bm
                qs = [int(qp[s]) for s in cs]
                # congruent rule: only the partial owner has (quota - remp) % bm == 0
                own = next((j for j in range(len(cs))
                            if qs[j] >= remp and (qs[j] - remp) % bm == 0), -1)
                if own >= 0:
                    u = [(qs[j] - (remp if j == own else 0)) // bm + (j == own)
                         for j in range(len(cs))]
            keep[e] = (u, own)
        # estimate T0 from kept units + this call's partial sizes
        y0, part0 = self.y.copy(), self.part.copy()
        for e, (u, own) in keep.items():
            cs = self.copies[e]
            if own >= 0:
                for j, s in enumerate(cs):
                    y0[s] = u[j] - (j == own)
                part0[e] = cs[own]
        q0 = y0 * bm
        m = part0 >= 0
        q0[part0[m]] += self.rem[m]
        t0 = self._eco_rank_time(q0)
        for e, (u, own) in keep.items():
            cs = self.copies[e]
            u = list(u)
            delta = int(self.k[e]) - sum(u)
            add = [0] * len(cs)
            while delta > 0:
                j = min(range(len(cs)),
                        key=lambda j: (int(t0[self.rank[cs[j]]]) + add[j] * unit, j))
                u[j] += 1; add[j] += 1; delta -= 1
            while delta < 0:
                j = max((j for j in range(len(cs)) if u[j] > 0),
                        key=lambda j: (int(t0[self.rank[cs[j]]]) + add[j] * unit, -j))
                u[j] -= 1; add[j] -= 1; delta += 1
            # the partial goes to the copy holding a unit with the highest estimate
            if own < 0 or u[own] == 0:
                own = max((j for j in range(len(cs)) if u[j] > 0),
                          key=lambda j: (int(t0[self.rank[cs[j]]]), -j))
            for j, s in enumerate(cs):
                self.y[s] = u[j] - (j == own)
            self.part[e] = cs[own]

    # ---- auctions (unit_ref L176-265) --------------------------------------
    def _eco_rounds(self, nr):
        for k in range(nr):
            if self._eco_do_auction_round(k < self.hop2_rounds) == 0:
                break
        return self._eco_emit_quota()

    def _eco_do_auction_round(self, hop2):
        t = self._eco_rank_time(self._eco_emit_quota())
        R, rank, phy = self.R, self.rank, self.phy
        tot = int(t.sum())
        best_out = {}
        if hop2:
            for e, cs in self.copies.items():
                for s1 in cs:
                    if not self._eco_has_unit(s1):
                        continue
                    r1 = int(rank[s1])
                    d1, ip = self._eco_out(s1)
                    for s2 in cs:
                        if s2 != s1:
                            key = (int(t[rank[s2]]) + self._eco_in(s2, ip), s1, s2)
                            if r1 not in best_out or key < best_out[r1][0]:
                                best_out[r1] = (key, d1, self._eco_in(s2, ip), ip)
        offers = []
        for r0 in range(R):
            if int(t[r0]) * R <= tot:
                continue
            excess, lim = int(t[r0]) * R - tot, int(t[r0])
            best = {}
            for s0 in range(r0 * self.S, (r0 + 1) * self.S):
                e = int(phy[s0])
                if e not in self.copies or not self.active[s0] or not self._eco_has_unit(s0):
                    continue
                d0, ip = self._eco_out(s0)
                for s1 in self.copies[e]:
                    if s1 == s0:
                        continue
                    r1 = int(rank[s1]); in1 = self._eco_in(s1, ip)
                    c = ((int(t[r1]) + in1, 1, s1, r1, in1, None, 0)
                         if int(t[r1]) + in1 < lim else None)
                    if r1 in best_out:
                        (k2, s1b, s2), d1, in2, ip2 = best_out[r1]
                        if (rank[s2] != r0 and phy[s1b] != e
                                and int(t[r1]) + in1 + d1 < lim and k2 < lim):
                            c2 = (max(k2, int(t[r1]) + in1 + d1), 2, s1,
                                  int(rank[s2]), in2, (s1b, s2), ip2)
                            if c is None or c2[0] < c[0]:
                                c = c2
                    if c is not None and (s0 not in best or c[:3] < best[s0][2][:3]):
                        best[s0] = (int(t[r0]) + d0, d0, c)
            shed = 0
            for s0, (src_new, d0, c) in sorted(
                    best.items(),
                    key=lambda kv: (max(kv[1][0], kv[1][2][0]), kv[1][2][:3], kv[0])):
                if shed * R >= excess:
                    break
                offers.append((c[3], (-int(t[r0]), c[0], s0), s0, lim, c,
                               int(self.part[phy[s0]] == s0)))
                shed -= d0
        return self._eco_do_accept(offers, t, tot)

    def _eco_do_accept(self, offers, t, tot):
        R = self.R
        ck = lambda o: (o[1][0], o[2])
        win2 = {}
        for o in offers:
            if o[4][5] is not None and (o[4][5][0] not in win2
                                        or ck(o) < win2[o[4][5][0]]):
                win2[o[4][5][0]] = ck(o)
        drop = set()
        own = {o[2]: o for o in offers}
        for x, kw in win2.items():
            if x in own:
                drop.add(('own', x) if kw < ck(own[x]) else ('transit', x))
        groups = {}
        for o in sorted(offers, key=lambda o: (o[0], o[1])):
            groups.setdefault(o[0], []).append(o)
        acc = []
        for rcv, grp in groups.items():
            cur = int(t[rcv])
            for o in grp:
                c = o[4]
                if ('own', o[2]) in drop:
                    continue
                if c[5] is not None and (('transit', c[5][0]) in drop
                                         or win2[c[5][0]] != ck(o)):
                    continue
                if cur + c[4] >= o[3]:
                    break
                if (cur + c[4] - self.slack) * R > tot:
                    break
                cur += c[4]
                acc.append(o)
        for o in acc:
            self._eco_move(o[2], o[4][2], o[5])
            if o[4][5] is not None:
                self._eco_move(o[4][5][0], o[4][5][1], o[4][6])
        return len(acc)
