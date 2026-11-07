"""SM90 CuTe dispatch, register-A WGMMA compute and combine roles."""
from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass import Boolean, Float32, Int32, Int64, Uint8, Uint32, Uint64

from ..common.math import (
    abs_f32, bits_float, f32x2_to_e4m3x2_bits, round_bf16_f32,
    ue8m0_fp8_inv, ue8m0_fp8_scale_exp,
)
from ..mma.sm90 import (
    canonical_offset, decode_fragment, smem_descriptor,
    wgmma_rs_issue, wgmma_rs_k128, wgmma_rs_k32_pair, wgmma_rs_wait,
)
from ..ptx.ld_st import (
    atom_acq_rel_add_u32, atom_add_u32, atom_release_gpu_add_u32,
    atom_shared_add_u32, copy_global_16, ld_acquire_gpu_u32,
    ld_acquire_sys_u32_sm90 as ld_acquire_sys_u32, ld_shared_u32x4_sm90 as load_shared_u32x4, load_bf16_8, load_scale_pair,
    load_shared_4, load_shared_8, st_release_u32, store_bf16_8,
    atom_add_u64, ld_volatile_u64, st_global_u64,
    red_global_min_u64, red_global_max_u64,
)
from ..ptx.tma import (
    bulk_g2s_1d, bulk_s2g_1d, copy_arrive, copy_async_4, copy_async_16,
    copy_commit, copy_wait, tma_store_fence,
)
from ..ptx.utils import (
    fence_gpu, fence_sys, sync_loader, sync_math, sync_tile,
    sync_activation, sync_weight, read_globaltimer, sync_aligned,
    elect_one_sync,
)
from ..scheduler.sm90_gc_moe import claim_sm90_task
from ..layout.sm90_gc_moe import get_role_shared_layout


@cute.jit
def cell(base: Uint64, offset: Uint64, dtype: cutlass.Constexpr):
    return cute.make_tensor(cute.make_ptr(dtype, base + offset, cute.AddressSpace.gmem,
                                         assumed_align=1), cute.make_layout((1,)))


@cute.jit
def shared_cell(slab: cute.Pointer, offset: Uint32, dtype: cutlass.Constexpr):
    return cute.make_tensor(cute.recast_ptr(slab + offset, dtype=dtype), cute.make_layout((1,)))


@cute.jit
def wait_epoch(ptr: cute.Pointer, target: Uint32, system: cutlass.Constexpr):
    # Same-order all-rank calls and the tail rendezvous prevent skipping a
    # still-needed epoch. Equality also remains valid at uint32 wraparound.
    if cutlass.const_expr(system):
        value = ld_acquire_sys_u32(ptr)
    else:
        value = ld_acquire_gpu_u32(ptr)
    while value != target:
        if cutlass.const_expr(system):
            value = ld_acquire_sys_u32(ptr)
        else:
            value = ld_acquire_gpu_u32(ptr)


@cute.jit
def wait_count(ptr: cute.Pointer, target: Uint32):
    value = ld_acquire_gpu_u32(ptr)
    while value < target:
        value = ld_acquire_gpu_u32(ptr)


class SM90GCMoERoleKernel:
    def __init__(self, config, layout, role, schedule, has_stats, routed_n_factor=1):
        self.role = role
        self.direct_route_scan = 0 < config.bound <= 2
        self.ranks = config.ranks
        # Logical experts per rank (for stats / weight layout) vs physical slots
        # per rank (what the dispatch loop walks). They are equal unless EPLB
        # added replica slots.
        self.logical_local = config.experts // config.ranks
        self.physical = int(getattr(config, 'physical', 0) or config.experts)
        self.local = self.physical // config.ranks
        self.replicated = self.physical > config.experts
        self.capacity, self.topk = config.capacity, config.topk
        self.hidden, self.intermediate = config.hidden, config.intermediate
        self.shared_experts = config.shared_experts
        self.shared_i = config.intermediate * config.shared_experts
        # Internal l1/l2 activation SF views store RN(originalSF *64).
        # Quantization inverse and public source x_sf retain original SF.
        self.rows = config.ranks * config.capacity
        self.fast_math, self.has_stats = config.fast_math, has_stats
        self.warp_specialized = ((role == 2 and schedule.threads[1] == 384)
                                 or role == 4)
        self.max_m = schedule.tile_m
        if routed_n_factor not in (1, 2) or (routed_n_factor == 2 and not (
                role == 2 and not self.warp_specialized and self.max_m == 8
                and self.hidden % 512 == 0 and self.intermediate % 256 == 0)):
            raise ValueError('invalid SM90 routed N factor for the role geometry')
        self.routed_n_factor = routed_n_factor
        shared_layout = get_role_shared_layout(self.max_m, self.warp_specialized,
                                               self.routed_n_factor)
        for name in shared_layout.__dataclass_fields__:
            setattr(self, name, getattr(shared_layout, name))
        for name, spec in layout.views.items():
            setattr(self, 'off_' + name, spec.offset)
        # ── ECO (EPLB placement + online plan) ───────────────────────────
        # [ECO-SM90] role 5 is the plan CTA. `eco_logical_experts > 0` turns
        # ECO on; the plan role only exists when it is on.
        self.eco = int(getattr(config, 'eco_logical_experts', 0)) > 0
        self.eco_logical = int(getattr(config, 'eco_logical_experts', 0))
        self.eco_wb = int(getattr(config, 'eco_wb', 0))
        self.eco_wa = int(getattr(config, 'eco_wa', 0))
        self.eco_rounds = int(getattr(config, 'eco_rounds', 1))
        self.eco_cold_rounds = int(getattr(config, 'eco_cold_rounds', 6))
        self.eco_max_copies = int(getattr(config, 'eco_max_copies', 8))
        self.eco_hop2_rounds = int(getattr(config, 'eco_hop2_rounds', 0))
        # [ECO-SM90] GC_MOE_ECO_DEBUG dumps the cold-start intermediates so a
        # GPU run can be diffed against the CPU reference.
        import os as _os
        # [ECO-SM90] GC_MOE_ECO_NOSPLIT is a compile-time triage switch: replica
        # slots pull their whole stream without waiting for the plan. It is
        # NEVER set in a real run (it defeats the plan); it exists so the
        # plan handshake can be isolated from the split.
        self.eco_nosplit = _os.environ.get('GC_MOE_ECO_NOSPLIT') == '1'
        # [ECO-MEAS] cost-decomposition gates (compile-time, no runtime branch)
        self.eco_null_plan = bool(getattr(config, 'eco_null_plan', False))
        self.eco_no_auction = bool(getattr(config, 'eco_no_auction', False))
        # [ECO-MEAS] timeline trace: COMPILE-TIME off unless requested, so the
        # default build carries no clock read at all.
        # [ECO-TRACE] Derived from the LAYOUT, not the env: the timeline table
        # only exists when the buffer was built with `trace=True`, and a kernel
        # that wrote it otherwise would touch an offset that was never
        # reserved (that is exactly how `off_eco_tl_ctr` went missing).
        self.eco_trace = 'eco_tl' in getattr(layout, 'views', {})
        # [PULL-TRACE] Independent of ECO: measures the pull phase on the
        # plain (no-replica) path too, which had no instrumentation before.
        self.pull_trace = _os.environ.get('GC_MOE_PULL_TRACE') == '1'
        # [ECO-PULL] A/B switch for the token copy path. Default ON (TMA); set
        # GC_MOE_SM90_PULL_TMA=0 to compile the upstream 16-byte synchronous
        # loop instead. Compile-time, so the disabled build pays nothing.
        self.pull_tma = _os.environ.get('GC_MOE_SM90_PULL_TMA', '1') == '1'
        # [ECO-MEAS] triage: GC_MOE_ECO_TRACE_RESET=0 keeps the trace points but
        # skips the per-call window clear, to separate "the instrument's writes
        # perturb the run" from "the trace points themselves".
        self.eco_trace_reset = _os.environ.get('GC_MOE_ECO_TRACE_RESET', '1') == '1'
        # slack = wb + bm + wa (unit_ref.py L30)
        self.eco_slack = self.eco_wb + schedule.tile_m + self.eco_wa
        if self.eco:
            self._eco_smem_offsets()
        if role == 5:
            # [ECO-SM90] The plan CTA has its own resources (one CTA) and does
            # not index the three-entry role schedule.
            #
            # 512 threads = 16 warps, matching SM100's `kPlanWarps = 16`
            # (`eco_unit_kernel.cu` L27): warps 0..15 run the plan, warps 1..15
            # also gather the counts, and warp 0 stages the static tables.
            # That split is what lets the counts overlap the static tables.
            self.grid, self.threads = 1, 512
            self.smem = self.eco_smem if self.eco else 0
        else:
            resource_role = 1 if role == 4 else role - 1
            self.grid, self.threads = schedule.grids[resource_role], schedule.threads[resource_role]
            self.smem = schedule.smem[resource_role]
        if self.routed_n_factor == 2 and 2 * self.stage_bytes + 128 > self.smem:
            raise ValueError('SM90 routed N512 requires two complete stages and control space')
        if role == 4:
            shared_stride = (36864 + self.max_m * 144 + 1023) // 1024 * 1024
            assert 2 * shared_stride + 128 <= self.smem
        self.cleanup_grids = min(self.grid, (self.local + 127) // 128)
        self.finalize_flags = layout.views['combine_block_done'].shape[0]
        if role == 3 and self.grid > self.finalize_flags:
            raise ValueError('combine grid exceeds the SM90 completion-flag capacity')

    def _eco_smem_offsets(self) -> None:
        """# [ECO-SM90] Compile-time shared-memory layout of the plan CTA.

        Every array the plan touches is listed here; nothing is allocated at
        runtime. All offsets are byte offsets from the slab base and every
        array holds int32 words (signed values as two's complement). Each array
        has a single writer per barrier interval, so no shared-memory atomics
        are needed and the result is deterministic.
        """
        P, E, R = self.physical, self.eco_logical, self.ranks
        if E > P or P % R:
            raise ValueError('ECO needs P % R == 0 and at least one slot per expert')
        off = 0
        def take(count: int) -> int:
            nonlocal off
            here = off
            off += count * 4
            return here
        S = P // R
        self.eco_y_off = take(P)        # units per slot
        self.eco_part_off = take(E)     # slot owning the partial unit (0xFFFFFFFF = none)
        self.eco_rem_off = take(E)      # partial size, per expert
        self.eco_k_off = take(E)        # unit count, per expert
        self.eco_n_off = take(E)        # token count, per expert
        self.eco_pn_off = take(E)       # warm: the estimate's partial owner
        self.eco_x_off = take(P)        # cold: even split / warm: q0
        self.eco_g_off = take(P)        # cold: units per copy / warm: y0
        self.eco_u_off = take(P)        # warm: inherited units per copy
        self.eco_phy_off = take(P)      # slot -> logical expert
        self.eco_act_off = take(P)      # slot is an active (planned) copy
        self.eco_T_off = take(R)        # rank time
        self.eco_T0_off = take(R)       # rank time estimate (cold / warm start)
        self.eco_cks_off = take(P)      # auction: candidate destination slot
        self.eco_ckv_off = take(P)      # auction: candidate value (receiver time)
        self.eco_cki_off = take(P)      # auction: candidate receiver increase
        self.eco_ckh_off = take(P)      # auction: candidate shed (source decrease)
        self.eco_ckp_off = take(P)      # auction: candidate unit is the partial (1/0)
        self.eco_hit_off = take(P)      # auction: candidate present
        self.eco_of_off = take(P * 8)   # auction: offers, 8 words each, base r*S
        self.eco_rcnt_off = take(R)     # auction: offers per source rank
        self.eco_gcnt_off = take(R)     # auction: offers per receiver / fill cursor
        self.eco_gof_off = take(R + 1)  # auction: per-receiver segment base
        self.eco_grp_off = take(P)      # auction: offer indices grouped by receiver
        self.eco_acc_off = take(P)      # auction: accepted flag per offer slot
        self.eco_nof_off = take(1)      # auction: number of offers
        self.eco_nacc_off = take(1)     # auction: number of accepted offers
        # [ECO-AUC] packed 64-bit key array: needs 8-byte alignment, so pad the
        # running offset before taking it. `take()` hands out int32-word
        # offsets, and the preceding arrays left this at a 4-byte boundary --
        # an unaligned Uint64 store silently corrupts the plan's smem.
        off = (off + 7) // 8 * 8
        self.eco_key_off = take(P * 2)  # auction: packed 64-bit total-order key
        self.eco_quota_off = take(P)    # quota per slot
        self.eco_tot_off = take(1)      # sum of T
        self.eco_flag_off = take(1)     # warm (1) / cold (0)
        self.eco_smem = (off + 1023) // 1024 * 1024
        if self.eco_smem > 200 * 1024:
            raise ValueError(f'ECO plan smem {self.eco_smem} exceeds the 200 KB budget')


    @cute.jit
    def _eco_s(self, slab: cute.Pointer, off: cutlass.Constexpr, idx: Uint32):
        """Read one int32 word of the plan's shared-memory slab (as Uint32).

        The explicit conversion keeps the DSL's value type stable when the same
        variable is also assigned from another expression.
        """
        return Uint32(shared_cell(slab, Uint32(off) + idx * Uint32(4), Uint32)[0])

    @cute.jit
    def _eco_set(self, slab: cute.Pointer, off: cutlass.Constexpr, idx: Uint32,
                 value: Uint32):
        shared_cell(slab, Uint32(off) + idx * Uint32(4), Uint32)[0] = value

    @cute.jit
    def _eco_inc(self, slab: cute.Pointer, off: cutlass.Constexpr, idx: Uint32,
                 value: Uint32):
        """Atomic add into shared memory; returns the previous value."""
        return atom_shared_add_u32(
            cute.recast_ptr(slab + Uint32(off) + idx * Uint32(4), dtype=Uint32),
            value)

    @cute.jit
    def _eco_meta_word(self, base: Uint64, idx: Uint32):
        """One int32 word of the host-built EcoMeta."""
        return cell(base, Uint64(self.off_eco_meta) + Uint64(idx) * Uint64(4),
                    Uint32)[0]

    @cute.jit
    def _eco_copy(self, base: Uint64, e: Uint32, k):
        """copies[e][k] as an int16 half-word (spec §6.3: SM90 has no Uint16).

        `k` is always a compile-time copy index (a literal or a
        `range_constexpr` loop variable), so the half-word selection folds at
        code-read time. SM90's DSL has no Uint16, hence the u32 read.
        """
        word = self._eco_meta_word(base, Uint32(16) + e * Uint32(16)
                                   + Uint32(k // 2))
        # select the half-word without branching: the DSL forbids an early
        # return inside a staged `if`
        return (word >> (Uint32(k % 2) * Uint32(16))) & Uint32(0xFFFF)

    @cute.jit
    def _eco_slot_of(self, base: Uint64, e: Uint32, j):
        """Physical slot of the expert's j-th ACTIVE copy."""
        return self._eco_copy(base, e, j) & Uint32(0x7FFF)

    @cute.jit
    def _eco_ncopies(self, base: Uint64, e: Uint32):
        """Number of ACTIVE copies of expert `e` (unit_ref.py L36).

        `copies[e]` lists the active copies first and then the passive ones
        (bit 15 set); passive copies are never planned, so they must not be
        counted here.
        """
        n = Uint32(0)
        for k in cutlass.range_constexpr(self.eco_max_copies):
            raw = self._eco_copy(base, e, k)
            if raw != Uint32(0x7FFF) and (raw & Uint32(0x8000)) == Uint32(0):
                n = Uint32(k + 1)
        return n

    @cute.jit
    def _eco_phy(self, base: Uint64, slot: Uint32):
        """phy2log[slot]."""
        return self._eco_meta_word(
            base, Uint32(16) + Uint32(self.eco_logical) * Uint32(16) + slot)

    @cute.jit
    def _eco_num_fixed(self, base: Uint64, rank: Uint32):
        """EcoMeta.num_fixed[rank]: single-copy slots are the rank's prefix."""
        return self._eco_meta_word(
            base, Uint32(16) + Uint32(self.eco_logical) * Uint32(16)
            + Uint32(self.physical) + rank)

    @cute.jit
    def _eco_rank_time(self, slab: cute.Pointer, quota_off: cutlass.Constexpr,
                       out_off: cutlass.Constexpr):
        """T[r] = sum_slots [wb*ceil(q/bm) + wa*(q>0) + q]  (unit_ref L49-51)."""
        tid, _, _ = cute.arch.thread_idx()
        R, P = self.ranks, self.physical
        S = P // R
        for r in cutlass.range(tid, R, self.threads):
            self._eco_set(slab, out_off, r, Uint32(0))
        cute.arch.sync_threads()
        for s in cutlass.range(tid, P, self.threads):
            q = self._eco_s(slab, quota_off, s)
            blocks = (q + Uint32(self.max_m - 1)) // Uint32(self.max_m)
            active = Uint32(1) if q > Uint32(0) else Uint32(0)
            cost = Uint32(self.eco_wb) * blocks + Uint32(self.eco_wa) * active + q
            self._eco_inc(slab, out_off, s // Uint32(S), cost)
        cute.arch.sync_threads()

    @cute.jit
    def _eco_emit_quota(self, slab: cute.Pointer):
        """quota[s] = y[s]*bm + (rem[e] if part[e] == s else 0) (unit_ref L80-84)."""
        tid, _, _ = cute.arch.thread_idx()
        P = self.physical
        bm = self.max_m
        for s in cutlass.range(tid, P, self.threads):
            phy = self._eco_s(slab, self.eco_phy_off, s)
            y = self._eco_s(slab, self.eco_y_off, s)
            part = self._eco_s(slab, self.eco_part_off, phy)
            rem = self._eco_s(slab, self.eco_rem_off, phy)
            q = y * Uint32(bm)
            if part == s:
                q = q + rem
            self._eco_set(slab, self.eco_quota_off, s, q)
        cute.arch.sync_threads()

    # ---- costs (unit_ref L53-78) --------------------------------------
    @cute.jit
    def _eco_costC(self, slab: cute.Pointer, s: Uint32, y: Uint32, hp: Uint32):
        """wb*(y+hp) + wa*[y+hp>0] + y*bm + hp*rem[phy[s]] (unit_ref L53-55)."""
        e = self._eco_s(slab, self.eco_phy_off, s)
        rem = self._eco_s(slab, self.eco_rem_off, e)
        blk = y + hp
        return (Uint32(self.eco_wb) * blk
                + (Uint32(self.eco_wa) if blk > Uint32(0) else Uint32(0))
                + y * Uint32(self.max_m) + hp * rem)

    @cute.jit
    def _eco_hp(self, slab: cute.Pointer, s: Uint32):
        """1 when slot `s` owns its expert's partial unit."""
        e = self._eco_s(slab, self.eco_phy_off, s)
        return Uint32(self._eco_s(slab, self.eco_part_off, e) == s)

    @cute.jit
    def _eco_has_unit(self, slab: cute.Pointer, s: Uint32):
        """y[s] > 0 or s owns the partial unit (unit_ref L57)."""
        return Uint32(self._eco_s(slab, self.eco_y_off, s) > Uint32(0)) | self._eco_hp(slab, s)

    @cute.jit
    def _eco_out(self, slab: cute.Pointer, s: Uint32):
        """Shed when one unit leaves `s` = cost(before) - cost(after) > 0 (L60-64).

        Returns (shed, is_partial); shed is non-negative because a unit's
        contribution to its rank's time is strictly positive.
        """
        y = self._eco_s(slab, self.eco_y_off, s)
        hp = self._eco_hp(slab, s)
        # Both branches are pre-declared: the DSL requires values used after a
        # staged `if` to exist before it. cost(partial) - cost(no partial) is
        # what leaves when the partial is shed; otherwise one full unit leaves.
        d_part = (self._eco_costC(slab, s, y, Uint32(1))
                  - self._eco_costC(slab, s, y, Uint32(0)))
        d_full = (self._eco_costC(slab, s, y, Uint32(0))
                  - self._eco_costC(slab, s, y - Uint32(1), Uint32(0)))
        shed = d_part if hp == Uint32(1) else d_full
        return shed, hp

    @cute.jit
    def _eco_in(self, slab: cute.Pointer, s: Uint32, ispart: Uint32):
        """Increase when one unit of type `ispart` enters `s` (L66-68)."""
        y = self._eco_s(slab, self.eco_y_off, s)
        hp = self._eco_hp(slab, s)
        y2 = y if ispart == Uint32(1) else y + Uint32(1)
        return (self._eco_costC(slab, s, y2, hp + ispart)
                - self._eco_costC(slab, s, y, hp))

    @cute.jit
    def _eco_move(self, slab: cute.Pointer, a: Uint32, b: Uint32, ispart: Uint32):
        """Move the partial (a owns it) or one full unit from slot a to slot b (L70-78)."""
        e = self._eco_s(slab, self.eco_phy_off, a)
        if ispart == Uint32(1):
            self._eco_set(slab, self.eco_part_off, e, b)
        else:
            self._eco_set(slab, self.eco_y_off, a,
                          self._eco_s(slab, self.eco_y_off, a) - Uint32(1))
            self._eco_set(slab, self.eco_y_off, b,
                          self._eco_s(slab, self.eco_y_off, b) + Uint32(1))

    # ---- cold start (unit_ref L107-129) -------------------------------
    @cute.jit
    def _eco_cold_key(self, base: Uint64, slab: cute.Pointer, e: Uint32,
                      xs, j: cutlass.Constexpr):
        """t0[rank]*bm - (x%bm)*(wb+bm) (unit_ref L118)."""
        s = self._eco_slot_of(base, e, j)
        S = self.physical // self.ranks
        t0 = self._eco_s(slab, self.eco_T0_off, s // Uint32(S))
        x = xs[j]
        # SIGNED: unit_ref's key `t0*bm - (x%bm)*(wb+bm)` is routinely negative,
        # and a Uint32 subtraction would wrap it to a huge positive value and
        # invert the ordering.
        return (Int32(t0 * Uint32(self.max_m))
                - Int32((x % Uint32(self.max_m)) * Uint32(self.eco_wb + self.max_m)))

    @cute.jit
    def _eco_do_cold(self, base: Uint64, slab: cute.Pointer):
        """Even split, load-aware rounding to units, partial owner (L107-129)."""
        tid, _, _ = cute.arch.thread_idx()
        P, E = self.physical, self.eco_logical
        bm = self.max_m

        # x[s]: even split over the ACTIVE copies; the first copy takes the rest
        for s in cutlass.range(tid, P, self.threads):
            phy = self._eco_s(slab, self.eco_phy_off, s)
            nc = self._eco_ncopies(base, phy)
            n = self._eco_s(slab, self.eco_n_off, phy)
            x = Uint32(0)
            if self._eco_s(slab, self.eco_act_off, s) != Uint32(0) and nc > Uint32(0):
                x = n // nc
                if self._eco_slot_of(base, phy, 0) == s:
                    x = x + (n - (n // nc) * nc)
            self._eco_set(slab, self.eco_x_off, s, x)
        cute.arch.sync_threads()
        self._eco_rank_time(slab, self.eco_x_off, self.eco_T0_off)
        # [ECO-SM90] debug: expose the cold-start estimate (x and T0) so a GPU
        # run can be diffed against the CPU reference bit for bit.
        # per replicated expert (one thread each)
        for e in cutlass.range(tid, E, self.threads):
            n = self._eco_s(slab, self.eco_n_off, e)
            nc = self._eco_ncopies(base, e)
            if nc > Uint32(1) and n > Uint32(0):
                k = self._eco_s(slab, self.eco_k_off, e)
                g = cute.make_rmem_tensor((4,), Uint32)
                xs = cute.make_rmem_tensor((4,), Uint32)
                order = cute.make_rmem_tensor((4,), Uint32)
                total_g = Uint32(0)
                for j in cutlass.range_constexpr(4):
                    v = Uint32(0)
                    xv = Uint32(0)
                    if Uint32(j) < nc:
                        s = self._eco_slot_of(base, e, j)
                        xv = self._eco_s(slab, self.eco_x_off, s)
                        v = xv // Uint32(bm)
                    g[j] = v
                    xs[j] = xv
                    total_g = total_g + v
                    order[j] = Uint32(j)
                # sum(floor(x/bm)) <= ceil(n/bm) always holds, so left >= 0.
                left = Uint32(0) if k < total_g else k - total_g
                # Selection sort on (key, j); positions >= nc keep their order
                # (unit_ref only sorts the expert's real copies).
                for i in cutlass.range_constexpr(4):
                    if Uint32(i) < nc:
                        best = Uint32(i)
                        best_key = Int32(self._eco_cold_key(base, slab, e, xs, i))
                        for j in cutlass.range_constexpr(4):
                            if cutlass.const_expr(j > i):
                                if Uint32(j) < nc:
                                    key = Int32(self._eco_cold_key(base, slab, e, xs, j))
                                    if key < best_key:
                                        best = Uint32(j)
                                        best_key = key
                        swap = order[i]
                        order[i] = order[best]
                        order[best] = swap
                for i in cutlass.range_constexpr(4):
                    if Uint32(i) < left:
                        j = order[i]
                        g[j] = g[j] + Uint32(1)
                # jp: max (g*bm - x); ties go to the lowest copy index
                # jp = argmax (g*bm - x), ties to the lowest copy index. The
                # difference is SIGNED (it is negative whenever a copy was
                # rounded down), so it must not be compared as Uint32.
                jp = Uint32(0)
                best_val = Int32(0)
                seen = Uint32(0)
                for j in cutlass.range_constexpr(4):
                    if Uint32(j) < nc and g[j] > Uint32(0):
                        val = Int32(g[j] * Uint32(bm)) - Int32(xs[j])
                        if seen == Uint32(0) or val > best_val:
                            best_val = val
                            jp = Uint32(j)
                            seen = Uint32(1)
                for j in cutlass.range_constexpr(4):
                    if Uint32(j) < nc:
                        s = self._eco_slot_of(base, e, j)
                        self._eco_set(slab, self.eco_y_off, s,
                                      g[j] - (Uint32(1) if j == jp else Uint32(0)))
                self._eco_set(slab, self.eco_part_off, e,
                              self._eco_slot_of(base, e, jp))
        cute.arch.sync_threads()

    # ---- warm start (unit_ref L131-173) -------------------------------
    @cute.jit
    def _eco_do_warm(self, base: Uint64, slab: cute.Pointer):
        """Congruent inverse of the previous plan, then delta adjust (L131-173).

        The previous quotas live in `eco_state[0:P]` and the previous counts in
        `eco_state[P:P+E]`, both written at the end of the previous call. One
        thread per replicated expert; the unit arrays are registers.
        """
        tid, _, _ = cute.arch.thread_idx()
        P, E = self.physical, self.eco_logical
        bm = self.max_m
        unit = Uint32(self.eco_wb + self.max_m)
        S = P // self.ranks
        state = self._field(base, 'eco_state', Uint32)

        # y0 / part0 start from the fixed single-copy state (plan step 4)
        for s in cutlass.range(tid, P, self.threads):
            self._eco_set(slab, self.eco_g_off, s, self._eco_s(slab, self.eco_y_off, s))
            self._eco_set(slab, self.eco_u_off, s, Uint32(0))
        for e in cutlass.range(tid, E, self.threads):
            self._eco_set(slab, self.eco_pn_off, e,
                          self._eco_s(slab, self.eco_part_off, e))
        cute.arch.sync_threads()

        # (1) congruent inverse: which copy held the previous partial?
        #     u[j] = (qs[j] - [j == own] * remp) / bm + [j == own]
        #     own == -1 (inconsistent previous plan) drops the expert's units.
        for e in cutlass.range(tid, E, self.threads):
            n = self._eco_s(slab, self.eco_n_off, e)
            nc = self._eco_ncopies(base, e)
            if nc > Uint32(1) and n > Uint32(0):
                npv = state[P + e]
                kp = (npv + Uint32(bm - 1)) // Uint32(bm)
                own = Uint32(0xFFFFFFFF)
                remp = Uint32(0)
                if kp > Uint32(0):
                    remp = npv - (kp - Uint32(1)) * Uint32(bm)
                    for j in cutlass.range_constexpr(4):
                        if Uint32(j) < nc:
                            qs = state[self._eco_slot_of(base, e, j)]
                            # only the partial owner has (quota - remp) % bm == 0
                            if (own == Uint32(0xFFFFFFFF) and qs >= remp
                                    and (qs - remp) % Uint32(bm) == Uint32(0)):
                                own = Uint32(j)
                if own != Uint32(0xFFFFFFFF):
                    for j in cutlass.range_constexpr(4):
                        if Uint32(j) < nc:
                            s = self._eco_slot_of(base, e, j)
                            qs0 = state[s] - (remp if j == own else Uint32(0))
                            u = qs0 // Uint32(bm) + (Uint32(1) if j == own else Uint32(0))
                            self._eco_set(slab, self.eco_u_off, s, u)
                            self._eco_set(slab, self.eco_g_off, s,
                                          u - (Uint32(1) if j == own else Uint32(0)))
                    self._eco_set(slab, self.eco_pn_off, e,
                                  self._eco_slot_of(base, e, own))
        cute.arch.sync_threads()

        # (2) rank-time estimate from the kept units + this call's partial sizes
        for s in cutlass.range(tid, P, self.threads):
            e = self._eco_s(slab, self.eco_phy_off, s)
            q0 = self._eco_s(slab, self.eco_g_off, s) * Uint32(bm)
            if self._eco_s(slab, self.eco_pn_off, e) == s:
                q0 = q0 + self._eco_s(slab, self.eco_rem_off, e)
            self._eco_set(slab, self.eco_x_off, s, q0)
        cute.arch.sync_threads()
        self._eco_rank_time(slab, self.eco_x_off, self.eco_T0_off)

        # (3) delta = k - sum(u) against ONE estimate, then move the partial
        for e in cutlass.range(tid, E, self.threads):
            n = self._eco_s(slab, self.eco_n_off, e)
            nc = self._eco_ncopies(base, e)
            if nc > Uint32(1) and n > Uint32(0):
                u = cute.make_rmem_tensor((4,), Uint32)
                add = cute.make_rmem_tensor((4,), Uint32)
                t0 = cute.make_rmem_tensor((4,), Uint32)
                total_u = Uint32(0)
                own = Uint32(0xFFFFFFFF)
                pn = self._eco_s(slab, self.eco_pn_off, e)
                for j in cutlass.range_constexpr(4):
                    v = Uint32(0)
                    t = Uint32(0)
                    if Uint32(j) < nc:
                        s = self._eco_slot_of(base, e, j)
                        v = self._eco_s(slab, self.eco_u_off, s)
                        t = self._eco_s(slab, self.eco_T0_off, s // Uint32(S))
                        if s == pn:
                            own = Uint32(j)
                    u[j] = v
                    add[j] = Uint32(0)
                    t0[j] = t
                    total_u = total_u + v
                k = self._eco_s(slab, self.eco_k_off, e)
                # delta > 0: give units to the lowest (t0 + additions * unit)
                for _step in cutlass.range(k):
                    if total_u + Uint32(_step) < k:
                        bj = Uint32(0)
                        bk = Uint32(0xFFFFFFFF)
                        for j in cutlass.range_constexpr(4):
                            if Uint32(j) < nc:
                                key = t0[j] + add[j] * unit
                                if key < bk:
                                    bk = key
                                    bj = Uint32(j)
                        u[bj] = u[bj] + Uint32(1)
                        add[bj] = add[bj] + Uint32(1)
                # delta < 0: take units from the highest (t0 + additions * unit)
                for _step in cutlass.range(total_u):
                    if total_u > k + Uint32(_step):
                        bj = Uint32(0)
                        bk = Uint32(0)
                        found = Uint32(0)
                        for j in cutlass.range_constexpr(4):
                            if Uint32(j) < nc and u[j] > Uint32(0):
                                key = t0[j] + add[j] * unit
                                # max (estimate, -j): ties pick the LOWEST index
                                if found == Uint32(0) or key > bk:
                                    bk = key
                                    bj = Uint32(j)
                                    found = Uint32(1)
                        if found != Uint32(0):
                            u[bj] = u[bj] - Uint32(1)
                            add[bj] = add[bj] - Uint32(1)
                # the partial stays with its owner while it holds a unit, else it
                # moves to the holding copy on the highest-estimate rank
                if own == Uint32(0xFFFFFFFF) or u[own] == Uint32(0):
                    bj = Uint32(0)
                    bk = Uint32(0)
                    found = Uint32(0)
                    for j in cutlass.range_constexpr(4):
                        if Uint32(j) < nc and u[j] > Uint32(0):
                            if found == Uint32(0) or t0[j] > bk:
                                bk = t0[j]
                                bj = Uint32(j)
                                found = Uint32(1)
                    own = bj
                for j in cutlass.range_constexpr(4):
                    if Uint32(j) < nc:
                        s = self._eco_slot_of(base, e, j)
                        self._eco_set(slab, self.eco_y_off, s,
                                      u[j] - (Uint32(1) if j == own else Uint32(0)))
                self._eco_set(slab, self.eco_part_off, e,
                              self._eco_slot_of(base, e, own))
        cute.arch.sync_threads()

    # ---- auction (unit_ref L176-265, one-hop rounds) -------------------
    @cute.jit
    def _eco_of(self, slab: cute.Pointer, idx: Uint32, field: cutlass.Constexpr):
        """Offer word `field` of offer `idx`."""
        return self._eco_s(slab, self.eco_of_off, idx * Uint32(8) + Uint32(field))

    @cute.jit
    def _eco_setof(self, slab: cute.Pointer, idx: Uint32, field: cutlass.Constexpr,
                   value: Uint32):
        self._eco_set(slab, self.eco_of_off, idx * Uint32(8) + Uint32(field), value)

    @cute.jit
    def _eco_do_auction_round(self, base: Uint64, slab: cute.Pointer):
        """One auction round; returns the number of accepted moves (L182-265).

        This is `unit_ref._auction_round` with hop2 = False. EP8/EP16 have a
        complete rank graph, so every two-hop path has an equivalent one-hop
        move; the SM100 build also uses zero two-hop rounds at these EP sizes
        (`eco_unit_config`), so nothing is simplified away.

        Offer layout (8 words): 0 receiver, 1 -t_source, 2 value,
        3 source slot, 4 limit (= t_source), 5 destination slot,
        6 receiver increase, 7 partial bit.
        """
        tid, _, _ = cute.arch.thread_idx()
        P, E, R = self.physical, self.eco_logical, self.ranks
        S = P // R
        # ---- 0. total time (L184) ----------------------------------------
        if tid == 0:
            tot = Uint32(0)
            for r in cutlass.range_constexpr(R):
                tot = tot + self._eco_s(slab, self.eco_T_off, Uint32(r))
            self._eco_set(slab, self.eco_tot_off, Uint32(0), tot)
            self._eco_set(slab, self.eco_nof_off, Uint32(0), Uint32(0))
        cute.arch.sync_threads()
        tot = self._eco_s(slab, self.eco_tot_off, Uint32(0))

        # ---- 1. per slot: its best destination (L196-217) ----------------
        # One thread per slot; parallel-safe because each writes only its own.
        for s0 in cutlass.range(tid, P, self.threads):
            e = self._eco_s(slab, self.eco_phy_off, s0)
            nc = self._eco_ncopies(base, e)
            r0 = s0 // Uint32(S)
            tmax = self._eco_s(slab, self.eco_T_off, r0)
            hit = Uint32(0)
            if (nc > Uint32(1)
                    and self._eco_s(slab, self.eco_act_off, s0) != Uint32(0)
                    and tmax * Uint32(R) > tot
                    and (self._eco_s(slab, self.eco_y_off, s0) > Uint32(0)
                         or self._eco_hp(slab, s0) == Uint32(1))):
                shed, ip = self._eco_out(slab, s0)
                bv = Uint32(0)
                bs = Uint32(0)
                bi = Uint32(0)
                for k in cutlass.range_constexpr(4):
                    if k < nc:
                        s1 = self._eco_slot_of(base, e, k)
                        if s1 != s0:
                            in1 = self._eco_in(slab, s1, ip)
                            v = self._eco_s(slab, self.eco_T_off, s1 // Uint32(S)) + in1
                            # keep the receiver below the source (L204)
                            if v < tmax:
                                if (hit == Uint32(0) or v < bv
                                        or (v == bv and s1 < bs)):
                                    hit = Uint32(1)
                                    bv = v
                                    bs = s1
                                    bi = in1
                if hit != Uint32(0):
                    self._eco_set(slab, self.eco_cks_off, s0, bs)
                    self._eco_set(slab, self.eco_ckv_off, s0, bv)
                    self._eco_set(slab, self.eco_cki_off, s0, bi)
                    self._eco_set(slab, self.eco_ckh_off, s0, shed)
                    self._eco_set(slab, self.eco_ckp_off, s0, ip)
                    # [ECO-AUC] Pack the total-order key once, here, so stage 2
                    # can compare slots with a single 64-bit load.
                    # Order matches unit_ref's `(key, value, hops, s1, s0)`;
                    # hops is always 1 (no two-hop), and s0 makes it a TOTAL
                    # order so equal-key slots never fail to exclude each other.
                    # width: key 24b | value 20b | s1 10b | s0 10b
                    after0 = tmax - shed
                    kk = after0 if after0 > bv else bv
                    self._eco_key_set(
                        slab, s0,
                        (Uint64(kk) << Uint64(40)) | (Uint64(bv) << Uint64(20))
                        | (Uint64(bs) << Uint64(10)) | Uint64(s0))
            self._eco_set(slab, self.eco_hit_off, s0, hit)
        cute.arch.sync_threads()

        # ---- 2'. offers: each slot self-determines the prefix (SM100 L138-175)
        # The reference sorts a rank's candidates and offers them while the
        # accumulated shed is still below the rank's excess. That is equivalent
        # to, for each candidate independently:
        #     shed of all candidates with a SMALLER key  <  excess   ->  offer
        # (strict `<` matches unit_ref's sorted-prefix truncation exactly). The
        # outer "which offer number" loop therefore disappears: one thread per
        # slot instead of one per source rank, 60 iterations instead of 3600.
        if tid == 0:
            self._eco_set(slab, self.eco_nof_off, Uint32(0), Uint32(0))
        cute.arch.sync_threads()
        for s0 in cutlass.range(tid, P, self.threads):
            if self._eco_s(slab, self.eco_hit_off, s0) != Uint32(0):
                r0 = s0 // Uint32(S)
                tmax = self._eco_s(slab, self.eco_T_off, r0)
                if tmax * Uint32(R) > tot:
                    # unit_ref breaks out of its sorted offer loop at
                    # `shed * R >= excess`, i.e. it offers while
                    # `shed * R < tmax * R - tot`. (SM100 L316 writes exactly
                    # this as `excl * kR < tmax * kR - tot`.) Comparing raw
                    # sheds without the `* R` scale keeps far too many offers.
                    excess = tmax * Uint32(R) - tot
                    my_key = self._eco_key_get(slab, s0)
                    excl = Uint32(0)
                    for j in cutlass.range(S):
                        sj = r0 * Uint32(S) + Uint32(j)
                        if self._eco_s(slab, self.eco_hit_off, sj) != Uint32(0):
                            if self._eco_key_get(slab, sj) < my_key:
                                excl = excl + self._eco_s(slab, self.eco_ckh_off, sj)
                    if excl * Uint32(R) < excess:
                        # `nof` lives in the plan's smem slab, so the atomic is
                        # a shared-memory one (same helper stage 3 uses).
                        idx = self._eco_inc(slab, self.eco_nof_off, Uint32(0),
                                            Uint32(1))
                        self._eco_setof(slab, idx, 0,
                                        self._eco_s(slab, self.eco_cks_off, s0) // Uint32(S))
                        self._eco_setof(slab, idx, 1, Uint32(0) - tmax)
                        self._eco_setof(slab, idx, 2,
                                        self._eco_s(slab, self.eco_ckv_off, s0))
                        self._eco_setof(slab, idx, 3, s0)
                        self._eco_setof(slab, idx, 4, tmax)
                        self._eco_setof(slab, idx, 5,
                                        self._eco_s(slab, self.eco_cks_off, s0))
                        self._eco_setof(slab, idx, 6,
                                        self._eco_s(slab, self.eco_cki_off, s0))
                        self._eco_setof(slab, idx, 7,
                                        self._eco_s(slab, self.eco_ckp_off, s0))
        cute.arch.sync_threads()
        cute.arch.sync_threads()

        # ---- 3. group the offers by receiver (L253-254) ------------------
        # Step 3a: count the offers of every receiver. Step 3b: prefix sums of
        # those COUNTS give each receiver's segment base. (Using the per-source
        # counts here instead would hand several receivers overlapping
        # segments and silently mix up their offers.)
        if tid == 0:
            total_offers = Uint32(0)
            for r in cutlass.range_constexpr(R):
                self._eco_set(slab, self.eco_gcnt_off, Uint32(r), Uint32(0))
            self._eco_set(slab, self.eco_nacc_off, Uint32(0), Uint32(0))
        cute.arch.sync_threads()
        # [ECO-AUC] Offers are now COMPACT (stage 2 claims them with a global
        # atomic on `nof`), so the scan is 0..nof instead of all R*S slots, and
        # the per-source `rcnt` guard is no longer needed.
        nof_now = self._eco_s(slab, self.eco_nof_off, Uint32(0))
        if nof_now > Uint32(P):
            nof_now = Uint32(P)
        for item in cutlass.range(tid, nof_now, self.threads):
            rcv = self._eco_of(slab, item, 0)
            if rcv < Uint32(R):
                self._eco_inc(slab, self.eco_gcnt_off, rcv, Uint32(1))
        cute.arch.sync_threads()
        if tid == 0:
            total_offers = Uint32(0)
            for r in cutlass.range_constexpr(R):
                self._eco_set(slab, self.eco_gof_off, Uint32(r), total_offers)
                total_offers = total_offers + self._eco_s(slab, self.eco_gcnt_off, Uint32(r))
                self._eco_set(slab, self.eco_gcnt_off, Uint32(r), Uint32(0))
            self._eco_set(slab, self.eco_gof_off, Uint32(R), total_offers)
            self._eco_set(slab, self.eco_nof_off, Uint32(0), total_offers)
        cute.arch.sync_threads()
        for item in cutlass.range(tid, nof_now, self.threads):
            rcv = self._eco_of(slab, item, 0)
            if rcv < Uint32(R):
                pos = self._eco_inc(slab, self.eco_gcnt_off, rcv, Uint32(1))
                self._eco_set(slab, self.eco_grp_off,
                              self._eco_s(slab, self.eco_gof_off, rcv) + pos, item)
        cute.arch.sync_threads()

        # ---- 4. one thread per receiver: sort its offers, accept a prefix -
        for rcv in cutlass.range(tid, R, self.threads):
            b = self._eco_s(slab, self.eco_gof_off, rcv)
            cnt = self._eco_s(slab, self.eco_gof_off, rcv + Uint32(1)) - b
            # selection sort on (-t_source, value, source slot); the source
            # slot makes the key a total order, so ties never matter
            for i in cutlass.range(cnt):
                oi = self._eco_s(slab, self.eco_grp_off, b + i)
                bi = i
                bk0 = self._eco_of(slab, oi, 1)
                bk1 = self._eco_of(slab, oi, 2)
                bk2 = self._eco_of(slab, oi, 3)
                for j in cutlass.range(cnt):
                    if j > i:
                        oj = self._eco_s(slab, self.eco_grp_off, b + j)
                        k0 = self._eco_of(slab, oj, 1)
                        k1 = self._eco_of(slab, oj, 2)
                        k2 = self._eco_of(slab, oj, 3)
                        if (k0 < bk0 or (k0 == bk0 and (k1 < bk1
                                or (k1 == bk1 and k2 < bk2)))):
                            bk0 = k0
                            bk1 = k1
                            bk2 = k2
                            bi = j
                if bi != i:
                    swap = self._eco_s(slab, self.eco_grp_off, b + i)
                    self._eco_set(slab, self.eco_grp_off, b + i,
                                  self._eco_s(slab, self.eco_grp_off, b + bi))
                    self._eco_set(slab, self.eco_grp_off, b + bi, swap)
            # accept the prefix; two breaks stop before the receiver passes
            # the source or mean + slack (L257-259)
            cur = self._eco_s(slab, self.eco_T_off, rcv)
            nacc = Uint32(0)
            stopped = Uint32(0)
            for i in cutlass.range(cnt):
                oi = self._eco_s(slab, self.eco_grp_off, b + i)
                inc = self._eco_of(slab, oi, 6)
                lim = self._eco_of(slab, oi, 4)
                nxt = cur + inc
                # The two acceptance constraints (unit_ref L257-259): never
                # push the receiver past the source, never past mean + slack.
                # A staged loop cannot `break`, so `stopped` gates the body.
                if (stopped == Uint32(0) and nxt < lim
                        and (Int32(nxt) - Int32(self.eco_slack)) * Int32(R) <= Int32(tot)):
                    cur = nxt
                    self._eco_set(slab, self.eco_grp_off, b + nacc, oi)
                    nacc = nacc + Uint32(1)
                elif stopped == Uint32(0):
                    stopped = Uint32(1)
            self._eco_set(slab, self.eco_acc_off, rcv, nacc)
        cute.arch.sync_threads()

        # ---- 5. apply every accepted move, in accept order (L261-265) ----
        # One thread: a receiver's moves are order-dependent (a partial unit
        # can leave a slot another offer fills), and the reference applies them
        # serially. Accepted counts are tiny (bounded by the excess).
        if tid == 0:
            for rcv in cutlass.range_constexpr(R):
                b = self._eco_s(slab, self.eco_gof_off, Uint32(rcv))
                nacc = self._eco_s(slab, self.eco_acc_off, Uint32(rcv))
                for i in cutlass.range(nacc):
                    oi = self._eco_s(slab, self.eco_grp_off, b + i)
                    self._eco_move(slab, self._eco_of(slab, oi, 3),
                                   self._eco_of(slab, oi, 5),
                                   self._eco_of(slab, oi, 7))
        cute.arch.sync_threads()
        if tid == 0:
            n = Uint32(0)
            for rcv in cutlass.range_constexpr(R):
                n = n + self._eco_s(slab, self.eco_acc_off, Uint32(rcv))
            self._eco_set(slab, self.eco_nacc_off, Uint32(0), n)
        cute.arch.sync_threads()
        cute.arch.sync_threads()
        return self._eco_s(slab, self.eco_nacc_off, Uint32(0))

    @cute.jit
    def _eco_plan(self, base: Uint64, peers: cute.Pointer, rank: Uint32,
                  tokens: Uint32, slab: cute.Pointer):
        """Compute this call's per-slot quota and publish it (spec §5.4).

        Mirrors eco_ref/plan_port.py::_eco_plan_assign:
          wait for counts -> static tables -> D_e over ALL ranks ->
          unit count / partial size -> single-copy units -> cold or warm ->
          auction rounds -> publish quota / quota_start / next warm state.
        """
        tid, _, _ = cute.arch.thread_idx()
        P, E, R = self.physical, self.eco_logical, self.ranks
        bm = self.max_m
        ready = self._field(base, 'eco_ready', Uint32)
        state = self._field(base, 'eco_state', Uint32)
        epoch = ld_acquire_gpu_u32(self._field(base, 'header', Uint32))

        # ---- 1. wait until every rank published its eco_counts row ---------
        # slot 3 = plan entry (before the wait), slot 4 = counts seen.
        # 4-3 is the cost of the counts channel itself.
        if tid == 0:
            self._eco_trace(base, 3)   # (older 30-slot view) plan entry
            self._eco_tl_write_tag(base, epoch, epoch)                 # step_tag
            self._eco_tl_write(base, epoch, 2, read_globaltimer())     # plan_start
        if tid == 0:
            wait_epoch(ready, epoch + Uint32(1), False)
        cute.arch.sync_threads()
        if tid == 0:
            self._eco_trace(base, 4)   # (older view) counts seen
            self._eco_tl_write(base, epoch, 3, read_globaltimer())     # counts_seen

        # ══════════════════════════════════════════════════════════════════
        # [ECO-WARP] Warp specialisation, exactly like SM100
        # (`eco_unit_kernel.cu` L132-360):
        #
        #   warp 0        -> stage the static tables      (96 threads in SM100)
        #   warps 1..15   -> counts + broadcast + peer wait (`helper_sync`)
        #   bar.sync 1,512 -> the single join point (`plan_sync`)
        #
        # Both halves run at the same time; the CTA only meets again at the
        # join. SM100 names the barriers 1 (plan, 512) and 2 (helper, 480).
        # ══════════════════════════════════════════════════════════════════
        warp = tid // Uint32(32)
        gen = epoch & Uint32(1)
        peer_table = cute.make_tensor(peers, cute.make_layout((self.ranks,)))
        head = self._field(base, 'header', Uint32)

        if warp == Uint32(0):
            # ---- static tables: phy[slot] and act[slot] -------------------
            # `act` marks the copies the plan may split an expert over; the
            # passive duplicates a rank holds are never planned (unit_ref L36).
            for s in cutlass.range(tid, P, 32):
                phy = self._eco_phy(base, s)
                self._eco_set(slab, self.eco_phy_off, s, phy)
                nc = self._eco_ncopies(base, phy)
                act = Uint32(0)
                for k in cutlass.range_constexpr(4):
                    if k < nc:
                        if self._eco_slot_of(base, phy, k) == s:
                            act = Uint32(1)
                self._eco_set(slab, self.eco_act_off, s, act)
        else:
            # ---- counts + broadcast (warps 1..15, helper barriers only) ---
            self._eco_count_and_broadcast(base, peers, rank, tokens, tid, gen)
            # After the helper barrier every helper has written its row.
            if tid == Uint32(32):
                st_release_u32(head + Uint32(24), epoch, True)

        # plan_sync: every warp meets here (SM100 `bar.sync 1, 512`).
        sync_aligned(512, 1)
        # ---- wait for every peer's row (all warps) -------------------------
        if tid < self.ranks:
            wait_epoch(self._field(Uint64(peer_table[tid]), 'header', Uint32)
                       + Uint32(24), epoch, True)
        cute.arch.sync_threads()

        # ---- 3. demand per logical expert, summed over ALL ranks -----------
        # Every active copy of an expert receives that expert's complete route
        # stream, so one copy's per-source histogram sums to D_e. Reading only
        # this rank's row would make D_e 1/R of the truth (spec §5.5).
        # [ECO-SM90] Read LOCAL only: every rank broadcast its row into OUR
        # buffer before the barrier (SM100 L293 write / L304-341 read).
        for e in cutlass.range(tid, E, self.threads):
            s0 = self._eco_copy(base, Uint32(e), 0)
            running = Uint32(0)
            if s0 != Uint32(0x7FFF):
                slot = s0 & Uint32(0x7FFF)
                for r in cutlass.range_constexpr(R):
                    # [ECO-DBUF] only this call's half of the double buffer
                    running += self._field(base, 'eco_counts', Uint32)[
                        gen * Uint32(R * P) + r * P + slot]
            self._eco_set(slab, self.eco_n_off, e, running)
        cute.arch.sync_threads()
        if tid == 0:
            self._eco_trace(base, 17)  # [ECO-MEAS] static tables + D_e done
            self._eco_tl_write(base, epoch, 4, read_globaltimer())     # static_done

        # ---- 4'. NULL PLAN gate [ECO-MEAS] ---------------------------------
        # Cost decomposition only: keep step 1-3 (wait / static tables / D_e)
        # and replace everything downstream with the trivial split -- every
        # active copy of an expert gets nothing except the FIRST one, which
        # takes the expert's whole stream. This is NOT a policy: it measures
        # what the plan's cold/warm + auction + fine-publish section costs.
        if cutlass.const_expr(self.eco_null_plan):
            quota_f = self._field(base, 'eco_quota', Uint32)
            qstart_f = self._field(base, 'eco_quota_start', Uint32)
            for e in cutlass.range(tid, E, self.threads):
                n = self._eco_s(slab, self.eco_n_off, e)
                first = Uint32(0xFFFFFFFF)
                # NOTE: `range_constexpr` unrolls in Python, so it cannot hold
                # a `continue`; the skip conditions are nested instead.
                for k in cutlass.range_constexpr(4):
                    raw = self._eco_copy(base, Uint32(e), k)
                    if raw != Uint32(0x7FFF):
                        sl = raw & Uint32(0x7FFF)
                        if self._eco_s(slab, self.eco_act_off, sl) != Uint32(0):
                            if first == Uint32(0xFFFFFFFF):
                                first = sl
                for k in cutlass.range_constexpr(4):
                    raw = self._eco_copy(base, Uint32(e), k)
                    if raw != Uint32(0x7FFF):
                        sl = raw & Uint32(0x7FFF)
                        is_first = Uint32(1) if sl == first else Uint32(0)
                        quota_f[sl] = n * is_first
                        qstart_f[sl] = Uint32(0)
            # Keep the warm-start state coherent so the next call still takes
            # the warm path: this gate must not silently turn steady state
            # into cold (that would inflate the very cost being measured).
            # (helper call: a `self._field(...)` expression inside a dynamic
            # `for` makes the DSL try to stage the kernel object itself)
            for e in cutlass.range(tid, E, self.threads):
                self._eco_publish_count_state(base, slab, e)
            cute.arch.sync_threads()
            fence_sys()
            if tid == 0:
                # [ECO-DBUF] No "counts consumed" token any more: the panel is
                # double buffered, so there is nothing for the next call to
                # wait on. `eco_ready[1]` is still the plan-published token the
                # replica slots wait for.
                st_release_u32(ready + Uint32(1), epoch + Uint32(1), False)
            return

        # ---- 4. plan state: no partial, no units, k / rem per expert ------
        for e in cutlass.range(tid, E, self.threads):
            self._eco_set(slab, self.eco_part_off, e, Uint32(0xFFFFFFFF))
            n = self._eco_s(slab, self.eco_n_off, e)
            k = (n + Uint32(bm - 1)) // Uint32(bm)
            self._eco_set(slab, self.eco_k_off, e, k)
            self._eco_set(slab, self.eco_rem_off, e,
                          Uint32(0) if n == Uint32(0)
                          else n - (k - Uint32(1)) * Uint32(bm))
        for s in cutlass.range(tid, P, self.threads):
            self._eco_set(slab, self.eco_y_off, s, Uint32(0))
        cute.arch.sync_threads()

        # ---- 5. single-copy experts: their units are fixed up front -------
        for s in cutlass.range(tid, P, self.threads):
            phy = self._eco_s(slab, self.eco_phy_off, s)
            if (self._eco_s(slab, self.eco_act_off, s) != Uint32(0)
                    and self._eco_ncopies(base, phy) == Uint32(1)):
                n = self._eco_s(slab, self.eco_n_off, phy)
                if n > Uint32(0):
                    self._eco_set(slab, self.eco_y_off, s,
                                  self._eco_s(slab, self.eco_k_off, phy) - Uint32(1))
                    self._eco_set(slab, self.eco_part_off, phy, s)
        cute.arch.sync_threads()

        # ---- 6. warm iff the previous call left counts (unit_ref L102-105) -
        if tid == 0:
            prev = Uint32(0)
            for e in cutlass.range_constexpr(E):
                prev = prev + state[P + e]
            self._eco_set(slab, self.eco_flag_off, Uint32(0),
                          Uint32(1) if prev > Uint32(0) else Uint32(0))
        cute.arch.sync_threads()
        warm = self._eco_s(slab, self.eco_flag_off, Uint32(0)) != Uint32(0)
        rounds = Uint32(0)
        if warm:
            self._eco_do_warm(base, slab)
            rounds = Uint32(self.eco_rounds)
        else:
            self._eco_do_cold(base, slab)
            rounds = Uint32(self.eco_cold_rounds)
        if tid == 0:
            self._eco_trace(base, 18)  # [ECO-MEAS] cold/warm done
            self._eco_tl_write(base, epoch, 5, read_globaltimer())     # coldwarm_done

        # [ECO-MEAS] Publish which path this call took, so the host can prove
        # steady state is warm (a cold steady state would run 6 auction rounds
        # instead of 1 and silently charge that to the plan's cost).
        if cutlass.const_expr(self.eco_trace):
            if tid == 0:
                tr = self._field(base, 'eco_trace', Uint64)
                st_global_u64(tr + Uint64(30), Uint64(1) if warm else Uint64(0))
                st_global_u64(tr + Uint64(31), Uint64(rounds))

        # ---- 7. auction rounds (stops early when nothing is accepted) -----
        self._eco_emit_quota(slab)
        # [ECO-MEAS] `eco_no_auction` is a compile-time gate: the whole block
        # disappears from the cubin, so it measures the auction itself and
        # nothing else (cold/warm still run above).
        if not cutlass.const_expr(self.eco_no_auction):
            more = Uint32(1)
            for _it in cutlass.range(rounds):
                if more != Uint32(0):
                    self._eco_rank_time(slab, self.eco_quota_off, self.eco_T_off)
                    if self._eco_do_auction_round(base, slab) == Uint32(0):
                        more = Uint32(0)
                    else:
                        self._eco_emit_quota(slab)
        if tid == 0:
            self._eco_trace(base, 19)  # [ECO-MEAS] auction rounds done
            self._eco_tl_write(base, epoch, 6, read_globaltimer())     # auction_done

        # ---- 8. publish quota / quota_start and the next warm state -------
        # quota_start is the offset of this copy's slice inside the expert's
        # canonical order: the sum of the quotas of the active copies that
        # precede it (unit_ref's ascending-slot copy order).
        for s in cutlass.range(tid, P, self.threads):
            self._eco_publish_slot(base, slab, s)
        for e in cutlass.range(tid, E, self.threads):
            self._eco_publish_count_state(base, slab, e)
        cute.arch.sync_threads()
        fence_sys()
        if tid == 0:
            self._eco_trace(base, 5)   # [ECO-MEAS] quota + warm state published
            self._eco_tl_write(base, epoch, 7, read_globaltimer())     # plan_end
            st_release_u32(ready + Uint32(1), epoch + Uint32(1), False)


    # [ECO-TRACE] per-step x per-rank raw timeline geometry
    ECO_TL_STEPS = 256
    ECO_TL_SLOTS = 21

    @cute.jit
    def _eco_tl_write(self, base: Uint64, step: Uint32,
                      slot: cutlass.Constexpr, ts: Uint64):
        """[ECO-TRACE] Plain store into `eco_tl[step][slot]` (us since boot).

        No min/max/sum reduction and no atomics: every call keeps its own row
        and every rank keeps its own buffer, so the analyser can still tell
        WHICH rank was slow and average over steps offline. A reduced value
        would throw exactly that information away.

        Microseconds, not nanoseconds: raw `globaltimer` ns does not fit int32
        past ~2.1 s of device uptime.
        """
        if cutlass.const_expr(self.eco_trace):
            tl = self._field(base, 'eco_tl', Uint32)
            # [ECO-TRACE] The row is the caller's OWN epoch, never a shared
            # gmem counter. Since the plan is woken BEFORE the push, call N's
            # plan can still be running when call N+1's dispatch starts; a
            # shared `eco_tl_step` would then have moved on and two different
            # calls would write into the same row (measured: rows whose
            # `plan_start` was 159 ms earlier than their own `plan_wake`, i.e.
            # timestamps from two calls mixed into one row). Each role already
            # knows its epoch: dispatch reads it from `header[0]`, and the plan
            # waits for `eco_ready[0] == epoch + 1`, which is exactly the token
            # that same dispatch published. So passing it in keeps the row
            # per-call and race-free.
            row = Uint32(step) % Uint32(self.ECO_TL_STEPS)
            tl[row * Uint32(self.ECO_TL_SLOTS) + Uint32(slot)] = \
                Uint32(ts // Uint64(1000))

    # [PULL-TRACE] Pull-phase timing for every arm (ECO or not).
    PULL_TL_STEPS = 256
    # [ECO-TRACE] 8 pull slots + slots 8/9 for combine, so a NON-ECO arm can
    # measure combine too (`eco_trace` only exists in an ECO layout, so the
    # combine marks could not be written at all in the vanilla arm).
    #   [8] first role-3 CTA enters combine
    #   [9] last role-3 CTA bumps the epoch (combine done)
    PULL_TL_SLOTS = 10

    @cute.jit
    def _pull_trace_on(self):
        return cutlass.const_expr(self.pull_trace)

    @cute.jit
    def _pull_tl_write(self, base: Uint64, epoch: Uint32,
                       slot: cutlass.Constexpr, ts: Uint64):
        """One cell of `pull_tl[epoch % 256][slot]`, in microseconds."""
        if cutlass.const_expr(self.pull_trace or self.eco_trace):
            if cutlass.const_expr(self.pull_trace):
                tl = self._field(base, 'pull_tl', Uint32)
                row = Uint32(epoch) % Uint32(self.PULL_TL_STEPS)
                tl[row * Uint32(self.PULL_TL_SLOTS) + Uint32(slot)] = \
                    Uint32(ts // Uint64(1000))
            # [ECO-TRACE] Mirror into the shared timeline so a NON-ECO arm
            # produces the same columns as ECO (10/12/13 are the pull slots).
            # `slot` is a Constexpr int, but the tracer may still symbolically
            # reach this line for slots 8/9 (combine), so branch explicitly
            # instead of indexing a tuple with `slot - 3`.
            if cutlass.const_expr(slot == 3):
                self._eco_tl_write(base, epoch, 10, ts)
            elif cutlass.const_expr(slot == 4):
                self._eco_tl_write(base, epoch, 12, ts)
            elif cutlass.const_expr(slot == 5):
                self._eco_tl_write(base, epoch, 13, ts)

    @cute.jit
    def _eco_key_set(self, slab: cute.Pointer, s: Uint32, v: Uint64):
        """[ECO-MEAS] Store the packed 64-bit total-order key of slot `s`."""
        shared_cell(slab, Uint32(self.eco_key_off) + s * Uint32(8), Uint64)[0] = v

    @cute.jit
    def _eco_key_get(self, slab: cute.Pointer, s: Uint32):
        """[ECO-MEAS] Load the packed 64-bit key of slot `s`."""
        return shared_cell(slab, Uint32(self.eco_key_off) + s * Uint32(8), Uint64)[0]

    @cute.jit
    def _eco_add_slot(self, base: Uint64, slot: cutlass.Constexpr, delta_ns: Uint64):
        """[ECO-MEAS] Accumulate a duration (ns input) into trace `slot`, in us.

        Accumulating across rounds AND calls keeps the per-call cost at one
        atomic; the host divides by `acc_calls` (slot 21).
        """
        if cutlass.const_expr(self.eco_trace and self.eco):
            tr = self._field(base, 'eco_trace', Uint64)
            atom_add_u64(tr + Uint64(slot), delta_ns // Uint64(1000))

    @cute.jit
    def _eco_tl_row(self, base: Uint64, epoch: Uint32):
        """[ECO-TRACE] Authoritative row index for this call.

        Written by dispatch before it wakes the plan, so every role agrees on
        the row even though the plan may read `header[0]` at a different moment.
        """
        return Uint32(self._field(base, 'eco_tl_step', Uint32)[0])

    @cute.jit
    def _eco_tl_mark_epoch(self, base: Uint64, epoch: Uint32, which: cutlass.Constexpr,
                           slot: cutlass.Constexpr):
        """[ECO-TRACE] First-writer mark, accepted only within this call's epoch.

        Resident roles (compute, plan) may reach the mark for call N while call
        N-1 is still running. Requiring `header[0] == epoch` at the moment of
        the mark keeps each row's timestamps inside its own call.
        """
        if cutlass.const_expr(self.eco_trace and self.eco):
            tid, _, _ = cute.arch.thread_idx()
            # Gate on "this call's plan has already been woken", not on
            # header[0] == epoch: a resident role keeps the epoch it read at
            # kernel entry, so that test stays true while the NEXT call is
            # still preparing.
            if tid == 0:
                ready = self._field(base, 'eco_ready', Uint32)
                if ld_acquire_gpu_u32(ready) == epoch + Uint32(1):
                    ctr = self._field(base, 'eco_tl_ctr', Uint32)
                    prev = atom_add_u32(ctr + Uint32(which), Uint32(1))
                    if prev == Uint32(0):
                        self._eco_tl_write(base, epoch, slot, read_globaltimer())

    @cute.jit
    def _eco_tl_write_tag(self, base: Uint64, step: Uint32, tag: Uint32):
        """[ECO-TRACE] Store the raw epoch as this row's tag (not a timestamp)."""
        if cutlass.const_expr(self.eco_trace):
            tl = self._field(base, 'eco_tl', Uint32)
            # Same rule as `_eco_tl_write`: the row is this call's own epoch.
            row = Uint32(step) % Uint32(self.ECO_TL_STEPS)
            tl[row * Uint32(self.ECO_TL_SLOTS)] = tag

    @cute.jit
    def _eco_tl_mark(self, base: Uint64, step: Uint32, which: cutlass.Constexpr,
                     target: Uint32, slot: cutlass.Constexpr,
                     want_first: cutlass.Constexpr):
        """[ECO-TRACE] First- or last-arriver marker on counter `which`.

        `target` is the number of arrivals that completes the phase (so
        `want_first=False` fires on arrival `target-1`). One atomic per CTA and
        one plain store by the winner: no barrier, so this never perturbs the
        schedule it is measuring.
        """
        if cutlass.const_expr(self.eco_trace and self.eco):
            tid, _, _ = cute.arch.thread_idx()
            if tid == 0:
                ctr = self._field(base, 'eco_tl_ctr', Uint32)
                prev = atom_add_u32(ctr + Uint32(which), Uint32(1))
                hit = (prev == Uint32(0)) if cutlass.const_expr(want_first) \
                    else (prev + Uint32(1) == target)
                if hit:
                    self._eco_tl_write(base, step, slot, read_globaltimer())

    @cute.jit
    def _eco_trace_reset(self, base: Uint64):
        """[ECO-MEAS] Zero the trace window and prime the min-slots (block 0).

        The rendezvous right after this orders the clear before any CTA's
        min/max write.
        """
        if cutlass.const_expr(not self.eco_trace_reset):
            return
        tid, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        if block == 0:
            # 32 words, one per thread: a single thread doing 32 serialized
            # global stores delayed block 0 enough to perturb the very timing
            # this instrumentation is meant to observe.
            if tid < 32:
                self._eco_write_slot(base, tid, Uint64(0))
            cute.arch.sync_threads()
            if tid == 0:
                # [ECO-TRACE] Clear the per-call first/last-arrival counters.
                # They are monotonic across calls: leaving them would make
                # every `want_first` mark fire only in the very first call of
                # the whole run (measured: pull_norep/pull_rep durations of
                # 0.1 us because slots 10/12 were written once per run).
                ctr = self._field(base, 'eco_tl_ctr', Uint32)
                for c in cutlass.range_constexpr(16):
                    ctr[c] = Uint32(0)
                for c in cutlass.range_constexpr(6):
                    self._eco_write_slot(base, (0, 6, 9, 10, 11, 14)[c],
                                         Uint64(0xFFFFFFFFFFFFFFFF))
                self._eco_trace(base, 24, 0)

    @cute.jit
    def _eco_write_slot(self, base: Uint64, slot: cutlass.Constexpr, value: Uint64):
        """[ECO-MEAS] Unconditional raw write of one trace word (reset path)."""
        tr = self._field(base, 'eco_trace', Uint64)
        st_global_u64(tr + Uint64(slot), value)

    @cute.jit
    def _eco_trace(self, base: Uint64, slot: cutlass.Constexpr,
                   mode: cutlass.Constexpr = 0):
        """[ECO-MEAS] Timeline trace point (globaltimer ns).

        mode 0 = store  (single writer: overwrite)
        mode 1 = min    (many writers: "the earliest one")
        mode 2 = max    (many writers: "the last one")

        min/max go through red.global.*, so they are safe against a stale zero
        being smaller than a real timestamp -- that is exactly why the
        many-writer slots use min instead of a plain store.
        Cross-rank comparisons must use this device-wide timer, never clock64().
        """
        # `eco_trace` only exists in an ECO layout, so the guard must include
        # `self.eco`; otherwise the vanilla arm (eco=False) faults.
        if cutlass.const_expr(self.eco_trace and self.eco):
            tr = self._field(base, 'eco_trace', Uint64)
            now = read_globaltimer()
            if cutlass.const_expr(mode == 0):
                st_global_u64(tr + Uint64(slot), now)
            elif cutlass.const_expr(mode == 1):
                red_global_min_u64(tr + Uint64(slot), now)
            else:
                red_global_max_u64(tr + Uint64(slot), now)

    @cute.jit
    def _eco_is_active(self, base: Uint64, e: Uint32, slot: Uint32):
        """[ECO-SM90] Is `slot` one of the expert's ACTIVE copies?

        EcoMeta lists the active copies first and the passive duplicates after
        them (bit 15 set). Only active copies are planned and only they pull.
        """
        active = Uint32(0)
        for k in cutlass.range_constexpr(self.eco_max_copies):
            raw = self._eco_copy(base, e, k)
            if raw != Uint32(0x7FFF) and (raw & Uint32(0x8000)) == Uint32(0):
                if (raw & Uint32(0x7FFF)) == slot:
                    active = Uint32(1)
        return active

    @cute.jit
    def _eco_count_and_broadcast(self, base: Uint64, peers: cute.Pointer,
                                 rank: Uint32, tokens: Uint32, tid: Uint32,
                                 gen: Uint32):
        """[ECO-MOVE] Build this rank's route histogram and broadcast it.

        Runs inside the PLAN kernel (role 5, one CTA), not in dispatch. It used
        to run in dispatch, where it sat in front of the single-copy slots'
        pull for no reason: those slots need neither the histogram nor the
        plan. Moving it here means dispatch goes

            wake plan -> push -> pull(single-copy slots)

        and the ECO-only work happens while the push and the single-copy pull
        are already running.

        SM100 does the same thing in one kernel
        (`eco_unit_kernel.cu` L278-283 count -> L293 broadcast -> L304-341 read
        back), which is possible because everything there is in one CTA. Here
        the plan owns a whole CTA with `self.threads` threads, so the count is
        parallel across all of them and the clear needs only a CTA-local
        `sync_threads()` -- no grid rendezvous, which is exactly the cost this
        change removes from the critical path.
        """
        # [ECO-WARP] This body only runs on warps 1..15 (helper threads
        # 32..511). Warp 0 is staging the static tables at the same time, so
        # every barrier here must be a HELPER-ONLY named barrier -- a CTA-wide
        # `sync_threads()` would wait for warp 0 and destroy the overlap
        # SM100 gets from `helper_sync()` (`bar.sync 2, 480`).
        # `cutlass.range`'s bounds must be static ints; the helper subset is
        # expressed as a REBASED thread index so the loop shape stays the same
        # as the CTA-wide version (`helpers` below is the helper count).
        helpers = self.threads - 32          # 480 helper threads
        helper = Uint32(0)
        helper = tid - Uint32(32)            # rebased index inside the helpers
        hist = self._field(base, 'eco_hist', Uint32)
        for e in cutlass.range(helper, self.eco_logical, helpers):
            hist[e] = Uint32(0)
        sync_aligned(480, 2)
        for route in cutlass.range(helper, tokens * self.topk, helpers):
            expert_id = cell(base, Uint64(self.off_topk_idx) + Uint64(route) * 8,
                             Int64)[0]
            if expert_id >= 0 and expert_id < Int64(self.eco_logical):
                atom_add_u32(hist + Uint32(expert_id), Uint32(1))
        sync_aligned(480, 2)
        # Broadcast into every peer's `eco_counts[rank][*]` row, then wait for
        # every peer to have done the same. This is ONE cross-rank barrier
        # instead of the two dispatch used to need.
        counts_out = self._field(base, 'eco_counts', Uint32)
        peer_table = cute.make_tensor(peers, cute.make_layout((self.ranks,)))
        # [ECO-DBUF] This call's half of the panel (SM100 `gen * kR * kP`).
        gen_off = Uint64(gen) * Uint64(self.ranks) * Uint64(self.physical)
        base_off = (gen_off + Uint64(rank) * Uint64(self.physical))
        # Bind the raw addresses BEFORE the dynamic loop. A `self._field(...)`
        # / `cute.make_ptr(...)` expression built from a loop variable is
        # evaluated once by the DSL front end and the loop then reads a stale
        # value (the same trap the rest of this file warns about); the observed
        # symptom was `conserved=False` from the very first call.
        local_row = counts_out + base_off
        local_row_off = Uint64(self.off_eco_counts) + base_off * 4
        for shard in cutlass.range(helper, Uint32(self.physical), helpers):
            value = hist[self._eco_phy(base, shard)]
            local_row[Uint64(shard)] = value
            # Use the same `cell(...)` helper `_eco_publish_count` uses: it is a
            # frozen function, so the DSL emits one store per peer. Building a
            # `make_ptr` from the loop variable instead made the front end
            # hoist a single address and only some peers got the row
            # (measured: rows 3..7 stayed zero while 0..2 were right).
            for r in cutlass.range_constexpr(self.ranks):
                if r != rank:
                    cell(Uint64(peer_table[r]), local_row_off + Uint64(shard) * 4,
                         Uint32)[0] = value
        fence_sys()
        # [ECO-WARP] HELPER-only: warp 0 is staging the static tables in the
        # other branch and never reaches this line. A CTA-wide
        # `sync_threads()` here waits for warp 0 forever -- that was the hang.
        sync_aligned(480, 2)

    @cute.jit
    def _eco_publish_count(self, base: Uint64, peers: cute.Pointer, rank: Uint32,
                           shard: Uint32):
        """[ECO-SM90] Broadcast slot `shard`'s canonical count to EVERY rank.

        SM100 equivalent: eco_unit_kernel.cu L293. Writing into every peer's
        buffer is what lets the plan read only LOCAL memory afterwards -- no
        remote loads inside the plan's critical path.
        """
        shard_expert = self._eco_phy(base, shard)
        value = self._field(base, 'eco_hist', Uint32)[shard_expert]
        offset = (Uint64(self.off_eco_counts)
                  + (Uint64(rank) * Uint64(self.physical) + Uint64(shard)) * 4)
        cell(base, offset, Uint32)[0] = value              # local
        peer_table = cute.make_tensor(peers, cute.make_layout((self.ranks,)))
        for r in cutlass.range_constexpr(self.ranks):
            cell(Uint64(peer_table[r]), offset, Uint32)[0] = value

    @cute.jit
    def _eco_publish_slot(self, base: Uint64, slab: cute.Pointer, s: Uint32):
        """Publish one slot's quota, its quota_start and its warm-start quota.

        quota_start is the sum of the quotas of the active copies that precede
        `s` in the expert's canonical (ascending slot) order.
        """
        q = self._eco_s(slab, self.eco_quota_off, s)
        e = self._eco_s(slab, self.eco_phy_off, s)
        nc = self._eco_ncopies(base, e)
        start = Uint32(0)
        for k in cutlass.range_constexpr(self.eco_max_copies):
            if Uint32(k) < nc:
                c = self._eco_slot_of(base, e, k)
                if c < s:
                    start = start + self._eco_s(slab, self.eco_quota_off, c)
        self._field(base, 'eco_quota', Uint32)[s] = q
        self._field(base, 'eco_quota_start', Uint32)[s] = start
        self._field(base, 'eco_state', Uint32)[s] = q

    @cute.jit
    def _eco_publish_count_state(self, base: Uint64, slab: cute.Pointer, e: Uint32):
        """Record this call's per-expert token count for the next warm start."""
        self._field(base, 'eco_state', Uint32)[Uint32(self.physical) + e] = \
            self._eco_s(slab, self.eco_n_off, e)

    @cute.jit
    def _field(self, base: Uint64, name: cutlass.Constexpr, dtype: cutlass.Constexpr):
        return cute.make_ptr(dtype, base + Uint64(getattr(self, 'off_' + name)),
                             cute.AddressSpace.gmem, assumed_align=16)

    @cute.jit
    def _dispatch_grid(self, head: cute.Pointer, epoch: Uint32,
                         counter: cutlass.Constexpr, signal: cutlass.Constexpr,
                         publish_source: cutlass.Constexpr):
        tid, _, _ = cute.arch.thread_idx()
        if cutlass.const_expr(publish_source):
            fence_sys()
        else:
            fence_gpu()
        cute.arch.sync_threads()
        if tid == 0:
            arrived = atom_acq_rel_add_u32(head + counter, Uint32(1))
            if arrived == self.grid - 1:
                if cutlass.const_expr(publish_source):
                    st_release_u32(head + 1, epoch, True)
                st_release_u32(head + signal, epoch, False)
            wait_epoch(head + signal, epoch, False)
        cute.arch.sync_threads()

    @cute.jit
    def _dispatch(self, base: Uint64, peers: cute.Pointer, rank: Uint32,
                   tokens: Uint32, stats: cute.Pointer):
        tid, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        lane, warp = tid % 32, tid // 32
        peer_table = cute.make_tensor(peers, cute.make_layout((self.ranks,)))
        head = self._field(base, 'header', Uint32)
        epoch = ld_acquire_gpu_u32(head)
        if cutlass.const_expr(self.eco_trace and self.eco):
            step_ptr = self._field(base, 'eco_tl_step', Uint32)
            if block == 0 and tid == 0:
                # [ECO-TRACE] publish this call's row index before anything else
                step_ptr[0] = epoch
            # [ECO-MEAS] block 0 clears the whole trace window for this call.
            # The pre-existing rendezvous #1 right below orders this before any
            # CTA writes a min/max slot, so no stale value can win.
            self._eco_trace_reset(base)
            # NOTE: slot 0 is written AFTER rendezvous #1 below, so it can only
            # observe timestamps taken after the clear.
        if block == 0 and tid == 0:
            token_out = cell(base, Uint64(8), Uint32)
            token_out[0] = tokens
        for e in cutlass.range(block * self.threads + tid, self.ranks * self.local,
                               self.grid * self.threads):
            out = cell(base, Uint64(self.off_send_count) + Uint64(e) * 4, Uint32)
            out[0] = Uint32(0)
        # Four launch-bounded resident CTAs per communication SM make this
        # local grid rendezvous safe even when the compute partition is busy.
        self._dispatch_grid(head, epoch, 11, 12, False)
        if cutlass.const_expr(self.eco_trace and self.eco):
            self._eco_trace(base, 0, 1)       # slot 0: earliest CTA (after clear)
            if block == 0 and tid == 0:
                self._eco_trace(base, 1, 0)   # slot 1: rendezvous #1 done
        # ══════════════════════════════════════════════════════════════════
        # [ECO-SM90] PHASE A -- the plan's input, fully decoupled from the push.
        # Mirrors SM100 (eco_unit_kernel.cu L128-129 count -> L293 broadcast ->
        # L304-341 local read): the demand comes from the route table, every
        # rank broadcasts its row into every peer, one cross-rank barrier
        # confirms arrival, and the plan then reads LOCAL memory only. The push
        # below therefore runs CONCURRENTLY with the plan instead of in front
        # of it.
        # ══════════════════════════════════════════════════════════════════
        if cutlass.const_expr(self.eco):
            # ══════════════════════════════════════════════════════════════
            # [ECO-MOVE] The counting and broadcasting that used to live here
            # now run inside the PLAN kernel (role 5), because they are
            # ECO-only work and had no business sitting on the critical path
            # of the single-copy slots. What remains here is one release: the
            # plan CTA is resident and only needs to be told "this call's
            # topk_idx is valid".
            #
            # Ordering: `topk_idx` is written by vLLM *before* the kernel
            # launches (host-side copy), and the push loop below only reads
            # it, so waking the plan first is safe. The plan reads the SAME
            # array to build its own histogram, exactly like SM100
            # (eco_unit_kernel.cu L128-129).
            #
            # `eco_ready[0]` keeps its epoch token semantics (`epoch + 1`).
            # ══════════════════════════════════════════════════════════════
            if block == 0 and tid == 0:
                self._eco_tl_write(base, epoch, 1, read_globaltimer())  # plan_wake
                st_release_u32(self._field(base, 'eco_ready', Uint32),
                               epoch + Uint32(1), False)
            cute.arch.sync_threads()
        if block == 0 and tid == 0:
            self._eco_tl_write(base, epoch, 8, read_globaltimer())      # push_begin
        for route in cutlass.range(block * self.threads + tid, tokens * self.topk,
                                   self.grid * self.threads):
            expert_id = cell(base, Uint64(self.off_topk_idx) + Uint64(route) * 8, Int64)[0]
            if cutlass.const_expr(self.eco):
                # [ECO-SM90] `topk_idx` holds a LOGICAL expert id. Every active
                # copy gets its own copy of the route, so each replica's pull
                # phase can take its quota slice of the expert's canonical
                # order (spec §5.6). A single-copy expert has one entry in
                # `copies`, so it is written exactly once.
                if expert_id >= 0 and expert_id < Int64(self.eco_logical):
                    e = Uint32(expert_id)
                    # ONE counter per logical expert: every copy of the expert
                    # must see the SAME canonical order, because each copy
                    # pulls the slice [quota_start, quota_start + quota) of it
                    # (spec §5.6). A per-slot counter would give each copy a
                    # different order, and the slices would overlap and lose
                    # tokens.
                    slot = atom_add_u32(
                        self._field(base, 'send_count', Uint32) + e, Uint32(1))
                    # NOTE: `range_constexpr` unrolls in Python, so it cannot
                    # contain `continue`; the two skip conditions are nested.
                    for k in cutlass.range_constexpr(self.eco_max_copies):
                        raw = self._eco_copy(base, e, k)
                        # 0x7FFF: unused entry; bit 15: passive copy
                        if raw != Uint32(0x7FFF) and (raw & Uint32(0x8000)) == Uint32(0):
                            s = raw & Uint32(0x7FFF)
                            dst = Uint64(peer_table[s // self.local])
                            addr = (Uint64(self.off_source_index)
                                    + ((Uint64(s % Uint32(self.local)) * self.ranks + rank)
                                       * self.capacity + slot) * 4)
                            cell(dst, addr, Uint32)[0] = Uint32(route)
            elif expert_id >= 0 and expert_id < self.ranks * self.local:
                expert_u = Uint32(expert_id)
                slot = atom_add_u32(self._field(base, 'send_count', Uint32) + expert_u, Uint32(1))
                destination = Uint64(peer_table[expert_u // self.local])
                address = Uint64(self.off_source_index) + ((Uint64(expert_u % self.local) * self.ranks + rank) * self.capacity + slot) * 4
                route_out = cell(destination, address, Uint32)
                route_out[0] = Uint32(route)
        self._dispatch_grid(head, epoch, 13, 14, True)
        if block == 0 and tid == 0:
            self._eco_tl_write(base, epoch, 9, read_globaltimer())      # push_end
            # [PULL-TRACE] push end, every arm
            self._pull_tl_write(base, epoch, 3, read_globaltimer())
        # [ECO-SM90] The plan was already woken before the push; nothing to do
        # here any more.
        if tid < self.ranks:
            wait_epoch(self._field(Uint64(peer_table[tid]), 'header', Uint32) + 1, epoch, True)
        cute.arch.sync_threads()
        # [ECO-SM90] The counts row was broadcast BEFORE the push (phase A), so
        # the plan is already running here; nothing left to do for ECO.
        alloc = cutlass.utils.SmemAllocator()
        # [ECO-PULL] TMA staging for the token copy. The old loop moved a token
        # with 320 synchronous `ld.global.v4`/`st.global.v4` pairs of 16 bytes
        # each (one per lane per 5120-byte row), which measured 4.4 GB/s --
        # about 90x below NVLink. SM100 pulls through shared memory with
        # `cp.async.bulk` instead (impls/sm100_fp8_fp4_gc_moe.py L1193-1200),
        # and so do we now: one bulk g2s of the whole row, one bulk s2g out.
        #
        # ONE staging row per warp -- exactly what SM100 reserves
        # (layout/gc_moe.py L76 `dispatch_warps * pull_bytes`; there is no
        # double buffer there). SM100 hides the copy latency by deferring the
        # `mbarrier_wait` of the last chunk to AFTER the weight/scale work
        # (sm100_fp8_fp4_gc_moe.py L1184-1269), not by alternating buffers.
        #
        # The size MATTERS: the dispatch role does a whole-grid rendezvous
        # (`_dispatch_grid`), so every CTA must be resident at once, and
        # `min_blocks_per_mp=4` means the smem budget is 233472/4 = 58368 B.
        # A second buffer (84864 B/CTA) only allows 2 CTAs/SM, so 12 of the 24
        # CTAs never start and the rendezvous hangs -- that was every earlier
        # TMA attempt's deadlock.
        warp_row_bytes = ((self.hidden + (self.hidden // 128) * 4 + 15) // 16) * 16
        n_warps = self.threads // 32
        # Place the staging area at a FIXED offset from the top of the dynamic
        # shared window instead of asking the allocator to continue from
        # whatever `counts`/`prefix`/`total_shared` left behind: the window is
        # 84864 B and the staging needs 84736 B, so 52 B of slack means an
        # allocator bump in the wrong direction silently hands back an address
        # outside the allocation and the TMA never lands (observed as a hang).
        # Take the whole window ONCE and slice it by hand. Continuing the
        # allocator from `counts`/`prefix`/`total_shared` leaves only 52 B of
        # slack in an 84864 B window, and an alignment bump then hands back an
        # address outside the allocation -- the TMA never lands and the kernel
        # hangs. Everything below is placed explicitly.
        pull_stage_bytes = n_warps * warp_row_bytes
        win = cute.recast_ptr(alloc.allocate(self.smem, 1024), dtype=Uint8)
        stage = win + Uint32(self.smem) - Uint32(pull_stage_bytes) - Uint32(384)
        pull_bar = cute.recast_ptr(win + Uint32(self.smem) - Uint32(384), dtype=Uint64)
        # the small shared arrays live at the bottom of the same window
        counts = cute.recast_ptr(win, dtype=Uint32)
        prefix = cute.recast_ptr(win + Uint32(4 * self.ranks), dtype=Uint32)
        total_shared = cute.recast_ptr(win + Uint32(4 * (2 * self.ranks + 1)), dtype=Uint32)
        if tid < Uint32(n_warps):
            cute.arch.mbarrier_init(pull_bar + Uint64(tid), 1)
        cute.arch.sync_threads()
        # The barrier was just written; the fence makes the initialisation
        # visible to the arriving TMA engine (same pattern as `_tile_sync`).
        cute.arch.mbarrier_init_fence()
        # [ECO-SM90] A replica slot only knows how many tokens it owns after
        # the plan published their quotas, so its slice is [quota_start,
        # quota_start + quota]. Single-copy slots keep the whole stream and are
        # never blocked on the plan, which is what lets them overlap.
        # [PULL-TRACE] pull start: every CTA has passed the cross-rank wait and
        # is about to touch the first local slot.
        if block == 0 and tid == 0:
            self._pull_tl_write(base, epoch, 4, read_globaltimer())
        # [ECO-SM90] Blocks grid-stride over the local slots exactly like the
        # plain path. A replica slot blocks here until the plan has published
        # its quota (the two-phase split of spec §5.3 without growing the grid,
        # which the dispatch role's whole-grid rendezvous forbids), so
        # single-copy slots keep pulling while the plan runs.
        # [ECO-PULL] mbarrier phase, hoisted ABOVE the slot loop. A barrier's
        # phase flips on every arrive/complete and the same warp reuses the
        # same barrier across slots, so resetting it per slot hangs the second
        # slot onward (this was the TMA deadlock).
        #
        # [ECO-PULL] The mbarrier phase lives in SHARED memory, not in a
        # Python-side variable. `for expert in cutlass.range(...)` is a staged
        # (device-side) loop, so a variable assigned inside it is re-initialised
        # on every iteration -- the phase would silently restart at 0 while the
        # hardware barrier had already flipped, and the wait would hang. A
        # shared word survives across iterations because it is real memory.
        # placed just after the staging rows, before the 256 B barrier block,
        # so it cannot collide with counts/prefix/total_shared at the bottom
        pull_phase_smem = cute.recast_ptr(
            win + Uint32(self.smem) - Uint32(128), dtype=Uint32)
        if tid < Uint32(n_warps):
            pull_phase_smem[tid] = Uint32(0)
        cute.arch.sync_threads()
        for expert in cutlass.range(block, self.local, self.grid):
            # [ECO-SM90] per logical expert, because the copies of an expert
            # share one canonical order; without ECO the key is the global
            # slot (rank * local + local slot), as before.
            # [ECO-SM90] `expert` is this rank's LOCAL slot index; EcoMeta,
            # eco_quota and eco_quota_start are indexed by GLOBAL slot.
            global_slot = rank * Uint32(self.local) + Uint32(expert)
            slot_expert = global_slot
            key = global_slot
            if cutlass.const_expr(self.eco):
                slot_expert = self._eco_phy(base, global_slot)
                # ECO's send_count is indexed by LOGICAL expert (one canonical
                # stream per expert, shared by all of its copies).
                key = slot_expert
            # [ECO-PULL] SM100 style (impls/sm100_fp8_fp4_gc_moe.py L1030-1060):
            # warp 0 builds the per-source counts from its LANE registers, so
            # there is no serial shared-memory walk and only ONE CTA barrier.
            # The old version stored the counts in shared memory and had
            # thread 0 walk them serially inside TWO CTA-wide `sync_threads()`;
            # that pair of 256-thread barriers plus the 8-step serial loop was
            # the fixed per-slot cost (~0.8 us/slot, 45 us over 56 slots).
            if warp == Uint32(0):
                my_count = Uint32(0)
                if lane < Uint32(self.ranks):
                    source_base = Uint64(peer_table[lane])
                    source_count = self._field(source_base, 'send_count', Uint32)
                    my_count = ld_acquire_sys_u32(source_count + key)
                if lane < Uint32(self.ranks):
                    counts[lane] = my_count
                cute.arch.sync_warp()
                if lane == Uint32(0):
                    running_count = Uint32(0)
                    prefix[0] = Uint32(0)
                    for source in cutlass.range_constexpr(self.ranks):
                        running_count += counts[source]
                        prefix[source + 1] = running_count
                    total_shared[0] = running_count
            cute.arch.sync_threads()      # one barrier instead of two
            total = Uint32(total_shared[0])
            # [ECO-SM90] replica slots: wait for the plan, then take our slice
            # [ECO-SM90] `replicated` is decided from EcoMeta (does this
            # slot's logical expert have more than one active copy?), not from
            # a num_fixed prefix: the EPLB packer does not guarantee that the
            # single-copy slots come first on a rank.
            lo = Uint32(0)
            hi = total
            if cutlass.const_expr(self.eco):
                # [ECO-SM90] PASSIVE copies pull nothing. A passive copy is a
                # duplicate an expert holds on a rank that already serves it
                # (unit_ref.py L36): the plan never gives it a quota, and every
                # copy receives the FULL canonical stream, so letting it pull
                # would duplicate that expert's tokens (measured: exactly the
                # 2 passive experts' demand came out twice).
                if cutlass.const_expr(True):
                    if not self._eco_is_active(base, slot_expert, global_slot):
                        hi = Uint32(0)
                n_copies_here = self._eco_ncopies(base, slot_expert)
                if n_copies_here <= Uint32(1):
                    # [ECO-TRACE] slot 10 = first single-copy slot to start
                    # pulling (it never waits for the plan)
                    self._eco_tl_mark(base, epoch, 1, Uint32(0), 10, True)
                if n_copies_here > Uint32(1):
                    # [ECO-TRACE] slot 12 = first replica slot to obtain the
                    # plan (i.e. the replica phase really starting)
                    self._eco_tl_mark(base, epoch, 0, Uint32(0), 12, True)
                    if cutlass.const_expr(not self.eco_nosplit):
                        # [ECO-MEAS] slot 27: how long a REPLICA slot waits for
                        # the plan. This is the direct measurement of the
                        # suspected "24 replica slots serialised behind the
                        # plan" cost; accumulated, so divide by the call count.
                        # slot 6 (min) = first replica CTA to start waiting,
                        # slot 7 (max) = last one to obtain the plan.
                        # 7-6 is the TAIL: from "somebody waits" to "everybody
                        # has the quotas".
                        if cutlass.const_expr(self.eco_trace and self.eco):
                            if tid == 0:
                                self._eco_trace(base, 6, 1)
                        wait_begin = read_globaltimer()
                        # [ECO-TRACE] slot 14: if the single-copy phase is
                        # already finished by the time this replica slot needs
                        # the plan, record how long the plan then still took.
                        # 0 means "no starvation": the plan was ready first.
                        norep_done = Uint32(0)
                        if cutlass.const_expr(self.eco_trace and self.eco):
                            if tid == 0:
                                ctr = self._field(base, 'eco_tl_ctr', Uint32)
                                n_fixed14 = self._eco_num_fixed(base, rank)
                                if (n_fixed14 == Uint32(0) or
                                        ld_acquire_gpu_u32(ctr + Uint32(3))
                                        >= n_fixed14):
                                    norep_done = Uint32(1)
                        if tid == 0:
                            wait_epoch(self._field(base, 'eco_ready', Uint32) + 1,
                                       epoch + Uint32(1), False)
                        cute.arch.sync_threads()
                        if cutlass.const_expr(self.eco_trace and self.eco):
                            if tid == 0:
                                if norep_done != Uint32(0):
                                    self._eco_tl_mark(base, epoch, 4, Uint32(0),
                                                      14, True)
                        if cutlass.const_expr(self.eco_trace and self.eco):
                            if tid == 0:
                                self._eco_trace(base, 7, 2)
                        if cutlass.const_expr(self.eco_trace and self.eco):
                            if tid == 0:
                                tr = self._field(base, 'eco_trace', Uint64)
                                atom_add_u64(tr + Uint64(27),
                                             read_globaltimer() - wait_begin)
                    lo = Uint32(self._field(base, 'eco_quota_start', Uint32)[global_slot])
                    hi = lo + Uint32(self._field(base, 'eco_quota', Uint32)[global_slot])
                    # [ECO-SM90] A COPY'S SLICE MAY BE SHORTER THAN ITS QUOTA.
                    # The plan splits an expert's demand into whole units
                    # (multiples of block M), but the canonical stream is a
                    # per-source concatenation the copies slice in order: the
                    # first copy can be handed a quota that runs past the end
                    # of its own share of the stream while the next copy then
                    # starts beyond it. Each copy therefore pulls
                    # [quota_start, min(quota_start + quota, D_e)) -- a contiguous
                    # range of the SAME canonical order, so the union over the
                    # copies is still exactly [0, D_e) and no token is lost or
                    # duplicated.
                    # The plan sizes a copy's slice in UNITS (multiples of
                    # block M), so a slice can extend past the expert's real
                    # stream -- in particular a quiet expert's second copy gets
                    # a start beyond the end. Clamp to the canonical order's
                    # length: the union of the clamped slices is still exactly
                    # [0, D_e) because the quotas sum to D_e and the starts are
                    # their prefix sums.
                    if lo > total:
                        lo = total
                    if hi > total:
                        hi = total
            meta_base = Uint64(self.off_route_meta) + Uint64(expert) * self.rows * 12
            weight_base = Uint64(self.off_route_weight) + Uint64(expert) * self.rows * 4
            for row in cutlass.range(lo + tid, hi, self.threads):
                source = Uint32(0)
                for candidate in cutlass.range_constexpr(1, self.ranks):
                    if row >= prefix[candidate]:
                        source = Uint32(candidate)
                source_slot = Uint32(row) - prefix[source]
                route = cell(base, Uint64(self.off_source_index) + ((Uint64(expert) * self.ranks + source) * self.capacity + source_slot) * 4, Uint32)[0]
                source_token, route_slot = route // self.topk, route % self.topk
                local_row = Uint32(row) - lo
                rank_out = cell(base, meta_base + Uint64(local_row * 3) * 4, Uint32)
                rank_out[0] = source
                token_out = cell(base, meta_base + Uint64(local_row * 3 + 1) * 4, Uint32)
                token_out[0] = source_token
                slot_out = cell(base, meta_base + Uint64(local_row * 3 + 2) * 4, Uint32)
                slot_out[0] = route_slot
                weight_out = cell(base, weight_base + Uint64(local_row) * 4, Float32)
                weight_out[0] = cell(Uint64(peer_table[source]), Uint64(self.off_topk_weights) + Uint64(route) * 4, Float32)[0]
            cute.arch.sync_threads()
            # A warp retains one source/token pair while copying an entire
            # route, avoiding metadata reloads for every sixteen data bytes.
            # [ECO-PULL] TMA copy, structurally identical to SM100
            # (impls/sm100_fp8_fp4_gc_moe.py L1184-1200):
            #
            #     if (elect_one_sync()) {
            #         bulk_g2s_1d(smem, src, N, bar);        // issue the TMA
            #         mbarrier_arrive_and_expect_tx(bar, N); // THEN expect
            #         mbarrier_wait(bar, phase);             // drain
            #         phase ^= 1;
            #         bulk_s2g_1d(dst, smem, N);             // write back
            #         cp_async_bulk_commit_group();
            #         cp_async_bulk_wait_group(0);
            #     }
            #
            # The old loop moved a token with 320 synchronous
            # `ld.global.v4`/`st.global.v4` pairs of 16 bytes (one per lane per
            # row) and measured 4.4 GB/s -- about 90x below NVLink.
            #
            # Order matters: the arrive/expect goes AFTER the bulk issue, and
            # `elect_one_sync()` pins the work to one lane so `pull_phase`
            # (hoisted above the slot loop, as in SM100 L1072) advances once
            # per row.
            warp_row = Uint32(warp) * Uint32(warp_row_bytes)
            for pull_row in cutlass.range(lo + warp, hi, self.threads // 32):
                local_row = Uint32(pull_row) - lo
                source_rank = cell(base, meta_base + Uint64(local_row * 3) * 4, Uint32)[0]
                token = cell(base, meta_base + Uint64(local_row * 3 + 1) * 4, Uint32)[0]
                remote = Uint64(peer_table[source_rank])
                src = remote + Uint64(self.off_x) + Uint64(token) * self.hidden
                dst = base + Uint64(self.off_l1_acts) + \
                    (Uint64(expert) * self.rows + Uint64(local_row)) * self.hidden
                smem_row = stage + warp_row
                my_bar = pull_bar + Uint64(warp)
                if cutlass.const_expr(self.pull_tma):
                  if elect_one_sync():
                    bulk_g2s_1d(smem_row,
                                cute.make_ptr(Uint8, src, cute.AddressSpace.gmem,
                                              assumed_align=16),
                                Uint32(self.hidden), my_bar)
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        my_bar, Uint32(self.hidden))
                    ph = pull_phase_smem[warp]
                    cute.arch.mbarrier_wait(my_bar, ph)
                    pull_phase_smem[warp] = ph ^ Uint32(1)
                    bulk_s2g_1d(cute.make_ptr(Uint8, dst, cute.AddressSpace.gmem,
                                              assumed_align=16),
                                smem_row, Uint32(self.hidden))
                    cute.arch.cp_async_bulk_commit_group()
                    cute.arch.cp_async_bulk_wait_group(0)
                else:
                    # upstream path: 320 synchronous 16-byte ld/st per row
                    for k in cutlass.range(lane, self.hidden // 16, 32):
                        copy_global_16(dst + Uint64(k) * 16, src + Uint64(k) * 16)
                cute.arch.sync_warp()
                # scales: 160 B per token, small enough to keep the direct path
                remote_sf = remote + Uint64(self.off_x_sf) + \
                    Uint64(token) * Uint64(self.hidden // 128 * 4)
                for sf_k in cutlass.range(lane, self.hidden // 128, 32):
                    sf_value = cell(Uint64(remote_sf), Uint64(sf_k) * 4, Float32)[0]
                    sf_out = cell(base, Uint64(self.off_l1_acts_sf) +
                                  ((Uint64(expert) * (self.hidden // 128) + Uint32(sf_k))
                                   * self.rows + Uint64(local_row)) * 4, Float32)
                    sf_out[0] = sf_value * Float32(64)
            fence_gpu()
            cute.arch.sync_threads()
            owned = hi - lo
            if cutlass.const_expr(self.eco_trace and self.eco):
                # [ECO-TRACE] completion marks, per class. Counting to the
                # rank's own slot total is what makes "all single-copy slots
                # done" exact without a barrier.
                # Targets are this rank's own slot counts: `num_fixed`
                # single-copy slots, then the remaining replica slots.
                n_fixed = self._eco_num_fixed(base, rank)
                n_rep = Uint32(self.local) - n_fixed
                if self._eco_ncopies(base, slot_expert) > Uint32(1):
                    # [ECO-TRACE] slot 17 = FIRST replica expert finished
                    # pulling: the moment the first replica's expert_ready is
                    # published. starvation = slot17 - slot16.
                    self._eco_tl_mark(base, epoch, 7, Uint32(0), 17, True)
                    if n_rep > Uint32(0):
                        self._eco_tl_mark(base, epoch, 2, n_rep, 13, False)
                else:
                    # [ECO-TRACE] slot 18 = FIRST single-copy slot finished
                    # pulling. The counterpart of slot 17 (first replica). The
                    # difference 18-10 is how long the fastest single-copy slot
                    # took, and 11-18 is the spread across the single-copy set.
                    self._eco_tl_mark(base, epoch, 9, Uint32(0), 18, True)
                    if n_fixed > Uint32(0):
                        self._eco_tl_mark(base, epoch, 3, n_fixed, 11, False)
            if tid == 0:
                count_out = cell(base, Uint64(self.off_expert_count) + Uint64(expert) * 4, Uint32)
                count_out[0] = owned
                if cutlass.const_expr(self.has_stats):
                    recv_stats = cute.make_tensor(stats, cute.make_layout((self.local,)))
                    recv_stats[expert] += owned
                st_release_u32(self._field(base, 'expert_ready', Uint32) + Uint32(expert), epoch, False)
                if owned > 0:
                    active_slot = atom_add_u32(head + 7, Uint32(1))
                    active_out = cell(base, Uint64(self.off_active_expert) + Uint64(active_slot) * 4, Uint32)
                    active_out[0] = Uint32(expert)
                    st_release_u32(self._field(base, 'active_ready', Uint32) + active_slot, epoch, False)
                finished = atom_release_gpu_add_u32(head + 8, Uint32(1))
                if finished == self.local - 1:
                    st_release_u32(head + 9, epoch, False)
                    # [PULL-TRACE] pull end: the LAST local slot finished.
                    self._pull_tl_write(base, epoch, 5, read_globaltimer())
            if cutlass.const_expr(self.eco_trace and self.eco):
                if tid == 0:
                    self._eco_trace(base, 8, 2)   # slot 8: role-1 CTA exits
            cute.arch.sync_threads()

    @cute.jit
    def _dispatch_direct(self, base: Uint64, peers: cute.Pointer, rank: Uint32,
                          tokens: Uint32, stats: cute.Pointer):
        tid, _, _ = cute.arch.thread_idx()
        expert, _, _ = cute.arch.block_idx()
        lane, warp = tid % 32, tid // 32
        peer_table = cute.make_tensor(peers, cute.make_layout((self.ranks,)))
        head = self._field(base, 'header', Uint32)
        epoch = ld_acquire_gpu_u32(head)
        if expert == 0 and tid == 0:
            scan_out1 = cell(base, Uint64(8), Uint32)
            scan_out1[0] = tokens
            st_release_u32(head + 1, epoch, True)
        alloc = cutlass.utils.SmemAllocator()
        counts = alloc.allocate_tensor(Uint32, cute.make_layout((self.ranks,)), 4)
        prefix = alloc.allocate_tensor(Uint32, cute.make_layout((self.ranks + 1,)), 4)
        route_count = alloc.allocate_tensor(Uint32, cute.make_layout((1,)), 4)
        if tid < self.ranks:
            remote = Uint64(peer_table[tid])
            wait_epoch(self._field(remote, 'header', Uint32) + 1, epoch, True)
            counts[tid] = cell(remote, Uint64(8), Uint32)[0] * self.topk
        cute.arch.sync_threads()
        if tid == 0:
            running_count = Uint32(0)
            prefix[0] = Uint32(0)
            for prefix_source in cutlass.range_constexpr(self.ranks):
                running_count += counts[prefix_source]
                prefix[prefix_source + 1] = running_count
            route_count[0] = Uint32(0)
        cute.arch.sync_threads()
        meta_base = Uint64(self.off_route_meta) + Uint64(expert) * self.rows * 12
        weight_base = Uint64(self.off_route_weight) + Uint64(expert) * self.rows * 4
        for candidate_route in cutlass.range(tid, prefix[self.ranks], self.threads):
            source = Uint32(0)
            for candidate in cutlass.range_constexpr(1, self.ranks):
                if candidate_route >= prefix[candidate]:
                    source = Uint32(candidate)
            source_route = Uint32(candidate_route) - prefix[source]
            remote = Uint64(peer_table[source])
            expert_id = cell(remote, Uint64(self.off_topk_idx) + Uint64(source_route) * 8, Int64)[0]
            if expert_id == Int64(rank * self.local + Uint32(expert)):
                row = atom_shared_add_u32(route_count.iterator, Uint32(1))
                scan_out3 = cell(base, meta_base + Uint64(row * 3) * 4, Uint32)
                scan_out3[0] = source
                scan_out4 = cell(base, meta_base + Uint64(row * 3 + 1) * 4, Uint32)
                scan_out4[0] = source_route // self.topk
                scan_out5 = cell(base, meta_base + Uint64(row * 3 + 2) * 4, Uint32)
                scan_out5[0] = source_route % self.topk
                scan_out6 = cell(base, weight_base + Uint64(row) * 4, Float32)
                scan_out6[0] = cell(remote, Uint64(self.off_topk_weights) + Uint64(source_route) * 4, Float32)[0]
        cute.arch.sync_threads()
        total = route_count[0]
        # A warp retains one source/token pair while copying an entire
        # route, avoiding metadata reloads for every sixteen data bytes.
        for pull_row in cutlass.range(warp, total, self.threads // 32):
            source_rank = cell(base, meta_base + Uint64(pull_row * 3) * 4, Uint32)[0]
            token = cell(base, meta_base + Uint64(pull_row * 3 + 1) * 4, Uint32)[0]
            remote = Uint64(peer_table[source_rank])
            src = remote + Uint64(self.off_x) + Uint64(token) * self.hidden
            dst = base + Uint64(self.off_l1_acts) + (Uint64(expert) * self.rows + Uint32(pull_row)) * self.hidden
            for k in cutlass.range(lane, self.hidden // 16, 32):
                copy_global_16(dst + Uint64(k) * 16, src + Uint64(k) * 16)
            for sf_k in cutlass.range(lane, self.hidden // 128, 32):
                sf_value = cell(remote, Uint64(self.off_x_sf) + (Uint64(token) * (self.hidden // 128) + Uint32(sf_k)) * 4, Float32)[0]
                sf_out = cell(base, Uint64(self.off_l1_acts_sf) + ((Uint64(expert) * (self.hidden // 128) + Uint32(sf_k)) * self.rows + Uint32(pull_row)) * 4, Float32)
                sf_out[0] = sf_value * Float32(64)
        fence_gpu()
        cute.arch.sync_threads()
        if tid == 0:
            count_out = cell(base, Uint64(self.off_expert_count) + Uint64(expert) * 4, Uint32)
            count_out[0] = total
            if cutlass.const_expr(self.has_stats):
                recv_stats = cute.make_tensor(stats, cute.make_layout((self.local,)))
                recv_stats[expert] += total
            st_release_u32(self._field(base, 'expert_ready', Uint32) + Uint32(expert), epoch, False)
            if total > 0:
                active_slot = atom_add_u32(head + 7, Uint32(1))
                active_out = cell(base, Uint64(self.off_active_expert) + Uint64(active_slot) * 4, Uint32)
                active_out[0] = Uint32(expert)
                st_release_u32(self._field(base, 'active_ready', Uint32) + active_slot, epoch, False)
            finished = atom_release_gpu_add_u32(head + 8, Uint32(1))
            if finished == self.local - 1:
                st_release_u32(head + 9, epoch, False)
        cute.arch.sync_threads()



    @cute.jit
    def _stage(self, base: Uint64, slab: cute.Pointer, expert: Uint32,
               ntile: Uint32, start: Uint32, valid: Uint32, kb: Uint32,
               stage: Uint32, weights: cute.Pointer, offsets: cute.Pointer,
               n: cutlass.Constexpr, l1: cutlass.Constexpr):
        tid, _, _ = cute.arch.thread_idx()
        k_size = self.hidden if l1 else self.intermediate
        n_size = 2 * self.intermediate if l1 else self.hidden
        input_offset = self.off_l1_acts if l1 else self.off_l2_acts
        stage_base = stage * self.stage_bytes
        if tid == 0:
            bar = cute.recast_ptr(slab + self.smem - 128 + stage * 8, dtype=Uint64)
            cute.arch.mbarrier_arrive_and_expect_tx(bar, 17408 * self.routed_n_factor)
            for sub in cutlass.range_constexpr(self.routed_n_factor):
                index = (Uint64(expert) * (n_size // 256) + ntile * self.routed_n_factor + sub) * (k_size // 128) + kb
                bulk_g2s_1d(slab + stage_base + sub * 16384, weights + index * 16384, Uint32(16384), bar)
                bulk_g2s_1d(slab + stage_base + 16384 * self.routed_n_factor + sub * 1024, offsets + index * 1024, Uint32(1024), bar)
        for j in cutlass.range(tid, n * 8, 256):
            row, k = Uint32(j // 8), Uint32((j % 8) * 16)
            size = Uint32(0)
            if row < valid:
                size = Uint32(16)
            address = base + Uint64(input_offset) + ((Uint64(expert) * self.rows + start + row) * k_size + kb * 128 + k)
            src = cute.make_ptr(Uint8, address, cute.AddressSpace.gmem, assumed_align=16)
            copy_async_16(slab + stage_base + self.routed_x_offset + canonical_offset(row, k), src, size)
        if tid < n:
            sf_size = Uint32(0)
            if tid < valid:
                sf_size = Uint32(4)
            sf_offset = self.off_l1_acts_sf if l1 else self.off_l2_acts_sf
            sf_address = base + Uint64(sf_offset) + ((Uint64(expert) * (k_size // 128) + kb) * self.rows + start + tid) * 4
            sf_src = cute.make_ptr(Uint8, sf_address, cute.AddressSpace.gmem, assumed_align=4)
            copy_async_4(slab + stage_base + self.sf_stage_offset + tid * 4, sf_src, sf_size)
        copy_commit()

    @cute.jit
    def _tile_sync(self, base: Uint64, peers: cute.Pointer, slab: cute.Pointer,
               expert: Uint32, ntile: Uint32, start: Uint32, valid: Uint32,
               weights: cute.Pointer, offsets: cute.Pointer, secondary: cute.Pointer,
               clamp: Float32, n: cutlass.Constexpr, l1: cutlass.Constexpr):
        k_size = self.hidden if l1 else self.intermediate
        tid, _, _ = cute.arch.thread_idx()
        lane, warp = tid % 32, (tid // 32) % 4
        wg = tid // 128
        if tid == 0:
            cute.arch.mbarrier_init(cute.recast_ptr(slab + self.smem - 128, dtype=Uint64), 1)
            cute.arch.mbarrier_init(cute.recast_ptr(slab + self.smem - 120, dtype=Uint64), 1)
        cute.arch.sync_threads()
        cute.arch.mbarrier_init_fence()
        final = cute.make_rmem_tensor((2 * self.routed_n_factor, n // 2), Float32)
        final.fill(Float32(0))
        secondary_value = cute.make_tensor(secondary, cute.make_layout((self.local,)))[expert]
        self._stage(base, slab, expert, ntile, start, valid, Uint32(0), Uint32(0), weights, offsets, n, l1)
        for kb in cutlass.range(k_size // 128):
            stage = Uint32(kb % 2)
            if tid == 0:
                cute.arch.mbarrier_wait(cute.recast_ptr(slab + self.smem - 128 + stage * 8, dtype=Uint64), Uint32((kb // 2) % 2))
            copy_wait()
            cute.arch.sync_threads()
            tma_store_fence()
            if kb + 1 < k_size // 128:
                self._stage(base, slab, expert, ntile, start, valid, Uint32(kb + 1), stage ^ Uint32(1), weights, offsets, n, l1)
            scales = cute.make_rmem_tensor((n // 4,), Float32)
            for group in cutlass.range_constexpr(n // 8):
                token = Uint32(group * 8 + 2 * (lane % 4))
                raw_sf = load_scale_pair(slab + stage * self.stage_bytes + Uint32(self.sf_stage_offset) + token * 4)
                scale0 = raw_sf[0] * secondary_value
                scale1 = raw_sf[1] * secondary_value
                scales[group * 2], scales[group * 2 + 1] = scale0, scale1
            for h in cutlass.range_constexpr(2 * self.routed_n_factor):
                weight_half = Uint32(wg * 2 * self.routed_n_factor + h)
                row = Uint32(warp * 16 + lane // 4)
                word_ptr = slab + stage * self.stage_bytes + weight_half * 4096 + Uint32(warp * 1024 + lane * 16)
                scale_ptr = slab + stage * self.stage_bytes + Uint32(16384 * self.routed_n_factor) + weight_half * 256 + Uint32(warp * 64 + (lane // 4) * 8)
                frag = decode_fragment(word_ptr, scale_ptr)
                part = wgmma_rs_k128(frag, smem_descriptor(slab + stage * self.stage_bytes + self.routed_x_offset), n)
                for group in cutlass.range_constexpr(n // 8):
                    final[h, group * 4] = cute.math.fma(part[group * 4], scales[group * 2], final[h, group * 4])
                    final[h, group * 4 + 1] = cute.math.fma(part[group * 4 + 1], scales[group * 2 + 1], final[h, group * 4 + 1])
                    final[h, group * 4 + 2] = cute.math.fma(part[group * 4 + 2], scales[group * 2], final[h, group * 4 + 2])
                    final[h, group * 4 + 3] = cute.math.fma(part[group * 4 + 3], scales[group * 2 + 1], final[h, group * 4 + 3])
        cute.arch.sync_threads()
        for h in cutlass.range_constexpr(2 * self.routed_n_factor):
            output_half = Uint32(wg * 2 * self.routed_n_factor + h)
            for group in cutlass.range_constexpr(n // 8):
                token = Uint32(group * 8 + 2 * (lane % 4))
                row = Uint32(warp * 16 + lane // 4)
                skew = Uint32((token % 8) // 2 * 8)
                # L1 needs row XOR only; L2 also permutes vector chunks.
                if cutlass.const_expr(l1):
                    shared_cell(slab, (token * (256 * self.routed_n_factor) + ((output_half * 64 + row) ^ skew)) * 4, Float32)[0] = final[h, group * 4]
                    shared_cell(slab, ((token + 1) * (256 * self.routed_n_factor) + ((output_half * 64 + row) ^ skew)) * 4, Float32)[0] = final[h, group * 4 + 1]
                    shared_cell(slab, (token * (256 * self.routed_n_factor) + ((output_half * 64 + row + 8) ^ skew)) * 4, Float32)[0] = final[h, group * 4 + 2]
                    shared_cell(slab, ((token + 1) * (256 * self.routed_n_factor) + ((output_half * 64 + row + 8) ^ skew)) * 4, Float32)[0] = final[h, group * 4 + 3]
                else:
                    shared_cell(slab, (token * (256 * self.routed_n_factor) + ((output_half * 64 + row) ^ skew ^ (((output_half * 64 + row) & 32) >> 3))) * 4, Float32)[0] = final[h, group * 4]
                    shared_cell(slab, ((token + 1) * (256 * self.routed_n_factor) + ((output_half * 64 + row) ^ skew ^ (((output_half * 64 + row) & 32) >> 3))) * 4, Float32)[0] = final[h, group * 4 + 1]
                    shared_cell(slab, (token * (256 * self.routed_n_factor) + ((output_half * 64 + row + 8) ^ skew ^ (((output_half * 64 + row + 8) & 32) >> 3))) * 4, Float32)[0] = final[h, group * 4 + 2]
                    shared_cell(slab, ((token + 1) * (256 * self.routed_n_factor) + ((output_half * 64 + row + 8) ^ skew ^ (((output_half * 64 + row + 8) & 32) >> 3))) * 4, Float32)[0] = final[h, group * 4 + 3]
        cute.arch.sync_threads()
        if cutlass.const_expr(l1):
            for ep_token in cutlass.range(tid // 32, valid, 8):
                for sub in cutlass.range_constexpr(self.routed_n_factor):
                    values = cute.make_rmem_tensor((4,), Float32)
                    amax = Float32(0)
                    weight = cell(base, Uint64(self.off_route_weight) + (Uint64(expert) * self.rows + start + ep_token) * 4, Float32)[0]
                    skew = Uint32((ep_token % 8) // 2 * 8)
                    raw = load_shared_8(slab + ep_token * (1024 * self.routed_n_factor) + sub * 1024 + ((lane * 4) ^ skew) * 4, True)
                    for ep_j in cutlass.range_constexpr(4):
                        gate = cute.math.min(raw[ep_j], clamp)
                        up = cute.math.min(cute.math.max(raw[ep_j + 4], -clamp), clamp)
                        sigmoid = cute.math.rcp(Float32(1) + cute.math.exp(-gate, fastmath=self.fast_math))
                        values[ep_j] = gate * sigmoid * up * weight
                        amax = cute.math.max(amax, abs_f32(values[ep_j]))
                    for shift in cutlass.range_constexpr(5):
                        amax = cute.math.max(amax, cute.arch.shuffle_sync_bfly(amax, 16 >> shift))
                    scale = cute.math.max(amax, Float32(1e-10)) * Float32(1.0 / 448.0)
                    inverse = cute.math.rcp(scale)
                    if lane == 0:
                        activation_scale_out = cell(base, Uint64(self.off_l2_acts_sf) + ((Uint64(expert) * (self.intermediate // 128) + ntile * self.routed_n_factor + sub) * self.rows + start + ep_token) * 4, Float32)
                        activation_scale_out[0] = scale * Float32(64)
                    low = f32x2_to_e4m3x2_bits(values[0] * inverse, values[1] * inverse)
                    high = f32x2_to_e4m3x2_bits(values[2] * inverse, values[3] * inverse)
                    activation_out = cell(base, Uint64(self.off_l2_acts) + (Uint64(expert) * self.rows + start + ep_token) * self.intermediate + ntile * (128 * self.routed_n_factor) + sub * 128 + lane * 4, Uint32)
                    activation_out[0] = Uint32(low) | (Uint32(high) << 16)
        else:
            peer_table = cute.make_tensor(peers, cute.make_layout((self.ranks,)))
            for ep_j in cutlass.range(tid, valid * (32 * self.routed_n_factor), 256):
                ep_token, ep_col = ep_j // (32 * self.routed_n_factor), (ep_j % (32 * self.routed_n_factor)) * 8
                meta = Uint64(self.off_route_meta) + (Uint64(expert) * self.rows + start + ep_token) * 12
                source = cell(base, meta, Uint32)[0]
                source_token = cell(base, meta + 4, Uint32)[0]
                slot = cell(base, meta + 8, Uint32)[0]
                remote = Uint64(peer_table[source])
                skew = Uint32((ep_token % 8) // 2 * 8)
                # DeepGEMM-style column-bit swizzle: address each LDS128
                # independently; the logical eight values and STG128 stay intact.
                values_lo = load_shared_4(slab + ep_token * (1024 * self.routed_n_factor) + (ep_col ^ skew ^ ((ep_col & 32) >> 3)) * 4)
                values_hi = load_shared_4(slab + ep_token * (1024 * self.routed_n_factor) + ((ep_col + 4) ^ skew ^ (((ep_col + 4) & 32) >> 3)) * 4)
                values8 = (*values_lo, *values_hi)
                destination = remote + Uint64(self.off_combine) + ((Uint64(source_token) * self.topk + slot) * self.hidden + ntile * (256 * self.routed_n_factor) + ep_col) * 2
                store_bf16_8(destination, values8)
        cute.arch.sync_threads()

    @cute.jit
    def _stage_weights(self, slab: cute.Pointer, expert: Uint32, ntile: Uint32,
                        kb: Uint32, stage: Uint32, weights: cute.Pointer,
                        offsets: cute.Pointer, l1: cutlass.Constexpr):
        tid, _, _ = cute.arch.thread_idx()
        if tid == (0 if self.max_m == 64 else 64):
            k_size = self.hidden if l1 else self.intermediate
            n_size = 2 * self.intermediate if l1 else self.hidden
            stage_base = stage * self.stage_bytes
            full = cute.recast_ptr(slab + self.full_offset + stage * 8, dtype=Uint64)
            cute.arch.mbarrier_arrive_and_expect_tx(full, 17408)
            index = (Uint64(expert) * (n_size // 256) + ntile) * (k_size // 128) + kb
            bulk_g2s_1d(slab + stage_base, weights + index * 16384, Uint32(16384), full)
            bulk_g2s_1d(slab + stage_base + 16384, offsets + index * 1024, Uint32(1024), full)

    @cute.jit
    def _stage_activation(self, base: Uint64, slab: cute.Pointer, expert: Uint32,
                           start: Uint32, valid: Uint32, kb: Uint32,
                           stage: Uint32, n: Uint32, l1: cutlass.Constexpr):
        tid, _, _ = cute.arch.thread_idx()
        k_size = self.hidden if l1 else self.intermediate
        input_offset = self.off_l1_acts if l1 else self.off_l2_acts
        stage_base = stage * self.stage_bytes
        full = cute.recast_ptr(slab + self.full_offset + stage * 8, dtype=Uint64)
        for j in cutlass.range(tid, n * 8, 128 if self.max_m == 64 else 64):
            row, k = Uint32(j // 8), Uint32((j % 8) * 16)
            size = Uint32(0)
            if row < valid:
                size = Uint32(16)
            address = base + Uint64(input_offset) + ((Uint64(expert) * self.rows + start + row) * k_size + kb * 128 + k)
            src = cute.make_ptr(Uint8, address, cute.AddressSpace.gmem, assumed_align=16)
            copy_async_16(slab + stage_base + 17408 + canonical_offset(row, k), src, size)
        if tid < n:
            sf_size = Uint32(0)
            if tid < valid:
                sf_size = Uint32(4)
            sf_offset = self.off_l1_acts_sf if l1 else self.off_l2_acts_sf
            sf_address = base + Uint64(sf_offset) + ((Uint64(expert) * (k_size // 128) + kb) * self.rows + start + tid) * 4
            sf_src = cute.make_ptr(Uint8, sf_address, cute.AddressSpace.gmem, assumed_align=4)
            copy_async_4(slab + stage_base + self.sf_stage_offset + tid * 4, sf_src, sf_size)
        copy_commit()
        copy_arrive(full)

    @cute.jit
    def _bucket_tile(self, base: Uint64, peers: cute.Pointer, slab: cute.Pointer,
                      expert: Uint32, ntile: Uint32, start: Uint32, valid: Uint32,
                      weights: cute.Pointer, offsets: cute.Pointer, secondary: cute.Pointer,
                      clamp: Float32, l1: cutlass.Constexpr):
        if cutlass.const_expr(self.max_m == 8):
            self._tile_sync(base, peers, slab, expert, ntile, start, valid,
                       weights, offsets, secondary, clamp, 8, l1)
        else:
            if valid <= 8:
                self._tile_sync(base, peers, slab, expert, ntile, start, valid,
                           weights, offsets, secondary, clamp, 8, l1)
            else:
                if cutlass.const_expr(self.max_m == 16):
                    self._tile_sync(base, peers, slab, expert, ntile, start, valid,
                               weights, offsets, secondary, clamp, 16, l1)
                else:
                    if valid <= 16:
                        self._tile_sync(base, peers, slab, expert, ntile, start, valid,
                                   weights, offsets, secondary, clamp, 16, l1)
                    elif valid <= 32:
                        self._tile_sync(base, peers, slab, expert, ntile, start, valid,
                                   weights, offsets, secondary, clamp, 32, l1)
                    else:
                        self._tile_sync(base, peers, slab, expert, ntile, start, valid,
                                   weights, offsets, secondary, clamp, 64, l1)

    @cute.jit
    def _claim_command(self, base: Uint64, command: cute.Tensor,
                         epoch: Uint32, is_l1: Boolean):
        tid, _, _ = cute.arch.thread_idx()
        head = self._field(base, 'header', Uint32)
        active_ptr = self._field(base, 'active_ready', Uint32)
        expert_ptr = self._field(base, 'expert_ready', Uint32)
        l1_ptr = self._field(base, 'l1_done', Uint32)
        if tid == 0:
            task, active_slot, ntile = Uint32(0), Uint32(0), Uint32(0)
            if is_l1:
                task = claim_sm90_task(head + 3)
                active_slot, ntile = task // Uint32(self.intermediate // (128 * self.routed_n_factor)), task % Uint32(self.intermediate // (128 * self.routed_n_factor))
            else:
                task = claim_sm90_task(head + 10)
                active_slot, ntile = task // Uint32(self.hidden // (256 * self.routed_n_factor)), task % Uint32(self.hidden // (256 * self.routed_n_factor))
            if cutlass.const_expr(self.eco_trace and self.eco):
                # slot 10 (min) = role-2 first polls for work
                # (SM100 calls this "counts seen"; here the equivalent is the
                #  first look at the dispatch's publication counters)
                self._eco_trace(base, 10, 1)
            active_count = ld_acquire_gpu_u32(head + 7)
            if is_l1:
                dispatch_done = ld_acquire_gpu_u32(head + 9)
                if cutlass.const_expr(self.eco_trace and self.eco):
                    # slot 11 (min) = first time a role-2 CTA blocks on a
                    # producer counter (the SM90 analogue of "wait for plan")
                    self._eco_trace(base, 11, 1)
                if active_slot >= active_count:
                    # [ECO-TRACE] slot 16 = FIRST consumer spin of THIS call:
                    # the earliest moment a role-2 CTA finds no published work.
                    # A resident role-2 CTA starts spinning for the NEXT call
                    # while the previous one is still finishing, so the mark
                    # must be gated on this call's epoch -- otherwise the row
                    # records a timestamp from the previous call (measured:
                    # 31 ms "starvation" on ranks 1-7).
                    self._eco_tl_mark_epoch(base, epoch, 6, 16)
                while active_slot >= active_count and dispatch_done != epoch:
                    active_count = ld_acquire_gpu_u32(head + 7)
                    dispatch_done = ld_acquire_gpu_u32(head + 9)
                if cutlass.const_expr(self.eco_trace and self.eco):
                    # slot 12 (max) = last CTA that got work
                    self._eco_trace(base, 12, 2)
                active_count = ld_acquire_gpu_u32(head + 7)
            command[0] = Uint32(0)
            if active_slot < active_count:
                # [ECO-TRACE] slot 15 = first role-2 CTA to actually get work
                # (epoch-gated: a resident CTA may reach this for the next call)
                self._eco_tl_mark_epoch(base, epoch, 5, 15)
                wait_epoch(active_ptr + active_slot, epoch, False)
                # [ECO-TRACE] slot 19 = a role-2 CTA got its first SLOT
                # (active_ready seen). Compare with 15: 19-15 is how long the
                # consumer actually blocked before it could start.
                self._eco_tl_mark_epoch(base, epoch, 10, 19)
                expert = cell(base, Uint64(self.off_active_expert) + Uint64(active_slot) * 4, Uint32)[0]
                wait_epoch(expert_ptr + expert, epoch, False)
                # [ECO-TRACE] slot 20 = data for that expert is readable, i.e.
                # the first GEMM can start.
                self._eco_tl_mark_epoch(base, epoch, 11, 20)
                if not is_l1:
                    wait_count(l1_ptr + expert, Uint32(self.intermediate // 128))
                command[0] = Uint32(1)
                command[1] = Uint32(expert)
                command[2] = Uint32(ntile)

    @cute.jit
    def _publish_task(self, slab: cute.Pointer, task_iter: Uint32,
                       phase: Uint32, expert: Uint32, ntile: Uint32,
                       start: Uint32, valid: Uint32, last: Uint32):
        tid, _, _ = cute.arch.thread_idx()
        slot = task_iter % 2
        if tid == 0 and task_iter >= 2:
            cute.arch.mbarrier_wait(cute.recast_ptr(slab + self.task_empty_offset + slot * 8, dtype=Uint64), Uint32((task_iter // 2 - 1) % 2))
        sync_loader()
        if tid == 0:
            record = slab + self.task_record_offset + slot * 32
            shared_cell(record, Uint32(0), Uint32)[0] = phase
            shared_cell(record, Uint32(4), Uint32)[0] = expert
            shared_cell(record, Uint32(8), Uint32)[0] = ntile
            shared_cell(record, Uint32(12), Uint32)[0] = start
            shared_cell(record, Uint32(16), Uint32)[0] = valid
            shared_cell(record, Uint32(20), Uint32)[0] = last
            cute.arch.mbarrier_arrive(cute.recast_ptr(slab + self.task_full_offset + slot * 8, dtype=Uint64))
        sync_loader()

    @cute.jit
    def _load_persistent_tile(self, base: Uint64, slab: cute.Pointer,
                               data_iter: Uint32, expert: Uint32, ntile: Uint32,
                               start: Uint32, valid: Uint32, weights: cute.Pointer,
                               offsets: cute.Pointer, l1: cutlass.Constexpr):
        tid, _, _ = cute.arch.thread_idx()
        k_size = self.hidden if l1 else self.intermediate
        load_m = Uint32(64)
        if valid <= 8:
            load_m = Uint32(8)
        elif valid <= 16:
            load_m = Uint32(16)
        elif valid <= 32:
            load_m = Uint32(32)
        elif valid <= 40:
            load_m = Uint32(40)
        elif valid <= 48:
            load_m = Uint32(48)
        if cutlass.const_expr(self.max_m == 64):
            for kb in cutlass.range(k_size // 128):
                iteration = data_iter + Uint32(kb)
                stage = iteration % self.stages
                if tid == 0 and iteration >= self.stages:
                    cute.arch.mbarrier_wait(cute.recast_ptr(slab + self.empty_offset + stage * 8, dtype=Uint64), Uint32((iteration // self.stages - 1) % 2))
                sync_loader()
                self._stage_weights(slab, expert, ntile, Uint32(kb), stage, weights, offsets, l1)
                self._stage_activation(base, slab, expert, start, valid, Uint32(kb), stage, load_m, l1)
        else:
            if tid < 64:
                for kb in cutlass.range(k_size // 128):
                    iteration = data_iter + Uint32(kb)
                    stage = iteration % self.stages
                    if tid == 0 and iteration >= self.stages:
                        cute.arch.mbarrier_wait(cute.recast_ptr(slab + self.empty_offset + stage * 8, dtype=Uint64), Uint32((iteration // self.stages - 1) % 2))
                    sync_activation()
                    self._stage_activation(base, slab, expert, start, valid, Uint32(kb), stage, load_m, l1)
            else:
                for weight_kb in cutlass.range(k_size // 128):
                    weight_iter = data_iter + Uint32(weight_kb)
                    weight_stage = weight_iter % self.stages
                    if tid == 64 and weight_iter >= self.stages:
                        cute.arch.mbarrier_wait(cute.recast_ptr(slab + self.empty_offset + weight_stage * 8, dtype=Uint64), Uint32((weight_iter // self.stages - 1) % 2))
                    sync_weight()
                    self._stage_weights(slab, expert, ntile, Uint32(weight_kb), weight_stage, weights, offsets, l1)
        sync_loader()
        return data_iter + Uint32(k_size // 128)

    @cute.jit
    def _produce_tasks(self, base: Uint64, slab: cute.Pointer, epoch: Uint32,
                        w1: cute.Pointer, s1: cute.Pointer,
                        w2: cute.Pointer, s2: cute.Pointer):
        command = cute.make_tensor(cute.recast_ptr(slab + self.command_offset, dtype=Uint32), cute.make_layout((3,)))
        is_l1, running = True, True
        task_iter, data_iter = Uint32(0), Uint32(0)
        while running:
            self._claim_command(base, command, epoch, is_l1)
            sync_loader()
            if command[0] == 0:
                if is_l1:
                    is_l1 = False
                else:
                    running = False
                    # [ECO-MEAS] slot 13 (max) = role-2 leaves its loop.
                    # 16-13 is the teardown (fence / sync) after the last tile.
                    if cutlass.const_expr(self.eco_trace and self.eco):
                        if cute.arch.thread_idx()[0] == 0:
                            self._eco_trace(base, 13, 2)
            else:
                expert = Uint32(command[1])
                ntile = Uint32(command[2])
                count = cell(base, Uint64(self.off_expert_count) + Uint64(expert) * 4, Uint32)[0]
                for start in cutlass.range(0, count, self.max_m):
                    valid = cute.math.min(Uint32(self.max_m), count - Uint32(start))
                    last = Uint32(0)
                    if Uint32(start) + valid == count:
                        last = Uint32(1)
                    phase = Uint32(1 if is_l1 else 2)
                    self._publish_task(slab, task_iter, phase, expert, ntile,
                                       Uint32(start), valid, last)
                    task_iter += Uint32(1)
                    if is_l1:
                        data_iter = self._load_persistent_tile(base, slab, data_iter,
                                      expert, ntile, Uint32(start), valid, w1, s1, True)
                    else:
                        data_iter = self._load_persistent_tile(base, slab, data_iter,
                                      expert, ntile, Uint32(start), valid, w2, s2, False)
        self._publish_task(slab, task_iter, Uint32(0), Uint32(0), Uint32(0),
                           Uint32(0), Uint32(0), Uint32(0))

    @cute.jit
    def _math_persistent_tile(self, base: Uint64, peers: cute.Pointer, slab: cute.Pointer,
                               data_iter: Uint32, expert: Uint32, ntile: Uint32,
                               start: Uint32, valid: Uint32, secondary: cute.Pointer,
                               clamp: Float32, n: cutlass.Constexpr, l1: cutlass.Constexpr):
        tid, _, _ = cute.arch.thread_idx()
        k_size = self.hidden if l1 else self.intermediate
        epi_slab = slab + (self.full_offset + 1024)
        math_tid = tid - 128
        lane, warp = math_tid % 32, (math_tid // 32) % 4
        wg = math_tid // 128
        final = cute.make_rmem_tensor((2, n // 2), Float32)
        final.fill(Float32(0))
        secondary_value = cute.make_tensor(secondary, cute.make_layout((self.local,)))[expert]
        row = Uint32(warp * 16 + lane // 4)
        first_stage = Uint32(data_iter % self.stages)
        if lane == 0:
            cute.arch.mbarrier_wait(cute.recast_ptr(slab + self.full_offset + first_stage * 8, dtype=Uint64), Uint32((data_iter // self.stages) % 2))
        cute.arch.sync_warp()
        tma_store_fence()
        initial_word = slab + first_stage * self.stage_bytes + Uint32(wg * 8192 + warp * 1024 + lane * 16)
        initial_scale = slab + first_stage * self.stage_bytes + Uint32(16384 + wg * 512 + warp * 64 + (lane // 4) * 8)
        frag0 = decode_fragment(initial_word, initial_scale)
        for kb in cutlass.range(k_size // 128):
            iteration = data_iter + Uint32(kb)
            stage = Uint32(iteration % self.stages)
            descriptor = smem_descriptor(slab + stage * self.stage_bytes + 17408)
            raw0 = wgmma_rs_issue(frag0, descriptor, n)
            word_ptr = slab + stage * self.stage_bytes + Uint32(wg * 8192 + warp * 1024 + lane * 16)
            scale_ptr = slab + stage * self.stage_bytes + Uint32(16384 + wg * 512 + warp * 64 + (lane // 4) * 8)
            frag1 = decode_fragment(word_ptr + 4096, scale_ptr + 256)
            scales = cute.make_rmem_tensor((n // 4,), Float32)
            for group in cutlass.range_constexpr(n // 8):
                token = Uint32(group * 8 + 2 * (lane % 4))
                raw_sf = load_scale_pair(slab + stage * self.stage_bytes + Uint32(self.sf_stage_offset) + token * 4)
                scale0 = raw_sf[0] * secondary_value
                scale1 = raw_sf[1] * secondary_value
                scales[group * 2], scales[group * 2 + 1] = scale0, scale1
            part0 = wgmma_rs_wait(raw0)
            for group in cutlass.range_constexpr(n // 8):
                final[0, group * 4] = cute.math.fma(part0[group * 4], scales[group * 2], final[0, group * 4])
                final[0, group * 4 + 1] = cute.math.fma(part0[group * 4 + 1], scales[group * 2 + 1], final[0, group * 4 + 1])
                final[0, group * 4 + 2] = cute.math.fma(part0[group * 4 + 2], scales[group * 2], final[0, group * 4 + 2])
                final[0, group * 4 + 3] = cute.math.fma(part0[group * 4 + 3], scales[group * 2 + 1], final[0, group * 4 + 3])
            raw1 = wgmma_rs_issue(frag1, descriptor, n)
            if kb + 1 < k_size // 128:
                next_stage = Uint32((iteration + 1) % self.stages)
                if lane == 0:
                    cute.arch.mbarrier_wait(cute.recast_ptr(slab + self.full_offset + next_stage * 8, dtype=Uint64), Uint32(((iteration + 1) // self.stages) % 2))
                cute.arch.sync_warp()
                tma_store_fence()
                next_word = slab + next_stage * self.stage_bytes + Uint32(wg * 8192 + warp * 1024 + lane * 16)
                next_scale = slab + next_stage * self.stage_bytes + Uint32(16384 + wg * 512 + warp * 64 + (lane // 4) * 8)
                frag0 = decode_fragment(next_word, next_scale)
            part1 = wgmma_rs_wait(raw1)
            for group in cutlass.range_constexpr(n // 8):
                final[1, group * 4] = cute.math.fma(part1[group * 4], scales[group * 2], final[1, group * 4])
                final[1, group * 4 + 1] = cute.math.fma(part1[group * 4 + 1], scales[group * 2 + 1], final[1, group * 4 + 1])
                final[1, group * 4 + 2] = cute.math.fma(part1[group * 4 + 2], scales[group * 2], final[1, group * 4 + 2])
                final[1, group * 4 + 3] = cute.math.fma(part1[group * 4 + 3], scales[group * 2 + 1], final[1, group * 4 + 3])
            tma_store_fence()
            cute.arch.sync_warp()
            if lane == 0:
                cute.arch.mbarrier_arrive(cute.recast_ptr(slab + self.empty_offset + stage * 8, dtype=Uint64))
        sync_math()
        for h in cutlass.range_constexpr(2):
            output_half = Uint32(wg * 2 + h)
            for group in cutlass.range_constexpr(n // 8):
                token = Uint32(group * 8 + 2 * (lane % 4))
                row = Uint32(warp * 16 + lane // 4)
                # XOR token-pair bits into columns, as in swizzled staging.
                skew = Uint32((token % 8) // 2 * 8)
                # L1 needs row XOR only; L2 also permutes vector chunks.
                if cutlass.const_expr(l1):
                    shared_cell(epi_slab, (token * 256 + ((output_half * 64 + row) ^ skew)) * 4, Float32)[0] = final[h, group * 4]
                    shared_cell(epi_slab, ((token + 1) * 256 + ((output_half * 64 + row) ^ skew)) * 4, Float32)[0] = final[h, group * 4 + 1]
                    shared_cell(epi_slab, (token * 256 + ((output_half * 64 + row + 8) ^ skew)) * 4, Float32)[0] = final[h, group * 4 + 2]
                    shared_cell(epi_slab, ((token + 1) * 256 + ((output_half * 64 + row + 8) ^ skew)) * 4, Float32)[0] = final[h, group * 4 + 3]
                else:
                    shared_cell(epi_slab, (token * 256 + ((output_half * 64 + row) ^ skew ^ (((output_half * 64 + row) & 32) >> 3))) * 4, Float32)[0] = final[h, group * 4]
                    shared_cell(epi_slab, ((token + 1) * 256 + ((output_half * 64 + row) ^ skew ^ (((output_half * 64 + row) & 32) >> 3))) * 4, Float32)[0] = final[h, group * 4 + 1]
                    shared_cell(epi_slab, (token * 256 + ((output_half * 64 + row + 8) ^ skew ^ (((output_half * 64 + row + 8) & 32) >> 3))) * 4, Float32)[0] = final[h, group * 4 + 2]
                    shared_cell(epi_slab, ((token + 1) * 256 + ((output_half * 64 + row + 8) ^ skew ^ (((output_half * 64 + row + 8) & 32) >> 3))) * 4, Float32)[0] = final[h, group * 4 + 3]
        sync_math()
        if cutlass.const_expr(l1):
            for ep_token in cutlass.range(math_tid // 32, valid, 8):
                values = cute.make_rmem_tensor((4,), Float32)
                amax = Float32(0)
                weight = cell(base, Uint64(self.off_route_weight) + (Uint64(expert) * self.rows + start + ep_token) * 4, Float32)[0]
                skew = Uint32((ep_token % 8) // 2 * 8)
                raw = load_shared_8(epi_slab + ep_token * 1024 + ((lane * 4) ^ skew) * 4, True)
                for ep_j in cutlass.range_constexpr(4):
                    gate = cute.math.min(raw[ep_j], clamp)
                    up = cute.math.min(cute.math.max(raw[ep_j + 4], -clamp), clamp)
                    sigmoid = cute.math.rcp(Float32(1) + cute.math.exp(-gate, fastmath=self.fast_math))
                    values[ep_j] = gate * sigmoid * up * weight
                    amax = cute.math.max(amax, abs_f32(values[ep_j]))
                for shift in cutlass.range_constexpr(5):
                    amax = cute.math.max(amax, cute.arch.shuffle_sync_bfly(amax, 16 >> shift))
                scale = cute.math.max(amax, Float32(1e-10)) * Float32(1.0 / 448.0)
                inverse = cute.math.rcp(scale)
                if lane == 0:
                    activation_scale_out = cell(base, Uint64(self.off_l2_acts_sf) + ((Uint64(expert) * (self.intermediate // 128) + ntile) * self.rows + start + ep_token) * 4, Float32)
                    activation_scale_out[0] = scale * Float32(64)
                low = f32x2_to_e4m3x2_bits(values[0] * inverse, values[1] * inverse)
                high = f32x2_to_e4m3x2_bits(values[2] * inverse, values[3] * inverse)
                activation_out = cell(base, Uint64(self.off_l2_acts) + (Uint64(expert) * self.rows + start + ep_token) * self.intermediate + ntile * 128 + lane * 4, Uint32)
                activation_out[0] = Uint32(low) | (Uint32(high) << 16)
        else:
            peer_table = cute.make_tensor(peers, cute.make_layout((self.ranks,)))
            for ep_j in cutlass.range(math_tid, valid * 32, 256):
                ep_token, ep_col = ep_j // 32, (ep_j % 32) * 8
                meta = Uint64(self.off_route_meta) + (Uint64(expert) * self.rows + start + ep_token) * 12
                source = cell(base, meta, Uint32)[0]
                source_token = cell(base, meta + 4, Uint32)[0]
                slot = cell(base, meta + 8, Uint32)[0]
                remote = Uint64(peer_table[source])
                skew = Uint32((ep_token % 8) // 2 * 8)
                # DeepGEMM-style column-bit swizzle: address each LDS128
                # independently; the logical eight values and STG128 stay intact.
                values_lo = load_shared_4(epi_slab + ep_token * 1024 + (ep_col ^ skew ^ ((ep_col & 32) >> 3)) * 4)
                values_hi = load_shared_4(epi_slab + ep_token * 1024 + ((ep_col + 4) ^ skew ^ (((ep_col + 4) & 32) >> 3)) * 4)
                values8 = (*values_lo, *values_hi)
                destination = remote + Uint64(self.off_combine) + ((Uint64(source_token) * self.topk + slot) * self.hidden + ntile * 256 + ep_col) * 2
                store_bf16_8(destination, values8)
        sync_math()
        return data_iter + Uint32(k_size // 128)

    @cute.jit
    def _consume_tasks(self, base: Uint64, peers: cute.Pointer, slab: cute.Pointer,
                        sec1: cute.Pointer, sec2: cute.Pointer, clamp: Float32):
        tid, _, _ = cute.arch.thread_idx()
        math_tid, lane = tid - 128, tid % 32
        task_iter, data_iter = Uint32(0), Uint32(0)
        running = True
        while running:
            slot = task_iter % 2
            if lane == 0:
                cute.arch.mbarrier_wait(cute.recast_ptr(slab + self.task_full_offset + slot * 8, dtype=Uint64), Uint32((task_iter // 2) % 2))
            cute.arch.sync_warp()
            record = slab + self.task_record_offset + slot * 32
            phase = Uint32(shared_cell(record, Uint32(0), Uint32)[0])
            expert = Uint32(shared_cell(record, Uint32(4), Uint32)[0])
            ntile = Uint32(shared_cell(record, Uint32(8), Uint32)[0])
            start = Uint32(shared_cell(record, Uint32(12), Uint32)[0])
            valid = Uint32(shared_cell(record, Uint32(16), Uint32)[0])
            last = Uint32(shared_cell(record, Uint32(20), Uint32)[0])
            sync_math()
            if math_tid == 0:
                cute.arch.mbarrier_arrive(cute.recast_ptr(slab + self.task_empty_offset + slot * 8, dtype=Uint64))
            task_iter += Uint32(1)
            if phase == 0:
                running = False
            else:
                if phase == 1:
                    data_iter = self._select_persistent_tile(base, peers, slab, data_iter,
                                     expert, ntile, start, valid, sec1, clamp, True)
                else:
                    data_iter = self._select_persistent_tile(base, peers, slab, data_iter,
                                     expert, ntile, start, valid, sec2, clamp, False)
                fence_gpu()
                sync_math()
                if math_tid == 0 and phase == 1 and last != 0:
                    atom_release_gpu_add_u32(self._field(base, 'l1_done', Uint32) + expert, Uint32(1))

    @cute.jit
    def _select_persistent_tile(self, base: Uint64, peers: cute.Pointer, slab: cute.Pointer,
                                 data_iter: Uint32, expert: Uint32, ntile: Uint32,
                                 start: Uint32, valid: Uint32, secondary: cute.Pointer,
                                 clamp: Float32, l1: cutlass.Constexpr):
        result = Uint32(data_iter)
        if cutlass.const_expr(self.max_m == 8):
            result = self._math_persistent_tile(base, peers, slab, data_iter, expert, ntile, start, valid, secondary, clamp, 8, l1)
        elif cutlass.const_expr(self.max_m == 16):
            if valid <= 8:
                result = self._math_persistent_tile(base, peers, slab, data_iter, expert, ntile, start, valid, secondary, clamp, 8, l1)
            else:
                result = self._math_persistent_tile(base, peers, slab, data_iter, expert, ntile, start, valid, secondary, clamp, 16, l1)
        elif cutlass.const_expr(self.max_m == 32):
            if valid <= 8:
                result = self._math_persistent_tile(base, peers, slab, data_iter, expert, ntile, start, valid, secondary, clamp, 8, l1)
            elif valid <= 16:
                result = self._math_persistent_tile(base, peers, slab, data_iter, expert, ntile, start, valid, secondary, clamp, 16, l1)
            else:
                result = self._math_persistent_tile(base, peers, slab, data_iter, expert, ntile, start, valid, secondary, clamp, 32, l1)
        else:
            if valid <= 8:
                result = self._math_persistent_tile(base, peers, slab, data_iter, expert, ntile, start, valid, secondary, clamp, 8, l1)
            elif valid <= 16:
                result = self._math_persistent_tile(base, peers, slab, data_iter, expert, ntile, start, valid, secondary, clamp, 16, l1)
            elif valid <= 32:
                result = self._math_persistent_tile(base, peers, slab, data_iter, expert, ntile, start, valid, secondary, clamp, 32, l1)
            elif valid <= 40:
                result = self._math_persistent_tile(base, peers, slab, data_iter, expert, ntile, start, valid, secondary, clamp, 40, l1)
            elif valid <= 48:
                result = self._math_persistent_tile(base, peers, slab, data_iter, expert, ntile, start, valid, secondary, clamp, 48, l1)
            else:
                result = self._math_persistent_tile(base, peers, slab, data_iter, expert, ntile, start, valid, secondary, clamp, 64, l1)
        return result

    @cute.jit
    def _compute_persistent(self, base: Uint64, peers: cute.Pointer, w1: cute.Pointer,
                             s1: cute.Pointer, sec1: cute.Pointer, w2: cute.Pointer,
                             s2: cute.Pointer, sec2: cute.Pointer, clamp: Float32):
        tid, _, _ = cute.arch.thread_idx()
        alloc = cutlass.utils.SmemAllocator()
        slab = cute.recast_ptr(alloc.allocate(self.smem, 1024), dtype=Uint8)
        if tid < self.stages:
            cute.arch.mbarrier_init(cute.recast_ptr(slab + self.full_offset + tid * 8, dtype=Uint64), 129 if self.max_m == 64 else 65)
            cute.arch.mbarrier_init(cute.recast_ptr(slab + self.empty_offset + tid * 8, dtype=Uint64), 8)
        if tid < 2:
            cute.arch.mbarrier_init(cute.recast_ptr(slab + self.task_full_offset + tid * 8, dtype=Uint64), 1)
            cute.arch.mbarrier_init(cute.recast_ptr(slab + self.task_empty_offset + tid * 8, dtype=Uint64), 1)
        cute.arch.sync_threads()
        cute.arch.mbarrier_init_fence()
        epoch = ld_acquire_gpu_u32(self._field(base, 'header', Uint32))
        # [ECO-MEAS] slot 9 (min) = earliest role-2 CTA in its scheduling loop
        if cutlass.const_expr(self.eco_trace and self.eco):
            if tid == 0:
                self._eco_trace(base, 9, 1)
        if tid < 128:
            cute.arch.setmaxregister_decrease(32 if self.max_m <= 32 else 64)
            self._produce_tasks(base, slab, epoch, w1, s1, w2, s2)
            fence_sys()
            sync_tile()
        else:
            cute.arch.setmaxregister_increase(104 if self.max_m <= 32 else 216)
            self._consume_tasks(base, peers, slab, sec1, sec2, clamp)
            fence_sys()
            sync_tile()
        # [ECO-MEAS] slot 16 (max) = last role-2 CTA to leave
        if cutlass.const_expr(self.eco_trace and self.eco):
            if tid == 0:
                self._eco_trace(base, 16, 2)

    @cute.jit
    def _compute_sync(self, base: Uint64, peers: cute.Pointer, w1: cute.Pointer,
                  s1: cute.Pointer, sec1: cute.Pointer, w2: cute.Pointer,
                  s2: cute.Pointer, sec2: cute.Pointer, clamp: Float32):
        tid, _, _ = cute.arch.thread_idx()
        alloc = cutlass.utils.SmemAllocator()
        slab = cute.recast_ptr(alloc.allocate(self.smem, 1024), dtype=Uint8)
        command = cute.make_tensor(cute.recast_ptr(slab + self.smem - 96, dtype=Uint32), cute.make_layout((3,)))
        head = self._field(base, 'header', Uint32)
        epoch = ld_acquire_gpu_u32(head)
        # Dispatch publishes only nonempty experts. Reserved queue entries
        # have an epoch flag so a consumer never reads an unpublished expert.
        # Exhaust L1 claims before any L2 wait: every dependency is in flight.
        is_l1 = True
        running = True
        while running:
            self._claim_command(base, command, epoch, is_l1)
            cute.arch.sync_threads()
            if command[0] == 0:
                if is_l1:
                    is_l1 = False
                else:
                    running = False
            else:
                expert = Uint32(command[1])
                ntile = Uint32(command[2])
                count = cell(base, Uint64(self.off_expert_count) + Uint64(expert) * 4, Uint32)[0]
                for start in cutlass.range(0, count, self.max_m):
                    valid = cute.math.min(Uint32(self.max_m), count - Uint32(start))
                    if is_l1:
                        self._bucket_tile(base, peers, slab, expert, ntile, Uint32(start), valid,
                                          w1, s1, sec1, clamp, True)
                    else:
                        self._bucket_tile(base, peers, slab, expert, ntile, Uint32(start), valid,
                                          w2, s2, sec2, clamp, False)
                fence_gpu()
                cute.arch.sync_threads()
                if is_l1 and tid == 0:
                    atom_release_gpu_add_u32(self._field(base, 'l1_done', Uint32) + expert, Uint32(self.routed_n_factor))
                cute.arch.sync_threads()
        fence_sys()

    @cute.jit
    def _compute(self, base: Uint64, peers: cute.Pointer, w1: cute.Pointer,
                  s1: cute.Pointer, sec1: cute.Pointer, w2: cute.Pointer,
                  s2: cute.Pointer, sec2: cute.Pointer, clamp: Float32):
        if cutlass.const_expr(self.warp_specialized):
            self._compute_persistent(base, peers, w1, s1, sec1, w2, s2, sec2, clamp)
        else:
            self._compute_sync(base, peers, w1, s1, sec1, w2, s2, sec2, clamp)

    @cute.jit
    def _shared_stage(self, base: Uint64, slab: cute.Pointer, weights: cute.Pointer,
                      scales: cute.Pointer, ntile: Uint32, start: Uint32,
                      valid: Uint32, kb: Uint32, bar: cute.Pointer,
                      n: cutlass.Constexpr, l1: cutlass.Constexpr):
        """Issue exact W/SF TMA and X/SF cp.async copies, without waiting."""
        tid, _, _ = cute.arch.thread_idx()
        k_size = self.hidden if l1 else self.shared_i
        input_offset = self.off_shared_l1_acts if l1 else self.off_shared_l2_acts
        sf_offset = self.off_shared_l1_acts_sf if l1 else self.off_shared_l2_acts_sf
        activation_offset = 36864
        sf_stage = activation_offset + n * 128
        if tid == 0:
            cute.arch.mbarrier_arrive_and_expect_tx(bar, 36864)
            tile = Uint64(ntile) * (k_size // 128) + kb
            bulk_g2s_1d(slab, weights + tile * 32768, Uint32(32768), bar)
            bulk_g2s_1d(slab + 32768, cute.recast_ptr(scales, dtype=Uint8) + tile * 4096, Uint32(4096), bar)
        for j in cutlass.range(tid, n * 8, self.threads):
            row, col = Uint32(j // 8), Uint32(j % 8) * 16
            size = Uint32(16) if row < valid else Uint32(0)
            address = base + Uint64(input_offset) + Uint64(start + row) * k_size + kb * 128 + col
            copy_async_16(slab + activation_offset + canonical_offset(row, col),
                          cute.make_ptr(Uint8, address, cute.AddressSpace.gmem, assumed_align=16), size)
        for j in cutlass.range(tid, n * 4, self.threads):
            row, kg = Uint32(j % n), Uint32(j // n)
            size = Uint32(4) if row < valid else Uint32(0)
            address = base + Uint64(sf_offset) + (Uint64(start + row) * (k_size // 32) + kb * 4 + kg) * 4
            copy_async_4(slab + sf_stage + Uint32(j) * 4,
                         cute.make_ptr(Uint8, address, cute.AddressSpace.gmem, assumed_align=4), size)
        copy_commit()

    @cute.jit
    def _shared_tile(self, base: Uint64, slab: cute.Pointer, weights: cute.Pointer,
                     scales: cute.Pointer, ntile: Uint32, start: Uint32,
                     valid: Uint32, clamp: Float32, n: cutlass.Constexpr,
                     l1: cutlass.Constexpr):
        """Native dense FP8 SE tile in the shared compute role, with separate row/K32 scales.

        A K128 stage has 32768B FP8 W, 4096B FP32 W SF, n*128B X,
        and n*16B X SF. Persistent CTAs prefetch the next of two stages
        during current-stage math in the independent shared role.
        The end-of-stage CTA barrier retires all readers before slot reuse;
        each slot has its own mbarrier, with parity advancing on every reuse.
        """
        tid, _, _ = cute.arch.thread_idx()
        math_tid = tid
        lane, warp = math_tid % 32, (math_tid // 32) % 4
        wg = math_tid // 128
        k_size = self.hidden if l1 else self.shared_i
        activation_offset = 36864
        sf_stage = activation_offset + n * 128
        bar = cute.recast_ptr(slab + self.smem - 128, dtype=Uint64)
        # The effective Role4 launch reserves both dense stages independently
        # of the routed geometry and its shared-memory budget.
        stage_stride = (36864 + n * 144 + 1023) // 1024 * 1024
        if tid < 2:
            cute.arch.mbarrier_init(bar + tid, 1)
        cute.arch.sync_threads()
        cute.arch.mbarrier_init_fence()
        final = cute.make_rmem_tensor((2, n // 2), Float32)
        final.fill(Float32(0))
        self._shared_stage(base, slab, weights, scales, ntile, start, valid,
                           Uint32(0), bar, n, l1)
        for kb in cutlass.range(k_size // 128):
            tile_slab = slab + Uint32(kb % 2) * stage_stride
            copy_wait()
            if tid == 0:
                cute.arch.mbarrier_wait(bar + Uint32(kb % 2), Uint32((kb // 2) % 2))
            cute.arch.sync_threads()
            tma_store_fence()
            if kb + 1 < k_size // 128:
                next_slot = Uint32((kb + 1) % 2)
                self._shared_stage(base, slab + next_slot * stage_stride,
                                   weights, scales, ntile, start, valid,
                                   Uint32(kb + 1), bar + next_slot, n, l1)
            for h in cutlass.range_constexpr(2):
                half = Uint32(wg * 2 + h)
                row = Uint32(warp * 16 + lane // 4)
                for pair in cutlass.range_constexpr(2):
                    fragment0 = load_shared_u32x4(tile_slab + ((half * 4 + pair * 2) * 128 + Uint32(math_tid % 128)) * 16)
                    fragment1 = load_shared_u32x4(tile_slab + ((half * 4 + pair * 2 + 1) * 128 + Uint32(math_tid % 128)) * 16)
                    descriptor = smem_descriptor(tile_slab + activation_offset) + Uint64(pair * 4)
                    parts = wgmma_rs_k32_pair(
                        fragment0, fragment1, descriptor, n)
                    for bank in cutlass.range_constexpr(2):
                        kg = pair * 2 + bank
                        part = tuple(parts[bank * (n // 2) + i] for i in range(n // 2))
                        sw0 = shared_cell(tile_slab, Uint32(32768) + (half * 256 + row * 4 + kg) * 4, Float32)[0]
                        sw1 = shared_cell(tile_slab, Uint32(32768) + (half * 256 + (row + 8) * 4 + kg) * 4, Float32)[0]
                        for group in cutlass.range_constexpr(n // 8):
                            token = Uint32(group * 8 + 2 * (lane % 4))
                            sa = load_scale_pair(tile_slab + sf_stage + Uint32(kg * n) * 4 + token * 4)
                            final[h, group * 4] = cute.math.fma(part[group * 4], sw0 * sa[0], final[h, group * 4])
                            final[h, group * 4 + 1] = cute.math.fma(part[group * 4 + 1], sw0 * sa[1], final[h, group * 4 + 1])
                            final[h, group * 4 + 2] = cute.math.fma(part[group * 4 + 2], sw1 * sa[0], final[h, group * 4 + 2])
                            final[h, group * 4 + 3] = cute.math.fma(part[group * 4 + 3], sw1 * sa[1], final[h, group * 4 + 3])
            cute.arch.sync_threads()
        for h in cutlass.range_constexpr(2):
            half = Uint32(wg * 2 + h)
            row = Uint32(warp * 16 + lane // 4)
            for group in cutlass.range_constexpr(n // 8):
                token = Uint32(group * 8 + 2 * (lane % 4))
                skew = Uint32((token % 8) // 2 * 8)
                for r in cutlass.range_constexpr(2):
                    for t in cutlass.range_constexpr(2):
                        col = half * 64 + row + r * 8
                        if cutlass.const_expr(l1):
                            ep_smem_offset = (token + t) * 1024 + (col ^ skew) * 4
                        else:
                            ep_smem_offset = (token + t) * 1024 + (col ^ skew ^ ((col & 32) >> 3)) * 4
                        shared_cell(slab, ep_smem_offset, Float32)[0] = final[h, group * 4 + r * 2 + t]
        cute.arch.sync_threads()
        if cutlass.const_expr(l1):
            for ep_token in cutlass.range(math_tid // 32, valid, 8):
                skew = Uint32((ep_token % 8) // 2 * 8)
                raw = load_shared_8(slab + ep_token * 1024 + ((lane * 4) ^ skew) * 4, True)
                values = cute.make_rmem_tensor((4,), Float32)
                amax = Float32(0)
                for j in cutlass.range_constexpr(4):
                    # Match standalone DeepGEMM BF16 GEMM1 output rounding.
                    gate = cute.math.min(round_bf16_f32(raw[j]), clamp)
                    up = cute.math.min(cute.math.max(round_bf16_f32(raw[j + 4]), -clamp), clamp)
                    values[j] = gate * cute.math.rcp(Float32(1) + cute.math.exp(-gate, fastmath=self.fast_math)) * up
                    amax = cute.math.max(amax, abs_f32(values[j]))
                for shift in cutlass.range_constexpr(3):
                    amax = cute.math.max(amax, cute.arch.shuffle_sync_bfly(amax, 4 >> shift))
                exponent = ue8m0_fp8_scale_exp(amax)
                inverse = ue8m0_fp8_inv(exponent)
                if lane % 8 == 0:
                    sf = cell(base, Uint64(self.off_shared_l2_acts_sf) + (Uint64(start + ep_token) * (self.shared_i // 32) + ntile * 4 + lane // 8) * 4, Float32)
                    sf[0] = bits_float(exponent << 23)
                lo = f32x2_to_e4m3x2_bits(values[0] * inverse, values[1] * inverse)
                hi = f32x2_to_e4m3x2_bits(values[2] * inverse, values[3] * inverse)
                out = cell(base, Uint64(self.off_shared_l2_acts) + Uint64(start + ep_token) * self.shared_i + ntile * 128 + lane * 4, Uint32)
                out[0] = Uint32(lo) | (Uint32(hi) << 16)
        else:
            for j in cutlass.range(math_tid, valid * 32, 256):
                token, col = Uint32(j // 32), Uint32(j % 32) * 8
                skew = Uint32((token % 8) // 2 * 8)
                lo = load_shared_4(slab + token * 1024 + (col ^ skew ^ ((col & 32) >> 3)) * 4)
                hi = load_shared_4(slab + token * 1024 + ((col + 4) ^ skew ^ (((col + 4) & 32) >> 3)) * 4)
                ep_global_address = base + Uint64(self.off_shared_y) + (Uint64(start + token) * self.hidden + ntile * 256 + col) * 2
                store_bf16_8(ep_global_address, (*lo, *hi))
        cute.arch.sync_threads()

    @cute.jit
    def _shared_select(self, base: Uint64, slab: cute.Pointer, weights: cute.Pointer,
                       scales: cute.Pointer, ntile: Uint32, start: Uint32,
                       valid: Uint32, clamp: Float32, l1: cutlass.Constexpr):
        if cutlass.const_expr(self.max_m == 8):
            self._shared_tile(base, slab, weights, scales, ntile, start, valid, clamp, 8, l1)
        elif cutlass.const_expr(self.max_m == 16):
            self._shared_tile(base, slab, weights, scales, ntile, start, valid, clamp, 16, l1)
        elif cutlass.const_expr(self.max_m == 32):
            self._shared_tile(base, slab, weights, scales, ntile, start, valid, clamp, 32, l1)
        else:
            self._shared_tile(base, slab, weights, scales, ntile, start, valid, clamp, 64, l1)

    @cute.jit
    def _compute_shared(self, base: Uint64, slab: cute.Pointer, tokens: Uint32, w1: cute.Pointer,
                        s1: cute.Pointer, w2: cute.Pointer, s2: cute.Pointer,
                        clamp: Float32):
        # Replicated local-token branch, independently queued from routed experts.
        # L1 ticket exhaustion precedes L2 waits, so no CTA can wait on an
        # unclaimed L1 dependency. T0 claims no shared tasks.
        tid, _, _ = cute.arch.thread_idx()
        command = cute.make_tensor(cute.recast_ptr(slab + self.smem - 96, dtype=Uint32), cute.make_layout((1,)))
        head = self._field(base, 'header', Uint32)
        blocks = (tokens + self.max_m - 1) // self.max_m
        for phase in cutlass.range_constexpr(2):
            tiles = self.shared_i // 128 if phase == 0 else self.hidden // 256
            running = True
            while running:
                if tid == 0:
                    command[0] = claim_sm90_task(head + 20 + phase)
                cute.arch.sync_threads()
                task = Uint32(command[0])
                if task >= blocks * tiles:
                    running = False
                else:
                    block, ntile = task // tiles, task % tiles
                    start = block * self.max_m
                    valid = cute.math.min(Uint32(self.max_m), tokens - start)
                    if cutlass.const_expr(phase == 0):
                        self._shared_select(base, slab, w1, s1, ntile, start, valid, clamp, True)
                        fence_gpu()
                        cute.arch.sync_threads()
                        if tid == 0:
                            atom_release_gpu_add_u32(self._field(base, 'shared_l1_done', Uint32) + block, Uint32(1))
                    else:
                        if tid == 0:
                            wait_count(self._field(base, 'shared_l1_done', Uint32) + block, Uint32(self.shared_i // 128))
                        cute.arch.sync_threads()
                        self._shared_select(base, slab, w2, s2, ntile, start, valid, clamp, False)
                cute.arch.sync_threads()
        fence_sys()

    @cute.jit
    def _combine(self, base: Uint64, peers: cute.Pointer, y: cute.Pointer, rank: Uint32, tokens: Uint32):
        tid, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        head = self._field(base, 'header', Uint32)
        epoch = ld_acquire_gpu_u32(head)
        # [ECO-MEAS] slot 14 (min) = earliest role-3 CTA in combine.
        # NOT gated on `self.eco`: the vanilla arm needs this too, otherwise
        # there is no way to compare the combine phase across arms.
        if cutlass.const_expr(self.eco_trace):
            if tid == 0:
                self._eco_trace(base, 14, 1)
        # [ECO-TRACE] combine start, EVERY arm: `eco_trace` only exists in an
        # ECO layout, so the vanilla arm measures this through `pull_tl`.
        if cutlass.const_expr(self.pull_trace):
            if block == 0 and tid == 0:
                self._pull_tl_write(base, epoch, 8, read_globaltimer())
        table = cute.make_tensor(peers, cute.make_layout((self.ranks,)))
        cute.arch.sync_threads()
        if tid < 128:
            # Local Role1/2 graph predecessors have retired. These fields
            # have ONLY local readers; cleanup may overlap peer scatter wait
            # and complete FP32 slot combine in the other 256 threads.
            if block == 0 and tid < self.ranks:
                # Push completion to each peer; consumers acquire local flags.
                st_release_u32(self._field(Uint64(table[tid]), 'compute_ready', Uint32) + rank, epoch, True)
            for expert in cutlass.range(block * 128 + tid, self.local, self.grid * 128):
                flag_out = cell(base, Uint64(self.off_active_ready) + Uint64(expert) * 4, Uint32)
                flag_out[0] = epoch
                done_out = cell(base, Uint64(self.off_l1_done) + Uint64(expert) * 4, Uint32)
                done_out[0] = Uint32(0)
            if block == 0 and tid == 0:
                for c in cutlass.range_constexpr(8 if self.shared_experts > 0 else 6):
                    counter_out = cell(base, Uint64((3, 7, 8, 10, 11, 13, 20, 21)[c] * 4), Uint32)
                    counter_out[0] = Uint32(0)
                if cutlass.const_expr(self.eco):
                    # [ECO-SM90] Clear both plan handshake words for the next
                    # call. The plan waits on eco_ready[0] == epoch + 1 and the
                    # replica pull on eco_ready[1] == epoch + 1, so a stale
                    # value from the previous call would let a rank skip its
                    # wait (and read last call's quotas).
                    eco_rd = self._field(base, 'eco_ready', Uint32)
                    eco_rd[0] = Uint32(0)
                    eco_rd[1] = Uint32(0)
                    # NOTE: eco_rd[2] ("counts consumed") must NOT be cleared
                    # here -- the next dispatch compares it against epoch-1 and
                    # relies on it surviving the combine stage.
                    
                    # [ECO-SM90] `header[25]/[26]` are phase A's whole-grid
                    # histogram rendezvous (counter + signal). They must be
                    # zeroed per call or the next rendezvous returns against a
                    # half-built histogram. NOTE: 20/21 are the shared-expert
                    # task queue and 18/19 the cleanup counters -- using 21 here
                    # collided and deadlocked.
                    for c in cutlass.range_constexpr(2):
                        a_out = cell(base, Uint64((25, 26)[c] * 4), Uint32)
                        a_out[0] = Uint32(0)
                    # [ECO-SM90] `header[29]/[30]` order the broadcast's grid
                    # rendezvous; `header[31]/[23]` order clear-before-count on
                    # the per-expert histogram.
                    for c in cutlass.range_constexpr(4):
                        b_out = cell(base, Uint64((29, 30, 31, 23)[c] * 4), Uint32)
                        b_out[0] = Uint32(0)
                    # [ECO-TRACE] clear the timeline's first/last-writer
                    # counters for the next call.
                    for c in cutlass.range_constexpr(8):
                        tl_out = cell(base, Uint64(self.off_eco_tl_ctr) + Uint64(c) * 4,
                                      Uint32)
                        tl_out[0] = Uint32(0)
            if cutlass.const_expr(self.shared_experts > 0):
                for block_m in cutlass.range(block * 128 + tid, (self.capacity + 7) // 8, self.grid * 128):
                    out = cell(base, Uint64(self.off_shared_l1_done) + Uint64(block_m) * 4, Uint32)
                    out[0] = Uint32(0)
            # Retain peer-readable send_count/source_index through this call.
            # The all-peer compute_ready rendezvous below retires their old readers
            # before any rank may begin its next dispatch initialization.
            fence_gpu()
            sync_loader()
            if tid == 0:
                if cutlass.const_expr(self.cleanup_grids == 1):
                    if block == 0:
                        st_release_u32(head + 19, epoch, False)
                else:
                    if block < self.cleanup_grids:
                        arrived = atom_acq_rel_add_u32(head + 18, Uint32(1))
                        if arrived == self.cleanup_grids - 1:
                            st_release_u32(head + 19, epoch, False)
                    if block == 0:
                        wait_epoch(head + 19, epoch, False)
            sync_loader()
        else:
            math_tid = tid - 128
            if math_tid < self.ranks:
                wait_epoch(self._field(base, 'compute_ready', Uint32) + math_tid, epoch, True)
            sync_math()
            for index in cutlass.range(block * 256 + math_tid, tokens * (self.hidden // 8), self.grid * 256):
                token, col = index // (self.hidden // 8), (index % (self.hidden // 8)) * 8
                total = cute.make_rmem_tensor((8,), Float32)
                total.fill(Float32(0))
                for slot in cutlass.range_constexpr(self.topk):
                    expert = cell(base, Uint64(self.off_topk_idx) + Uint64(token * self.topk + slot) * 8, Int64)[0]
                    if expert >= 0:
                        address = base + Uint64(self.off_combine) + (Uint64(token * self.topk + slot) * self.hidden + col) * 2
                        values8 = load_bf16_8(address)
                        for i in cutlass.range_constexpr(8):
                            total[i] += values8[i]
                if cutlass.const_expr(self.shared_experts > 0):
                    shared = load_bf16_8(base + Uint64(self.off_shared_y) + Uint64(index) * 16)
                    for i in cutlass.range_constexpr(8):
                        total[i] = round_bf16_f32(total[i]) + shared[i]
                store_bf16_8(Uint64(y.toint()) + Uint64(index) * 16, tuple(total[i] for i in range(8)))
            fence_sys()
            sync_math()
            if math_tid == 0:
                st_release_u32(self._field(base, 'combine_block_done', Uint32) + block, epoch, False)
            if block == 0:
                for other_block in cutlass.range(math_tid, self.grid, 256):
                    wait_epoch(self._field(base, 'combine_block_done', Uint32) + other_block, epoch, False)
                sync_math()
                # Refresh EVERY unused flag after all old readers retire.
                # This prevents stale zero from aliasing epoch FFFFFFFF->0,
                # including when successive calls use different grid sizes.
                for flag in cutlass.range(math_tid, self.finalize_flags, 256):
                    flag_out = cell(base, Uint64(self.off_combine_block_done) + Uint64(flag) * 4, Uint32)
                    flag_out[0] = epoch
                fence_gpu()
                sync_math()
        # Retire every LOCAL combine/cleanup group before the successor.
        # A next-call scatter cannot overwrite a peer's old combine slots:
        # both dispatch paths await every peer's next input_ready before
        # publishing any expert_ready, so that peer has finished this call.
        # The earlier all-peer compute_ready wait also retires all old source readers.
        # This removes a redundant all-rank exit wait, including for T0.
        sync_tile()
        if block == 0 and tid == 0:
            final_out = cell(base, Uint64(16), Uint32)
            final_out[0] = Uint32(0)
            cleanup_out = cell(base, Uint64(18 * 4), Uint32)
            cleanup_out[0] = Uint32(0)
            # [ECO-MEAS] slot 15 (max) = last role-3 CTA done (epoch bump).
            # Ungated for the same reason as slot 14.
            if cutlass.const_expr(self.eco_trace):
                self._eco_trace(base, 15, 2)
            if cutlass.const_expr(self.pull_trace):
                self._pull_tl_write(base, epoch, 9, read_globaltimer())
            st_release_u32(head, epoch + Uint32(1), False)

    @cute.kernel
    def kernel(self, workspace: cute.Pointer, peers: cute.Pointer, y: cute.Pointer,
                w1: cute.Pointer, s1: cute.Pointer, sec1: cute.Pointer,
                w2: cute.Pointer, s2: cute.Pointer, sec2: cute.Pointer,
                stats: cute.Pointer, sw1: cute.Pointer, ss1: cute.Pointer,
                sw2: cute.Pointer, ss2: cute.Pointer, rank: Uint32, tokens: Uint32, clamp: Float32):
        base = Uint64(workspace.toint())
        if cutlass.const_expr(self.role == 1):
            if cutlass.const_expr(self.direct_route_scan):
                self._dispatch_direct(base, peers, rank, tokens, stats)
            else:
                self._dispatch(base, peers, rank, tokens, stats)
        elif cutlass.const_expr(self.role == 2):
            self._compute(base, peers, w1, s1, sec1, w2, s2, sec2, clamp)
        elif cutlass.const_expr(self.role == 4):
            alloc = cutlass.utils.SmemAllocator()
            slab = cute.recast_ptr(alloc.allocate(self.smem, 1024), dtype=Uint8)
            self._compute_shared(base, slab, tokens, sw1, ss1, sw2, ss2, clamp)
        elif cutlass.const_expr(self.role == 5):
            # [ECO-SM90] one CTA per rank computes the identical plan
            alloc = cutlass.utils.SmemAllocator()
            slab = cute.recast_ptr(alloc.allocate(self.eco_smem, 1024), dtype=Uint8)
            self._eco_plan(base, peers, rank, tokens, slab)
        else:
            self._combine(base, peers, y, rank, tokens)

    @cute.jit
    def __call__(self, workspace: cute.Pointer, peers: cute.Pointer, y: cute.Pointer,
                  w1: cute.Pointer, s1: cute.Pointer, sec1: cute.Pointer,
                  w2: cute.Pointer, s2: cute.Pointer, sec2: cute.Pointer,
                  stats: cute.Pointer, sw1: cute.Pointer, ss1: cute.Pointer,
                  sw2: cute.Pointer, ss2: cute.Pointer, rank: Uint32, tokens: Uint32, clamp: Float32):
        self.kernel(workspace, peers, y, w1, s1, sec1, w2, s2, sec2,
                    stats, sw1, ss1, sw2, ss2, rank, tokens, clamp).launch(
            # [ECO-SM90] role 5 is a single CTA with its own smem budget.
            grid=((1 if self.role == 5 else self.grid), 1, 1),
            block=(self.threads, 1, 1),
            max_number_threads=(self.threads, 1, 1),
            min_blocks_per_mp=(4 if self.role == 3 else ((8 if self.threads == 128 else 4) if self.role == 1 else ((3 if self.max_m == 8 else 2) if self.role in (2, 4) and not self.warp_specialized else (2 if self.role in (2, 4) and self.max_m <= 32 else 1)))))
