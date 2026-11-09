"""SM90 role preparation and pointer ABI; compiler utilities are shared with GC MoE."""
from __future__ import annotations

import ctypes
from dataclasses import dataclass, replace
from pathlib import Path
import re

import torch
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Uint8, Uint32, Uint64
from cutlass.cute.runtime import make_ptr

from ...impls.sm90_fp8_fp4_gc_moe import SM90GCMoERoleKernel
from ...runtime.sm90_gc_moe import RoleBinary, get_context
from ...runtime.jit import (
    COMPILE_LOCK, CompilerIdentity, assemble_ptx, compile_kernel, compiler_identity,
)
from ..heuristics.sm90_gc_moe import (
    get_sm90_role_config, get_routed_role_config, get_shared_role_config,
)


@dataclass(frozen=True)
class _Prepared:
    binaries: tuple[RoleBinary, ...]


_CACHE: dict[tuple, _Prepared] = {}
_BINARY_CACHE: dict[tuple, tuple[RoleBinary, ...]] = {}


def _compiler_identity(target: str) -> CompilerIdentity:
    return compiler_identity(target, SM90GCMoERoleKernel.__call__,
                             Path(__file__).resolve().parents[2], 'gc_moe_sm90_role')


def _assemble_role_ptx(ptx, role, dump_dir, identity):
    ptx = _clean_cute_ptx(ptx)
    formals = _entry_formals(ptx)
    if len(formals) != 17:
        raise RuntimeError('SM90 CuTe launch ABI must have fourteen pointers and three scalars')
    for ordinal, formal in enumerate(formals):
        kind = (r'\.u64\s+\.ptr\s+\.global\s+\.align\s+16' if ordinal < 14
                else r'\.u32' if ordinal < 16 else r'\.f32')
        if re.search(kind + r'\s+\S+_param_' + str(ordinal) + r',?\s*$', formal) is None:
            raise RuntimeError(f'SM90 CuTe launch formal {ordinal} changed type or order')
    if role in (2, 4) and ('wgmma.mma_async' not in ptx or 'cp.async.bulk' not in ptx):
        raise RuntimeError('SM90 CuTe compute lost its Hopper WGMMA/TMA operations')
    return assemble_ptx(ptx, dump_dir, f'sm90_role{role}', target='sm_90a',
                        ptxas_options=identity.ptxas_options)


def _clean_cute_ptx(ptx: str) -> str:
    if ptx.endswith("\x00"):
        ptx = ptx[:-1]
    if "\x00" in ptx:
        raise RuntimeError('unexpected NUL inside CuTe PTX')
    return ptx

def _entry_formals(ptx: str) -> list[str]:
    """Read only the launch ABI, excluding function and call-local params."""
    entries = list(re.finditer(
        r"(?m)^[ \t]*\.visible[ \t]+\.entry[ \t]+[A-Za-z0-9_.$]+\([ \t]*$",
        ptx))
    if len(entries) != 1:
        raise RuntimeError('CuTe kernel entry count changed')
    start = entries[0].end()
    closing = re.search(r"(?m)^\)[ \t]*$", ptx[start:])
    if closing is None:
        raise RuntimeError('CuTe kernel entry signature is unterminated')
    lines = [line for line in ptx[start:start + closing.start()].splitlines()
             if line.strip()]
    if not lines or any(re.fullmatch(r"[ \t]*\.param\b[^\n]*", line) is None
                        for line in lines):
        raise RuntimeError('CuTe kernel entry formals changed')
    return lines

# [SM-REUSE] Per-SM shared-memory budget on this H20.  Used to derive the
# role-6 CTA count (one CTA per dispatch SM at 199 KB).  A wrong value can only
# make the estimate conservative.
_SMEM_BUDGET = 233472

_POINTER_TYPES = (Uint8, Uint64, BFloat16, Uint8, Uint8, Float32,
                  Uint8, Uint8, Float32, Uint32, Uint8, Float32, Uint8, Float32)


def _role_specs(config, layout, schedule, has_stats):
    routed = get_routed_role_config(config, schedule)
    # [SM-REUSE] Size of the DISPATCH green partition, which role 6 inherits.
    schedule_sms = int(getattr(config, 'dispatch_partition_sms', 0) or 0)
    shared = get_shared_role_config(schedule) if config.shared_experts else None
    # [ECO-SM90] role 5 = the ECO plan CTA, only when the buffer reserved it.
    eco = bool(getattr(config, 'eco_logical_experts', 0))
    base_roles = (1, 4, 2, 3) if config.shared_experts else (1, 2, 3)
    roles = (base_roles[:-1] + (5,) + base_roles[-1:]) if eco else base_roles
    # [SM-REUSE] Role 6 = compute helper that shares the DISPATCH green context
    # and parks until pull is done. Must be compiled/launched BEFORE role 3
    # (combine depends on every producer finishing).
    if getattr(config, 'sm_reuse', False):
        roles = roles[:-1] + (6,) + roles[-1:]
    specs = []
    for role in roles:
        if role == 6:
            # Same geometry as role 2 (384 threads) so `_tile_sync`'s
            # tid%32 / (tid//32)%4 / tid//128 decomposition is unchanged; the
            # CTA count is the dispatch grid (24), i.e. what those 6 SMs hold.
            # [SM-REUSE] Role 6 uses the SAME geometry as role 2 (tile_m and
            # therefore 199680 B at BS=256).  A smaller tile_m would let two
            # CTAs share an SM, but filling all 6 dispatch SMs was measured to
            # deadlock ("No available shared memory broadcast block found in
            # 60 seconds"): the context has no room left for the next call's
            # dispatch.  One CTA per SM is the hard ceiling here.
            _sched6 = routed.schedule
            k = SM90GCMoERoleKernel(replace(config, shared_experts=0),
                                    layout, 6, _sched6, has_stats,
                                    routed_n_factor=routed.n_factor)
            # Role 6 inherits the 6 SMs of the DISPATCH green context, and its
            # shared memory is the routed role's (199 KB at tile_m=64), so only
            # ONE CTA fits per SM: the grid must be 6, not dispatch's 24.
            # Using 24 would leave 18 CTAs unschedulable forever and the call
            # would hang.  `workers` is 1 at this geometry; at tile_m<=32
            # (99 KB) two fit, so scale by occupancy.
            # The dispatch context holds `dispatch_sms` SMs and its 24 CTAs
            # need ceil(24 / occ_dispatch) of them; whatever is left is free
            # once pull is done.  Role 6 then takes those SMs at ITS occupancy
            # (1 CTA/SM at 199 KB).  At this geometry: 6 - 5 = 1 SM -> 1 CTA.
            # [SM-REUSE] dispatch RETURNS after the last slot is pulled, so
            # its whole partition is free for role 6.  Fill every one of those
            # SMs at role 6's own occupancy (1 CTA/SM at 199 KB, 2 at 99 KB).
            # [SM-REUSE] One CTA per dispatch SM, NOT occupancy-many.
            # `_part * occ` (12 CTAs at tile_m=32) fills all 6 SMs completely,
            # and then the graph can no longer schedule the OTHER roles that
            # share this context -- observed as "No available shared memory
            # broadcast block found in 60 seconds", the same deadlock the
            # SM-SHARE experiment hit.  Keep a CTA per SM so each SM retains
            # room to host the next call's dispatch.
            _part = schedule_sms if schedule_sms else 6
            _grid6 = max(1, _part)
            specs.append((k, (6, _sched6.smem[1],
                              _sched6.threads[1], _grid6, ())))
            continue
        if role == 5:
            k = SM90GCMoERoleKernel(config, layout, 5, schedule, has_stats)
            # [ECO-WARP] thread count comes from the kernel (512 since the
            # warp split), not a literal.
            specs.append((k, (5, k.eco_smem, k.threads, 1, ())))
            continue
        index = 1 if role == 4 else role - 1
        # Workspace routed offsets are a strict prefix of the shared layout.
        role_config = replace(config, shared_experts=0) if role == 2 else config
        role_schedule = shared if role == 4 else (routed.schedule if role == 2 else schedule)
        kernel = SM90GCMoERoleKernel(role_config, layout, role, role_schedule, has_stats,
                                    routed_n_factor=routed.n_factor if role == 2 else 1)
        wait_for = (4,) if config.shared_experts and role == 2 and schedule.tile_m >= 32 else ()
        specs.append((kernel, (role, role_schedule.smem[index], role_schedule.threads[index],
                               role_schedule.grids[index], wait_for)))
    return tuple(specs)


def _compile_roles(specs, identity):
    pointers = tuple(make_ptr(dtype, 0, cute.AddressSpace.gmem, assumed_align=16)
                     for dtype in _POINTER_TYPES)
    binaries = []
    for kernel, launch in specs:
        role = launch[0]
        cubin = compile_kernel(
            kernel, (*pointers, Uint32(0), Uint32(0), Float32(0)), identity,
            lambda ptx, dump_dir: _assemble_role_ptx(ptx, role, dump_dir, identity))
        binaries.append(RoleBinary(role, cubin, *launch[1:]))
    return tuple(binaries)


def _binary_key(identity, specs):
    # Pointer compile arguments have fixed types. Kernel attributes and launch
    # resources cover every constexpr/layout field, without CUDA addresses.
    return (identity, tuple((tuple(sorted(vars(kernel).items())), launch)
                            for kernel, launch in specs))


def _get_or_compile_binaries(key, identity, specs):
    with COMPILE_LOCK:
        binaries = _BINARY_CACHE.get(key)
        if binaries is None:
            binaries = _compile_roles(specs, identity)
            expected = tuple(launch[0] for _, launch in specs)
            if (tuple(item.role for item in binaries) != expected):
                raise RuntimeError('SM90 compilation did not produce all role binaries')
            if _compiler_identity('sm_90a') != identity:
                raise RuntimeError('SM90 compiler/source identity changed while compiling')
            _BINARY_CACHE[key] = binaries
        return binaries



def prepare_role_binaries(config, y, w1, w2, buffer, stats=None, context=None,
                          shared_l1_weights=None, shared_l2_weights=None):
    if torch.cuda.get_device_capability(y.device) != (9, 0):
        raise ValueError('SM90 CuTe GC requires compute capability 9.0')
    if context is None:
        context = get_context()
    # A prepared entry holds immutable configuration and cubins, never live
    # CUDA addresses. Resident JIT code does not need filesystem scans on each
    # call. CuTe resolves its compiler environment at initialization; identity
    # is audited on preparation/compilation misses, not by rescanning os.environ.
    key = (config, buffer.layout, stats is not None, context.dispatch_sms,
           context.compute_sms)
    prepared = _CACHE.get(key)
    if prepared is None:
        with COMPILE_LOCK:
            prepared = _CACHE.get(key)
            if prepared is None:
                schedule = get_sm90_role_config(config, context)
                identity = _compiler_identity('sm_90a')
                specs = _role_specs(config, buffer.layout, schedule, stats is not None)
                binaries = _get_or_compile_binaries(_binary_key(identity, specs), identity, specs)
                prepared = _Prepared(binaries)
                _CACHE[key] = prepared
    addresses = [buffer.buffer.data_ptr(), buffer.peer_table.data_ptr(), y.data_ptr(),
                 *(t.data_ptr() for t in (*w1, *w2)), 0 if stats is None else stats.data_ptr(),
                 *(t.data_ptr() for pair in (shared_l1_weights, shared_l2_weights) for t in pair)
                 ] if config.shared_experts else [buffer.buffer.data_ptr(), buffer.peer_table.data_ptr(), y.data_ptr(),
                 *(t.data_ptr() for t in (*w1, *w2)), 0 if stats is None else stats.data_ptr(), 0, 0, 0, 0]
    arguments = [*(ctypes.c_void_p(p) for p in addresses),
                 ctypes.c_uint32(buffer.rank_idx), ctypes.c_uint32(config.tokens),
                 ctypes.c_float(config.clamp)]
    return prepared.binaries, arguments


def launch_sm90_gc_moe(config, y, w1, w2, buffer, stats,
                       shared_l1_weights=None, shared_l2_weights=None):
    context = get_context()
    binaries, arguments = prepare_role_binaries(
        config, y, w1, w2, buffer, stats, context,
        shared_l1_weights, shared_l2_weights)
    context.launch(binaries, arguments)
