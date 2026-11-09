"""SM90 role geometry and shared-memory budgets for resident compute CTAs."""
from dataclasses import dataclass, replace
from functools import lru_cache


@dataclass(frozen=True)
class SM90RoleConfig:
    grids: tuple[int, int, int]
    tile_m: int = 32
    threads: tuple[int, int, int] = (256, 256, 256)
    # Dispatch allocates a dynamic shared uint32 counter; the graph must reserve it.
    smem: tuple[int, int, int] = (128, 65792, 0)


@lru_cache(maxsize=256)
def _get_sm90_role_config(config, dispatch_sms: int, compute_sms: int):
    small_pipelined = 0 < config.bound <= 2
    pipelined = small_pipelined or config.bound == 0 or config.bound >= 32
    # Common public bound controls geometry only. Every received row is still
    # processed by the kernel's complete M-chunk loop, including skewed loads.
    expected_rows = (config.bound or config.capacity) * config.ranks * config.topk / config.experts
    if pipelined:
        tile_m = 8 if small_pipelined else (16 if expected_rows <= 12 else (32 if expected_rows <= 24 else 64))
    else:
        tile_m = 8 if config.bound <= 16 else 16
    workers = (1 if tile_m == 64 else 2) if pipelined else (3 if tile_m == 8 else 2)
    if pipelined:
        stage_bytes = (17408 + tile_m * 132 + 1023) // 1024 * 1024
        stages = 5 if tile_m == 64 else (3 if tile_m == 32 else 4)
        compute_smem = stages * stage_bytes + 1024 + tile_m * 1024 + (0 if tile_m == 64 else 256)
    else:
        compute_smem = 53376
    direct_scan = 0 < config.bound <= 2
    combine_bound = config.bound or config.capacity
    combine_grid = min(compute_sms * 4, max(1, (combine_bound * config.hidden + 2047) // 2048))
    # [ECO-SM90] The ECO dispatch keeps the same grid as the plain path: its
    # blocks grid-stride over the local slots and a replica slot simply waits
    # for the plan before pulling its quota. Growing the grid is NOT safe here
    # -- the dispatch role has a whole-grid rendezvous, so every block must be
    # resident, which caps the grid at 4 CTAs per communication SM.
    dispatch_grid = config.experts // config.ranks if direct_scan else dispatch_sms * 4
    # [ECO-SM90] The plan role (5) must be RESIDENT while the call runs: the
    # dispatch role blocks on it before it can pull a replica's quota. The
    # routed role is persistent and holds every SM of the compute partition for
    # the whole call, so with ECO the routed grid gives up one SM -- the same
    # trade SM100 makes ("compute launches one cluster fewer ... leaves the
    # plan its own SM"). Measured on this H20: a routed CTA is 99 KB (small
    # batch) to 199 KB (tile_m=64) of shared memory, and a leftover *half* SM
    # is not enough -- the plan CTA is only scheduled on a completely free SM
    # (2 CTAs per SM co-residency starved it), hence one whole SM.
    routed_grid = compute_sms * workers
    if getattr(config, 'eco_logical_experts', 0):
        # [ECO-SM142] The routed role is PERSISTENT: `workers` CTAs per SM
        # co-reside and grid-stride over the M blocks, so `grid == CTAs`, and
        # CTAs per SM == workers. The old value `compute_sms - 1` silently
        # dropped `workers`, so at bound=128 (tile_m=32, workers=2) the
        # compute partition ran 71 CTAs on 36 SMs and left half of it idle
        # (native arm: 144). Keep one whole SM for the plan CTA -- it is only
        # schedulable on a completely free SM -- and fill the rest with
        # `workers` CTAs each.
        routed_grid = max(1, (compute_sms - 1) * workers)
    # [ECO-PULL] The dispatch role stages each pulled token through shared
    # memory so it can use TMA (`cp.async.bulk`) instead of a 16-byte
    # `ld.global.v4`/`st.global.v4` pair. One token row is `hidden` bytes (fp8)
    # plus its scales; one staging slot per warp with a double buffer covers
    # `2 * warps` rows so a warp can have the next pull in flight.
    # Falls back to the tiny barrier-only allocation when the window does not
    # fit (the plain path never uses it).
    _pull_row_bytes = ((config.hidden + (config.hidden // 128) * 4 + 15) // 16) * 16
    _pull_warps = (256 if not direct_scan else 128) // 32
    dispatch_smem = max(128, ((2 * config.ranks + 2) * 4 + 127) // 128 * 128)
    # ONE buffer per warp, like SM100 (no double buffering there).
    _want = _pull_warps * _pull_row_bytes + 256
    if dispatch_smem + _want <= 200 * 1024:
        # Round UP to 1024: the dispatch path hands the whole window to
        # `SmemAllocator.allocate(size, 1024)`, and that alignment means a size
        # like 84864 asks for 84992 bytes -- 128 B past the real limit. The
        # barrier block at the top of the window then lands in the overflow and
        # the TMA's arrive/expect never becomes visible (observed as a hang).
        dispatch_smem = ((dispatch_smem + _want + 1023) // 1024) * 1024

    return SM90RoleConfig((dispatch_grid,
                           routed_grid, combine_grid),
                          tile_m=tile_m,
                          threads=(128 if direct_scan else 256, 384 if pipelined else 256, 384),
                          smem=(dispatch_smem, compute_smem, 0))


def get_sm90_role_config(config, context):
    return _get_sm90_role_config(config, context.dispatch_sms, context.compute_sms)


@dataclass(frozen=True)
class RoutedRoleConfig:
    schedule: SM90RoleConfig
    n_factor: int


@lru_cache(maxsize=256)
def get_routed_role_config(config, schedule):
    if (config.shared_experts > 0 and schedule.threads[1] == 256
            and schedule.tile_m == 8 and config.hidden % 512 == 0
            and config.intermediate % 256 == 0):
        return RoutedRoleConfig(
            replace(schedule, smem=(schedule.smem[0], 73856, schedule.smem[2])), 2)
    return RoutedRoleConfig(schedule, 1)


@lru_cache(maxsize=256)
def get_shared_role_config(schedule):
    smem = schedule.smem
    if schedule.threads[1] != 384:
        stage_stride = (36864 + schedule.tile_m * 144 + 1023) // 1024 * 1024
        shared_smem = (2 * stage_stride + 128 + 1023) // 1024 * 1024
        smem = (smem[0], shared_smem, smem[2])
    return replace(schedule, threads=(schedule.threads[0], 256, schedule.threads[2]),
                   smem=smem)


def get_reuse_role_config(config, schedule):
    """[SM-REUSE] Geometry for role 6 (the helper compute role).

    Role 6 runs the same math as role 2 but on the 6 SMs dispatch vacates, so
    it wants the SMALLEST per-CTA shared memory that still fits the routed
    kernel: `tile_m=32` needs 101632 B (occupancy 2/SM) versus 199680 B at
    tile_m=64 (occupancy 1), i.e. 12 CTAs on 6 SMs instead of 6.

    Correctness: `tile_m` only controls how one claimed task is split into M
    chunks (`for start in range(0, count, max_m)`); the chunks write disjoint
    row ranges of the same expert and `l1_done` is bumped once per TASK, not
    per chunk, so the L2 dependency count (`intermediate // 128`) is unchanged.
    """
    from dataclasses import replace as _replace
    from gc_moe.jit_kernels.heuristics.sm90_gc_moe import SM90RoleConfig
    tile_m = 32
    stage_bytes = (17408 + tile_m * 132 + 1023) // 1024 * 1024
    stages = 3
    smem = stages * stage_bytes + 1024 + tile_m * 1024 + 256
    return _replace(schedule,
                    tile_m=tile_m,
                    smem=(schedule.smem[0], smem, schedule.smem[2]))
