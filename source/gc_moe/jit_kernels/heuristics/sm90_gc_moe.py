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
    direct_route_scan: bool = False
    split_shared: bool = False
    shared_grid: int = 0
    shared_l2_grid: int = 0
    compute_sms: int = 0
    shared_tile_m: int = 32


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
    # [T4-BMSCAN] Force the M-block height so the block_m scan can be measured
    # on real hardware. SM90's shared-memory cap allows 16/32/64 only
    # (tile_m=64 already needs 199 KB of the 232 KB budget); an unsupported
    # value must fail loudly instead of silently falling back, otherwise the
    # scan would compare two identical builds.
    import os as _osb
    _bm = int(_osb.environ.get('GC_MOE_BLOCK_M_FORCE', '0') or 0)
    if _bm:
        if _bm not in (16, 32, 64):
            raise ValueError(f'GC_MOE_BLOCK_M_FORCE={_bm} unsupported on SM90 '
                             '(only 16/32/64 fit the shared-memory budget)')
        tile_m = _bm
    workers = (1 if tile_m == 64 else 2) if pipelined else (3 if tile_m == 8 else 2)
    import os as _osw
    if _osw.environ.get('GC_MOE_FORCE_WORKERS'):
        workers = int(_osw.environ['GC_MOE_FORCE_WORKERS'])
    # [T4-ECO-WORKERS] With ECO the plan role (5) must own a WHOLE SM: it is
    # only schedulable once an SM is completely free (2 co-resident CTAs already
    # starved it, see the note below). At tile_m=64 the compute partition runs
    # `workers=1`, so `(compute_sms - 1) * 1` CTAs leave exactly one SM idle and
    # the plan fits. At tile_m=32/16 `workers=2`, and `(compute_sms - 1) * 2`
    # CTAs pack 2 CTAs onto each of the first 71 SMs -- the 72nd SM is free but
    # the plan CTA still never gets scheduled, and the call DEADLOCKS (measured:
    # BS32 smoke hung in CUDA-graph capture at 34/51 buckets, GPU 100%, forever;
    # forcing workers=1 made the same build finish capture and run at
    # 13.29 ms/step). Pin workers to 1 when ECO is on: the routed role is
    # persistent and grid-strides over M blocks, so one CTA per SM still covers
    # all the work, it just runs a deeper per-CTA loop.
    if getattr(config, 'eco_logical_experts', 0) and workers > 1:
        workers = 1
    if pipelined:
        stage_bytes = (17408 + tile_m * 132 + 1023) // 1024 * 1024
        stages = 5 if tile_m == 64 else (3 if tile_m == 32 else 4)
        compute_smem = stages * stage_bytes + 1024 + tile_m * 1024 + (0 if tile_m == 64 else 256)
    else:
        compute_smem = 53376
    # The direct scanner consumes physical IDs and has no ECO plan handshake.
    direct_scan = small_pipelined and not getattr(config, 'eco_logical_experts', 0)
    combine_bound = config.bound or config.capacity
    combine_grid = min(compute_sms * 4, max(1, (combine_bound * config.hidden + 2047) // 2048))
    # [ECO-SM90] The ECO dispatch keeps the same grid as the plain path: its
    # blocks grid-stride over the local slots and a replica slot simply waits
    # for the plan before pulling its quota. Growing the grid is NOT safe here
    # -- the dispatch role has a whole-grid rendezvous, so every block must be
    # resident, which caps the grid at 4 CTAs per communication SM.
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
    if getattr(config, 'eco_logical_experts', 0) and not getattr(config, 'plan_dispatch', False):
        # [ECO-SM142] The routed role is PERSISTENT: `workers` CTAs per SM
        # co-reside and grid-stride over the M blocks, so `grid == CTAs`, and
        # CTAs per SM == workers. The old value `compute_sms - 1` silently
        # dropped `workers`, so at bound=128 (tile_m=32, workers=2) the
        # compute partition ran 71 CTAs on 36 SMs and left half of it idle
        # (native arm: 144). Keep one whole SM for the plan CTA -- it is only
        # schedulable on a completely free SM -- and fill the rest with
        # `workers` CTAs each.
        routed_grid = max(1, (compute_sms - 1) * workers)
    # _dispatch lays counts/prefix at the bottom, one staging row per warp
    # in the middle, and reserves the last 384 bytes for barriers and phases.
    row_bytes = ((config.hidden + (config.hidden // 128) * 4 + 15) // 16) * 16
    warps = (128 if direct_scan else 256) // 32
    counts_bytes = ((2 * config.ranks + 2) * 4 + 127) // 128 * 128
    tail_bytes = 256 if direct_scan else 384
    dispatch_smem = ((counts_bytes + warps * row_bytes + tail_bytes + 1023) // 1024) * 1024

    if direct_scan:
        dispatch_grid = (config.physical or config.experts) // config.ranks
    else:
        # All dispatch CTAs must reach the grid barrier. Large hidden widths
        # fit only three staging windows per SM90 SM (228 KiB; 1 KiB reserved per CTA).
        resident = min(4, (228 * 1024) // (dispatch_smem + 1024))
        if getattr(config, 'eco_logical_experts', 0) and getattr(config, 'plan_dispatch', False):
            # The graph admits dispatch only after plan is resident.
            dispatch_grid = (dispatch_sms - 1) * resident
        else:
            dispatch_grid = dispatch_sms * resident

    split_shared = bool(config.shared_experts and tile_m in (32, 64))
    shared_grid = 0
    shared_tile_m = 32
    shared_l2_grid = 0
    if split_shared:
        # Preserve upstream M32 scheduling unless whole M64 tiles can fill the
        # compute partition. L1 launches one task per (token block, output tile).
        blocks = (combine_bound + 31) // 32
        l1_tiles = config.intermediate * config.shared_experts // 128
        tasks = blocks * max(l1_tiles, config.hidden // 256)
        limit = 2 * compute_sms if tile_m == 64 else routed_grid
        if tile_m == 64:
            shared_l2_grid = max(1, compute_sms // 4)
        full_m64_tasks = config.tokens // 64 * l1_tiles
        if (config.split_shared_m == 64 and tile_m == 64 and config.tokens % 64 == 0
                and full_m64_tasks >= compute_sms):
            shared_tile_m = 64
            tasks = full_m64_tasks
            # M64 uses one resident CTA/SM (242 registers/thread on H20).
            # Keep L2's in-flight token rows at the M32 cap while routed runs.
            limit = compute_sms
            shared_l2_grid = max(1, shared_l2_grid // 2)
        shared_grid = min(limit, max(1, tasks))

    return SM90RoleConfig((dispatch_grid,
                           routed_grid, combine_grid),
                          tile_m=tile_m,
                          threads=(128 if direct_scan else 256, 384 if pipelined else 256, 384),
                          smem=(dispatch_smem, compute_smem, 0), direct_route_scan=direct_scan,
                          compute_sms=compute_sms, shared_grid=shared_grid, split_shared=split_shared,
                          shared_tile_m=shared_tile_m,
                          shared_l2_grid=shared_l2_grid)


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
    if schedule.split_shared:
        tile_m = schedule.shared_tile_m
        # The original M32 reservation also fits two 46,080-byte M64 stages.
        smem = (smem[0], 101632, smem[2])
    else:
        tile_m = schedule.tile_m
    return replace(schedule, tile_m=tile_m,
                   grids=(schedule.grids[0], schedule.shared_grid or schedule.grids[1], schedule.grids[2]),
                   threads=(schedule.threads[0], 256, schedule.threads[2]),
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
