# SM100 V41 upstream GCMoE vs vendored MegaMoE (BS1024)

Execution role: **SM100 only**. No SM90 task book was run.

## Task and revisions

- Task: `docs/sm100_megamoe/V41_UPSTREAM_NATIVE_PAIR.md`
- Task SHA256: `987acc7a099d446ccc98ad29c02ffff6c8e236823612a9da99f48a07e73c0971`
- Executed repository revision: `ba462195bac57a1241a13f45033d07c60979baff`
- Those four repair commits were later rebased onto current `main` as
  `ae82e37`, `10ecef5`, `1f17204`, and `89b8577`.
- Task branch base: `8f52fb0`
- Upstream GCMoE: `https://github.com/Bruce-Lee-LY/gc-moe.git`
- Upstream revision: `7902831ec4a6691b6268b308bcff5ec0cb15ec86`
- Upstream worktree status at run time: clean

## Configuration

- Model: `/data0/home/admin/workspace/megamoe/native-v41-KO1hTbTj/model`
- DeepSeek V4.1 Flash, TP8 / EP8, native expert layout, fused shared expert.
- 40 real expert layers per sample, top-6 routing, 48 local experts per rank.
- Input: the previous BS1024 fixture set at
  `/data0/home/admin/workspace/megamoe/v41-native-pair-bs1024-01/step{0,8,16}-rank{0..7}.pt`.
- The 24 fixtures were SHA256-verified before the run; see `fixture-check.txt`
  and `measurements.sha256`.
- Mega backend: vLLM vendored `vllm.third_party.deep_gemm.mega`.
- GC backend: pinned upstream `gc_moe.gc`, package version 0.6.2.
- Both arms used the same model, inputs, physical GPUs, launcher, and timing path.
- EPLB, ECO, RR reordering, and additional tracing were disabled.
- Timing: 3 eager forwards, one 40-layer graph capture, 10 warmup replays,
  15 CUDA-event groups of 20 replays. Each group uses the slowest rank; the
  reported value is the median of those 15 group times.

The task's output-buffer reuse path was used for both arms, so eager warmup and
parent-graph capture address the same layer output. Eager references are cloned
before replay comparison. No device-to-device copy was added to the timed graph.

## Timing

Positive delta means GC is slower than Mega.

| Step | Mega (ms) | GC (ms) | 100 x (GC/Mega - 1) | Strict |
|---:|---:|---:|---:|:---:|
| 0 | 7.887888 | 7.895562 | +0.0973% | no |
| 8 | 7.852768 | 7.827145 | -0.3263% | no |
| 16 | 7.899139 | 7.869527 | -0.3749% | no |

No step qualifies for the strict performance judgement because GC differs from
the live Mega reference at every step. The timings are descriptive observations,
not a strict performance result.

### Slowest-rank group distribution

| Arm | Step | Min (ms) | Median (ms) | Max (ms) | Spread |
|---|---:|---:|---:|---:|---:|
| Mega | 0 | 7.885387 | 7.887888 | 7.891906 | 0.0827% |
| Mega | 8 | 7.850117 | 7.852768 | 7.857034 | 0.0881% |
| Mega | 16 | 7.896449 | 7.899139 | 7.901536 | 0.0644% |
| GC | 0 | 7.889677 | 7.895562 | 7.912362 | 0.2875% |
| GC | 8 | 7.822422 | 7.827145 | 7.836041 | 0.1741% |
| GC | 16 | 7.861742 | 7.869527 | 7.878930 | 0.2186% |

All 90 slowest-rank group values are in `groups.csv`.

## Correctness

- Eager/replay consistency: all 960 layer outputs per arm
  (8 ranks x 3 steps x 40 layers) passed `torch.equal`; all outputs were finite.
- Mega graph versus live Mega reference: `0 / 0 / 0` differing elements at
  steps 0, 8, and 16.
- GC graph versus live Mega reference:

| Step | Different elements / total | Max abs | Per-rank counts | Per-rank max abs |
|---:|---:|---:|---|---|
| 0 | 21 / 209,715,200 | 2.0 | `[1,0,1,1,9,4,4,1]` | `[2.98e-8,0,1.49e-8,4.88e-4,1,2,2,3.73e-9]` |
| 8 | 4 / 209,715,200 | 0.015625 | `[0,0,1,0,0,0,0,3]` | `[0,0,6.10e-5,0,0,0,0,1.5625e-2]` |
| 16 | 11 / 209,715,200 | 8.0 | `[0,2,0,9,0,0,0,0]` | `[0,4.88e-4,0,8,0,0,0,0]` |

GC differs in 36 / 629,145,600 elements overall (5.7220e-6 percent).
The number of differing layers is 12 at step 0, 4 at step 8, and 4 at step 16.
These GC differences exactly match the previous vendored-GC `V41_NATIVE_PAIR`
result, despite this run using the pinned upstream GC package.

Full per-rank details are in `output_diff_by_rank.csv`; all 960 per-layer records
are in `output_diff_by_layer.csv`; raw per-rank outputs metadata is in
`mega.json` and `gc.json`.

## Backends and SM partition

- Mega: 8/8 ranks reported `vllm.third_party.deep_gemm.mega` from the shared
  venv, capability `[10,3]`, package version 0.6.2.
- GC: 8/8 ranks reported `gc_moe.gc` from the pinned upstream checkout,
  capability `[10,3]`, package version 0.6.2.
- GC on every rank: `dispatch_sms=12`, `compute_sms=136`, `primary=False`.
- Exact import paths are in `imports.txt` and `backends.txt`.

## Compiler and common environment

- `ptxas`: CUDA 13.5, V13.5.37, build `cuda_13.5.r13.5/compiler.39125733_0`.
- `nvcc`: CUDA 13.0, V13.0.48, build `cuda_13.0.r13.0/compiler.36260728_0`.
- Driver: 580.105.08.
- Common environment: vLLM 0.31.0, Torch 2.13.0+cu130, FlashInfer
  0.7.0.post1, NCCL 2.29.7, Triton 3.7.1.
- The task-specific private environment used CuTe DSL 4.8.1; the previous shared
  environment had 4.7.1. The DSL wheel set was identical for both arms.
- Difference from the previous pair run: GC was replaced by the pinned clean
  upstream checkout at `7902831`; the private interpreter used DSL 4.8.1. The
  Mega vendored backend, model, fixtures, GPUs, and timing path were unchanged.

## Repair stack

The task-supplied output reuse implementation initially failed four launch
attempts on Mega with `AssertionError: expert output allocation changed`.
All failed logs and exit codes are retained:

- `launcher-attempt0.log`, `mega-attempt0.log`, `mega-attempt0.exit-code=1`
- `launcher-attempt1.log`, `mega-attempt1.log`, `mega-attempt1.exit-code=1`
- `launcher-attempt2.log`, `mega-attempt2.log`, `mega-attempt2.exit-code=1`
- `launcher-attempt3.log`, `mega-attempt3.log`, `mega-attempt3.exit-code=1`

Attempt 3 duplicated the attempt 2 log and PID/timestamp evidence. The final
successful repair stack is:

- `ae82e37` (executed as `69de893`): match output reuse by source allocation.
- `10ecef5` (executed as `c3718f6`): wrap the expert `torch.empty_like`
  allocation directly.
- `1f17204` (executed as `cc9343f`): match the reused allocation by shape,
  dtype, device, and layout.
- `89b8577` (executed as `ba46219`): add environment-gated reuse diagnostics.

The complete final diff against the task branch base is in
`output_reuse_fix.diff`. It changes only the allocation-reuse adapter. No model,
kernel, runtime arithmetic, input, reference, or timing logic was changed.

## GPU release and cleanup

- Measurements and lock acquisitions/releases are recorded in `acquired.log`
  and `released.log`.
- Final arm exit codes: Mega 0, GC 0.
- Final GPU release: `2026-11-22T16:37:10+08:00`.
- Post-release check at `2026-11-22T08:38:21Z`: no compute applications and no
  holders on `/dev/nvidia0` through `/dev/nvidia7`; see
  `gpu_release_after_run.txt`.
- The run lock was acquired before the successful measurement and released
  afterward; see `lock-release.txt`.
- Removed after the run: private `dsl-env` (498 MiB), `upstream` (21 MiB),
  `dg-cache` (1.2 MiB), `wheels` (180 MiB), and 24 fixture symlinks.
- Retained: the shared model/venv and all 24 original measurement files at
  `/data0/home/admin/workspace/megamoe/v41-native-pair-bs1024-01/`.
- Cleanup details are in `cleanup.txt`.

## Evidence index

`comparison.json`, `mega.json`, `gc.json`, both success logs, all failed
attempt logs and exit codes, `groups.csv`, `output_diff_by_rank.csv`,
`output_diff_by_layer.csv`, `upstream.sha`, `upstream-inspect.txt`,
`imports.txt`, `fixture-check.txt`, `compiler.txt`, `versions.txt`,
`weights.sha256`, `gpus.txt`, `acquired.log`, `released.log`,
`gpu_release_after_run.txt`, `lock-release.txt`, `cleanup.txt`,
`environment.txt`, `backends.txt`, `task.md`, `task.sha256`,
`measurements.sha256`, the DSL provenance logs, the repair diff, and
`SHA256SUMS`.
