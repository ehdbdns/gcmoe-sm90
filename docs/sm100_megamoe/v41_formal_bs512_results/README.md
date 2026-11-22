# SM100 V41 formal BS512 - stopped by commander

Execution role: **SM100 only**. No SM90 task book was run.

## Status

This run was stopped at the commander's request because the formal fairness
workload was too slow. The requested replacement is to remove fairness
verification and run the five arms directly, but no replacement run was started
in this turn because the instruction was to stop the experiment and release the
GPUs first.

- Task: `docs/sm100_megamoe/V41_FORMAL_BS512.md`
- Task SHA256: `c11cbf86e95fc57af6bb78b737c958d10ed9c140753253d380e4f4771afaef98`
- Executed repository SHA: `0eaeec98d885d6ef0d521473daede8fb0c1fd62d`
- Upstream GCMoE: `fb8b65fa829c97962eb2dbc9a4ad8b759b9cd736`
- Batch size: 512 requests per decode step, 40 expert layers, TP8/EP8.
- Per-arm formal plan: 1000 prefill/statistics steps plus 1000 measured steps.

## Progress at stop

- `mega` started at `2026-11-22T17:59:03+08:00` after the earlier failures were
  repaired. It completed all 1000 statistics steps and generated
  `bs512/plan.npz`.
- The 8 per-rank statistics count archives were written.
- The 8 KV checkpoint files for the statistics boundary were written and their
  per-rank JSON manifests report successful SHA256 values.
- The run had entered the measure boundary and was writing partial
  `kv-measure/rank*.bin` files when it was stopped.
- `mega/result.json` was never written. No valid timing exists for any arm.
- `gc`, `mega_rr`, `gc_rr`, and `gc_eco` were not started.
- No cross-arm native top-1, controlled-replay, or strict-pair result exists.

## Earlier attempts

1. The first launch failed after model initialization because the input archive
   was absent from the sparse worktree:
   `docs/trace_executor_20261119/sm90_main24_stable_decode_bs256_results/delivery/bs256/vanilla/rank0.npz`.
   The file is tracked in Git; adding its directory to sparse checkout fixed the
   path. Its SHA256 is
   `ceaa52dc3f3b09fa95e08c59b83b56b97d90b8335b3437a506cb5dbd42fe09f2`.
2. The second launch failed immediately because `natureal_matrix_audit.patch`
   had already been applied to the private upstream clone. The clone was
   restored to the pinned SHA and its copied untracked files were removed before
   the final attempt.

Both failed launcher and model logs are retained.

## Stop and release

- Stopped process group: `1027075` (pipeline, `run_five.sh`, `five_matrix.py`,
  and its eight VLLM workers). Only this run's process group was terminated.
- Final release record: `2026-11-22T18:21:21+08:00`.
- Post-stop check at `2026-11-22T18:21:58+08:00`: no compute applications and
  no holders on `/dev/nvidia0` through `/dev/nvidia7`.
- Evidence: `released.log`, `processes-stopped.txt`, and
  `gpu_release_after_stop.txt`.

## Cleanup

The private run filesystem was reduced from 263 GiB to 208 KiB. Removed items:

- `bs512/` including the statistics/measure KV binaries, route tensors, and
  partial measure checkpoints;
- `overlay/` (39 MiB);
- `dg-cache/` (1.5 MiB);
- private `dsl-env/` (498 MiB);
- private `upstream/` (18 MiB);
- `wheels/` and `eco-cache/` were absent or removed.

Small evidence was copied into this repository before cleanup. Details are in
`cleanup.txt`.

## Replacement-run direction

Per the commander's instruction, the next run should remove fairness
verification from the critical path and execute the five arms directly. In
concrete terms this means omitting:

- the shared KV backing-store write/restore and boundary comparison;
- cross-arm native top-1 / controlled-replay checks;
- the 10-pair strict comparison.

Each arm should still keep its own finite-output and eager/replay checks and
report complete MoE timing. This replacement workload was not executed in this
turn.

## Evidence index

- `task.md`, `task.sha256`, `revision`, `upstream.sha`
- `launcher.log`, `attempt0-input-missing/`, `attempt1-patch-reapply/`
- `mega/model.log`, `mega/start`, `mega/stats-rank*.npz`
- `plan.npz`, `stats-routes.sha256`, `kv-stats.sha256`, `measure-partial.txt`
- `kv-stats/rank*.json`
- `acquired.log`, `released.log`, `processes-stopped.txt`,
  `gpu_release_after_stop.txt`, `cleanup.txt`
- `versions.txt`, `gpus.txt`, `weights.sha256`, `ptxas.txt`, `SHA256SUMS`
