# SM100 direct BS512 evidence

Executor: **SM100**. This directory is the lightweight evidence bundle for
`docs/sm100_megamoe/V41_DIRECT_BS512.md` and the CPU phase analysis requested by
`docs/sm100_megamoe/V41_CPU_PHASES.md`.

## GPU run

- Task SHA256: `1d25419614efb385ee10bc00bf4f32ae34cbe8575039d3cbf2ee920d0820f024`
- Execution source: `f6808405a920faedc9d781d56dbe0fcc22267c48`
- Packaging base: `fe58cba2948a85a7e57cda4820a1f9622db304e7`
- Upstream GCMoE: `fb8b65fa829c97962eb2dbc9a4ad8b759b9cd736`
- Arms: `mega`, `gc`, `mega_rr`, `gc_rr`, `gc_eco`; each exit code is 0.
- Each arm ran 1000 statistics steps then 1000 measured steps at BS512, TP8/EP8,
  40 layers, 32 GiB KV per rank, and native greedy decode.
- `bs512/metrics.csv` is the five-row task table. `bs512/summary.json` has
  `failures: []` and deliberately has zero cross-arm strict pairs: the task
  specifies independent natural generation, not route/token replay.
- Placement: `bs512/plan.npz` SHA256
  `ddcb82d5151cb353b62095a935a0141d5bd3f59ad095e2418b5d88e09b2a0ecf`;
  all three EPLB arms used it.

## CPU run

- Task SHA256: `09ec0eef7738f8ecbd7a531ca6b7aebdeb8e144a688c13c4d1da17a128b3bc07`
- Source commit: `44c63e8ccbc1298a35e83d942b50484345fcde6e`
- Output: `docs/sm100_megamoe/v41_old_pair_phases.csv`, SHA256
  `1ec2016ddc1a953eb3330a85aebb3ab56e8d963d17b766ae16871bb4a79cf1ce`
- No CUDA device was visible and no model was launched. Validation details are in
  `cpu_phase_validation.txt`.

## Evidence map

- `delivery/`: launcher-generated light delivery, including per-arm start/end,
  exit codes, results, metrics, plan, versions, GPUs, and the 80-file rank-NPZ
  hash manifest.
- `logs/`: complete model stdout/stderr for each arm.
- `gpu_release.txt`, `processes-stopped.txt`, `lock-release.txt`: post-run process,
  device, and lock checks.
- `retained-measurements.txt`: retention path for the 80 measurement/statistics
  rank NPZ files. The large files are intentionally not committed to Git.
- `cleanup.txt`: private checkout, venv, overlay, and compile-cache cleanup.

The rank NPZ manifest is `delivery/rank-files.sha256` with paths relative to the
run directory. It was checked after the run and after cleanup with
`sha256sum --quiet -c delivery/rank-files.sha256`; result was `SHA_OK` (80/80).
