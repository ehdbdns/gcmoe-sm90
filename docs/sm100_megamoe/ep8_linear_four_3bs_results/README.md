# SM100 EP8 linear four-arm, BS256 only: blocked before GPU acquisition

Executor: **SM100**. This directory contains the retained preflight and resource-blocker
evidence for `docs/sm100_megamoe/SM100_EP8_LINEAR_FOUR_3BS.md`.

The commander limited this attempt to one batch size and then requested an immediate
receipt. BS256 was selected. BS512 and BS1024 were not attempted. The run was stopped
before any arm acquired the eight-GPU group because four GPUs remained occupied by
external `rtp_llm_rank-0..3` processes. No model was loaded and no measurement row was
produced.

## State

- Task SHA256: `7f0d9281edf818eef1e5c52e8994166f2dd554a5b410632195ea82c3ab79abf1`
- Task commit: `55ef4ff27176c9c839d2feaa44348e2055ceb2fa`
- Execution commit: `d16bf90047024504d0c974a70dc2da13b6d3102c`
- Fixed upstream: `fb8b65fa829c97962eb2dbc9a4ad8b759b9cd736`
- Requested scope: BS256, arms `vanilla`, `random`, `rr`, `eco`
- Arms started: 0
- Measurement rows: 0
- Task GPU acquisition: no
- Task model start: no
- Queue stopped: `2026-11-25T18:07:42+08:00`
- Task workers after stop: none

## Why no performance table

The task requires all eight site-authorized GPUs to be free. At preflight, GPUs 4-7
were each using about 262 GiB and reporting 100% utilization from four external
`rtp_llm` processes started at `16:37:34`; the task explicitly says to wait when any
device is occupied and not to disturb another user's job. The four remaining GPUs were
insufficient, so no arm could begin.

Consequently `main.csv`, `layers.csv`, `pairs.csv`, `auction.csv`, route banks, KV
payloads, and model logs do not exist. Their absence is intentional and must not be
interpreted as zero-cost results. No performance or fairness conclusion is made.

## Retained evidence

- `preflight.log`: task/model/input/patch hashes and startup checks.
- `prompts.sha256`: exact three-cohort prompt hashes, including the repaired SM90-compatible selector output.
- `upstream.diff`: exact upstream patch stack used by the attempted run.
- `run_ep8_onebs.sh`: launcher restricted to the selected BS.
- `versions.txt`: Python package environment.
- `gpu-occupancy.csv`, `gpu-compute-apps.csv`: pre-stop GPU blocker.
- `gpu-final-check.txt`: final occupancy and zero-task-worker check.
- `queue-stop.log`: queue and launcher shutdown evidence.
- `launcher.log`: launcher output.
- `SHA256SUMS`: checksums for all delivered files.

The external jobs were not terminated. No token, model weight, or shared environment was
modified or deleted.
