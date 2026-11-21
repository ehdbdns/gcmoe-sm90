# SM100 V4.1 native loading retry results

This directory contains only SM100 execution evidence. No SM90 task was run and no SM90 result is mixed in.

Result: **FAIL**, exit code `1`. The model loaded all 48 checkpoint shards and resolved the native MoE backend, but `VllmWorker-0` died before `LLM.generate()`. Therefore no four-token `generated_ids` exist.

Start with `status.txt`, `timeline.txt`, `first_error.txt`, `external_termination.txt`, `backend_evidence.txt`, and `cgroup-evidence.txt`. The raw logs are `model-full.log` and `model.log`.

The local XFS weight copy was made because the host had sufficient space and the task requested local reads when possible. Model weights and compiled caches are not committed.
