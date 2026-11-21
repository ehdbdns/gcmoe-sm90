# SM100 V41 GCMoE JIT retry results

This directory contains two cold-start attempts for task
`docs/sm100_megamoe/V41_GC_JIT_RETRY.md` at source commit
`82987eb0a1da70e0c3b3cbb6493ec6c501fe6a08`.

- `attempt1-system-ptxas-13.0/`: the task's original command path. The CuTe DSL
  4.7.1 cache-key failure is gone, but the default system assembler is CUDA
  13.0 and cannot assemble the generated PTX 9.1 for `sm_103f`. Exit code 1.
- `attempt2-venv-ptxas-13.5/`: the same task path plus a one-line optional
  `GC_MOE_PTXAS` selector, pointed at the CUDA 13.5 assembler already installed
  in the verified vLLM venv. Exit code 0; all four prefix audits and the 8x40
  layer introspection completed.

The source fix is recorded in `local-fix.diff`; its default is still
`/usr/local/cuda/bin/ptxas`, so default behavior is unchanged. No model,
weight, sampling, vLLM, or harness semantics were changed.

Key results in `attempt2-venv-ptxas-13.5/`:

- `prefix-comparison.json` / `prefix-comparison.txt`: full GC vs Mega top10.
- `current-prefix-top10.json`: raw GC top10 output.
- `inspect-v41-moe.json` / `introspection-summary.json`: 8 workers x 40 layers.
- `cgroup-summary.txt`, `gpu-release-recheck.txt`, `journal-access.txt`.
- `model-full.log`, `model.log`, launcher, versions, preflight, and exit codes.

Top1 IDs are identical to the prior Mega reference: `75, 300, 200, 19`.
Every top10 entry and logprob also compares exactly equal in
`prefix-comparison.json`.

`SHA256SUMS` covers every other file in this directory tree.
