# SM100 V4.1 DeepInfer RR decode results

Two independent cold starts at source commit
`6fd3e35954328769e0b864e3099198c561e02af2`, using the existing checkpoint,
vLLM venv, TP8/EP8, EPLB initial placement, and the same four-token prompt.

- `deep_gemm_mega_moe/`: `--backend deep_gemm_mega_moe --eplb-decode --deepinfer-rr`
- `gc_moe/`: `--backend gc_moe --eplb-decode --deepinfer-rr`

Before each startup, with eight GPUs idle and the group lock held,
`scripts/sm100/check_v41_rr.py` passed with exit `0`. The launcher waited for
an unrelated GPU allocation to finish before taking the lock; it did not
interrupt that allocation.

Results:

- Both arms returned `rr_calls = [160, 160, 160, 160, 160, 160, 160, 160]`
  for the eight ranks, i.e. 40 layers times four decode steps.
- Both arms generated `[75, 200, 18, 200]`; first ID difference: none.
- All 40 per-layer placement hashes equal each other and all 40 hashes from
  `v41_eplb_decode_results`; first hash difference: none.
- Both model exits and launcher exits are `0`; both loaded 48/48 shards.
- No `ERROR`, `Traceback`, or `RuntimeError` lines were found.
- cgroup OOM counters remained `0 / 0 / 0`; GPUs were released after both arms.
- System journal access is `PERMISSION_LIMITED` for both arms.

| arm | model start | model end / release | rr_calls | IDs |
|---|---|---:|---|---|
| `deep_gemm_mega_moe` | `2026-11-22T00:08:25+08:00` | `2026-11-22T00:11:05+08:00` | `[160] x 8` | `[75,200,18,200]` |
| `gc_moe` | `2026-11-22T00:11:53+08:00` | `2026-11-22T00:15:13+08:00` | `[160] x 8` | `[75,200,18,200]` |

Worker PIDs:

| rank | MegaMoE | GCMoE |
|---:|---:|---:|
| 0 | `2611416` | `2938922` |
| 1 | `2611417` | `2938923` |
| 2 | `2611421` | `2938924` |
| 3 | `2611422` | `2938925` |
| 4 | `2611424` | `2938926` |
| 5 | `2611426` | `2938927` |
| 6 | `2611427` | `2938928` |
| 7 | `2611428` | `2938929` |

The GCMoE adapter installed in all eight distinct worker PIDs; the adapter PID
set exactly equals the worker PID set.

Top-level artifacts include `arm-summaries.json`, `summary.json`,
`replica-comparison.json`, the 40-layer table in
`replica-comparison.txt`, the per-rank RR counts in `rr-calls.json`, the decode
IDs in `decode-ids.json`, and the pass assertions in `layout-check.txt`.
Each arm directory contains its full model log, RR preflight, versions,
memory/cgroup snapshots, GPU release evidence, worker PIDs, and arm summary.
`SHA256SUMS` covers every other file in this directory tree. No performance
numbers were collected for this round.
