# SM100 V4.1 DeepInfer layer-placement results

Two independent cold starts at source commit
`fe53bb7d5e4a7d86e61e03cfb240ecf6ac981135`. Both arms used the same
prebuilt plan, the existing checkpoint, vLLM venv, CUDA 13.5 ptxas, TP8/EP8,
and the fixed four-token prompt.

- `deep_gemm_mega_moe/`: `--backend deep_gemm_mega_moe --eplb-decode --deepinfer-rr --placement-file .../plan.npz`
- `gc_moe/`: `--backend gc_moe --eplb-decode --deepinfer-rr --placement-file .../plan.npz`

The plan was built on CPU from the archived SM90 BS512 logical-demand fixture
(`input.sha256`) and has SHA-256
`4048ec19e4051ca7df85ccc3efe5c8d9b9e4eabe6d09e4b3d8305a2b6a6faf81`.
Both arms report the same plan SHA. The launcher waited for an unrelated GPU
allocation to finish before taking the group lock.

Results:

- Every layer moved: `17,873` changed physical slots total; per-layer counts
  are in `changed-slots.json` and `placement-comparison.txt`.
- Packed-weight sampling found `0` mismatching slots in both arms. Each layer
  component sampled 32 bytes per slot; this is a sampled check, not a full
  byte-for-byte audit.
- Both arms returned `[160, 160, 160, 160, 160, 160, 160, 160]` RR calls
  (40 layers times four decode steps).
- Both arms produced 40 unique final layer hashes and the two arms agree on all
  40. No layer hash equals the original placement from the preceding RR round.
- Both arms generated `[75, 200, 18, 200]`, equal to the pre-migration baseline;
  no first ID difference.
- Preflight, model, and launcher exits are `0`; both loaded 48/48 shards.
- No `ERROR`, `Traceback`, or `RuntimeError` lines; cgroup OOM counters stayed
  `0 / 0 / 0`.
- All 16 SM100 worker PIDs from both arms are absent in the final release
  recheck. A separate external `rtp_llm` job occupied the GPUs after this run;
  it is not part of these results.

| arm | model start | model end / release | changed slots | sampled mismatches | IDs |
|---|---|---:|---:|---:|---|
| `deep_gemm_mega_moe` | `2026-11-22T02:24:30+08:00` | `2026-11-22T02:27:50+08:00` | `17,873` | `0` | `[75,200,18,200]` |
| `gc_moe` | `2026-11-22T02:28:44+08:00` | `2026-11-22T02:34:05+08:00` | `17,873` | `0` | `[75,200,18,200]` |

Top-level artifacts include `summary.json`, `plan.npz`, `plan.sha256`,
`arm-summaries.json`, `placement-comparison.json`, `placement-comparison.txt`,
`changed-slots.json`, `rr-calls.json`, `decode-ids.json`, `layout-check.txt`,
`gpu-release-recheck.txt`, and `SHA256SUMS`. Each arm directory contains its
full model log, RR preflight, versions, worker PIDs, cgroup summaries, and
release evidence. No performance numbers were collected.
