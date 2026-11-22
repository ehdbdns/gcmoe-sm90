# SM100 V41 native Mega vs native GC (BS1024)

Task: `docs/sm100_megamoe/V41_NATIVE_PAIR.md`
Task SHA256: `41826b5bb360e94696c2af17a93812e2b05f123d3d128c4dfb79601f2964471b`
Executed revision: `e477f5a752aeedcf29e759f7b330b92bbda1643c`
GC capture fix: `3e24814b272aaef0cc4c84c14ca1b2a48d2e9961`

## Configuration

- Model: `/data0/home/admin/workspace/megamoe/native-v41-KO1hTbTj/model`
- TP8 / EP8, 40 real expert layers per sample, top-6, 48 local experts plus fused shared expert.
- Identical real BS1024 decode inputs from `docs/trace_executor_20261119/sm90_main24_stable_decode_bs256_results/delivery/bs256/vanilla/rank0.npz`.
- Input archive SHA256: `ceaa52dc3f3b09fa95e08c59b83b56b97d90b8335b3437a506cb5dbd42fe09f2`.
- Each sample captures all 40 layers in one graph, replays 20 times per group, uses 15 CUDA-event groups, takes the slowest rank per group, then the median of those 15 group values.
- Fixture SHA equality: 24/24 step-rank pairs match between Mega and GC.

## Timing

| Step | Mega (ms) | GC (ms) | GC vs Mega | Strict? |
|---:|---:|---:|---:|:---:|
| 0 | 7.887664 | 8.362762 | +6.0233% | no |
| 8 | 7.852147 | 8.279971 | +5.4485% | no |
| 16 | 7.898392 | 8.316273 | +5.2907% | no |

No sample qualifies for a strict performance judgement: GC differs from the live Mega reference in every sample. The timing values are descriptive observations only.

### Slowest-rank group distribution (15 groups)

The full per-group values are in `groups.csv`; the table below summarizes them.

| Arm | Step | Min (ms) | Median (ms) | Max (ms) | Spread |
|---|---:|---:|---:|---:|---:|
| Mega | 0 | 7.883768 | 7.887664 | 7.903784 | 0.2539% |
| Mega | 8 | 7.849867 | 7.852147 | 7.854722 | 0.0618% |
| Mega | 16 | 7.895248 | 7.898392 | 7.903971 | 0.1105% |
| GC | 0 | 8.358704 | 8.362762 | 8.369374 | 0.1276% |
| GC | 8 | 8.271320 | 8.279971 | 8.286448 | 0.1829% |
| GC | 16 | 8.311121 | 8.316273 | 8.323008 | 0.1430% |

## Correctness

- Eager/replay consistency: 960/960 layer outputs per arm (8 ranks x 3 steps x 40 layers) are bitwise equal (`torch.equal`).
- Mega graph versus live Mega reference: 0 differing elements at all three steps.
- GC graph versus live Mega reference:

| Step | Different elements / total | Max abs | Per-rank different counts |
|---:|---:|---:|---|
| 0 | 21 / 209,715,200 | 2.0 | `[1,0,1,1,9,4,4,1]` |
| 8 | 4 / 209,715,200 | 0.015625 | `[0,0,1,0,0,0,0,3]` |
| 16 | 11 / 209,715,200 | 8.0 | `[0,2,0,9,0,0,0,0]` |

Total GC differences: 36 / 629,145,600 elements (5.72e-8 fraction, 5.72e-6 percent). All per-rank and per-layer counts and max-absolute errors are retained in `gc.json`, `output_diff_by_rank.csv`, and `output_diff_by_layer.csv`.

## Backends and exits

- Mega: all 8 ranks report `vllm.third_party.deep_gemm.mega`; exit 0.
- GC: all 8 ranks report `gc_moe.gc`; corrected retry exit 0.
- First GC attempt exit 1 at pair-time graph capture. It produced no timing sample. Full first stack is in `gc-attempt0.log`.
- `gc-launcher.exit-code` is the original launcher exit (1); `gc-retry1.exit-code` is the canonical measurement exit (0).

## Capture fix

The first GC attempt failed inside `torch.cuda.graph` with `CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED`: `_ChildGraph.__init__` instantiated the child graph inside the active parent capture. Commit `3e24814` removes that eager instantiation; ordinary launches still instantiate before `cuGraphLaunch`, while captured launches let the parent graph own instantiation. The full diff is `gc_capture_fix.diff`. No computation or layout semantics were changed.

## Retained evidence

- 24 measurement/reference files: `/data0/home/admin/workspace/megamoe/v41-native-pair-bs1024-01/step{0,8,16}-rank{0..7}.pt`
- Their SHA256 values: `measurements.sha256`
- Input archive: `docs/trace_executor_20261119/sm90_main24_stable_decode_bs256_results/delivery/bs256/vanilla/rank0.npz`; a persistent 2.7 MiB copy is retained at `/data0/home/admin/workspace/megamoe/v41-native-pair-bs1024-01/input-rank0.npz`, SHA256 `ceaa52dc3f3b09fa95e08c59b83b56b97d90b8335b3437a506cb5dbd42fe09f2`.
- The 24 `.pt` fixtures are 105,272,703 bytes each, above GitHub's 100 MiB per-file limit. They remain at the path above with hashes pushed; they are not copied into Git.
- Private `dg-cache` was removed after the run; see `cleanup.txt`.
- GPU release after retry: `gpu_release_after_retry.txt` (no compute apps, no device holders).
