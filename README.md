# GC MoE on SM90 — ECO (EPLB-placed Capped Online split)

**This repository follows and builds on [Bruce-Lee-LY/gc-moe](https://github.com/Bruce-Lee-LY/gc-moe).**
The `source/` tree is that project (MIT, © 2026 Bruce-Lee-LY) with an additional
SM90 implementation of the **ECO** online token-split plan; every change of ours is
marked with an `[ECO-SM90]` comment.

This is a self-contained experiment directory: pinned kernel source, the vLLM
backend, the CPU/GPU verification harnesses, and the raw result JSON of the runs
quoted below.

---

## What this is

GC MoE is a CuTe-DSL FP8/FP4 MoE kernel using CUDA Green Contexts to overlap
expert-parallel dispatch with expert compute. The upstream project implements
**ECO** (an EPLB placement + an in-kernel "how many tokens does each replica
take" plan) for **SM100 only**.

The port here brings that plan to **SM90 (Hopper / H20)** and makes it work
next to the SM90 role graph (dispatch / routed compute / shared expert /
finalize), which the SM100 layout does not have.

### Two arms on top of the placement layer

| arm | placement | per-token split |
|---|---|---|
| `eplb-rr` | EPLB replicas | host-side round-robin over an expert's copies |
| `eplb-eco` | EPLB replicas | **in-kernel role 5**, per-copy `quota` |

ECO decides the split by minimizing the slowest rank's time
`Σ_slots [wb·⌈q/bm⌉ + wa·(q>0) + q]`, starting from either a cold (even split,
load-aware rounding) or a warm (reuse of the previous plan, congruent inverse +
δ adjustment) state, followed by auction rounds that move whole/partial unit
blocks from overloaded ranks.

The algorithm is a **line-by-line translation** of the upstream reference
implementation `eco/unit_ref.py`; there is no re-design and no simplification.

## Results (8×H20, DeepSeek-V4.1-Flash, TP=8/EP=8, 25% redundancy)

4 seq/rank, 100 decode steps to learn + rearrange, then 100 measured steps:

| arm | ms/step | Δ vs vanilla | IR (rank load) | tokens/rank |
|---|---:|---:|---:|---:|
| vanilla | **20.25** | — | 1.0284 | 32 |
| eplb-rr | 20.51 | +1.3% | 1.0672 | 32 |
| eplb-eco | 21.99 | +8.6% | **1.0358** | 32 |

Slot-level IR (per-layer, averaged): vanilla 1.662 / rr 1.338 / **eco 1.310**.

ECO balances better (it moves whole M-blocks and the partial unit separately,
round-robin can only round-robin tokens) but costs ~1.5 ms/step at this batch
size. The reason is structural, not a regression:

* the plan CTA must be **resident while the call runs** (dispatch blocks on it),
  and the routed role is persistent and fills the compute partition, so the
  routed grid gives up one whole SM whenever ECO is on;
* at 32 tokens the per-slot load is ~0.4 token/step, so **M-blocks == active
  slots** and the `wb`/`wa` terms of the cost model are collinear — the plan has
  almost nothing to optimize. Seeing the latency win needs a larger batch
  and/or a real H20 calibration of `wb`/`wa` (see `docs/ECO_SM90_PORT.md` §5).

See `docs/RESULTS.md` for per-rank detail, and the caveat block there about two
load metrics that were mis-reported and have since been retracted and fixed.

## Correctness

Three independent layers of evidence, all reproducible without loading a model:

1. **CPU** — `eco_ref/plan_port.py` (Python port) and
   `eco_ref/test_kernel_full.py` (CPU simulation of the kernel's control flow)
   are diffed bit-for-bit against `eco_ref/unit_ref.py`; cold, warm, auction,
   accept and `quota_start` are all covered, including the full EP8 geometry
   (P=480, E=384) and random small layouts.
2. **GPU, plan only** — `scripts/gpu_plan_only.py` runs role 5 alone on 8 ranks
   against real `eco_counts` and compares the published quota with
   `unit_ref.UnitPlanner.plan()`. 9 configurations × (cold, warm) all match
   exactly.
3. **GPU, end to end** — `scripts/verify_eco_ep8.py` runs the real kernel with
   random weights on 8 GPUs and checks the ECO arm against the replica-free
   reference path:

```
ECO_EP8 finite=True cos=1.00000000 rel=0.000e+00 conserved=True
        quota_start_contiguous=True plan_identical=True hot_copies_used=3/3 OK
```

i.e. bit-identical output, quotas conserve every expert's tokens, slices are
contiguous, and all 8 ranks compute the same plan.

## Layout

```
source/       gc_moe — upstream tree + the [ECO-SM90] port (MIT, see source/LICENSE)
integration/  vLLM "gc_moe" backend: EPLB placement, rearrangement, load metrics
eco_ref/      unit_ref.py (authoritative spec) + plan_port.py + EcoMeta + CPU diffs
tests/        CPU regressions (EPLB logic, load-metric definitions)
scripts/      run_arm_eco.sh (3 arms), gpu_plan_only.py, verify_eco_ep8.py, ...
docs/         ECO_SM90_PORT (port record), RESULTS, METRICS, REPRODUCE, TASK_SPEC
runs/ logs/   raw JSON + logs of the runs above
```

## Where the port lives

| piece | file |
|---|---|
| plan (cold/warm/auction/accept), role 5 | `source/gc_moe/impls/sm90_fp8_fp4_gc_moe.py` |
| dispatch: logical-id routing, push to every active copy | same file, `_dispatch` |
| pull: per-copy quota slice, passive copies skipped | same file, `_dispatch` |
| EcoMeta host builder | `source/gc_moe/eco/meta_sm90.py` |
| ECO views (disabled ⇒ byte layout unchanged) | `source/gc_moe/layout/sm90_gc_moe.py` |
| role-5 compile / resources / SM reserved for the plan | `jit_kernels/*`, `heuristics/sm90_gc_moe.py` |
| vLLM wiring (`GC_MOE_ECO=1`) | `integration/deepep_compare.py` |

## Reproduce

```bash
cd <this repo>
P=<python with cutlass-dsl + torch>
for f in eco_ref/test_*.py tests/test_*.py; do PYTHONPATH=$PWD/source $P $f; done

source scripts/env.sh
export GC_MOE_SRC=$PWD/source PYTHONPATH=$GC_MOE_SRC:$GC_SHIM:$PWD/integration
$GC_ENV/bin/torchrun --nproc_per_node 8 scripts/gpu_plan_only.py   # plan vs unit_ref
$GC_ENV/bin/torchrun --nproc_per_node 8 scripts/verify_eco_ep8.py   # end to end

SEQS_PER_RANK=4 WARM_STEPS=100 MEASURE_STEPS=100 ./scripts/run_arm_eco.sh 0 0 vanilla
SEQS_PER_RANK=4 WARM_STEPS=100 MEASURE_STEPS=100 GC_MOE_EPLB_REDUNDANCY=0.25 \
    ./scripts/run_arm_eco.sh 1 0 eplb_rr
SEQS_PER_RANK=4 WARM_STEPS=100 MEASURE_STEPS=100 GC_MOE_EPLB_REDUNDANCY=0.25 \
    ./scripts/run_arm_eco.sh 1 1 eplb_eco
```

`scripts/env.sh` pins absolute paths of the cluster this ran on; adjust
`GC_ENV` / `GC_SHIM` / `TOOLCHAIN` for your machine.

## Not done

* **two-hop auction** — disabled. EP8 is a complete rank graph, so a one-hop
  move always exists; the upstream build also uses 0 two-hop rounds at ≤32
  ranks, so no effective path is dropped.
* **`wb` / `wa` calibration** — still `0 / 0` (token-dominant objective, which
  is upstream's `GC_MOE_ECO_OBJECTIVE=token`). The B300-fitted weights must not
  be reused on H20.
* Metrics for M-blocks / active experts shown in `docs/RESULTS.md` were
  mis-computed in the first pass and are retracted there; the measuring code is
  fixed and covered by `tests/test_metrics_cpu.py`.

## License

`source/` is [Bruce-Lee-LY/gc-moe](https://github.com/Bruce-Lee-LY/gc-moe),
MIT, © 2026 Bruce-Lee-LY — its `LICENSE` is kept verbatim in `source/LICENSE`.
The port, harnesses and documents added here are released under the same MIT
terms.
