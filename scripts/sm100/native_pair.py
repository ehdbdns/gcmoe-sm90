"""Only native Mega and native GC. Compare one captured sequence of 40 real MoEs."""

import argparse
import json
import os
from pathlib import Path
import statistics

import numpy as np

STEPS = (0, 8, 16)


def report(root):
    runs = [json.loads((root / f"{arm}.json").read_text()) for arm in ("mega", "gc")]
    summary = []
    for step in STEPS:
        arms = [sorted(run[str(step)], key=lambda r: r["rank"]) for run in runs]
        for rows in arms:
            assert [r["rank"] for r in rows] == list(range(8))
        assert [r["fixture"] for r in arms[0]] == [r["fixture"] for r in arms[1]]
        data = []
        for rows in arms:
            times = np.asarray([r["ms"] for r in rows])
            assert (
                times.shape == (8, 15) and np.isfinite(times).all() and times.min() > 0
            )
            data.append(statistics.median(times.max(axis=0)))
        diffs = [
            sum(v["different"] for r in rows for v in r["outputs"]) for rows in arms
        ]
        summary.append(
            dict(
                step=step,
                mega_ms=data[0],
                gc_ms=data[1],
                gc_time_delta_pct=100 * (data[1] / data[0] - 1),
                unequal_elements_vs_live_mega=diffs,
                strict=not any(diffs),
            )
        )
    (root / "comparison.json").write_text(json.dumps(summary, indent=2))
    print(summary, flush=True)


def main(root, arm):
    os.environ.update(GC_MOE_V41_ENGINE=arm, GC_MOE_V41_DEEPINFER_RR="0")
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import RequestOutputKind

    llm = LLM(
        model=os.environ["SM100_V41_MODEL"],
        tensor_parallel_size=8,
        enable_expert_parallel=True,
        enable_eplb=False,
        enable_prefix_caching=False,
        enforce_eager=True,
        worker_extension_cls="native_pair_worker.NativePairWorker",
        max_num_seqs=1024,
        max_num_batched_tokens=2048,
        max_model_len=2144,
        kernel_config=dict(
            enable_flashinfer_autotune=False, moe_backend="deep_gemm_mega_moe"
        ),
        dtype="bfloat16",
        gpu_memory_utilization=0.8,
        seed=0,
        async_scheduling=False,
    )
    engine = llm.llm_engine

    def rpc(name, *args):
        values = engine.collective_rpc(name, args=args)
        assert len(values) == 8
        return values

    backend = rpc("pair_setup", str(root), arm)
    if arm == "mega":
        path = (
            Path(__file__).resolve().parents[2]
            / "docs/trace_executor_20261119/sm90_main24_stable_decode_bs256_results/delivery/bs256/vanilla/rank0.npz"
        )
        with np.load(path) as f:
            prompts = f["native"][:8].T.tolist() * 4
        assert len(prompts) == 1024
        params = SamplingParams(
            max_tokens=256,
            temperature=0,
            ignore_eos=True,
            output_kind=RequestOutputKind.DELTA,
        )
        for i, ids in enumerate(prompts):
            engine.add_request(f"req{i}", {"prompt_token_ids": ids}, params)
        seen = set()
        for _ in range(128):
            seen.update(
                o.request_id
                for o in engine.step()
                if o.outputs and o.outputs[0].token_ids
            )
            if len(seen) == 1024:
                break
        assert len(seen) == 1024
        for step in range(max(STEPS) + 1):
            if step in STEPS:
                rpc("pair_record")
            assert sum(len(o.outputs[0].token_ids) for o in engine.step()) == 1024
            if step in STEPS:
                rpc("pair_save", step)
    results = {"backend": backend}
    for step in STEPS:
        results[step] = rpc("pair_time", step)
    (root / f"{arm}.json").write_text(json.dumps(results))
    if arm == "gc":
        report(root)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("arm", choices=("mega", "gc"))
    a = p.parse_args()
    main(a.root, a.arm)
