"""Run one GLM-5.3 BS256 full-model arm, or reduce its matched EP8 cohort."""

import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

ARMS = ("vanilla", "rr", "eco")
LAYERS = (8, 20, 32)
BS = 256


def prepare(root, model):
    from transformers import AutoTokenizer

    root.mkdir(exist_ok=True, parents=True)
    cfg = json.loads((model / "config.json").read_text())
    q = cfg["quantization_config"]
    if (q.get("expert_dtype"), q.get("scale_fmt"), cfg["num_hidden_layers"]) != ("fp4", "ue8m0", 78):
        raise ValueError("unexpected GLM checkpoint")
    # Local HF view: explicitly select vLLM's MXFP4 routed method while
    # preserving the 128x128 FP8 method for every other quantized linear.
    q["store_dtype"] = "mxfp4"
    overlay = root / "model"
    overlay.mkdir()
    for src in model.iterdir():
        if src.name != "config.json":
            (overlay / src.name).symlink_to(src.resolve())
    (overlay / "config.json").write_text(json.dumps(cfg))
    tok = AutoTokenizer.from_pretrained(str(model), local_files_only=True)
    topics = [
        "交通",
        "物流",
        "仓储",
        "气象",
        "地质",
        "土壤",
        "城市",
        "河流",
        "森林",
        "能源",
        "电网",
        "桥梁",
        "语言",
        "历史",
        "音乐",
        "教育",
    ]
    tasks = [
        "测量方法",
        "发展过程",
        "数据来源",
        "典型应用",
        "成本构成",
        "评价标准",
        "设计思路",
        "季节变化",
        "观察误差",
        "数学模型",
        "历史案例",
        "比较方法",
        "资源分配",
        "实验设计",
        "研究现状",
        "长期影响",
    ]
    prompts = [
        tok.encode(f"请结合具体实例，解释{topic}研究中的{task}，并说明你的推理过程。")
        for topic in topics
        for task in tasks
    ]
    if len(set(map(tuple, prompts))) != BS:
        raise ValueError("prompt rows are not unique")
    (root / "prompts.json").write_text(json.dumps(prompts, separators=(",", ":")))
    for phase in ("stats", "measure"):
        (root / f"kv-{phase}").mkdir()


def plan(root, extra=8):
    import torch
    from gc_moe.eco.meta_sm90 import reorder_single_copy_first
    from gc_moe.eplb.deepinfer_periodic.policy import rebalance_experts_global

    slots = 32 + extra
    counts = [load(root, "vanilla", "stats", r, "counts") for r in range(8)]
    logical = torch.from_numpy(np.concatenate(counts, axis=-1))
    if not torch.all(logical.sum(-1) == BS * 8):
        raise ValueError("native dispatch lost or duplicated tokens")
    # Independent layer problems bound the packer's CPU working set.
    torch.set_num_threads(4)
    rows = []
    for j in range(75):
        placed, _, _ = rebalance_experts_global(logical[:, j : j + 1], 8 * slots, slots)
        rows.append(reorder_single_copy_first(placed[0].numpy(), 8)[0])
    np.save(root / "placement.npy", np.stack(rows))


def load(root, arm, phase, rank, key):
    with np.load(root / arm / f"{phase}-rank{rank}.npz") as v:
        return v[key]


def arm_run(root, arm):
    import torch
    from eco_trace import export
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import RequestOutputKind

    if torch.cuda.device_count() != 8 or any(
        torch.cuda.get_device_capability(r)[0] != 10 for r in range(8)
    ):
        raise ValueError("need exactly eight allocated Blackwell GPUs")
    if os.getenv("GLM_RESUME_FROM") and arm == "vanilla":
        raise ValueError("resume runs only RR and ECO")
    if os.getenv("GLM_ECO_ONLY") == "1" and arm != "eco":
        raise ValueError("only ECO may start for the completed RR cohort")
    if os.getenv("GLM_RESUME_FROM"):
        native = json.loads((root / "vanilla/result.json").read_text())
        expected = [str(x["gpu"]).lower().removeprefix("gpu-") for x in native["setup"]]
        actual = [str(torch.cuda.get_device_properties(r).uuid).lower().removeprefix("gpu-") for r in range(8)]
        if actual != expected:
            raise ValueError(f"GPU rank order differs from the completed vanilla arm: {actual}")
    os.environ.update(
        GLM_ARM=arm,
        GLM_RUN=str(root),
        GC_MOE_MATRIX_ARM="vanilla" if arm == "vanilla" else "consumer",
        GC_MOE_MATRIX_BS=str(BS),
    )
    from vllm.v1.engine.core_client import InprocClient

    prompts = json.loads((root / "prompts.json").read_text())
    folder = root / arm
    (folder / "perfetto").mkdir(parents=True)
    llm = LLM(
        model=str(root / "model"),
        tokenizer=os.environ["GLM_MODEL"],
        tensor_parallel_size=8,
        enable_expert_parallel=True,
        enable_ep_weight_filter=True,
        load_format="safetensors",
        safetensors_load_strategy="lazy",
        enable_prefix_caching=False,
        enforce_eager=False,
        worker_extension_cls="glm53_decode_worker.GlmWorker",
        max_num_seqs=BS,
        max_num_batched_tokens=2048,
        max_model_len=2464,
        kv_cache_memory_bytes=48 << 30,
        seed=0,
        dtype="bfloat16",
        async_scheduling=False,
        compilation_config={
            "mode": 0,
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [BS],
            "max_cudagraph_capture_size": BS,
            "cudagraph_num_of_warmups": 2,
        },
        kernel_config={"enable_flashinfer_autotune": False, "sparse_indexer_topk_backend": "torch"},
    )
    engine = llm.llm_engine
    if not isinstance(engine.engine_core, InprocClient):
        raise TypeError("expected V2 in-process engine client")

    def rpc(method, *args):
        values = engine.collective_rpc(method, args=args)
        if len(values) != 8:
            raise RuntimeError("missing worker")
        return values

    result = {
        "arm": arm,
        "setup": rpc("glm_setup"),
        "prompt_sha256": hashlib.sha256((root / "prompts.json").read_bytes()).hexdigest(),
    }
    tokens = {f"req{i}": [] for i in range(BS)}
    params = SamplingParams(
        temperature=0, max_tokens=2400, ignore_eos=True, output_kind=RequestOutputKind.DELTA
    )
    for key, prompt in zip(tokens, prompts):
        engine.add_request(key, {"prompt_token_ids": prompt}, params)

    def step():
        outputs = engine.step()
        for out in outputs:
            tokens[out.request_id].extend(out.outputs[0].token_ids)
        return sum(len(out.outputs[0].token_ids) for out in outputs)

    while not all(tokens.values()):
        step()
        if max(map(len, tokens.values())) >= 400:
            raise ValueError("prefill exceeds capacity")
    for phase in ("stats", "measure"):
        rpc("glm_begin", phase)
        t = time.monotonic()
        for i in range(1000):
            sample = phase == "measure" and i == 1
            if sample:
                rpc("glm_profile", True)
            if step() != BS:
                raise ValueError(f"{phase} step {i + 1} is not a full decode batch")
            if sample:
                rpc("glm_profile", False)
        result[phase] = rpc("glm_end")
        result[phase + "_wall_s_with_audit"] = time.monotonic() - t
        print(f"{arm} {phase} complete", flush=True)
    if arm == "vanilla":
        (root / "tokens.json").write_text(json.dumps({"tokens": tokens}))
    elif tokens != json.loads((root / "tokens.json").read_text())["tokens"]:
        raise ValueError("returned feedback differs")
    for layer in LAYERS:
        export(folder / "perfetto", f"step0002-layer{layer:02d}")
    (folder / "result.json").write_text(json.dumps(result, indent=2))


def reduce(root, arms=ARMS, filename="main.csv"):
    plan_rows = np.load(root / "placement.npy")
    native = json.loads((root / "vanilla/result.json").read_text())
    summaries = []
    for arm in arms:
        data = json.loads((root / arm / "result.json").read_text())
        paired = data["prompt_sha256"] == native["prompt_sha256"]
        for phase in ("stats", "measure"):
            for r, recv in enumerate(data[phase]):
                ref = native[phase][r]
                if recv["boundary"] != ref["boundary"] or recv["batch_order"] != ref["batch_order"]:
                    raise ValueError(f"{arm} {phase} rank{r} checkpoint or request order differs")
                paired &= not any(
                    recv[k]
                    for k in (
                        "raw_gate_id_diff",
                        "raw_gate_weight_diff",
                        "native_top1_diff",
                        "prefill_native_diff",
                    )
                )
        counts = np.stack([load(root, arm, "measure", r, "counts") for r in range(8)], axis=2)
        clocks = np.stack([load(root, arm, "measure", r, "clocks") for r in range(8)], axis=2)
        bm_values = {x["block_m"] for x in data["setup"]}
        width = 32 if arm == "vanilla" else plan_rows.shape[1] // 8
        if counts.shape != (1000, 75, 8, width) or any(s["slots"] != width for s in data["setup"]):
            raise ValueError(f"{arm}: measured slots differ from installed placement")
        if len(bm_values) != 1 or np.any(counts < 0):
            raise ValueError("invalid physical counts or block M")
        bm = bm_values.pop()
        if arm == "vanilla":
            ref_counts = counts.reshape(1000, 75, 256)
        else:
            for j, row in enumerate(plan_rows):
                totals = np.zeros((256, 1000), np.int32)
                np.add.at(totals, row, counts[:, j].reshape(1000, plan_rows.shape[1]).T)
                if not np.array_equal(totals.T, ref_counts[:, j]):
                    raise ValueError(f"{arm} decoder layer {j + 3} logical dispatch differs")
        t = counts.sum(-1)
        m = ((counts + bm - 1) // bm).sum(-1)
        a = (counts > 0).sum(-1)
        start = clocks[:, :, :, 0].copy()
        for k in [1, 15] if arm == "eco" else [1]:
            start = np.minimum(start, clocks[:, :, :, k])
        span = clocks[:, :, :, 13] - start
        if not np.all(t.sum(-1) == BS * 8) or np.any(start <= 0) or np.any(span <= 0):
            raise ValueError("invalid graph timing or incomplete dispatch")
        row = {
            "arm": arm,
            "bs": BS,
            "extra_slots_per_rank": width - 32,
            "bm": bm,
            "full_native_match": bool(paired),
            "moe_us": span.max(-1).mean() / 1000,
            "token_ir": (t.max(-1) / t.mean(-1)).mean(),
            "total_m": m.sum(-1).mean(),
            "peak_m": m.max(-1).mean(),
            "active_slots": a.sum(-1).mean(),
            "peak_active": a.max(-1).mean(),
            "m_ir": (m.max(-1) / m.mean(-1)).mean(),
        }
        if arm == "eco":
            score = clocks[:, :, :, 19]
            routed = np.where(t > 0, clocks[:, :, :, 7] - clocks[:, :, :, 4], 0)
            accepted, offered = (clocks[:, :, :, k] for k in (17, 18))
            if (
                np.any(routed[t > 0] <= 0)
                or np.any(score.sum(-1) <= 0)
                or np.any(accepted < 0)
                or np.any(accepted > offered)
                or np.any(accepted != accepted[:, :, :1])
                or np.any(offered != offered[:, :, :1])
            ):
                raise ValueError("invalid ECO clocks or replica auction counters")
            row.update(
                routed_mae_pp=100
                * np.abs(
                    score / score.sum(-1, keepdims=True) - routed / routed.sum(-1, keepdims=True)
                ).mean(),
                accepted_per_call=accepted[:, :, 0].mean(),
                accepted=int(accepted[:, :, 0].sum()),
                offered=int(offered[:, :, 0].sum()),
                zero_accept_calls=int((accepted[:, :, 0] == 0).sum()),
            )
            row["accept_pct"] = 100 * row["accepted"] / row["offered"] if row["offered"] else None
        summaries.append(row)
    for row in summaries:
        for key, ref in (("vs_native_pct", summaries[0]), ("vs_rr_pct", summaries[1])):
            row[key] = (
                100 * (row["moe_us"] / ref["moe_us"] - 1)
                if row["full_native_match"] and ref["full_native_match"]
                else None
            )
    with (root / filename).open("w") as f:
        writer = csv.DictWriter(f, list(dict.fromkeys(k for row in summaries for k in row)))
        writer.writeheader()
        writer.writerows(summaries)
    return summaries


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("prepare", "plan", "report", *ARMS))
    p.add_argument("root", type=Path)
    p.add_argument("--extra", type=int, choices=(4, 8), default=8)
    args = p.parse_args()
    root = args.root.resolve()
    if args.action == "prepare":
        prepare(root, Path(os.environ["GLM_MODEL"]))
    elif args.action == "plan":
        plan(root, args.extra)
    elif args.action == "report":
        reduce(root)
    else:
        arm_run(root, args.action)
