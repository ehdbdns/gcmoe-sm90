"""One full-model arm per process; share the reference trajectory across the five."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

ARMS = ("mega", "gc", "mega_rr", "gc_rr", "gc_eco")
STEPS = 1000


def make_plan(root, bs, device="cpu"):
    from gc_moe.eco.meta_sm90 import reorder_single_copy_first
    from gc_moe.eplb.policy import rebalance_experts_global

    hist = np.concatenate(
        [np.load(root / "mega" / f"stats-rank{r}.npz")["counts"] for r in range(8)],
        axis=2,
    )
    assert hist.shape == (1000, 40, 384)
    assert (hist >= 0).all() and (hist.sum(2) == bs * 6).all()
    # Keep the complete time vector for DeepInfer's simultaneous-peak policy.
    layout, inverse, counts = rebalance_experts_global(
        torch.from_numpy(hist).to(device), 448, 56
    )
    layout, inverse, counts = [a.cpu().numpy() for a in (layout, inverse, counts)]
    for l in range(40):
        ordered, local_order = reorder_single_copy_first(layout[l], 8)
        new_to_old = (
            np.asarray(local_order).reshape(8, 56) + 56 * np.arange(8)[:, None]
        ).ravel()
        valid = inverse[l] >= 0
        inverse[l][valid] = np.argsort(new_to_old)[inverse[l][valid]]
        layout[l] = ordered
        assert np.array_equal(layout[l][inverse[l][valid]], np.nonzero(valid)[0])
    np.savez_compressed(
        root / "plan.npz", layout=layout, inverse=inverse, counts=counts
    )


def main():
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import RequestOutputKind
    from vllm.v1.engine.core_client import InprocClient

    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("arm", choices=ARMS)
    p.add_argument("bs", type=int)
    args = p.parse_args()
    root, arm, bs = args.root.resolve(), args.arm, args.bs
    dest = root / arm
    assert bs in (256, 512, 1024, 2048)
    publisher = arm == "mega"
    os.environ.update(
        GC_MOE_V41_ENGINE="gc" if arm.startswith("gc") else "mega",
        GC_MOE_V41_DEEPINFER_RR=str(int("_" in arm)),
        GC_MOE_MATRIX_BS=str(bs),
        GC_MOE_MATRIX_ARM="vanilla" if publisher else "consumer",
    )
    if os.getenv("SM100_FORMAL_512"):
        assert bs == 512
        if publisher:
            import shutil
            required = 2 * 8 * (32 << 30) + (38 << 30)
            if shutil.disk_usage(root).free < required:
                raise RuntimeError("BS512 needs 550 GiB free on the private run filesystem")
    assert torch.cuda.device_count() == 8 and all(
        torch.cuda.get_device_capability(i) == (10, 3) for i in range(8)
    )
    eplb = "_" in arm
    llm = LLM(
        model=os.environ["SM100_V41_MODEL"],
        tensor_parallel_size=8,
        enable_expert_parallel=True,
        enable_prefix_caching=False,
        enforce_eager=True,
        enable_eplb=eplb,
        **(
            {
                "eplb_config": dict(
                    num_redundant_experts=64, step_interval=100000, use_async=False
                )
            }
            if eplb
            else {}
        ),
        worker_extension_cls="five_worker.MatrixWorker",
        max_num_seqs=bs,
        max_num_batched_tokens=max(2048, bs),
        max_model_len=2144,
        kernel_config=dict(
            enable_flashinfer_autotune=False, moe_backend="deep_gemm_mega_moe"
        ),
        dtype="bfloat16",
        gpu_memory_utilization=0.8,
        **({"kv_cache_memory_bytes": 32 << 30} if os.getenv("SM100_FORMAL_512") else {}),
        seed=0,
        async_scheduling=False,
    )
    engine = llm.llm_engine
    assert isinstance(engine.engine_core, InprocClient)

    def rpc(name, *a):
        result = engine.collective_rpc(name, args=a)
        assert len(result) == 8
        return result

    modules = rpc("inspect_v41_moe")
    assert all(
        len(ms) == 40 and all(m["sequence_parallel"] and m["fused_shared"] for m in ms)
        for ms in modules
    )
    rpc("matrix_setup", str(root), arm, bs)
    # Fixed eight-token prompts from the actual V4.1 SM90 output archive.
    # All BS use repetitions of the same 256-sequence mix.
    archive = (
        Path(__file__).resolve().parents[2]
        / "docs/trace_executor_20261119/sm90_main24_stable_decode_bs256_results/delivery/bs256/vanilla/rank0.npz"
    )
    with np.load(archive) as f:
        prompts = f["native"][:8].T.tolist() * (bs // 256)
    buf = {f"req{i}": [] for i in range(bs)}
    params = SamplingParams(
        max_tokens=2128,
        temperature=0,
        ignore_eos=True,
        output_kind=RequestOutputKind.DELTA,
    )
    for i, ids in enumerate(prompts):
        engine.add_request(f"req{i}", {"prompt_token_ids": ids}, params)

    def step():
        n = 0
        for o in engine.step():
            ids = o.outputs[0].token_ids
            buf[o.request_id].extend(ids)
            n += len(ids)
        return n

    while not all(buf.values()):
        step()
        if max(map(len, buf.values())) >= 128:
            raise ValueError("prefill did not converge within decode budget")
    result = dict(
        arm=arm,
        bs=bs,
        prompt_sha256=hashlib.sha256(json.dumps(prompts).encode()).hexdigest(),
    )
    for phase in ("stats", "measure"):
        if phase == "measure" and eplb:
            moved = rpc("install_v41_placement", str(root / "plan.npz"))
            assert all(x == moved[0] for x in moved)
            result["placement"] = moved[0]
            live = rpc("inspect_v41_replicas")
            with np.load(root / "plan.npz") as plan:
                expected = [
                    hashlib.sha256(
                        str((counts.tolist(), inv[inv >= 0].tolist())).encode()
                    ).hexdigest()
                    for counts, inv in zip(plan["counts"], plan["inverse"], strict=True)
                ]
            assert all(
                [item["placement_sha256"] for item in rank] == expected for rank in live
            )
            result["layout_sha256"] = expected
            if arm == "gc_eco":
                rpc("matrix_eco", str(root / "plan.npz"))
        rpc("matrix_begin", phase)
        rr_before = rpc("inspect_rr_calls")
        times = []
        t = time.perf_counter()
        for i in range(STEPS):
            if step() != bs:
                raise ValueError(f"{phase} step {i} is not full decode")
            times.append(time.perf_counter() - t)
        rpc("matrix_sync")
        elapsed = time.perf_counter() - t
        result[phase] = dict(
            wall_ms=1000 * elapsed / STEPS,
            cumulative_seconds=times,
            ranks=rpc("matrix_finish"),
        )
        rr_after = rpc("inspect_rr_calls")
        result[phase]["rr_calls"] = [
            a - b for a, b in zip(rr_after, rr_before, strict=True)
        ]
        expect_rr = eplb and (arm != "gc_eco" or phase == "stats")
        assert result[phase]["rr_calls"] == [STEPS * 40 * int(expect_rr)] * 8
        if publisher and phase == "stats":
            make_plan(root, bs)
    if publisher:
        (root / "tokens.json").write_text(json.dumps(dict(tokens=buf)))
    else:
        assert buf == json.loads((root / "tokens.json").read_text())["tokens"]
    (dest / "result.json").write_text(json.dumps(result))


if __name__ == "__main__":
    main()
