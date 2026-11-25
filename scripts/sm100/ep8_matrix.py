"""Run one arm in its own process on the exact EP8 prompt cohort."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from ep8_inputs import ARMS, BATCHES, PROMPT_HASH, compact, initial_plans
from five_matrix import make_plan


def run(group, arm, bs):
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import RequestOutputKind
    from vllm.v1.engine.core_client import InprocClient

    publisher = arm == "vanilla"
    os.environ.update(SM100_EP8_ARM=arm, SM100_EP8_GROUP=str(group),
                      GC_MOE_MATRIX_ARM="vanilla" if publisher else "consumer", GC_MOE_MATRIX_BS=str(bs),
                      GC_MOE_V41_ENGINE="gc", GC_MOE_V41_DEEPINFER_RR=str(int(arm in ("random", "rr"))))
    if torch.cuda.device_count() != 8 or any(torch.cuda.get_device_capability(i) != (10, 3) for i in range(8)):
        raise RuntimeError("expected the eight B300 GPUs")
    data = (group / "prompts.json").read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    prompts = json.loads(data)
    if digest != PROMPT_HASH[bs] or len(prompts) != bs:
        raise ValueError(f"unexpected BS{bs} cohort")
    # Leave room for the earliest requests to decode during later chunked prefill.
    chunk = 2048
    capacity = 2000 + 2 * ((sum(map(len, prompts)) + chunk - 1) // chunk) + 128
    if publisher:
        initial_plans(group)
    dest = group / arm
    llm = LLM(model=os.environ["SM100_V41_MODEL"], tensor_parallel_size=8, enable_expert_parallel=True,
              enable_prefix_caching=False, enforce_eager=False, enable_eplb=not publisher,
              **({"eplb_config": dict(num_redundant_experts=64, step_interval=100000, use_async=False)}
                 if not publisher else {}),
              compilation_config=dict(mode=0, cudagraph_mode="FULL_DECODE_ONLY", cudagraph_capture_sizes=[bs],
                                      max_cudagraph_capture_size=bs),
              worker_extension_cls="ep8_worker.EP8Worker", max_num_seqs=bs, max_num_batched_tokens=chunk,
              max_model_len=max(map(len, prompts)) + capacity + 64, kv_cache_memory_bytes=32 << 30,
              kernel_config=dict(enable_flashinfer_autotune=False, moe_backend="deep_gemm_mega_moe",
                                 sparse_indexer_topk_backend="torch"),
              seed=0, dtype="bfloat16", async_scheduling=False)
    engine = llm.llm_engine
    assert isinstance(engine.engine_core, InprocClient)

    def rpc(name, *args):
        result = engine.collective_rpc(name, args=args)
        assert len(result) == 8
        return result

    modules = rpc("inspect_v41_moe")
    if not all(len(ms) == 40 and all(m["sequence_parallel"] and m["fused_shared"] for m in ms) for ms in modules):
        raise ValueError("incomplete native fused shared / sequence-parallel model")
    result = dict(bs=bs, arm=arm, prompt_sha256=digest, fused_shared=True)
    result["green_contexts"] = rpc("ep8_setup", str(group), arm, bs, capacity)
    if arm in ("random", "eco"):
        result["initial_placement"] = rpc("install_v41_placement",
                                          str(group / ("random.npz" if arm == "random" else "eco-initial.npz")))
    buf = {f"req{i}": [] for i in range(bs)}
    params = SamplingParams(max_tokens=capacity, temperature=0, ignore_eos=True,
                            output_kind=RequestOutputKind.DELTA)
    for name, prompt in zip(buf, prompts, strict=True):
        engine.add_request(name, {"prompt_token_ids": prompt}, params)

    def step():
        outputs = engine.step()
        for o in outputs:
            buf[o.request_id].extend(o.outputs[0].token_ids)
        return sum(len(o.outputs[0].token_ids) for o in outputs)

    while not all(buf.values()):
        step()
        if max(map(len, buf.values())) + 2000 >= capacity:
            raise ValueError("exceeded the fixed prefill allowance")
    for phase in ("stats", "measure"):
        if phase == "measure" and arm in ("rr", "eco"):
            moved = rpc("install_v41_placement", str(group / "plan.npz"))
            if any(x != moved[0] for x in moved):
                raise ValueError("ranks disagree on the weight move")
            result["placement"] = moved[0]
            replicas = rpc("inspect_v41_replicas")
            if any(x != replicas[0] for x in replicas):
                raise ValueError("replica maps differ across ranks")
        rpc("ep8_begin", phase)
        t = time.perf_counter()
        for i in range(1000):
            if step() != bs:
                raise ValueError(f"incomplete batch: {arm} {phase} step {i}")
        # All workers synchronize in finish, outside device MoE stamps.
        ranks = rpc("ep8_finish")
        result[phase] = dict(ranks=ranks)
        print(f"{arm} {phase} finished in {time.perf_counter() - t:.1f}s including audit export", flush=True)
        if publisher and phase == "stats":
            make_plan(group, bs, publisher="vanilla")
    if publisher:
        (group / "tokens.json").write_bytes(compact(dict(tokens=buf)))
    elif buf != json.loads((group / "tokens.json").read_text())["tokens"]:
        raise ValueError("returned token stream differs from reference")
    result["feedback_sha256"] = hashlib.sha256(compact(buf)).hexdigest()
    (dest / "result.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("group", type=Path)
    p.add_argument("arm", choices=ARMS)
    p.add_argument("bs", type=int, choices=BATCHES)
    args = p.parse_args()
    run(args.group.resolve(), args.arm, args.bs)
