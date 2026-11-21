import argparse
import json
import os

import torch
from vllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("auto", "deep_gemm_mega_moe", "gc_moe"), default="auto")
    parser.add_argument("--prefix-audit", action="store_true")
    args = parser.parse_args()
    if args.backend == "gc_moe":
        os.environ["GC_MOE_V41_ENGINE"] = "gc"
    else:
        os.environ.pop("GC_MOE_V41_ENGINE", None)
    extension = ({"worker_extension_cls": "v41_gc_worker.MoeAuditWorker"}
                 if args.prefix_audit or args.backend == "gc_moe" else {})
    assert torch.cuda.device_count() == 8
    assert all(torch.cuda.get_device_capability(i) == (10, 3) for i in range(8))
    model = LLM(
        model=os.environ["SM100_V41_MODEL"],
        tensor_parallel_size=8,
        enable_expert_parallel=True,
        enable_prefix_caching=False,
        enforce_eager=True,
        kernel_config={"enable_flashinfer_autotune": False, "moe_backend": "deep_gemm_mega_moe" if args.backend == "gc_moe" else args.backend},
        max_model_len=2048,
        max_num_seqs=8,
        max_num_batched_tokens=2048,
        **extension,
        seed=0,
        dtype="bfloat16",
        async_scheduling=False,
        gpu_memory_utilization=0.8,
    )
    prompt = {"prompt_token_ids": [100, 200, 300, 400]}
    if args.prefix_audit:
        modules = model.collective_rpc("inspect_v41_moe")
        if len(modules) != 8 or any(len(rank) != 40 for rank in modules):
            raise RuntimeError(f"incomplete MoE coverage: {[len(rank) for rank in modules]}")
        print(json.dumps(dict(backend=args.backend, workers=modules)), flush=True)
        tokens = list(prompt["prompt_token_ids"])
        # Keep the first-step workload identical. Later calls use exactly the
        # same prefix in both engines, independent of their previous prediction.
        for ref_id in (200, 19, 200, 18):
            predicted = model.generate([{"prompt_token_ids": list(tokens)}],
                                       SamplingParams(temperature=0, max_tokens=1, logprobs=10))
            output = predicted[0].outputs[0]
            scores = sorted(((int(idx), float(v.logprob)) for idx, v in output.logprobs[0].items()),
                            key=lambda row: -row[1])
            print(json.dumps(dict(backend=args.backend, prefix=list(tokens), top=scores,
                                  generated=int(output.token_ids[0]))), flush=True)
            tokens.append(ref_id)
        return
    out = model.generate(
        [prompt],
        SamplingParams(temperature=0, max_tokens=4, ignore_eos=True),
    )
    ids = list(out[0].outputs[0].token_ids)
    assert len(ids) == 4
    print(json.dumps(dict(backend=args.backend, prompt=prompt, generated_ids=ids)), flush=True)


if __name__ == "__main__":
    main()
