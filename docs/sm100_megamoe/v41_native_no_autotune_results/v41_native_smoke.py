import json
import os

import torch
from vllm import LLM, SamplingParams


def main():
    assert torch.cuda.device_count() == 8
    assert all(torch.cuda.get_device_capability(i) == (10, 3) for i in range(8))
    model = LLM(
        model=os.environ["SM100_V41_MODEL"],
        tensor_parallel_size=8,
        enable_expert_parallel=True,
        enable_prefix_caching=False,
        enforce_eager=True,
        kernel_config={"enable_flashinfer_autotune": False},
        max_model_len=2048,
        max_num_seqs=8,
        max_num_batched_tokens=2048,
        seed=0,
        dtype="bfloat16",
        async_scheduling=False,
        gpu_memory_utilization=0.8,
    )
    prompt = {"prompt_token_ids": [100, 200, 300, 400]}
    out = model.generate(
        [prompt],
        SamplingParams(temperature=0, max_tokens=4, ignore_eos=True),
    )
    ids = list(out[0].outputs[0].token_ids)
    assert len(ids) == 4
    print(json.dumps(dict(prompt=prompt, generated_ids=ids)), flush=True)


if __name__ == "__main__":
    main()
