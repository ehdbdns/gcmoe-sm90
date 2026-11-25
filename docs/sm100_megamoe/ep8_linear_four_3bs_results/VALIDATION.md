# SM100 preflight validation and blocker checks

## Task identity

```text
7f0d9281edf818eef1e5c52e8994166f2dd554a5b410632195ea82c3ab79abf1  docs/sm100_megamoe/SM100_EP8_LINEAR_FOUR_3BS.md
```

Task commit: `55ef4ff27176c9c839d2feaa44348e2055ceb2fa`.
Execution commit: `d16bf90047024504d0c974a70dc2da13b6d3102c`.
Fixed upstream: `fb8b65fa829c97962eb2dbc9a4ad8b759b9cd736`.

## Prompt cohort repair

The checked-in selector initially rejected the historical JSONL files because only 187
and 158 prompts fall in the required 256-2048 token interval. The selector was repaired
to match the SM90 cohort procedure: sort each eligible pool, stride evenly, refill a short
pool by striding it again, interleave the two pools for BS512, and reuse BS512 twice for
BS1024. BS256 uses only the first file.

The resulting prompt hashes are:

```text
254af9d8a2752eab68f6419f085c1973e22e582a3f50a3c99a9853efeffdaec8  bs256/prompts.json
e7c4aee9db91a34c018bd54464e67355681b74112d515401844f473c053cfcef  bs512/prompts.json
d7eeeb50bb0428af010e6cad5ed56598f8c559a06525070a4d09ebd023d7e74c  bs1024/prompts.json
```

These exactly match the SM90 cohort hashes. Only BS256 was in execution scope; the other
two hashes are preflight evidence, not runs.

## Inputs, model, and patches

```text
3664daca1619617b76026e6a75e4b3738316320aceb2852c76f2897873e99073  dsv4-flash_online_input_ids_26-10-08_1.jsonl
957d06fed999f9fe517dab4581a2fe79f162debc1c21468502d3e988c83511f0  dsv4-flash_online_input_ids_26-10-08_2.jsonl
8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879  model/config.json
74b0686a3d2891980d5e303251b075a3bccae2c2ff650747db2620a649b98fa8  model/model.safetensors.index.json
7c7dd8165a45c75cc98c8fe19b3e684a9ec7174cb9a1338de299912ba0fb9d87  scripts/sm100/natureal_matrix_audit.patch
915d93ea08805d13ddb72536aabbad9a17385b947e252d5ad0349cf31f01fe7d  docs/sm100_megamoe/v41_full_decode_pair_results/compat/gc_capture_fix.diff
a761a9380db90aef6d9e988c6661b12eb1dfce538cdee12f509f60c81bf763f4  scripts/sm100/vllm_combine_order.patch
1f7000e828ca450931af2cea74c015aeda8b81d79fbdd68679185eae076b4791  scripts/sm100/ep8_linear_roles.patch
```

All checks passed before the GPU group gate.

## Eight-GPU gate

Visible devices:

```text
0  GPU-3aad6312-0e20-d01b-4249-71cb090c2127
1  GPU-b0ac7f5d-738f-cdad-423f-456f4c26947e
2  GPU-c5a93c78-be0c-2648-564a-4b4666671960
3  GPU-77463de9-85e0-a6f3-2c7e-f1147d6f7f49
4  GPU-4a493131-84b9-4c88-18ee-50fc6f213a36
5  GPU-045e62f4-d04d-2f9e-e456-1fb82810029f
6  GPU-6b44066c-6712-30f2-02d0-fa0cb91733a3
7  GPU-31a509fd-e741-5ab8-b837-443ad986c404
```

GPUs 4-7 had external compute applications:

```text
3351983  rtp_llm_rank-0  262 GiB  100%
3351987  rtp_llm_rank-1  262 GiB  100%
3351989  rtp_llm_rank-2  262 GiB  100%
3351991  rtp_llm_rank-3  262 GiB  100%
```

They started at `2026-11-25T16:37:34+08:00` and were still present at the final check.
The task requires waiting when any device is occupied. The external jobs were not
terminated. This was the decisive blocker.

The session labels the devices `NVIDIA L20D` while the task text says B300; hardware
identity should also be confirmed by the commander before retrying. It is not the only
blocker and was not used to justify stopping by itself.

## Stop and release checks

- Queue stop: `2026-11-25T18:07:42+08:00`.
- Launcher PID `2153811`: absent at the final check.
- `run/acquired.log`: not created, proving the task never passed the eight-GPU gate.
- Task model processes: none.
- Task workers: none.
- `this_task_acquired_gpu=no`.
- `this_task_started_model=no`.

## Required-output check

The following are absent because no arm ran:

```text
main.csv
layers.csv
pairs.csv
auction.csv
*/result.json
*/model.log
*-routes-*.pt
kv-*/rank*.bin
```

No partial timing value is promoted to a result.
