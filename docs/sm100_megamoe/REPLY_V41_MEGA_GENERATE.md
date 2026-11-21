# 【SM100】V4.1-Flash 整模 DeepGEMM MegaMoE 四 token 生成回执

执行者：**SM100**。本轮未执行任何 SM90 任务书，也未混入 SM90 结果。

## 结论

**功能验证通过。** 整模入口使用显式 `moe_backend=deep_gemm_mega_moe`，加载 48/48 个 safetensors 分片，成功生成恰好 4 个贪心 token；模型进程与 launcher 退出码均为 `0`。本轮只证明整模入口、DeepGEMM MegaMoE 路径和生成能力，不构成端到端性能结论。

- 任务提交：`84c19c8123ba6a2c5a0ced41e31dcc5f4f77d4d7`
- 任务书 blob：`a96b7e2c00f4f4822c46a4d46a487c432d16ca70`
- 工作目录：`/data0/home/admin/workspace/megamoe/native-v41-ETKNbmD7`
- 运行窗口：`2026-11-21T21:29:10+08:00` 至 `2026-11-21T21:33:40+08:00`
- 最终退出码：`0`；launcher 退出码：`0`
- checkpoint：`48/48`，`00:42`
- 实际后端：`deep_gemm_mega_moe`
- 生成 ID：`[75, 200, 18, 200]`
- GPU KV cache：`8,258,440` tokens
- 八卡释放：`2026-11-21T21:33:40+08:00`；`21:36:47` 复核仍为空
- 外部定时终止：`none`

## 生成结果

请求保持原样：

```text
prompt_token_ids=[100, 200, 300, 400]
temperature=0
max_tokens=4
ignore_eos=true
```

原始脚本输出：

```json
{"backend": "deep_gemm_mega_moe", "prompt": {"prompt_token_ids": [100, 200, 300, 400]}, "generated_ids": [75, 200, 18, 200]}
```

`generated_ids` 恰为 4 个 token，长度断言通过。

已记录的原生 auto 基线为 `[200, 19, 200, 18]`。本轮首个不同位置是索引 `0`：`75` vs `200`；四个位置均不同。任务没有 bit-exact 或数值容差要求，因此这里只记录差异，不宣称 bit-exact，也不把差异归因为单一原因。

本轮继续使用已记录的 `enable_flashinfer_autotune=False`；相对于同参数、仅后端为 `auto` 的原生基线，唯一主动改变的模型后端选项是 `moe_backend`。未更换模型、卡数、框架版本、dtype、TP8/EP8、请求或采样参数。

## 实际后端与加载路径

启动参数中明确记录：

```text
KernelConfig(... moe_backend='deep_gemm_mega_moe' ...)
```

运行时 DeepGEMM 解析路径：

```text
deep_gemm not found in site-packages, trying vendored vllm.third_party.deep_gemm
Imported deep_gemm module from vllm.third_party.deep_gemm
```

实际模块路径：

```text
/data0/home/admin/workspace/megamoe/native-v41-stsFS4w7/venv/lib64/python3.12/site-packages/vllm/third_party/deep_gemm/__init__.py
```

同时观察到：

```text
Detected quantization_config.scale_fmt=ue8m0; enabling UE8M0 for DeepGEMM.
Using DeepGemmMxfp8BmmLinearKernel for MXFP8 GEMM
DeepGEMM PDL enabled on vllm.third_party.deep_gemm.
DeepGEMM E8M0 enabled on current platform.
Skipping FlashInfer autotune because it is disabled.
```

完整日志中没有 `fallback`、fatal error、traceback 或 MegaMoE shared-expert 降级警告。

加载数据：

- 每 rank 权重占用：`37.67 GiB`
- 每 rank non-KV memory：`40.54 GiB`
- memory profiling：`27.71` 至 `27.81 s`
- KV cache：`173.61 GiB`
- `init engine (profile, create kv cache, warmup model)`：`43.71 s`

## Shared expert 是否融合

**判定：融合已启用。** 需要明确限制：`model-full.log` 没有直接打印 `has_fused_shared_experts` 布尔值，也没有显式“shared expert fused”行；因此这不是日志直出的结论，而是以下证据链：

1. 模型 `config.json` 的 `text_config.n_shared_experts=1`。
2. 本轮为 TP8/EP8/PP1、`deep_gemm_mega_moe`，满足 sequence-parallel 的 shared-expert 融合条件。
3. `VLLM_DISABLE_DSV4_MEGAMOE_SHARED_EXPERT_FUSION` 在运行 venv 中求值为 `False`，且 launcher/env.sh 未设置该变量。
4. 实际加载的 DeepGEMM API 同时支持 `get_symm_buffer_for_mega_moe(..., num_shared_experts=...)`、`fp8_fp4_mega_moe(..., shared_l1_weights=..., shared_l2_weights=...)`、`transform_weights_for_mega_moe` 和 `get_block_m_for_mega_moe`。
5. 加载源码在 finalize 时把共享权重写入 `_transformed_shared_l1_weights`/`_transformed_shared_l2_weights`；forward 的 `has_fused_shared_experts` 分支把这些权重传给同一个 `fp8_fp4_mega_moe` launch，并跳过 serial shared MLP 加和。
6. 完整日志中 `Disabling native MegaMoE shared-expert fusion` 出现次数为 `0`。该警告覆盖 API 不支持、scale 布局不可转换和权重/scale 形状不符等全部已知降级路径。

源码指纹：

```text
deepseek_v41 model.py f3b109e980d12652e024fd9b20396e11d9810544c0f4a285fccb6206ce8a3bc8
deepseek_v4  model.py 5d5a74e339cd110def9b5e37a56fa602a42cdece98811013fb69700a0274a224
```

详细证据链与源码摘录见 `docs/sm100_megamoe/v41_mega_generate_results/shared-expert-evidence.txt` 和 `shared-expert-source.txt`。

## cgroup 与资源

cgroup：`/sys/fs/cgroup/user.slice/user-1384661.slice/session-1882009.scope`；`memory.max=max`。

| 项目 | 值 |
|---|---:|
| `memory.current` 前 | 423,325,437,952 bytes |
| `memory.current` 峰值 | 833,362,128,896 bytes (`21:32:10`) |
| `memory.current` 后 | 521,786,744,832 bytes |
| `oom` 前 / 后 / 观测峰值 | 0 / 0 / 0 |
| `oom_kill` 前 / 后 / 观测峰值 | 0 / 0 / 0 |
| `oom_group_kill` 前 / 后 / 观测峰值 | 0 / 0 / 0 |
| `MemAvailable` 前 / 最低 / 后 | 1,159,072,000 / 882,601,468 / 1,076,500,836 kB |

`memory.current` 是整个 session cgroup 的计数，包含可回收页缓存，不是模型进程 RSS；没有观察到 cgroup OOM 或内存限额事件。

## 外部终止、journal 与八卡释放

- launcher 为 `setsid-detached`，`external_timeout=none`，没有设置 `timeout`；Python 自行退出 `0`，未观察到外部定时终止。
- 生成完成后 teardown 出现 `force killing remaining process EngineCore` 和 worker SIGTERM 告警，但生成输出已先产生，模型进程与 launcher 退出码均为 `0`；这些是收尾告警，不把本轮判为失败。
- 八卡均为 `NVIDIA L20D`，每卡 `275040 MiB`，capability `(10, 3)`；使用 `ENV.json` 登记的 8 个 UUID，启动前 compute-apps 为空。
- 释放快照 `2026-11-21T21:33:40+08:00` 为空；较晚复核 `2026-11-21T21:36:47+08:00` 的 `nvidia-smi --query-compute-apps` 和 `/dev/nvidia0..7` 占用仍为空。
- system journal 权限受限：`system_journal_access=PERMISSION_LIMITED`。没有从不可读日志推断 OOM/Xid；cgroup 计数和进程结果单独保留。

## 范围限制

- 未启用 EPLB 重平衡或 ECO，未产生五臂性能数据。
- 约 4 分 30 秒的运行窗口只用于功能验证，不作为调优后性能结果。
- 外部 journal 不可读，因此不能声称完成了 system-wide Xid/OOM 审计；本轮能确认的是进程退出、cgroup 无 OOM、生成输出完整和八卡已释放。

## 环境

- vLLM `0.31.0`
- torch `2.13.0+cu130`，CUDA `13.0`
- flashinfer `0.7.0.post1`
- transformers `5.17.0`
- triton `3.7.1`
- Python `3.12.7`
- 独立 venv：`/data0/home/admin/workspace/megamoe/native-v41-stsFS4w7/venv`
- 权重：`/data0/home/admin/workspace/megamoe/native-v41-KO1hTbTj/model`，XFS，48 分片
- `config.json` SHA-256：`8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879`
- 权重索引 SHA-256：`74b0686a3d2891980d5e303251b075a3bccae2c2ff650747db2620a649b98fa8`

## 产物

`docs/sm100_megamoe/v41_mega_generate_results/` 包含完整日志、生成输出、后端与 shared-expert 证据、cgroup 前后计数、GPU 释放、journal 结果、launcher、真实入口、安装/版本、cgroup 汇总、时间线和 `SHA256SUMS`。未提交模型权重、wheel 或编译缓存。
