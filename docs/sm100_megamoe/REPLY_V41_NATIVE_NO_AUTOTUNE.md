# 【SM100】V4.1-Flash 关闭启动期 FlashInfer autotune 的生成验证回执

执行者：**SM100**。本轮未执行任何 SM90 任务书，也未混入 SM90 结果。

## 结论

**验收通过。** 复用上一轮的本机 XFS 权重与 FlashInfer 缓存，仅关闭 `enable_flashinfer_autotune`；整模入口加载 48/48 分片并成功生成 4 个贪心 token，进程退出码为 `0`。

- 任务提交：`13af85aac8ded860e5e5655ad2682ceebb349dbe`
- 任务书 blob：`bad719a1d09e8c660b84fd93d6e5ca81d08efc3f`
- 工作目录：`/data0/home/admin/workspace/megamoe/native-v41-utXJnGyD`
- 运行窗口：`2026-11-21T21:19:07+08:00` 至 `2026-11-21T21:25:07+08:00`
- 最终退出码：`0`
- checkpoint：`48/48`，加载耗时 `03:09`
- FlashInfer autotune：`21:24:46` 明确记录 `Skipping FlashInfer autotune because it is disabled.`
- 实际 MoE 后端：`FLASHINFER_TRTLLM_MXFP4_MXFP8`
- 生成 ID：`[200, 19, 200, 18]`
- 八卡释放：`2026-11-21T21:25:07+08:00`，`nvidia-smi --query-compute-apps` 为空

本轮只证明整模入口与四 token 生成能力。`360 s` 墙钟时间不作为调优后的五臂性能结果。

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
{"prompt": {"prompt_token_ids": [100, 200, 300, 400]}, "generated_ids": [200, 19, 200, 18]}
```

`generated_ids` 恰为 4 个 token，长度断言通过，脚本退出 `0`。

## 关闭 autotune 的证据

启动参数中明确记录：

```text
KernelConfig(... enable_flashinfer_autotune=False, ...)
```

运行时日志明确记录：

```text
2026-11-21T21:24:46+08:00 Worker_TP0_EP0:
Skipping FlashInfer autotune because it is disabled.
```

本轮没有 candidate-cubin autotune 下载风暴：

- `Fetching ... cubin`：`0` 行
- `File downloaded successfully`：`0` 行
- 复用已有 FlashInfer 缓存

加载与初始化明显缩短为：

- checkpoint 48/48：`03:09`
- 每 rank memory profiling：`11.21` 至 `11.48 s`
- `init engine (profile, create kv cache, warmup model) took 17.07 s`

这不构成性能结论；只说明关闭 autotune 后功能验证不再被候选 cubin 下载阻塞。

## 权重复用与实际后端

本轮没有再次复制 510 GB 权重，直接复用：

- 本机 XFS 权重：`/data0/home/admin/workspace/megamoe/native-v41-KO1hTbTj/model`
- 挂载类型：`xfs`
- `48` 个 safetensors 分片可读
- `config.json` SHA-256：`8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879`
- 权重索引 SHA-256：`74b0686a3d2891980d5e303251b075a3bccae2c2ff650747db2620a649b98fa8`
- 配置：`hidden_size=5120`、`moe_intermediate_size=2304`、`n_routed_experts=384`、`num_experts_per_tok=6`、`num_hidden_layers=40`

`moe_backend='auto'` 的实际解析仍为：

```text
Mxfp4 MoE backend 'FLASHINFER_TRTLLM_MXFP4_MXFP8' does not support the deployment configuration since kernel does not support routing method 100.
Using 'FLASHINFER_TRTLLM_MXFP4_MXFP8' Mxfp4 MoE backend.
```

随后 expert 实现选择：

```text
Using TrtLlmMxfp4ExpertsModular
```

后端支持性警告与实际选择均原样保留；未更换模型、卡数、框架版本或后端。

## 资源与释放

八张 `NVIDIA L20D`，每卡 `275040 MiB`，capability `(10, 3)`；使用 `ENV.json` 登记的八个 UUID。启动前 `gpu-compute-before.txt` 为空。

cgroup 为 `/sys/fs/cgroup/user.slice/user-1384661.slice/session-1882009.scope`：

| 项目 | 值 |
|---|---:|
| `memory.max` | `max` |
| `memory.current` 前 | 345,433,944,064 bytes |
| `memory.current` 峰值 | 814,085,054,464 bytes (`21:24:37`) |
| `memory.current` 后 | 574,074,408,960 bytes |
| `oom` / `oom_kill` / `oom_group_kill` | 0 / 0 / 0 |
| `MemAvailable` 前 / 后 | 1,133,037,420 / 1,030,809,544 kB |

没有 cgroup OOM 或内存限额事件。`memory.current` 包含可回收页缓存。

## 外部终止与收尾

- launcher 为 `setsid-detached`，未设置 `timeout`；`external_timeout=none`
- Python 自行退出 `0`，未观察到外部终止
- 生成完成后 vLLM 在 teardown 中仍有 `force killing remaining process EngineCore` 与 worker SIGTERM 告警，但主进程退出码仍为 `0`，且输出 ID 已先产生；这是收尾日志，不影响本轮生成验收
- 八卡在 `21:25:07` 已空；`21:25:44` 复核仍为空

system journal 仍为权限受限：

```text
system_journal_access=PERMISSION_LIMITED
Hint: You are currently not seeing messages from other users and the system.
coredump_readable=no_coredumps_or_permission_limited
```

因此没有从不可读日志推断 OOM/Xid；cgroup 计数与进程结果均单独保留。

## 环境

- vLLM `0.31.0`
- torch `2.13.0+cu130`，CUDA `13.0`
- flashinfer `0.7.0.post1`
- transformers `5.17.0`
- triton `3.7.1`
- Python `3.12.7`
- 独立 venv：`/data0/home/admin/workspace/megamoe/native-v41-stsFS4w7/venv`
- 入口：仓库内真实文件 `scripts/sm100/v41_native_smoke.py`，带 `if __name__ == "__main__"` guard

## 产物

目录：`docs/sm100_megamoe/v41_native_no_autotune_results/`

- 结论：`status.txt`、`timeline.txt`、`first_error.txt`、`output_ids.txt`
- autotune：`autotune_evidence.txt`、`jit-summary.txt`
- 后端：`model-full.log`、`config.log`、`weight_reuse.txt`
- 环境：`versions.txt`、`freeze.txt`、`install.log`、`gpu-capability.txt`、`gpu-before.txt`、`cuda-visible-devices.txt`
- 资源：`memory.before`、`memory.log`、`memory.after`、`cgroup-evidence.txt`、`cgroup-access.txt`、`gpu-release.txt`
- 日志：`model.log`、`model-full.log`、`console.log`、`launcher.sh`、`v41_native_smoke.py`
- 权限：`journal-access.txt`、`journal-probe.log`、`journal-probe.err`、`journal-probe-extended.txt`
- 退出状态：`exit-code.txt`、`launcher-exit-code.txt`
- `SHA256SUMS`

未提交模型权重或编译缓存。

## 判定

**PASS（功能验证）。** 关闭 `enable_flashinfer_autotune` 后，整模入口、48/48 权重加载、native auto MoE 后端和四 token 贪心生成全部通过。该结果不用于推断调优后五臂性能。
