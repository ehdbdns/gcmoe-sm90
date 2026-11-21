# 【SM100】V4.1 GCMoE 整模路径生成回执

执行者：**SM100**。本轮未执行任何 SM90 任务书，也未混入 SM90 结果。

## 结论

**验收失败，进程退出 `1`。** `gc_moe` 的 worker adapter 已开始安装，但模型在权重加载后的量化后处理阶段触发 `AssertionError`，EngineCore 初始化失败。没有进入四前缀推理，没有产生 top1/top10，也没有生成 token。

- 任务提交：`b823c329a6a8af7e68310efdb4fb704ddd0c4c37`
- 任务书 blob：`b337afaac983734a534c903a8dbd4ed12750c276`
- 工作目录：`/data0/home/admin/workspace/megamoe/native-v41-ijPsTrVT`
- 运行窗口：`2026-11-21T22:24:21+08:00` 至 `22:27:01+08:00`
- 模型进程退出码：`1`；launcher 退出码：`1`
- checkpoint 分片：`48/48` 已加载
- 预期后端：`gc_moe`（host 仍使用 MegaMoE 路径，由 worker adapter 替换专家转换、对称缓冲区、M 块选择和 launch）
- 实际推理：**未开始**
- 输出 ID：**无**
- 外部定时终止：`none`
- 八卡释放：`2026-11-21T22:27:01+08:00`；`22:27:48` 复核为空

## 首个错误

首个后台 worker 错误出现在 `2026-11-21T22:26:20+08:00`：

```text
File ".../vllm/model_executor/layers/quantization/modelopt.py", line 2541, in process_weights_after_loading
    self.wkey.process(layer, WEIGHT)
File ".../vllm/model_executor/layers/quantization/modelopt.py", line 2261, in process
    assert layer.weight.ndim == 2 and layer.weight.dtype == MXFP8_VALUE_DTYPE
AssertionError
```

随后 EngineCore 报告：

```text
Exception: WorkerProc initialization failed due to an exception in a background process.
RuntimeError: Engine core initialization failed. See root cause above.
```

首个错误不是 GCMoE kernel launch 失败，也不是编译失败；它发生在 vLLM 的模型权重后处理阶段。按任务书“其他故障原样回执”的要求，本轮没有修改 GCMoE、vLLM 或 harness 代码来绕过该断言。

## GC adapter 安装情况

worker extension 使用任务书新增的具名 RPC/扩展路径 `v41_gc_worker.MoeAuditWorker`。失败日志中观察到 4 条 adapter 安装记录，对应 4 个不同 worker PID：

| worker | PID | 状态 |
|---|---:|---|
| TP4/EP4 | 1167425 | adapter installed |
| TP5/EP5 | 1167427 | adapter installed |
| TP6/EP6 | 1167428 | adapter installed |
| TP7/EP7 | 1167432 | adapter installed |

任务要求的 8 个不同 worker PID 的安装行没有全部出现；只有上述 4 条在失败前被记录。由于 EngineCore 没有完成初始化，不能把这 4 条安装记录解释为 GCMoE 路径已经可用，也没有可核对的 8×40 层融合结果。

## 四前缀与对照

任务要求的四次固定前缀推理均未执行，因此没有新的 top1/top10。对照对象仍为 `v41_fixed_prefix_results/prefix-audit.json` 中上一轮 `deep_gemm_mega_moe` 的结果：

```text
[100,200,300,400]                     -> 75
[100,200,300,400,200]                 -> 300
[100,200,300,400,200,19]              -> 200
[100,200,300,400,200,19,200]          -> 19
```

本轮没有 GCMoE top1 可与之比较，因而首选 ID 名次、top2 logprob 差值及后续固定前缀稳定性均为 **N/A / 未测得**。

## 环境与资源

- vLLM `0.31.0`
- torch `2.13.0+cu130`，CUDA `13.0`
- flashinfer `0.7.0.post1`
- transformers `5.17.0`
- triton `3.7.1`
- Python `3.12.7`
- 独立 venv：`/data0/home/admin/workspace/megamoe/native-v41-stsFS4w7/venv`
- GCMoE：本工作树 `source/gc_moe`，版本 `0.3.1`
- 没有安装或覆盖共享环境依赖；`pip-freeze.log` 和 `import.log` 已保存
- cgroup：`/sys/fs/cgroup/user.slice/user-1384661.slice/session-1882009.scope`，`memory.max=max`
- `memory.current`：前 `433,432,305,664` bytes；峰值 `850,903,605,248` bytes（`22:26:21`）；后 `636,349,378,560` bytes
- `oom` / `oom_kill` / `oom_group_kill`：前、观测峰值、后均为 `0 / 0 / 0`
- `MemAvailable` 最低：`891,035,020 kB`；结束：`1,099,914,204 kB`
- system journal：`PERMISSION_LIMITED`，无法据 system journal 完成 Xid/OOM 审计
- GPU：8 × `NVIDIA L20D`，capability `(10, 3)`；启动前 compute-apps 为空

## 产物

`docs/sm100_megamoe/v41_gc_generate_results/` 包含：

- `model-full.log`、`model.log`：完整后端输出与 launcher 汇总
- `first-error.txt`：首个 `AssertionError` 的原始 traceback
- `adapter-install-evidence.txt`：4 条 adapter 安装记录及 PID/rank 映射
- `versions.txt`、`import.log`、`pip-freeze.log`、`install.log`、`config.log`、`index.sha256`
- `cgroup-access.txt`、`cgroup-summary.txt`、`memory.*`
- `gpu-before.txt`、`gpu-release.txt`、`gpu-release-verification.txt`
- `journal-probe.*`、`journal-access.txt`
- `source.sha`、`output-status.txt`、`reference-prefix-audit.json`、`reference-prefix-audit.sha256`
- `launcher.sh`、`launcher-actual.sh`
- `SHA256SUMS`

未提交模型权重、venv 或编译缓存。
