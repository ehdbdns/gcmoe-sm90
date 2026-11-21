# 【SM100】8 卡 V4.1-Flash 整模原生 smoke 回执

执行者：SM100。未执行任何 SM90 任务书，也未混入 SM90 结果。

## 结论

**验收失败。** 精确 V4.1 整模入口已解析，原生 MoE 后端已解析，八卡 worker 已启动并加载到 32/48 个 checkpoint 分片，但最终进程退出 1，未到达 `LLM.generate`，因此没有 4-token 输出 ID。

- 任务提交：`8f4254a6339a9fe70e439ff37103d8c3a80146e4`
- 任务书 blob：`ed8a6c48f07f32d01b2204dbcbdc9aab0699674b`
- 本回执基线：`0dd9579b0abbe8aefab8bfaa57d68defdd2cba93`
- 运行窗口：`2026-11-21T18:33:08+08:00` 至约 `19:34:13+08:00`
- 最终退出码：`1`
- checkpoint 最后进度：`32/48`，已运行 `57:12`
- `generated_ids`：未产生

## 两次启动与首个错误

### 第一次：任务书原始 heredoc

按任务书用 `python - <<'PYTHON'` 启动。架构已解析为 `DeepseekV41ForCausalLM` 后，vLLM 因 CUDA 已初始化而强制使用 `spawn`；子进程尝试把 `/tmp/sm100-native-20261121-175023/<stdin>` 当作真实 Python 文件重新导入，随即失败：

```text
FileNotFoundError: [Errno 2] No such file or directory: '/tmp/sm100-native-20261121-175023/<stdin>'
RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {'EngineCore': 1}
```

这是启动入口形式导致的错误，不是权重、GPU、编译或后端错误。原始完整日志保存在 `model.heredoc-failure.log`；外层 `set -eo pipefail` 下退出码为 1。

### 第二次：等价文件入口

仅把入口从 stdin 改为真实文件 `v41_native_smoke.py`，使 vLLM 的 `spawn` 可以重新导入；**模型、tokenizer、dtype、TP8/EP8、贪心采样、4 token、eager 模式、`gpu_memory_utilization`、自动 MoE 后端均未改变**。脚本内容随回执提交。

该次运行在 `19:33:49` 开始关闭 executor；当时 checkpoint 进度为 32/48（57:12）。首个可见终态异常序列为：

```text
[shutdown] Executor: waiting for worker exit count=8
[shutdown] Executor: workers still running after grace period; sending SIGTERM count=8
[shutdown] Executor: workers still running after SIGTERM; sending SIGKILL count=6
EngineCore failed to start.
Exception: WorkerProc initialization failed due to an exception in a background process. See stack trace for root cause.
RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}
```

vLLM 没有打印出发异常 worker 的 Python traceback，父进程只报告 background worker 异常。未观察到 OOM、NCCL、Xid 或编译错误文本；也**不能**在没有证据的情况下把根因归为这些类别。原始完整日志保留在 `model.log`。最终退出码为 `1`。

## 实际解析结果

- 模型：`/mnt/nas1/hf/DeepSeek-V4.1-Flash`，约 476 GiB，48 个 safetensors 分片。
- 架构：`DeepseekV41ForCausalLM`；tokenizer mode：`deepseek_v41`。
- 配置：`hidden_size=5120`、`moe_intermediate_size=2304`、`n_routed_experts=384`、`num_experts_per_tok=6`、`num_hidden_layers=40`；`config.json` SHA-256 为 `8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879`。
- 权重索引 SHA-256：`74b0686a3d2891980d5e303251b075a3bccae2c2ff650747db2620a649b98fa8`。
- 声明量化：`deepseek_v4_fp8`；expert dtype：`fp4`。
- MXFP8 GEMM：`FlashInferCutedslMxfp8LinearKernel`。
- KV cache：`DeepSeek's nvfp4_ds_mla`。
- EP 布局：8 个 EP rank，每 rank 48/384 个专家，linear placement。
- Model runner：V2；`enforce_eager=True`，CUDAGraph 关闭。
- 请求的 `moe_backend='auto'` 实际解析为 **`FLASHINFER_TRTLLM_MXFP4_MXFP8`**。

因此，本次 native 入口**没有**走到 DeepGEMM MegaMoE；不能把本回执当作 MegaMoE 集成、数值一致性或性能结论。

## 安装与设备

- vLLM：`0.31.0`
- torch：`2.13.0+cu130`，`torch.version.cuda=13.0`
- flashinfer-python：`0.7.0.post1`
- transformers：`5.17.0`
- triton：`3.7.1`
- wheel：`vllm-0.31.0-cp38-abi3-manylinux_2_28_x86_64.whl`
- wheel SHA-256：`39132b433967f0962d03d8a5f633b9bee6b32708f566c7d5637b79f8d0ef3433`，与公开 PyPI 该 wheel 的摘要一致。

安装来源需要明确：公开 PyPI 文件端点在当前机器上仅约 17 KB/s，安装尝试中止；随后改用清华镜像 `https://pypi.tuna.tsinghua.edu.cn/simple` 获取 **同一个 wheel**，并对实际文件重算 SHA-256 后确认与 PyPI 摘要一致。未安装不同版本或本地修改过的 vLLM。

八张 `NVIDIA L20D` 均报告 capability `(10, 3)`，每卡 275040 MiB。`CUDA_VISIBLE_DEVICES` 使用 `docs/sm100_megamoe/ENV.json` 中登记的 8 个唯一 UUID。运行中八个 `VLLM::Worker_TP0_EP0` 至 `TP7_EP7` 各占约 40058 MiB。最后一次确认全组释放的时间为 `2026-11-21T19:35:34+08:00`，`nvidia-smi --query-compute-apps` 已为空；`19:36:25` 复核仍为空。

## 产物

`docs/sm100_megamoe/v41_native_results/`：

- `status.txt`、`timeline.txt`、`exit-code.txt`、`first_error.txt`、`output_ids.txt`
- `versions.txt`、`backend_evidence.txt`、`gpu_allocation.txt`
- `install.log`、`install.pypi-public-aborted.log`、`freeze.txt`、`config.log`
- `model.log`、`model.heredoc-failure.log`、`console.log`
- `source.sha`、`v41_native_smoke.py`、`README.md`
- `SHA256SUMS`

未提交模型权重、wheel 本体或编译缓存。未进行第三次重跑，避免在 worker 根因未明时重复占用八卡。
