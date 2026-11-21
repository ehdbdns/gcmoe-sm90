# 【SM100】V4.1-Flash 整模原生权重加载重试回执

执行者：**SM100**。本轮未执行任何 SM90 任务书，也未混入 SM90 结果。

## 结论

**验收失败。** 48/48 个 checkpoint 分片已从本机 XFS 盘完整加载，原生 MoE 后端已解析为 `FLASHINFER_TRTLLM_MXFP4_MXFP8`，但 `LLM(...)` 构造阶段未完成，进程退出码为 `1`，因此没有产生 4-token `generated_ids`。

- 任务提交：`15835680f58e0052d4cd16eba76d5757558dbede`
- 任务书 blob：`975473159a35dd7149c4ca02f3104915ee471304`
- 本回执基线：`956bb3fa16deb9472a1493c265e69fbba7ff0b55`
- 运行窗口：`2026-11-21T20:07:43+08:00` 至 `2026-11-21T21:09:14+08:00`
- 工作目录：`/data0/home/admin/workspace/megamoe/native-v41-KO1hTbTj`
- 独立 venv：`/data0/home/admin/workspace/megamoe/native-v41-stsFS4w7/venv`
- 最终退出码：`1`
- 首个致命日志：`2026-11-21T21:08:32+08:00`，`Worker proc VllmWorker-0 died unexpectedly (exit code: None), shutting down executor.`
- 输出：无；`generated_ids=none`
- 八卡首次确认释放：`2026-11-21T21:09:16+08:00`；`21:12:24` 复核仍为空

## 失败链与首个错误

按时间顺序，原始日志中的首个异常链为：

```text
2026-11-21 21:08:19  No available shared memory broadcast block found in 60 seconds. This typically happens when some processes are hanging or doing some time-consuming work (e.g. compilation, weight/kv cache quantization).
2026-11-21 21:08:21  Worker_TP2_EP2: WorkerProc handling signal 15, raising SystemExit
2026-11-21 21:08:21  Worker_TP3_EP3: WorkerProc handling signal 15, raising SystemExit
2026-11-21 21:08:32  ERROR Worker proc VllmWorker-0 died unexpectedly (exit code: None), shutting down executor.
2026-11-21 21:08:37  WARNING [shutdown] sending SIGTERM count=7
2026-11-21 21:08:41  WARNING [shutdown] sending SIGKILL count=6
2026-11-21 21:09:14  END EXIT=1
```

父进程最后报告：

```text
RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}
```

`VllmWorker-0` 的原始 Python traceback 没有出现在日志中；日志只给出进程退出码 `None`。两台 worker 在 `21:08:21` 收到信号 15 并抛出 `SystemExit`，而引擎自己的关闭信号直到 `21:08:37` 才发出，晚于首个 worker 死亡。因此 `21:08:37` 之后的 SIGTERM/SIGKILL 是关闭过程的次生事件，不能作为根因。

## 外部终止与时间包络

- launcher 使用 `setsid` 分离，未使用 `timeout`；`versions.txt` 明确记录 `launcher=setsid-detached`、`external_timeout=none`。
- 八个 worker 于 `20:08:22` 启动；首个信号 15 出现在 `21:08:21`，相差 `3599 s`。
- systemd scope `session-1882009.scope` 的 `RuntimeMaxUSec` 未设置，`MemoryMax=infinity`，`TasksMax=infinity`。
- 首次 SIGTERM 早于引擎正常关闭路径，来源在本次 launcher/model 进程之外；由于 system journal 与 Audit 读取权限不足，无法进一步确认发送者。
- 因此本轮没有“执行器主动加了外部定时终止”；但运行实际在约 1 小时处被外部信号截断。

系统日志访问情况：

```text
system_journal_access=PERMISSION_LIMITED
journal_probe_bytes=0
journal_stderr=Hint: You are currently not seeing messages from other users and the system.
ausearch_config=Permission denied
ausearch_log=/var/log/audit/audit.log: Permission denied
coredump_readable=no_coredumps_or_permission_limited
```

没有从不可读的 journal 中推断 OOM 或 Xid；这些证据应标记为不可得。

## 权重、加载与内存

源权重在 NFS4，工作目录在 XFS。按任务书在空间充足时顺序复制到本次专用目录，未覆盖或清理既有文件：

- 模型：`/mnt/nas1/hf/DeepSeek-V4.1-Flash`
- 本机副本：`/data0/home/admin/workspace/megamoe/native-v41-KO1hTbTj/model`
- 复制窗口：`2026-11-21T19:45:42+08:00` 至 `2026-11-21T20:07:42+08:00`
- 复制大小：`510313473710` bytes；`48` 个 safetensors 分片
- `config.json` SHA-256：`8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879`
- 权重索引 SHA-256：`74b0686a3d2891980d5e303251b075a3bccae2c2ff650747db2620a649b98fa8`
- 配置：`hidden_size=5120`、`moe_intermediate_size=2304`、`n_routed_experts=384`、`num_experts_per_tok=6`、`num_hidden_layers=40`
- checkpoint 加载：`48/48`，最后进度为 `[06:18]`
- 每 rank memory profiling：`657.5 s`；weights `36.12 GiB`；non-KV memory `38.21 GiB`；KV cache `175.94 GiB`
- GPU KV cache：`8,369,325` tokens；2,048-token 请求的理论并发为 `4086.58x`

cgroup 为 `/sys/fs/cgroup/user.slice/user-1384661.slice/session-1882009.scope`：

| 项目 | 前 | 峰值 | 后 |
|---|---:|---:|---:|
| `memory.current` (bytes) | 591,567,134,720 | 977,049,341,952 (`20:16:13`) | 345,279,668,224 |
| `memory.max` | `max` | `max` | `max` |
| `oom` / `oom_kill` / `oom_group_kill` | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |

`memory.current` 包含可回收页缓存；cgroup 没有内存限额或 OOM 事件。`memory.log` 共 123 个采样点。

## 实际 native MoE 后端

`moe_backend='auto'` 实际解析为：

```text
Mxfp4 MoE backend 'FLASHINFER_TRTLLM_MXFP4_MXFP8' does not support the deployment configuration since kernel does not support routing method 100.
Using 'FLASHINFER_TRTLLM_MXFP4_MXFP8' Mxfp4 MoE backend.
```

同时：

- `routing_method=<RoutingMethodType.DeepseekV4: 100>`
- `ep_size=8`、`use_ep=True`、`all2all_backend='allgather_reducescatter'`
- 每 rank 48/384 个专家
- FlashInfer MNNVL all-reduce workspace 在八个 rank 上初始化成功

后端支持性警告与实际选择均原样保留；未更换后端、模型、卡数或框架版本。

## JIT 与证据保留

- 首次 FlashInfer cubin fetch：`20:23:46`
- 最后一次成功下载：`21:08:12`
- `Fetching ... cubin` 452 行，成功下载 422 行，缓存锁失败 85 行
- 观察到 60 个 cubin 文件，FlashInfer cache 约 128 MiB
- 锁竞争、重复下载和 cubin 获取处于非致命状态，直到 `21:08:32` worker 死亡

这些行完整保存在 `jit-download-evidence.txt`；原始全量 stdout/stderr 保存在 `model-full.log`。

## 环境

- vLLM `0.31.0`
- torch `2.13.0+cu130`，CUDA `13.0`
- flashinfer `0.7.0.post1`
- transformers `5.17.0`
- triton `3.7.1`
- Python `3.12.7`
- wheel SHA-256：`39132b433967f0962d03d8a5f633b9bee6b32708f566c7d5637b79f8d0ef3433`
- 8 × `NVIDIA L20D`，每卡 `275040 MiB`，capability `(10, 3)`
- `CUDA_VISIBLE_DEVICES` 使用 `ENV.json` 登记的八个 UUID；`gpu-compute-before.txt` 为空

## 产物

目录：`docs/sm100_megamoe/v41_native_retry_results/`

- 结论与时间线：`status.txt`、`timeline.txt`、`first_error.txt`、`output_ids.txt`
- 环境与后端：`environment.txt`、`versions.txt`、`backend_evidence.txt`、`gpu-before.txt`、`gpu-capability.txt`
- 终止与资源：`external_termination.txt`、`cgroup-evidence.txt`、`cgroup-scope.txt`、`memory.before`、`memory.log`、`memory.after`、`gpu-release.txt`
- JIT：`jit-download-evidence.txt`、`jit-cache-summary.txt`
- 日志：`model-full.log`、`model.log`、`console.log`、`launcher.sh`、`v41_native_smoke.py`
- 权重与安装：`copy.log`、`copy-progress.log`、`copied-checkpoints.txt`、`source.sha`、`config.log`、`install.log`、`install.pypi-public-aborted.log`、`freeze.txt`
- 文件系统与权限：`source-filesystem.txt`、`load-filesystem.txt`、`disk-before.txt`、`disk-after-copy.txt`、`disk-decision.txt`、`cgroup-access.txt`、`journal-access.txt`、`journal-probe.log`、`journal-probe.err`、`journal-probe-extended.txt`
- `SHA256SUMS`

未提交模型权重或 JIT/编译缓存。

## 判定

任务未通过：整模入口和 48/48 权重加载已完成，但没有到达 `LLM.generate()`，四条 token ID 不存在。失败发生在约一小时处的 worker 终止，且早于任何可归因于模型或 MoE 后端的断言；外部信号来源需在日志权限更高的条件下继续确认。
