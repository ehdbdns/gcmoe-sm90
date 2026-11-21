# SM100 V4.1 DeepGEMM MegaMoE full-model generation

执行者：SM100。本轮未执行任何 SM90 任务书，也未混入 SM90 结果。

## 结论

功能验证通过。使用与原生基线相同的独立 venv、本地 48 分片权重、TP8/EP8 和相同四 token 请求，唯一主动改变的模型后端选项为 `moe_backend=deep_gemm_mega_moe`。模型成功加载 48/48 分片，实际走到 `deep_gemm_mega_moe`，生成恰好 4 个 token，进程和 launcher 退出码均为 `0`。

- 生成 ID：`[75, 200, 18, 200]`
- 原生 auto 基线：`[200, 19, 200, 18]`
- 首个不同位置：索引 `0`，`75` vs `200`
- 四个位置均不同；没有 bit-exact 或数值容差验收要求，本轮只记录差异
- checkpoint：`48/48`，`00:42`
- GPU KV cache：`8,258,440` tokens
- `init engine`：`43.71 s`
- 八卡释放：`2026-11-21T21:33:40+08:00`；`21:35:32` 复核仍为空

本轮不产生性能结论，不包含 EPLB 重平衡或 ECO 五臂数据。

## 后端与 shared expert

运行日志明确显示 `KernelConfig(... moe_backend='deep_gemm_mega_moe' ...)`，并从 vendored `vllm.third_party.deep_gemm` 导入 DeepGEMM。共享专家的融合判定为 **启用**：配置含 1 个 shared expert，TP8/EP8/PP1 满足 sequence-parallel 分支，禁用变量为 false，且已加载 DeepGEMM API 同时支持 `num_shared_experts`、`shared_l1_weights` 和 `shared_l2_weights`。完整日志中没有出现任何 `Disabling native MegaMoE shared-expert fusion` 降级警告。运行日志没有直接打印 `has_fused_shared_experts` 布尔值，证据链与限制见 `shared-expert-evidence.txt`。

## 资源与收尾

cgroup `memory.max=max`；`oom=0`、`oom_kill=0`、`oom_group_kill=0`。`memory.current` 从 `423,325,437,952` bytes 上升到峰值 `833,362,128,896` bytes，结束时为 `521,786,744,832` bytes。该值是整个 session cgroup 的计数，包含可回收页缓存，不是进程 RSS。

launcher 为 `setsid-detached`，`external_timeout=none`，没有外部定时终止。生成后 teardown 出现 force-kill/SIGTERM 告警，但输出 ID 已先产生且两个退出码都为 `0`。system journal 权限受限，因此没有从不可读日志推断 OOM/Xid。

## 文件索引

- `model-full.log`、`model.log`：完整模型日志与 launcher 合并输出
- `generated-output.json`、`output_ids.txt`：实际生成结果
- `backend-evidence.txt`、`shared-expert-evidence.txt`、`shared-expert-source.txt`：后端及 shared expert 证据
- `cgroup-summary.txt`、`memory.before`、`memory.log`、`memory.after`：资源计数
- `versions.txt`、`freeze.txt`、`install.log`、`config.log`、`source.sha`：环境与提交
- `gpu-before.txt`、`gpu-compute-before.txt`、`gpu-release.txt`、`late-gpu-release.txt`：八卡占用与释放
- `journal-access.txt`、`journal-probe.log`、`journal-probe.err`：journal 权限
- `launcher.sh`、`v41_native_smoke.py`：真实启动入口
- `source-hashes.txt`、`SHA256SUMS`：源码与产物校验
