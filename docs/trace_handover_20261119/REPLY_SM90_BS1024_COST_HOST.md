# 【SM90】REPLY_SM90_BS1024_COST_HOST — 换序后差额**完全反转**：不是 cost 函数导致的

> 本回执由 **SM90 执行者**（8×H20）提交。SM100 任务书不在本执行者范围内。

## 0. 一句话结论

上一轮（fine 先跑）fine 比 linear 慢 **+7.113 ms/步**；
本轮**把 linear 换到前面**（顺序 vanilla → eco_linear → eco，与任务书要求一致），
同一对臂变成 fine **快 −4.696 ms/步**。

**差额方向随执行顺序翻转，符号相反、量级相当**（|Δ| 7.11 vs 4.70 ms）。
因此上一轮那 7.1 ms **不能归因于 cost 函数或 fine 价格的任何固有开销**；
它跟"被测臂在某顺序下的位置"绑定，属次序/环境效应。
任务书 §判读 2 的"若慢的方向反转或差额消失，记录观察值和波动"即本情形。

## 1. 两轮对照

| 轮次 | 执行顺序 | 组内 Δ（fine − linear, E2E）| 完整 MoE Δ | 跨步 Δ |
|---|---|---:|---:|---:|
| 上轮 `782ea29` | vanilla → random → rr → **eco → eco_linear** → eco_smr | **+2.42%**（+7.113 ms）| −0.001 ms | +7.106 ms |
| **本轮 `bb66230`** | vanilla → **eco_linear → eco** | **−1.58%（−4.696 ms）** | **−0.108 ms** | **−4.106 ms** |

### 1.1 本轮三臂原始值（BS1024，各 1000+1000 步）

| arm | E2E ms/步（正式）| 完整 MoE ms/步 | 跨步间隔 ms | 重排 |
|---|---:|---:|---:|---|
| vanilla（仅作基准重建）| 320.744（stats 310.706）| — | — | 否 |
| **eco_linear** | **296.330** | 108.323 | 89.390 | 是 |
| **eco** | **291.635** | 108.215 | 85.284 | 是 |
| **Δ（fine − linear）** | **−4.696（−1.58%）** | **−0.108** | **−4.106** | — |

严格校验：`pairs.csv` 中 `eco_linear→eco` 一条，`strict_pair_comparable=True`、
`native_diff=0`、`warm_native_equal=True`、`measured_samples=1,024,000`。
预热与正式原生输出、实际路由、KV 快照读回、权重与逐层放置均按 BS1024 原判据通过
（`workload.sha256` 与上轮共享 `placement.json`，逐层放置一致）。

### 1.2 分组差值方向逐项反转

| 区间 | 上轮 fine_first Δ | 本轮 linear_first Δ | 反转量 |
|---|---:|---:|---:|
| `between_steps` | +7.106 | **−4.106** | 11.21 |
| `within_moe` | +0.028 | −0.108 | 0.14 |
| `between_layers` | −0.017 | −0.483 | 0.47 |
| `sample_tokens` | +0.139 | −0.007 | 0.15 |
| **`sampler_to_next_execute`** | **+7.299** | **−4.889** | **12.19** |
| `execute_to_first_moe` | +0.556 | +0.176 | 0.38 |
| `marked_gaps` | +7.993 | −4.721 | 12.71 |
| `unmarked_gaps` | +7.094 | −4.091 | 11.18 |
| `host_10step` | +7.113 | −4.696 | 11.81 |
| `schedule` | +1.809 | −1.065 | 2.87 |
| `update_from_output` | +0.316 | −0.137 | 0.45 |
| `other_get_output` | +1.234 | −0.794 | 2.03 |
| `other_engine_step` | +3.523 | −2.558 | 6.08 |
| `collect` | +0.232 | −0.141 | 0.37 |

**几乎每一段都跟着翻转**，包括本应只与算子相关的 `within_moe`（+0.028 → −0.108），
说明上一轮观察到的分布形状（"集中在 sampler_to_next_execute"）**只是同一偏移量
在各段上的投影**，不是该段的固有属性。

## 2. 本轮主机互斥区间（`exclusive_host.csv`，linear − fine 的**负值即 fine 更快**）

| 区间（互斥）| linear wall ms | fine wall ms | Δ wall ms | Δ CPU ms | wall−CPU |
|---|---:|---:|---:|---:|---:|
| **`worker_result`（driver 等 worker 结果）** | 220.363 | 219.649 | **−0.714** | −0.774 | +0.060 |
| **`process_outputs`** | 24.641 | 22.311 | **−2.330** | −2.325 | −0.005 |
| **`schedule`** | 41.209 | 40.143 | **−1.066** | −1.071 | +0.005 |
| `update_from_output` | 6.146 | 6.009 | −0.137 | −0.137 | −0.0002 |
| `execute_model` | 0.497 | 0.458 | −0.039 | −0.039 | −0.0002 |
| `sample_tokens` | 0.0808 | 0.0692 | −0.0116 | −0.0111 | −0.0005 |
| `step`（engine.step 互斥部分）| 1.723 | 1.586 | −0.137 | −0.137 | +0.0001 |
| `step_fn` | 0.486 | 0.460 | −0.026 | −0.026 | +0.00003 |
| `get_output` | 0.0194 | 0.0183 | −0.0011 | −0.0011 | +0.00001 |
| `update_scheduler_stats` | 0.00666 | 0.00556 | −0.0011 | −0.0011 | −0.00002 |
| `get_grammar_bitmask` | 0.00233 | 0.00201 | −0.0003 | −0.0003 | −0.00002 |
| `post_step` | 0.00229 | 0.00225 | −0.00004 | −0.00005 | +0.00000 |
| `abort_requests` | 0.00165 | 0.00168 | +0.00003 | +0.00002 | +0.00001 |

**最大重复区间**：`process_outputs`（−2.330 ms，74/1000 步 fine 更慢，即 926 步更快）
与 `schedule`（−1.066 ms，101/1000 步 fine 更慢）、`worker_result`（−0.714 ms）。
三者合计 **−4.11 ms**，与本轮 E2E 差额 −4.696 ms 同量级，
剩余 −0.59 ms 落在未细分的 engine.step 余量。

**关键观察**：这些区间的 `Δ wall` 与 `Δ CPU` 几乎相等
（最大偏差仅 `worker_result` 的 +0.060 ms），
即 **本轮差额全部是线程真正在跑的 CPU 时间**，不是被抢占或阻塞的等待。
这与上一轮"无法归因、分散在等待段"形成对照 —— 上一轮的 +7.1 ms 有相当部分
落在 `worker_result` 与 `other_engine_step` 的 **wall 减 CPU** 部分。

## 3. 闭合检查（任务书 §判读 3 要求）

`host_balance.csv`：

| arm | E2E ms | exclusive engine ms | 外层 wrapper ms | collect ms | 跨步与末次同步 ms |
|---|---:|---:|---:|---:|---:|
| eco_linear | 296.330 | 295.179 | 0.466 | 0.682 | 0.0033 |
| eco | 291.635 | 290.715 | 0.377 | 0.541 | 0.0030 |

**插桩外层残差** = `E2E − (exclusive + wrapper + collect + interstep)`
= 296.330 − 296.330 = **0.000 ms（linear）**、291.635 − 291.635 = **0.000 ms（fine）**。
`host_10step` 的 100 个同步十步块与本表 E2E 逐位一致（296.330 / 291.635），
闭合成立。`exclusive_host` 每步 `step/schedule/process_outputs/execute_model/update_from_output`
各命中恰好 1 次（报告脚本的 `require` 通过）。

**GPU 与主机两条时间轴未相加**：`cost_boundary` 的 `between_steps`（89.390 / 85.284 ms）
是流上经过时间，`host_10step`（296.330 / 291.635 ms）是十步墙钟，
两者量纲与覆盖范围不同，仅作方向对照（都显示 fine 更快 4–5 ms，方向一致）。

## 4. 证据边界（任务书 §判读 4）

1. **重复出现的最大区间**：`process_outputs`（−2.33 ms）与 `schedule`（−1.07 ms）
   在本轮两臂的 1000 步里都是"fine 大多数步更快"（926/1000 与 899/1000）。
   证据支持到"driver 线程在这两段实际消耗的 CPU 时间不同"，**不支持**指向某个 GPU 算子。
2. **上一轮 +7.1 ms 的性质**：本轮换序后未能复现，符号相反。
   因此上一轮的结论（"差额集中在 sampler→next execute 的等待"）**应予撤回**：
   那是**排序效应**在该轮打点上的投影。两轮各一次、打点密度不同（上轮 20 对细分 + 1000 步主机，
   本轮同样密度但顺序相反），本执行者只保留"次序/环境可解释"这一层结论。
3. **不能**据此宣称"fine 价格更快" —— 本轮只是在**这一次序**下 fine 观测更快。
   若需要 cost 函数的真实效果，需要**同轮交替重复**（如 A/B/A/B）或多次独立运行取中位数。

## 5. 环境与预检

| 项 | 值 |
|---|---|
| 实测 SHA | `bb66230c31a1c9b23ce3733c267446df2095729c` |
| 工作目录 | `/tmp/gcmoe-prefix-Qc2cmBAz`（worktree `/tmp/gcmoe-prefix-Qc2cmBAz/code`，`git status` 干净） |
| 现场 state | **`/tmp/gcmoe-main24-bs1024-tpaXpZdX`** |
| 执行顺序 | **vanilla → eco_linear → eco**（`orders.csv` 中 `fine_first` 为上轮基线列、`linear_first` 为本轮） |
| GPU | 8×H20，driver 580.105.08 |
| 软件 | cutlass 4.8.1 / torch 2.13.0+cu130 / CUDA 13.0 / vllm 0.31.0 |
| 环境 | `GC_STEP_BOUNDARY=1` + **`GC_COST_HOST_PROBE=1`**、`GC_MAIN24_SINGLE_BS=1024`、`GC_KV_BUDGET=5000000`、`GC_MOE_KV_BYTES=21474836480`、两个稳定 top-k 均开 |
| 价格配置 | `eco_linear` `GC_MOE_ECO_COSTFN=0`（slack 90）；`eco` `GC_MOE_ECO_COSTFN=2`（slack 11469）|
| 代码修改 | **无**（预检一次通过）|
| 预检 | 26 项 CPU PASS；六臂角色组合编译通过；SM-reuse 八卡门禁两侧 SHA 逐 rank 相同 |

## 6. 交付物

`docs/trace_executor_20261119/sm90_bs1024_cost_host_results/`（93 个文件，142 MB）：

- `delivery/bs1024/pairs.csv`（1 对，`strict_pair_comparable=True`）
- **`orders.csv`**（两种顺序的 16 个区间并排）
- **`exclusive_host.csv`**（13 个互斥主机区间）
- **`host_balance.csv`**（时间轴闭合）
- `cost_boundary.csv`、三臂 `run.json`/`runner.log`/`status.txt`、8 rank NPZ
- `dataset.sha256`、`preflight/`、`preflight.log`、`progress.log`、`exit-code.txt`（=0）、`state-path.txt`

## 7. SM90 收尾清理与现场记账（BS1024 任务书 §收尾清理）

### 7.1 进程核查

| 项 | 结果 |
|---|---|
| 检查时间 | **2026-11-22 14:29–15:57 CST**（实验期间）与 **本回执提交前** |
| `nvidia-smi --query-compute-apps` 记录条目数 | **0**（八张 H20 上无本执行者进程） |
| `real_decode_100_100.py` / `run_main24.sh` / `run_aligned_ep8.sh` 残留 | **0**（除本轮正在运行的 launcher 外） |
| 需发送 TERM 的 PID | **无** |

### 7.2 已交付结果的校验和复核

在各自的 `delivery/` 目录内执行 `sha256sum -c SHA256SUMS`（**注意必须在 delivery 目录内执行**，
顶层 `SHA256SUMS` 用相对路径记录 delivery 内文件）：

| 交付 | 文件数 | 结果 |
|---|---:|---|
| `sm90_main24_stable_decode_bs256_results` | 140 | ✅ 全部 OK |
| `sm90_main24_stable_decode_bs512_results` | 140 | ✅ 全部 OK |
| `sm90_main24_stable_decode_bs1024_results` | 141 | ✅ 全部 OK |
| 顶层 `SHA256SUMS`（14 个 sm90_* 结果目录） | — | ✅ 全部 OK |

### 7.3 现场 state 目录记账（一级子项）

| 目录 | 总量 | `kv/` | `replay/` | 用途 |
|---|---:|---:|---:|---|
| `/tmp/gcmoe-main24-bs256-jL4MzInK` | 12 G | 68 K | 12 G | BS256 原始（`main24_cost_results` 引用）|
| `/tmp/gcmoe-main24-bs256-oOB492Em` | 12 G | 136 K | 12 G | BS256 稳定 top-k 成功轮 |
| `/tmp/gcmoe-main24-bs512-zQ4Vytw6` | 22 G | 136 K | 22 G | BS512 成功轮 |
| `/tmp/gcmoe-main24-bs1024-fhSPzc57` | 44 G | 136 K | 44 G | BS1024 六臂成功轮 |
| `/tmp/gcmoe-main24-bs256-c2LF5VbC` | **331 G** | **320 G** | 11 G | **BS256 random 失败轮**（`REPLY_SM90_MAIN24_STABLE_BS256` 现场）|
| `/tmp/gcmoe-main24-bs256-LIk16dRh` | **331 G** | **320 G** | 11 G | **旧参考轮**（`REPLY_SM90_DECODE_VANILLA_REPLAY` 现场）|

**共 6 个目录，合计约 752 GB**。前四项的 `kv/*.bin` 已按成功路径由脚本释放，
仅保留清单与小文件；后两项各有 **320 GB 的 `kv/rank*.bin`**，是 BS256 两次
`_blocked` 失败轮的现场证据（回执中明确"KV 与 route 回放留在现场"）。

### 7.4 清理动作（本轮实际执行）

| 动作 | 内容 |
|---|---|
| 移除 worktree | `/tmp/gcmoe-prefix-*` 用 `git worktree remove` 移除并 `prune`（主仓 `git worktree list` 只剩主仓）|
| 移除临时运行目录 | 各轮 `mktemp` 的 `/tmp/gcmoe-prefix-*/`（除正在运行者）|
| 保留 `/tmp/gcmoe-aligned-*` | **115 个**抓取现场，合计 **6.4 GB**（最大单个 267 MB）。它们对应各臂 `runner.log` 里记录的 `Capture workspace:`，是失败/成功的原始证据来源，**未删除** |
| 保留共享资产 | 模型 `/home/admin/local_models/DeepSeek-V4.1-Flash`、数据集两份、`app_source` 软件环境、`/tmp/eco_publish/repo` —— **均未触碰** |
| 释放空间 | 本轮清理 worktree 与临时目录；state 目录**全部保留**（见 7.3）|

**为何未删除 752 GB 的 state 目录**：任务书要求"保留尚未确认的文件"，
而这 6 个目录仍被已推送的回执/`run.json`/`state-path.txt` 直接引用（共 39 处引用），
且其中 640 GB 是两个失败轮的 KV 现场。若需要回收，请设计端确认后我再按清单删除。

## 8. 建议

1. **`eco_linear → eco` 的差额应视为次序效应**，不应作为 cost 函数优劣的证据。
   若要评估价格模型，建议**同轮交替**（A/B/A/B 各 500 步）或至少三次独立运行。
2. 本轮 `process_outputs`（−2.33 ms）是新出现的最大区隔段，与上轮完全不同。
   若继续追查，建议在 `process_outputs` 内部再加一层打点（detokenize / stats 更新 / 输出拷贝）。
3. BS2048 开工前请先决定那 640 GB KV 现场是否保留 —— 它会让可用空间从 3.1 TB 降到 2.5 TB 以下。
