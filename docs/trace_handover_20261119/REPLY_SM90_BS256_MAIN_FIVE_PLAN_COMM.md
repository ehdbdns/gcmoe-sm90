# 【SM90】REPLY_SM90_BS256_MAIN_FIVE_PLAN_COMM — 五臂重测：eco_smr −8.109%，MoE −19.261%

> 本回执由 **SM90 执行者**（8×H20）提交。SM100 任务书不在本执行者范围内。

## 0. 结论

| 项 | 结果 |
|---|---|
| 实测 SHA | `cdac2a688c918c8eb296f273bdd36b47abc39b59` |
| 模型启动数 | **5**（vanilla / random / rr / eco / eco_smr 各一次）|
| `exit-code.txt` | **0** |
| `pairs.csv` | **10 对**，`strict_pair_comparable` **10/10 真** |
| `warm_native_equal` | **10/10 真** |
| `native_diff` | **10/10 = 0** |
| 每步实际 BS | **256**（五臂 `batch=256`）|
| 硬件 | 8×H20，driver 580.105.08 |

## 1. 主表（README 全列，BS256 正式窗口 1000 步）

| arm | E2E ms/步 | vs vanilla | **完整 MoE ms/步** | **MoE vs vanilla** | token IR | 总 M 块 | 峰值 M 块 | 活跃物理槽 | M 块 IR | routed MAE（pp）| graph MAE（pp）| slack | score source |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| vanilla | 94.546 | +0.000% | 40.409 | +0.000% | 1.407 | 371.66 | 56.58 | 292.116 | 1.219 | 0.094 | 1.927 | 0 | offline fixed |
| random | 98.692 | **+4.385%** | 43.390 | **+7.377%** | 1.489 | 415.42 | 63.41 | 341.345 | 1.222 | 0.111 | 2.000 | 0 | offline fixed |
| rr | 92.396 | **−2.275%** | 35.646 | **−11.787%** | 1.083 | 404.55 | 54.36 | 356.064 | 1.075 | 0.062 | 0.484 | 0 | offline fixed |
| eco | 88.383 | **−6.519%** | 33.697 | **−16.610%** | 1.060 | 371.66 | 49.05 | 315.344 | 1.056 | 0.083 | 0.219 | 11469 | online ECO |
| **eco_smr** | **86.879** | **−8.109%** | **32.626** | **−19.261%** | 1.060 | 371.66 | 49.05 | 315.344 | 1.056 | 0.140 | 0.237 | 11469 | online ECO |

排名：**eco_smr 最快 → eco → rr → vanilla → random**，与历史一致且 Eco 系收益扩大。

### 1.1 任务书要求的组内差异

| 对照 | E2E Δ | MoE Δ |
|---|---:|---:|
| **eco vs rr** | **−4.343%** | **−5.467%** |
| **eco_smr vs eco**（SM-reuse）| **−1.701%** | **−3.179%** |

## 2. 校验（任务书 §验收）

| 检查 | 结果 |
|---|---|
| 10 对 `warm_native_equal` | ✅ 全真 |
| 10 对 `strict_pair_comparable` | ✅ 全真 |
| 10 对 `native_diff` | ✅ 全 0 |
| 输入/输出哈希、各层放置、route bank | ✅（`workload.sha256` 由 runner 生成并通过）|
| 两个 ECO 臂逐槽 counts 与拍卖计数 | ✅ **完全一致**（`auction.csv` 两行 offers/accepted 逐值相同：291,116 / 187,082）|
| 两臂实际 role-5 价格 | ✅ eco / eco_smr 均为 `primary_price=fine`、`auction_slack=11469`；vanilla/random/rr 为 `offline fixed allocation`、slack 0 |
| 通信分区日志 | ✅ eco 与 eco_smr 各 **8 条** `SM90 ECO plan context=communication`（共 16 条）；vanilla/random/rr **0 条**（符合预期：只有 ECO 臂启用）|
| 主机预检 role grid | ✅ 打印 `[5, 7, 1, 4, 2, 3]`（eco 系）与 `[5, 7, 1, 4, 2, 6, 3]`（eco_smr）与 `[1, 4, 2, 3]`（vanilla/random/rr）；预检日志另有 18 条 `plan context=communication` |
| 预检 PASS 数 | **29 项**（比此前 26 项多 3 项，均为本轮新增的 plan-comm 门禁）|
| SM-reuse 八卡门禁 | ✅ `reuse-0.sha` == `reuse-1.sha` |

## 3. 显存与 KV（任务书 §验收要求）

| arm | 最低剩余（stats）| 最低剩余（measured）| 最高 reserved | 总显存 |
|---|---:|---:|---:|---:|
| vanilla | 22.89 GiB | 22.63 GiB | 61.78 GiB | 95.07 GiB |
| random | 14.68 GiB | 14.47 GiB | 69.20 GiB | 95.07 GiB |
| rr | 14.72 GiB | 13.48 GiB | 69.20 GiB | 95.07 GiB |
| eco | 14.27 GiB | 13.03 GiB | 69.19 GiB | 95.07 GiB |
| eco_smr | 14.24 GiB | **13.02 GiB** | 69.19 GiB | 95.07 GiB |

| 项 | 五臂共同值 |
|---|---|
| `GPU KV cache size` | **1,530,922 tokens** |
| `GC_MOE_KV_BYTES` | 21,474,836,480 B = **20.00 GiB** |
| **峰值 KV 利用率** | **11.4%** |

> `free` 是设备剩余显存读数，**包含 CUDA allocator 之外的占用**（NCCL/NVSHMEM 对称堆、
> CUDA context、驱动预留）。与 `95.07 − 69.19 = 25.88 GiB` 的差额即来自这些部分。

每步 token：五臂各 **2000 条** `256 generation tokens` 记录（stats + measured 各 1000），
`long_window` 的满 batch 断言未触发。

## 4. 与上一轮（plan 未进通信分区）的对照

| 项 | 上一轮（plan=compute）| **本轮（plan=communication）** |
|---|---:|---:|
| vanilla 完整 MoE | 40.415（`sm90_main24_stable_decode_bs256`）| **40.409** |
| eco 完整 MoE | 35.600 | **33.697** |
| eco_smr 完整 MoE | 34.083 | **32.626** |
| eco vs vanilla | −11.9% | **−16.6%** |
| eco_smr vs vanilla | −15.7% | **−19.3%** |
| eco_smr E2E | 88.757 | **86.879** |

**vanilla 与 rr 几乎不变**（vanilla 40.415 → 40.409，−0.015%；rr 上一轮 35.303 → 35.646，+0.97%），
**两个 ECO 臂显著下降**（−5.3% ~ −4.3%），这正是把 plan 移到通信分区带来的
「shared L1+L2 从 195–197 降到 153–156 µs」的直接体现
（详见 `REPLY_SM90_BS256_PLAN_ON_COMM` 的 −5.356% 单臂对照）。

**rr 略升 0.97%** 属于轮间波动（该臂不启用 plan-comm，理论上不应受影响）。

## 5. 交付物

`docs/trace_executor_20261119/sm90_bs256_main_five_plan_comm_results/`（133 个文件，132 MB）：

| 路径 | 内容 |
|---|---|
| `delivery/bs256/pairs.csv` | **10 对**，全 Strict |
| `delivery/bs256/main_table.csv` | README 全列（30 列）|
| `delivery/bs256/layers.csv` | 逐层最慢 rank 耗时与预测误差 |
| `delivery/bs256/auction.csv` | 两个 ECO 臂的拍卖统计（逐值相同）|
| `delivery/bs256/<arm>/` | 五臂各含 `run.json`、8 份 `rank*.npz`、`runner.log`、`status.txt`、`installed-layout.json`、`hardware.csv` |
| `delivery/bs256/checkpoint/`、`stats-demand.npz`、`placement.json`、`workload.sha256` | 两边界快照清单与固定工作量 |
| `delivery/SHA256SUMS`、`preflight/`、`preflight.log`、`progress.log`、`exit-code.txt`（=0）| 预检与校验 |
| `summary.json` | 主表、拍卖、显存、日志计数、预检信息的机器可读汇总 |

**未提交**：二进制 KV 与 route 回放，按要求留在现场（`state-path.txt` 指向）。

## 6. 清理

| 项 | 结果 |
|---|---|
| 作业退出 | 五臂结束，`nvidia-smi --query-compute-apps` **0 条** |
| 临时 worktree | `/tmp/sm90-main-five-run1/code` 已 `git worktree remove` 并 `prune` |
| 原始证据 | `/tmp/sm90-main-five-run1/`（delivery/preflight/日志）已全部归档后保留 |
| 历史现场 | `/tmp/sm90-bs256-trace-pbpMbH4u`（KV + replay）**未触碰** |
