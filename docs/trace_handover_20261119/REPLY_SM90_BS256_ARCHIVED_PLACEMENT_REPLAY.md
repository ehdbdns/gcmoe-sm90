# 回执 · SM90 · BS256 按整模归档放置补测 RR 与 ECO

执行者：SM90 八卡执行者。任务书 `docs/trace_executor_20261119/SM90_BS256_ARCHIVED_PLACEMENT_REPLAY.md`。

| 项 | 值 |
|---|---|
| 本次执行 SHA | `4728dfe35cebdf39136d09e47e1474303e65bc2b` |
| 复用对照 SHA | `f80ba3915d06c7c69c7688a5aefe46e297a5de50`（vanilla、random 两臂原样复用）|
| `run.exit` | **0** |
| `compare.exit` | **0** |
| 分配/拍卖审计 | **288/288 matched** |
| 启动的语言模型 | **0 次**（无需整模）|
| 启动的算子进程组 | **2 次**（rr、eco 各一次，各跑 L8/L20/L32）|

## 结论：上轮未配对的 L20/L32 现已全部对齐

上轮 `compare_bs256_replay.py` 以 exit=3 结束，缺的是 rr/eco 在 L20、L32 的放置。本轮按归档 `installed-layout.json` 安装、并用 `[1000,40,384]` int32 需求一次性恢复 RR 的 CUDA 副本顺序后：

- `history_comparison.csv` **288 行**（4 臂 × 3 层 × 3 步 × 8 rank）**全部匹配**（上一轮为 245/288）。
- `timing_layers.csv` **12 行全部 `workload_matched=True`**，`layout_equal` 与 `slots_and_auction_equal` 全为 True。
- `timing_summary.csv` **4 行全部 `workload_matched=True`**。

## 四臂主表（µs/层，三层等权平均，sample 1 = 第二步）

| 臂 | 旧整模样本 µs | 本次回放 µs | 回放−整模 | 百分比 | 旧主表全窗口 µs | 回放 vs 窗口 | 样本 vs 窗口 | 复用上轮 |
|---|---|---|---|---|---|---|---|---|
| vanilla | 1127.691 | 1110.933 | -16.757 | **-1.49%** | 1010.22 | +9.97% | +11.63% | 是 |
| random | 1082.219 | 1063.445 | -18.773 | **-1.73%** | 1084.74 | -1.96% | -0.23% | 是 |
| rr | 889.931 | 862.272 | -27.659 | **-3.11%** | 891.15 | -3.24% | -0.14% | 否（本次新跑） |
| eco | 834.304 | 853.397 | +19.093 | **+2.29%** | 842.43 | +1.30% | -0.96% | 否（本次新跑） |

正值表示回放偏慢。旧整模三平均值 1127.691 / 1082.219 / 889.931 / 834.304 与任务书给定值逐项吻合。

### 逐层差值

| 臂 | 层 | 旧整模 µs | 回放 µs | 差 µs | 差 % |
|---|---|---|---|---|---|
| vanilla | L8 | 1068.32 | 1067.87 | -0.45 | -0.04% |
| vanilla | L20 | 1294.82 | 1253.31 | -41.50 | -3.21% |
| vanilla | L32 | 1019.94 | 1011.62 | -8.32 | -0.82% |
| random | L8 | 1125.44 | 1093.57 | -31.87 | -2.83% |
| random | L20 | 1115.68 | 1111.20 | -4.48 | -0.40% |
| random | L32 | 1005.54 | 985.57 | -19.97 | -1.99% |
| rr | L8 | 876.06 | 867.84 | -8.22 | -0.94% |
| rr | L20 | 909.09 | 861.18 | -47.90 | -5.27% |
| rr | L32 | 884.64 | 857.79 | -26.85 | -3.03% |
| eco | L8 | 832.29 | 825.79 | -6.50 | -0.78% |
| eco | L20 | 843.42 | 883.01 | +39.58 | +4.69% |
| eco | L32 | 827.20 | 851.39 | +24.19 | +2.92% |

三层均值差：vanilla −1.49%、random −1.73%、**rr −3.11%**、**eco +2.29%**。

### 关键观察

- **rr**：三层一致偏快（−0.94% / −5.27% / −3.03%），均值 −3.11%。
- **eco**：方向不一致 —— L8 偏快 −0.78%，但 **L20 +4.69%、L32 +2.92%** 偏慢，均值 +2.29%。这与放置无关（`layout_equal` 与 `slots_and_auction_equal` 均为 True，288/288 匹配），是本次新代码下 ECO 在这两层上的真实耗时表现。
- 因此**在新版代码下 eco 相对 rr 的优势缩小**：本轮回放 eco 853.40 vs rr 862.27（差 8.87 µs），而旧整模样本是 eco 834.30 vs rr 889.93（差 55.63 µs）。

### 新版回放 vs 旧版整模（旧版 shared 为单个 kernel）

这一列的差值里同时含两部分：①样本位置偏差（旧三层样本相对 40 层全窗口：vanilla +11.63%、random −0.23%、rr −0.14%、eco −0.96%）；②版本影响（旧版 shared 单 kernel，新版拆成 L1/L2，并含本轮新增的 `GC_MOE_M18_SMS`/stream/split 组合）。

| 臂 | 新版回放 µs | 旧版整模三层样本 µs | 差 % | 旧版整模全窗口 µs | 回放 vs 全窗口 % |
|---|---|---|---|---|---|
| vanilla | 1110.933 | 1127.691 | -1.49% | 1010.22 | +9.97% |
| random | 1063.445 | 1082.219 | -1.73% | 1084.74 | -1.96% |
| rr | 862.272 | 889.931 | -3.11% | 891.15 | -3.24% |
| eco | 853.397 | 834.304 | +2.29% | 842.43 | +1.30% |

## 同版本实测误差（引用上一轮）

上一轮 `SM90_BS256_REPLAY_VS_MODEL` 的 `live_split / vanilla`：整模真值 1102.443 µs/层，回放 1110.933 µs/层，**+8.491 µs（+0.77%）**；逐层 +1.50% / +0.26% / +0.65%。`capture_comparison.csv` 72 项全部 `different_slots=0`。

本轮复用同一份 vanilla trace（`reused-controls.sha = f80ba39…`，与上轮逐字节相同），所以该误差可直接沿用。

## 回放主表其余指标

| 臂 | MoE µs/层 | token IR | 总 M 块 | 峰值 rank M 块 | 活跃槽 | M 块 IR | routed MAE (pp) | graph MAE (pp) |
|---|---|---|---|---|---|---|---|---|
| vanilla | 1110.93 | 1.6649 | 365 | 60.0 | 274 | 1.3164 | 0.2487 | 3.156 |
| random | 1063.45 | 1.4668 | 405 | 62.7 | 321 | 1.2383 | 0.1770 | 1.900 |
| rr | 862.27 | 1.0438 | 399 | 53.3 | 338 | 1.0702 | 0.0658 | 0.464 |
| eco | 853.40 | 1.0608 | 365 | 48.3 | 320 | 1.0602 | 0.1068 | 0.240 |

## 12 份 Perfetto（每层每臂，8 rank，仅第二步）

| 层 | vanilla（复用）| random（复用）| rr（本次）| eco（本次）|
|---|---|---|---|---|
| L8 | [vanilla](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/trace_executor_20261119/sm90_bs256_archived_placement_results/replay/layer8/vanilla.replay2.perfetto.json) | [random](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/trace_executor_20261119/sm90_bs256_archived_placement_results/replay/layer8/random.replay2.perfetto.json) | [rr](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/trace_executor_20261119/sm90_bs256_archived_placement_results/replay/layer8/rr.replay2.perfetto.json) | [eco](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/trace_executor_20261119/sm90_bs256_archived_placement_results/replay/layer8/eco.replay2.perfetto.json) |
| L20 | [vanilla](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/trace_executor_20261119/sm90_bs256_archived_placement_results/replay/layer20/vanilla.replay2.perfetto.json) | [random](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/trace_executor_20261119/sm90_bs256_archived_placement_results/replay/layer20/random.replay2.perfetto.json) | [rr](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/trace_executor_20261119/sm90_bs256_archived_placement_results/replay/layer20/rr.replay2.perfetto.json) | [eco](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/trace_executor_20261119/sm90_bs256_archived_placement_results/replay/layer20/eco.replay2.perfetto.json) |
| L32 | [vanilla](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/trace_executor_20261119/sm90_bs256_archived_placement_results/replay/layer32/vanilla.replay2.perfetto.json) | [random](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/trace_executor_20261119/sm90_bs256_archived_placement_results/replay/layer32/random.replay2.perfetto.json) | [rr](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/trace_executor_20261119/sm90_bs256_archived_placement_results/replay/layer32/rr.replay2.perfetto.json) | [eco](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/trace_executor_20261119/sm90_bs256_archived_placement_results/replay/layer32/eco.replay2.perfetto.json) |

## 校验

- 路由：由已知 BS256 bank 的 SHA256 验证后抽取（`extract_se_routes.py`），与上一轮逐字节相同。
- RR 正向映射：三层逐槽等于归档 `installed-layout.json`（`layout_equal=True` ×3）。逆映射按原策略副本顺序显式传入。
- ECO：使用其归档的单副本优先排列，三层同样逐槽相等。
- 物理槽与拍卖：`history_comparison.csv` 288 行全部 `different_slots=0` 且 `auction_equal=True`。
- 四臂逐 rank 输入/输出哈希一致（`se_replay_report.py` 断言通过）。
- 复用校验：`replay/reused-controls.sha = f80ba3915d06c7c69c7688a5aefe46e297a5de50`，且本轮 vanilla/random 的 `rank0-device.json` 与上轮 `diff` 为空。

## 归档与退卡

- 全部产物：https://github.com/ehdbdns/gcmoe-sm90/tree/main/docs/trace_executor_20261119/sm90_bs256_archived_placement_results
- 含 `run.log`、`run.exit`(=0)、`compare.log`、`compare.exit`(=0)、`replay/execution.sha`、`replay/reused-controls.sha`、12 份 Perfetto、原始 `rank*.json` 与 `rank*-device.json`、`history_comparison.csv`(288)、`timing_summary.csv`(4)、`timing_layers.csv`(12)、`main_table.csv`(4)、`layers.csv`(12)、`ranks.csv`(96)、`moe.svg`。
- 现场完整目录：`/mnt/fuse/sm90-bs256-layout-GBCN8by`。
- 两臂 23:57:13 退出，rc=0；八卡已释放（各 4 MiB、无 compute app），worker 全部退出，协作锁已释放。
- 本任务未编译新 kernel、无私有缓存需清理；历史 bank 保持只读、未改动。
