# 回执 · SM90 · BS256 真实路由回放与整模计时对照

执行者：SM90 八卡执行者。任务书 `docs/trace_executor_20261119/SM90_BS256_REPLAY_VS_MODEL.md`，测量 SHA `f80ba3915d06c7c69c7688a5aefe46e297a5de50`。
比较脚本退出码 **3**（旧样本未全部配对），按任务书规定仍交付新采回放、本次 vanilla 真值和两张 CSV。

## 实际启动次数与释放

- vanilla 整模启动 **1 次**（1000 统计步 + 3 采集步，全程满批 `256 generating`，无 context 请求混入）。
- 算子进程组 **4 次**（vanilla / random / rr / eco），每组一次跑 L8、L20、L32。
- 未追加第二套历史 bank 回放（见下文「为何没走 archive-replay 分支」）。
- 最后一臂 23:31:12 退出；八卡已释放（各 4 MiB、无 compute app），worker 全部退出，协作锁已释放。

## 主表（µs/层，三层等权平均，sample 1 = 第二步）

| 臂 | 回放 µs/层 | 本次整模真值 µs/层 | 回放−整模 | 相对百分比 | 旧主表窗口 µs/层 | 回放 vs 窗口 | 样本 vs 窗口 |
|---|---|---|---|---|---|---|---|
| vanilla | 1110.93 | 1102.44 | +8.49 | +0.77% | 1010.22 | +9.97% | +9.13% |
| random | 1063.45 | 1082.22 | -18.77 | -1.73% | 1084.74 | -1.96% | -0.23% |
| rr | 849.99 | 889.93 | —（未配对） | — | 891.15 | -4.62% | -0.14% |
| eco | 822.69 | 834.30 | —（未配对） | — | 842.43 | -2.34% | -0.96% |

`vanilla` 行用 **live_split**（本次整模直接记录的时钟，不依赖历史 bank）；其余三臂用 **archive_pre_split**。
正值表示回放偏慢。

### 同版本误差（本次同 SHA，live_split / vanilla）

| 指标 | 数值 |
|---|---|
| 整模真值 | 1102.443 µs/层 |
| 回放 | 1110.933 µs/层 |
| 回放 − 整模 | **+8.491 µs（+0.77%）** |

逐层：L8 +15.78 µs（+1.50%）、L20 +3.20 µs（+0.26%）、L32 +6.50 µs（+0.65%）

即：同一版本下算子回放比整模实测仅慢 **+0.77%**（8.49 µs/层）；逐层 +1.50% / +0.26% / +0.65%，最大单层 1.50%。`capture_comparison.csv` 全部 72 项逐槽相等、第二步八卡时钟完整。

### 与旧主表同样本差距（archive_pre_split）

| 臂 | 旧整模样本 µs/层 | 本次回放 µs/层 | 差 | 配对 |
|---|---|---|---|---|
| vanilla | 1127.691 | 1110.933 | -16.76 µs（-1.49%） | 是 |
| random | 1082.219 | 1063.445 | -18.77 µs（-1.73%） | 是 |
| rr | 889.931 | 849.995 | — | **否** |
| eco | 834.304 | 822.688 | — | **否** |

旧整模三平均值与任务书给出的 1127.691 / 1082.219 / 889.931 / 834.304 **逐项吻合**（说明取样位置正确）。

## 为何 exit=3：L20、L32 的 RR/ECO 放置与历史不同

`timing_layers.csv` 15 行中 11 行 `workload_matched=True`。未配对的 4 行全部在 `rr`/`eco` 的 L20、L32：

| 层 | 臂 | 放置相同 | 槽计数相同 | 机制 |
|---|---|---|---|---|
| L20 | rr | **否** | 是 | 见下 |
| L32 | rr | **否** | **否** | 见下 |
| L20 | eco | **否** | 是 | 见下 |
| L32 | eco | **否** | **否** | 见下 |

**根因**（已用相同输入独立复现）：

- 新旧两轮的 1000 步需求统计 **逐字节完全相同**（`stats-demand.npz` 的 `logical` 数组 `np.array_equal` 为 True，最大绝对差 0）。
- 用这份相同需求调用当前的 `rebalance_experts_global(history[:, L:L+1], 448, 56)`，L8 与历史放置**完全一致**，但 L20 差 5/448 槽、L32 差 34/448 槽。
- **两次连续调用结果相同**（确定性），且 `sorted(new) == sorted(old)` —— **专家集合完全相同，只是副本在 rank 间/槽内的排列顺序不同**。
- 因此这是 `rebalance_experts_global` 在 L20/L32 的并列需求处理上与历史版本产生了不同的确定性输出，不是随机性、不是取样错位。
- 后果：令牌→物理槽映射变化，L32 上造成 19–24 个槽计数不同；L20 仅排列不同、槽计数相同（不同 rank 上分布不同但合计对上）。
- 显示第二步之后的三臂耗时**只作为实测值列出，不作配对结论**。

### 为何没走 archive-replay 分支

任务书说「若旧 bank 已通过 SHA 核验、但与新采样本不同，可在同次预约空闲期单独回放该历史 bank 配对四臂」。
本轮的实际情形是：**旧 bank 与本次新采样本逐字节相同**（15 行 `route_verified=True`，`same_routes()` 未报差异），差异只在**放置重算**上。
因此再回放一次历史 bank 不会改变结局（放置仍会由同一份代码重算成同样的 L20/L32 排列），故不为凑配对而重复占用八卡。
若设计端希望看到「用历史 `installed-layout.json` 强制的四臂」数据，需要脚本支持跳过 `rebalance_experts_global` 重算路径 —— 这属于语义变更，超出执行者权限。

## 逐层明细（µs/层）

| 参考 | 臂 | 层 | 整模 µs | 回放 µs | 配对 |
|---|---|---|---|---|---|
| archive_pre_split | vanilla | L8 | 1068.32 | 1067.87 | 是 |
| archive_pre_split | vanilla | L20 | 1294.82 | 1253.31 | 是 |
| archive_pre_split | vanilla | L32 | 1019.94 | 1011.62 | 是 |
| archive_pre_split | random | L8 | 1125.44 | 1093.57 | 是 |
| archive_pre_split | random | L20 | 1115.68 | 1111.20 | 是 |
| archive_pre_split | random | L32 | 1005.54 | 985.57 | 是 |
| archive_pre_split | rr | L8 | 876.06 | 836.00 | 是 |
| archive_pre_split | rr | L20 | 909.09 | 861.06 | **否** |
| archive_pre_split | rr | L32 | 884.64 | 852.93 | **否** |
| archive_pre_split | eco | L8 | 832.29 | 822.72 | 是 |
| archive_pre_split | eco | L20 | 843.42 | 823.74 | **否** |
| archive_pre_split | eco | L32 | 827.20 | 821.60 | **否** |
| live_split | vanilla | L8 | 1052.10 | 1067.87 | 是 |
| live_split | vanilla | L20 | 1250.11 | 1253.31 | 是 |
| live_split | vanilla | L32 | 1005.12 | 1011.62 | 是 |

## 回放主表其余指标（replay/main_table.csv）

| 臂 | MoE µs/层 | token IR | 总 M 块 | 峰值 rank M 块 | 活跃槽 | M 块 IR | routed MAE (pp) | graph MAE (pp) |
|---|---|---|---|---|---|---|---|---|
| vanilla | 1110.93 | 1.6649 | 365 | 60.0 | 274 | 1.3164 | 0.2487 | 3.156 |
| random | 1063.45 | 1.4668 | 405 | 62.7 | 321 | 1.2383 | 0.1770 | 1.900 |
| rr | 849.99 | 1.0419 | 399 | 53.3 | 338 | 1.0702 | 0.0689 | 0.423 |
| eco | 822.69 | 1.0677 | 365 | 48.3 | 319 | 1.0602 | 0.0891 | 0.213 |

## 代表性（旧主表 40 层 × 1000 步均值）

| 臂 | 旧主表全窗口 µs/层 | 旧三层样本相对全窗口 | 本次回放相对全窗口 |
|---|---|---|---|
| vanilla | 1010.22 | +11.63% | +9.97% |
| random | 1084.74 | -0.23% | -1.96% |
| rr | 891.15 | -0.14% | -4.62% |
| eco | 842.43 | -0.96% | -2.34% |

旧三层样本与全窗口的差异：vanilla 偏慢 +11.63%、random 基本持平 −0.23%、rr −0.14%、eco −0.96%。说明 `L8/L20/L32` 这三层对 vanilla 而言偏重（尤其 L20），因此**不能把三层均值直接当作 40 层均值**。

## 四臂排序与相对变化（本次回放，实测）

- vanilla：1110.93 µs/层，相对本次回放 vanilla +0.00%
- random：1063.45 µs/层，相对本次回放 vanilla -4.27%
- rr：849.99 µs/层，相对本次回放 vanilla -23.49%
- eco：822.69 µs/层，相对本次回放 vanilla -25.95%

排序 eco < rr < random < vanilla 与旧主表一致。

## 12 份 Perfetto（每层每臂，8 rank，仅第二步）

| 层 | vanilla | random | rr | eco |
|---|---|---|---|---|
| L8 | [vanilla](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/trace_executor_20261119/sm90_bs256_replay_vs_model_results/bs256/replay/layer8/vanilla.replay2.perfetto.json) | [random](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/trace_executor_20261119/sm90_bs256_replay_vs_model_results/bs256/replay/layer8/random.replay2.perfetto.json) | [rr](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/trace_executor_20261119/sm90_bs256_replay_vs_model_results/bs256/replay/layer8/rr.replay2.perfetto.json) | [eco](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/trace_executor_20261119/sm90_bs256_replay_vs_model_results/bs256/replay/layer8/eco.replay2.perfetto.json) |
| L20 | [vanilla](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/trace_executor_20261119/sm90_bs256_replay_vs_model_results/bs256/replay/layer20/vanilla.replay2.perfetto.json) | [random](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/trace_executor_20261119/sm90_bs256_replay_vs_model_results/bs256/replay/layer20/random.replay2.perfetto.json) | [rr](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/trace_executor_20261119/sm90_bs256_replay_vs_model_results/bs256/replay/layer20/rr.replay2.perfetto.json) | [eco](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/trace_executor_20261119/sm90_bs256_replay_vs_model_results/bs256/replay/layer20/eco.replay2.perfetto.json) |
| L32 | [vanilla](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/trace_executor_20261119/sm90_bs256_replay_vs_model_results/bs256/replay/layer32/vanilla.replay2.perfetto.json) | [random](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/trace_executor_20261119/sm90_bs256_replay_vs_model_results/bs256/replay/layer32/random.replay2.perfetto.json) | [rr](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/trace_executor_20261119/sm90_bs256_replay_vs_model_results/bs256/replay/layer32/rr.replay2.perfetto.json) | [eco](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/trace_executor_20261119/sm90_bs256_replay_vs_model_results/bs256/replay/layer32/eco.replay2.perfetto.json) |

轨道：count exchange 与 pull token 同行、shared expert L1/L2、wait token、首末实际计算 CTA 间的 routed GEMM、combine；ECO 另有 plan。原始 Kineto（`rank*.json`）与设备戳（`rank*-device.json`）按臂分目录提交。

## 采集与一致性核验

- `capture.json`：batch=256、stats_steps=1000、capture_steps=3、layers=[8,20,32]。prefill 阶段完成后 1000 步统计 + 3 步采集全部满批。
- 24 份路由切片（3 层 × 8 rank）+ `manifest.json`，每源 rank 256 行、`[3, 256, 6]` 的 int64 ID 与 float32 权重。
- 每层每步八卡路由项 12 288（= 256×8×6），与任务书一致。
- 汇总源需求 == 本次 vanilla 整模逐槽接收数（`prepare_se_capture.py` 断言通过）。
- 本次采集需求统计与历史 bank 的 1000 步需求**逐字节相同**。
- 本次新采路由切片与历史 bank 切片**逐字节相同**（`same_routes()` 15/15 通过）。
- `capture_comparison.csv` 72 行（vanilla）全部 `different_slots=0`。
- ECO 七项拍卖：`eco` 三层逐槽分配与拍卖计数在 `history_comparison.csv` 中 L8 全对；L20 分配排列不同，L32 分配与计数均不同（同上根因）。

## 权重加载与环境

- 48 shard 加载 **6 分 03 秒（7.57 s/it）**，走 `/mnt/dsv41` NFS4 路径（脚本 `f80ba39` 固定）。
- 加载完成后额外有约 16 分钟的 mmap 缺页读（worker 全程 `D` 状态，`read_bytes` 从 29 GB 爬到 58 GB，CPU 约 70%），之后才进入 `ROLE4` 注册与 graph 捕获。这一段在 BS512/1024 那轮指向本地 ext4 副本时不存在。
- NCCL 使用六个活跃端口 `mlx5_4,5,9,17,21,29`（脚本已硬编码）；未改动任何共享环境。
- 未做任何代码修改（diff 为空），无编译失败。

## 归档

- 全部产物：https://github.com/ehdbdns/gcmoe-sm90/tree/main/docs/trace_executor_20261119/sm90_bs256_replay_vs_model_results
- `bs256/collection/`：`capture.json`、24 份真实路由切片 + `manifest.json`、`stats-demand.npz`、`vanilla/`（8 rank NPZ）、四臂 `installed-layout.json`、`placement.json`、`dataset.sha256`、`collector.log`。
- `bs256/replay/`：12 份 `*.replay2.perfetto.json`、原始 `rank*.json` 与 `rank*-device.json`、`main_table.csv`(4)、`layers.csv`(12)、`ranks.csv`(96)、`timing_summary.csv`、`timing_layers.csv`、`capture_comparison.csv`(72)、`history_comparison.csv`(288，245 匹配)、`moe.svg`。
- 另有 `launch.log`、`compare.log`、`comparison.exit`(=3)。
- 现场完整目录：`/mnt/fuse/sm90-bs256-replay-model-SftZZMqS`。历史 bank 保留在 `app_source/sm90_persist_20261123/retained_bs256/replay`（只读，未改动）。

## 待设计端确认

1. `rebalance_experts_global` 在 L20/L32 上对并列需求的排列与历史版本不同（集合相同）。若要严格复现历史四臂，需要脚本使用归档的 `installed-layout.json` 而不是重算 —— 这是语义变更，请指示。
2. 旧三层样本对 vanilla 偏重 +11.6%，40 层全窗口均值（1010.22）明显低于三层均值（1127.69）。请确认主表口径是否要改用全窗口。
