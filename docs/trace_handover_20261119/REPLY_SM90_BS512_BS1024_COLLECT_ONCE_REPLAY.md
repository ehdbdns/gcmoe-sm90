# 回执 · SM90 · BS512 / BS1024 各采集一次真实路由 + 五臂回放

执行者：SM90 八卡执行者。任务书 `docs/trace_executor_20261119/SM90_BS512_BS1024_COLLECT_ONCE_REPLAY.md`。
测量 SHA：`029b009f501d014c0663c342f37a420e16be69b6`（两档 `execution.sha` 相同）。两档 exit-code 均为 `0`。

## 结论

两档各完成 **1 次 vanilla 整模启动**（非 5 次），采集到真实路由后 **五臂各 1 次算子启动**，全部一次通过。
`capture_comparison.csv` 两档各 72 行（3 步 × 3 层 × 8 rank，**vanilla 专用**，与 `audit_se_history.py --fresh` 一致），**全部 `different_slots=0`**，即本次算子回放的 vanilla 物理槽分配与本次整模采集逐槽一致；提交日志均为 `Historical allocation/auction audit: 72/72 matched`。其余四臂的一致性（random/RR 目标槽、ECO/eco_smr 分配与拍卖）由 `prepare_se_capture.py` 与 `se_replay_report.py` 的断言逐层校验，全部通过。

## 主表（µs/层，三层 L8/L20/L32 等权平均，sample 1 = 第二步）

| BS | 臂 | MoE µs/层 | 相对 vanilla | token IR | 总 M 块 | 峰值 rank M 块 | 活跃槽 | M 块 IR | routed MAE (pp) | graph MAE (pp) | ECO 成交(均值/合计) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 512 | vanilla | 1951.18 | +0.00% | 1.5460 | 568 | 98.7 | 328 | 1.3900 | 0.2595 | 2.931 | — |
| 512 | random | 1906.07 | -2.31% | 1.4592 | 615 | 99.0 | 383 | 1.2876 | 0.1712 | 2.116 | — |
| 512 | rr | 1499.84 | -23.13% | 1.0634 | 607 | 80.3 | 392 | 1.0593 | 0.0604 | 0.370 | — |
| 512 | eco | 1446.38 | -25.87% | 1.0252 | 568 | 74.7 | 387 | 1.0510 | 0.0465 | 0.161 | 4.3 / 13 (5/5/3) |
| 512 | eco_smr | 1377.89 | -29.38% | 1.0252 | 568 | 74.7 | 387 | 1.0510 | 0.0479 | 0.179 | 4.3 / 13 (5/5/3) |
| 1024 | vanilla | 3934.81 | +0.00% | 1.6671 | 945 | 180.0 | 358 | 1.5257 | 0.2805 | 3.341 | — |
| 1024 | random | 3455.61 | -12.18% | 1.3844 | 989 | 162.3 | 420 | 1.3126 | 0.1854 | 2.267 | — |
| 1024 | rr | 2779.96 | -29.35% | 1.0473 | 977 | 128.0 | 422 | 1.0482 | 0.0426 | 0.353 | — |
| 1024 | eco | 2693.50 | -31.55% | 1.0226 | 945 | 121.0 | 418 | 1.0247 | 0.0383 | 0.103 | 6.3 / 19 (8/4/7) |
| 1024 | eco_smr | 2924.87 | -25.67% | 1.0226 | 945 | 121.0 | 418 | 1.0247 | 0.0932 | 0.361 | 6.3 / 19 (8/4/7) |

相对变化分母为同档同批 vanilla 三层均值，负值为快。MoE 取每层最慢 rank 的本地 graph→combine 跨度。

### 与上一轮（合成输入）的对照

| 档 | 上一轮 replay2 的 eco_smr 收益 | 本轮真实路由的 eco_smr 收益 |
|---|---|---|
| BS512 | −4.96% | **−29.38%** |
| BS1024 | −5.07% | **−25.67%** |

差异来源：真实路由下 vanilla 的 token IR 高达 1.55（BS512）/ 1.67（BS1024），而 EPLB 系（rr/eco/eco_smr）为 1.02–1.06。合成输入（每源 rank 等量 token、均匀 id）无法体现这种热点，因此上一轮只看到 ~5% 收益；本轮真实路由下 EPLB 的摊平效果是主要贡献。

### BS1024 上 eco 优于 eco_smr（与 BS512 相反，如实报告）

- BS1024：`eco` 2693.50 µs（−31.55%）优于 `eco_smr` 2924.87 µs（−25.67%），差 231 µs（+8.6%）。
- 两臂的逐槽分配与七项拍卖计数**逐层完全一致**（已核对 L8/L20/L32 每层的 `slot_counts` 与 `auction` 全部相等），因此差异不是分配不同导致。
- **逐层看，慢只发生在 L8**：

| BS1024 层 | eco µs | eco_smr µs | 差 |
|---|---|---|---|
| L8 | 2696.64 | **3737.76** | **+1041.12 (+38.6%)** |
| L20 | 2705.31 | **2516.70** | −188.61 (−7.0%) |
| L32 | 2678.56 | **2520.16** | −158.40 (−5.9%) |

  L20/L32 上 SM-reuse 与 BS512 一样是加速的（约 −6~7%）；三层等权平均被 L8 的单层异常拉成净慢。该异常出现在 `eco_smr` 的 L8 捕获/回放上，属实测结果，未做任何剔除。
- BS512：`eco_smr` 1377.89 µs 优于 `eco` 1446.38 µs（−29.38% vs −25.87%），三层方向一致（L8 −2.8%、L20 −5.9%、L32 −5.5%）。
- 任务书要求“保留实测排名及差异”，故此处不做取舍，两行都如实列出。

## 逐层明细（µs/层）

### BS512

| 层 | 臂 | MoE µs | 总 M 块 | 峰值 M 块 | 活跃槽 | ECO 成交 | token IR | M 块 IR |
|---|---|---|---|---|---|---|---|---|
| L8 | vanilla | 1920.58 | 566 | 98 | 330 | — | 1.5078 | 1.3852 |
| L20 | vanilla | 2300.06 | 565 | 114 | 319 | — | 1.8984 | 1.6142 |
| L32 | vanilla | 1632.90 | 574 | 84 | 336 | — | 1.2318 | 1.1707 |
| L8 | random | 1999.90 | 611 | 104 | 386 | — | 1.5352 | 1.3617 |
| L20 | random | 1995.97 | 614 | 101 | 375 | — | 1.5938 | 1.3160 |
| L32 | random | 1722.34 | 621 | 92 | 389 | — | 1.2487 | 1.1852 |
| L8 | rr | 1476.61 | 604 | 80 | 394 | — | 1.0609 | 1.0596 |
| L20 | rr | 1490.14 | 605 | 79 | 383 | — | 1.0609 | 1.0446 |
| L32 | rr | 1532.77 | 611 | 82 | 400 | — | 1.0684 | 1.0736 |
| L8 | eco | 1451.90 | 566 | 76 | 385 | 5 | 1.0417 | 1.0742 |
| L20 | eco | 1442.08 | 565 | 73 | 380 | 5 | 1.0130 | 1.0336 |
| L32 | eco | 1445.15 | 574 | 75 | 395 | 3 | 1.0208 | 1.0453 |
| L8 | eco_smr | 1410.88 | 566 | 76 | 385 | 5 | 1.0417 | 1.0742 |
| L20 | eco_smr | 1356.61 | 565 | 73 | 380 | 5 | 1.0130 | 1.0336 |
| L32 | eco_smr | 1366.18 | 574 | 75 | 395 | 3 | 1.0208 | 1.0453 |

### BS1024

| 层 | 臂 | MoE µs | 总 M 块 | 峰值 M 块 | 活跃槽 | ECO 成交 | token IR | M 块 IR |
|---|---|---|---|---|---|---|---|---|
| L8 | vanilla | 3778.05 | 947 | 174 | 366 | — | 1.5872 | 1.4699 |
| L20 | vanilla | 4677.22 | 937 | 215 | 344 | — | 2.0391 | 1.8356 |
| L32 | vanilla | 3349.15 | 950 | 151 | 365 | — | 1.3750 | 1.2716 |
| L8 | random | 3598.56 | 995 | 172 | 429 | — | 1.4486 | 1.3829 |
| L20 | random | 3559.58 | 982 | 164 | 405 | — | 1.4492 | 1.3360 |
| L32 | random | 3208.67 | 991 | 151 | 425 | — | 1.2554 | 1.2190 |
| L8 | rr | 2803.68 | 977 | 127 | 430 | — | 1.0581 | 1.0399 |
| L20 | rr | 2780.35 | 970 | 129 | 408 | — | 1.0505 | 1.0639 |
| L32 | rr | 2755.84 | 984 | 128 | 429 | — | 1.0332 | 1.0407 |
| L8 | eco | 2696.64 | 947 | 121 | 425 | 8 | 1.0273 | 1.0222 |
| L20 | eco | 2705.31 | 937 | 120 | 404 | 4 | 1.0143 | 1.0245 |
| L32 | eco | 2678.56 | 950 | 122 | 424 | 7 | 1.0260 | 1.0274 |
| L8 | eco_smr | 3737.76 | 947 | 121 | 425 | 8 | 1.0273 | 1.0222 |
| L20 | eco_smr | 2516.70 | 937 | 120 | 404 | 4 | 1.0143 | 1.0245 |
| L32 | eco_smr | 2520.16 | 950 | 122 | 424 | 7 | 1.0260 | 1.0274 |

## 比较图

- BS512：[moe.svg](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/trace_executor_20261119/sm90_collect_once_replay_results/bs512/replay/moe.svg)
- BS1024：[moe.svg](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/trace_executor_20261119/sm90_collect_once_replay_results/bs1024/replay/moe.svg)

## 30 份 Perfetto（每层每臂，8 rank，仅第二步 replay）

### BS512

| 层 | vanilla | random | rr | eco | eco_smr |
|---|---|---|---|---|---|
| L8 | [vanilla](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/trace_executor_20261119/sm90_collect_once_replay_results/bs512/replay/layer8/vanilla.replay2.perfetto.json) | [random](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/trace_executor_20261119/sm90_collect_once_replay_results/bs512/replay/layer8/random.replay2.perfetto.json) | [rr](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/trace_executor_20261119/sm90_collect_once_replay_results/bs512/replay/layer8/rr.replay2.perfetto.json) | [eco](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/trace_executor_20261119/sm90_collect_once_replay_results/bs512/replay/layer8/eco.replay2.perfetto.json) | [eco_smr](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/trace_executor_20261119/sm90_collect_once_replay_results/bs512/replay/layer8/eco_smr.replay2.perfetto.json) |
| L20 | [vanilla](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/trace_executor_20261119/sm90_collect_once_replay_results/bs512/replay/layer20/vanilla.replay2.perfetto.json) | [random](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/trace_executor_20261119/sm90_collect_once_replay_results/bs512/replay/layer20/random.replay2.perfetto.json) | [rr](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/trace_executor_20261119/sm90_collect_once_replay_results/bs512/replay/layer20/rr.replay2.perfetto.json) | [eco](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/trace_executor_20261119/sm90_collect_once_replay_results/bs512/replay/layer20/eco.replay2.perfetto.json) | [eco_smr](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/trace_executor_20261119/sm90_collect_once_replay_results/bs512/replay/layer20/eco_smr.replay2.perfetto.json) |
| L32 | [vanilla](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/trace_executor_20261119/sm90_collect_once_replay_results/bs512/replay/layer32/vanilla.replay2.perfetto.json) | [random](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/trace_executor_20261119/sm90_collect_once_replay_results/bs512/replay/layer32/random.replay2.perfetto.json) | [rr](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/trace_executor_20261119/sm90_collect_once_replay_results/bs512/replay/layer32/rr.replay2.perfetto.json) | [eco](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/trace_executor_20261119/sm90_collect_once_replay_results/bs512/replay/layer32/eco.replay2.perfetto.json) | [eco_smr](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/trace_executor_20261119/sm90_collect_once_replay_results/bs512/replay/layer32/eco_smr.replay2.perfetto.json) |

### BS1024

| 层 | vanilla | random | rr | eco | eco_smr |
|---|---|---|---|---|---|
| L8 | [vanilla](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/trace_executor_20261119/sm90_collect_once_replay_results/bs1024/replay/layer8/vanilla.replay2.perfetto.json) | [random](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/trace_executor_20261119/sm90_collect_once_replay_results/bs1024/replay/layer8/random.replay2.perfetto.json) | [rr](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/trace_executor_20261119/sm90_collect_once_replay_results/bs1024/replay/layer8/rr.replay2.perfetto.json) | [eco](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/trace_executor_20261119/sm90_collect_once_replay_results/bs1024/replay/layer8/eco.replay2.perfetto.json) | [eco_smr](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/trace_executor_20261119/sm90_collect_once_replay_results/bs1024/replay/layer8/eco_smr.replay2.perfetto.json) |
| L20 | [vanilla](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/trace_executor_20261119/sm90_collect_once_replay_results/bs1024/replay/layer20/vanilla.replay2.perfetto.json) | [random](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/trace_executor_20261119/sm90_collect_once_replay_results/bs1024/replay/layer20/random.replay2.perfetto.json) | [rr](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/trace_executor_20261119/sm90_collect_once_replay_results/bs1024/replay/layer20/rr.replay2.perfetto.json) | [eco](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/trace_executor_20261119/sm90_collect_once_replay_results/bs1024/replay/layer20/eco.replay2.perfetto.json) | [eco_smr](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/trace_executor_20261119/sm90_collect_once_replay_results/bs1024/replay/layer20/eco_smr.replay2.perfetto.json) |
| L32 | [vanilla](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/trace_executor_20261119/sm90_collect_once_replay_results/bs1024/replay/layer32/vanilla.replay2.perfetto.json) | [random](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/trace_executor_20261119/sm90_collect_once_replay_results/bs1024/replay/layer32/random.replay2.perfetto.json) | [rr](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/trace_executor_20261119/sm90_collect_once_replay_results/bs1024/replay/layer32/rr.replay2.perfetto.json) | [eco](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/trace_executor_20261119/sm90_collect_once_replay_results/bs1024/replay/layer32/eco.replay2.perfetto.json) | [eco_smr](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/trace_executor_20261119/sm90_collect_once_replay_results/bs1024/replay/layer32/eco_smr.replay2.perfetto.json) |

轨道：count exchange 与 pull token 同行、shared expert L1/L2、wait token、首末实际计算 CTA 间的 routed GEMM、combine；ECO 两臂另有 ECO plan。原始 Kineto（`rank*.json`）与设备戳（`rank*-device.json`）按臂分目录一并提交。

## 采集与一致性核验

**BS512**：batch=512，stats_steps=1000，capture_steps=3，layers=[8, 20, 32]，prefill_steps=267，prompt_sha256=`e7c4aee9db91a34c…`；route slices=24 个（24 = 3 层 × 8 rank），demand_sha256=`f967a7c99699c62d…`。

**BS1024**：batch=1024，stats_steps=1000，capture_steps=3，layers=[8, 20, 32]，prefill_steps=645，prompt_sha256=`d7eeeb50bb0428af…`；route slices=24 个（24 = 3 层 × 8 rank），demand_sha256=`5accba1c8c6ca76d…`。

- 每档整模启动 **1 次**（vanilla），未重复启动。1000 统计步 + 3 采集步全程满批：BS512 恒定 `512 generating`、BS1024 恒定 `1024 generating`，无 context 请求混入。
- 每层每步八卡路由项：BS512 24 576、BS1024 49 152，与任务书表一致（由 `prepare_se_capture.py` 断言 `history.sum(-1) == 8*bs*6`）。
- 汇总源需求 == 本次 vanilla 整模逐槽接收数（`prepare_se_capture.py` 逐层比对 `vanilla/rank*.npz` 的 counts）。
- 展示样本（第二步）五臂逐 rank 输入哈希一致（同 `routes/` 切片）与输出哈希一致（`se_replay_report.py` 的 `replay input/output mismatch` 未触发）。
- random/RR 实收物理槽 == 源端算出的目标槽；ECO 与 eco_smr 三步逐槽分配及七项拍卖计数逐项相等。
- 数据集：两份 jsonl 的 SHA256 已落盘（`collection/dataset.sha256`）。模型：`/mnt/dsv41/models/DeepSeek-V4.1-Flash`。

## 执行环境与一处保持语义的路径修复

本次测量 SHA `029b009` 的 `collect_and_replay_se.sh` 中模型/数据集路径为 `/mnt/nas2/...`，本机不存在该挂载。
修复方式：**未修改仓库代码**，仅建立指向同一存储的符号链接 `/mnt/nas2 -> /mnt/dsv41`，并让模型走已核对的本地副本 `/mnt/fuse/models/DeepSeek-V4.1-Flash`（与 NAS 逐位相同，见上一轮回执）。
数据内容不受影响：链接目标与原 NAS 目录是同一份文件，`dataset.sha256` 已记录实际读取的两个 jsonl 哈希。
设计方随后推送的 `8e82578` 已把脚本内的路径改为 `/mnt/dsv41` 并加入输入校验，与本轮的符号链接指向一致。

## 权重加载（优化已开启，实测）

| 档 | 48 shard 加载耗时 | 条件 |
|---|---|---|
| BS512 | **42 s**（1.12 it/s） | 页缓存刚预热（`warm_page_cache.sh`） |
| BS1024 | **3 s**（14.09 it/s） | 页缓存已热 + 本地副本 |

生效的优化：`MODEL` 指向本地盘副本（消除 NAS 读放大）、`warm_page_cache.sh` 预热 48 shard、`GC_MOE_NO_PREFETCH=1`（关闭预取，避免 657 s 路径）。对比历史未优化值 6 分 03 秒，本轮 BS512 快 8.6 倍、BS1024 快 121 倍。

## 归档

- 全部产物：https://github.com/ehdbdns/gcmoe-sm90/tree/main/docs/trace_executor_20261119/sm90_collect_once_replay_results
- 结构：`bs512/{collection,replay}`、`bs1024/{collection,replay}`，另有 `bs512.log`、`bs1024.log`、两个 exit-code。
- `collection/`：`capture.json`、`routes/`（24 个 `layer{N}-rank{R}.npz` 真实路由切片 + `manifest.json`）、`stats-demand.npz`、`vanilla/`（8 rank 的 counts/clocks/epochs）、五臂 `installed-layout.json`、`placement.json`、`dataset.sha256`、`collector.log`。**本轮回放可完全从 GitHub 重建。**
- `replay/`：15 份 `*.replay2.perfetto.json`/档、原始 `rank*.json` 与 `rank*-device.json`、`capture_comparison.csv`、`main_table.csv`、`layers.csv`、`ranks.csv`、`moe.svg`、`input-manifest.json`。
- 现场完整目录（含 KV 与 log）：`/mnt/fuse/sm90-collect-once-5tVyQJ6f`。

## 退卡与清理

- 最后一臂（BS1024 eco_smr）于 22:36:42 退出，两档 rc 均为 0；八卡已释放（各 4 MiB、无 compute app），本任务 worker 全部退出。
- 站点设备掩码未改动；协作锁 `h20-eight.lock` 已释放。
- 本任务无可重建的私有编译产物（未编译新 kernel，DeepGEMM JIT 缓存沿用站点私有目录）。
