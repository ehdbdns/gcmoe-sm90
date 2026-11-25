# 回执 · SM90 · shared L1 优先启动，线性 ECO 取 wa=1

执行者：SM90 八卡执行者。任务书 `docs/trace_executor_20261119/SM90_SHARED_FIRST_LINEAR_WA1.md`（提交 `d6b5be9`）。

| 项 | 值 |
|---|---|
| 测量 SHA | `d6b5be9e6d720e1fd8a1cdae62a70500a25721db` |
| 模型启动 | **12 次**（6 档 × 参考 + ECO）|
| `exit-code.txt` | **1** |
| 六档 exit | bs32=0、bs64=0、bs128=0、bs256=0、**bs512=1、bs1024=1** |
| 完成报告的档 | **4/6**（BS32/64/128/256）|

## 核心结论：wa=1 只在 BS256 明显胜出，小批量反而更慢

| BS | wa=2 µs/层 | wa=1 µs/层 | 差 µs | 100×(wa1/wa2−1) | 更快权重 | wa=1 更快的层 (/40) | 更快的 50 步段 (/20) |
|---|---|---|---|---|---|---|---|
| 32 | 355.40 | 356.10 | +0.71 | **+0.20%** | wa2 | 8/40 | 0/20 |
| 64 | 437.55 | 452.92 | +15.37 | **+3.51%** | wa2 | 0/40 | 0/20 |
| 128 | 586.67 | 586.31 | -0.36 | **-0.06%** | wa1 | 23/40 | 13/20 |
| 256 | 872.49 | 863.26 | -9.22 | **-1.06%** | wa1 | 40/40 | 19/20 |

负值表示 wa=1 更快。逐档解读：

- **BS32**：wa=1 慢 0.20%，40 层中仅 8 层更快、20 段中 0 段更快 —— 方向一致地略差。
- **BS64**：wa=1 **慢 3.51%**（+15.37 µs/层），**40/40 层全部更慢、20/20 段全部更慢** —— 最明确的反例。
- **BS128**：wa=1 略快 0.06%（−0.36 µs），23/40 层、13/20 段更快 —— 处于拐点，差异在波动量级内。
- **BS256**：wa=1 **快 1.06%**（−9.22 µs），**40/40 层全部更快、19/20 段更快** —— 唯一显著且方向完整的胜出。

**结论**：把激活槽权重从 2 降到 1（`T + 26M + 2A` → `T + 26M + A`）并非普遍改进。
它在 BS256 及以上批量有效，在 BS32/64 反而有害。任务书假设的「wa=1 更优」只在 BS256 这一档成立。

## BS512 / BS1024：GPU 测量完成，报告校验失败

两档的 GPU 工作**全部跑完且数据齐全**：

| 档 | 参考 exit | ECO exit | ECO 八 rank NPZ | ECO run.json |
|---|---|---|---|---|
| 512 | 0 | 0 | **8/8** | OK |
| 1024 | 0 | 0 | **8/8** | OK |

失败发生在 CPU 报告阶段 `shared_first_wa1_report.py::check_pair`：

```
ValueError: different routes, feedback, KV, placement or native warm outputs
```

### 根因（已逐字段定位）

`workload()` 采集 115 个字段。BS512 的逐字段比对结果：

| 字段类别 | 结果 |
|---|---|
| `prompt_hash` / `output_hash` / `prefill_steps` | ✅ 完全相同 |
| 逐 rank 路由 `ids` / `weights` / `demand` | ✅ 相同 |
| `batch_order` / `warm_native` / `placement` / `logical_weights` | ✅ 相同 |
| **KV 快照读回哈希**（`prefill_snapshot` / `snapshot` 各 8 rank）| **❌ 16 处不同** |

原因：本轮 BS512/1024 的历史 KV 已被清理，脚本按任务书第 3 条「如果本档 KV 已清理，脚本调用一次 vanilla publisher 重建参考」重建了本档 KV。
**重建的 KV 与上轮 wa=2 那次的 KV 二进制不可能逐位相同**（缓冲区布局、分块边界等运行期细节不同）。

这与上一轮 `SM90_EP8_LINEAR_FOUR_6BS` 回执中报告的 BS512/1024 `controlled_inputs_match=False` **是同一个问题**，当时同样只差 KV 哈希。
BS32/64/128/256 四档之所以通过，是因为历史 KV 仍在现场被直接复用，与 wa=2 逐位一致。

### 建议（超出执行者权限，未执行）

`check_pair` 把 KV 读回哈希当作硬约束。对**重建参考**的档应放宽：

1. 只比对 `prompt_hash` / `output_hash` / `prefill_steps` / 路由 `ids`/`weights`/`demand` / `placement` / `logical_weights`，把 KV 哈希单列为「重建参考」标记；或
2. 由任务书提供「允许参考重建」开关，重建档不判 KV 逐位等同。

未获明确指令前我未修改任何校验代码（diff 为空）。BS512/1024 的 GPU 数据已完整保留，可在放宽校验后直接重出报告，无需重跑 GPU。

## 四档主表（已完成部分）

| BS | 臂 | wa | MoE µs/层 | vs vanilla | vs RR | token IR | 峰值 rank M 块 | 活跃槽 |
|---|---|---|---|---|---|---|---|---|
| 32 | eco | 2 | 355.40 | +8.38% | -1.97% | 1.249 | 16.23 | 115.8 |
| 32 | eco | 1 | 356.10 | +8.60% | -1.78% | 1.247 | 16.24 | 115.8 |
| 64 | eco | 2 | 437.55 | -1.61% | -4.81% | 1.179 | 24.54 | 175.8 |
| 64 | eco | 1 | 452.92 | +1.84% | -1.46% | 1.177 | 24.57 | 175.8 |
| 128 | eco | 2 | 586.67 | -8.28% | -6.29% | 1.108 | 35.25 | 248.2 |
| 128 | eco | 1 | 586.31 | -8.33% | -6.35% | 1.106 | 35.27 | 248.3 |
| 256 | eco | 2 | 872.49 | -14.26% | -2.66% | 1.058 | 49.00 | 311.7 |
| 256 | eco | 1 | 863.26 | -15.17% | -3.69% | 1.057 | 49.02 | 311.8 |

## BS256 profiler 采样（任务书第 3 条）

已完成独立 profiler 采样，产物：

- `bs256-eco-wa1-step001-layer08.perfetto.json`（31387 B）
- `bs256-eco-wa1-step001-layer20.perfetto.json`（31465 B）
- `bs256-eco-wa1-step001-layer32.perfetto.json`（31455 B）
- `launch-order.csv`（2729 B）
- `phase-times.csv`（14236 B）

- `launch-order.csv` 共 **24 行**（任务书要求 24 行）
- `phase-times.csv` 共 **192 行**（任务书要求 192 行）

## 归档

- https://github.com/ehdbdns/gcmoe-sm90/tree/main/docs/trace_executor_20261119/sm90_shared_first_wa1_results（370 MB，420 个文件）
- 含六档 `delivery-{32,64,128,256,512,1024}/`（BS512/1024 的 GPU 数据完整，仅缺报告 CSV）、六档 exit 与日志、`launcher.log`、`exit-code.txt`、`measurement.sha`。
- 根目录汇总 CSV 为**已完成四档**的合并（comparison 4 行、main_table 8 行、layers 320 行、auction 8 行、rank_shares 64 行）。任务书要求的 6/12/480/12/96 需补齐 BS512/1024 才能达到。
- 现场：`/mnt/fuse/sm90-shared-first-wa1-fEYZbAI`。

## 退卡与清理

- 最后一臂（BS1024 eco）04:05 完成，`bs1024.exit=1`（报告校验失败，非 GPU 失败）。
- 八卡已释放（各 4 MiB、无 compute app），worker 全部退出。
- 未修改任何代码（diff 为空）。

## 后续任务（fine cost）状态

任务书 `SM90_SHARED_FIRST_FINE_COST.md` 依赖本轮的 `comparison.csv` 与六档 `delivery-*/SHA256SUMS`。
由于 BS512/1024 未生成 SHA256SUMS，**fine 任务的 wa1 参考目录当前不完整**；我挂在后台的链式启动器已检测到本轮 exit=1 而**未启动 fine**。需先解决上述 KV 校验口径问题。
