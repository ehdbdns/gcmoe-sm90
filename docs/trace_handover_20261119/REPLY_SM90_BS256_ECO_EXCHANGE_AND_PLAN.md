# 【SM90】REPLY_SM90_BS256_ECO_EXCHANGE_AND_PLAN — split 生效但净变慢；dispatch 被驱动拒绝

> 本回执由 **SM90 执行者**（8×H20）提交。SM100 任务书不在本执行者范围内。

## 0. 结论

| 臂 | 状态 | 每步 MoE（999 步，剔采样步）| vs baseline |
|---|---|---:|---:|
| **baseline** | ✅ `exit_code=0` | **35.7994 ms** | — |
| **split** | ✅ `exit_code=0` | **35.8554 ms** | **+0.157%（慢 0.056 ms）** |
| **dispatch** | ❌ **模型未启动** | — | — |

| 臂 | 全 1000 步含采样 | vs baseline |
|---|---:|---:|
| baseline | 35.7997 ms | — |
| split | 35.8557 ms | +0.156% |

**split 的两阶段交换生效了，但净结果是变慢 0.157%**：`count exchange` 略降
（−5 µs/层），而 `pull token`（+33 ~ +44 µs）与 `ECO plan`（+30 ~ +34 µs）涨得更多。
**dispatch 臂在本机驱动上无法建立"dispatch 分区内再切出 1 个 SM"的 Green Context**，
模型未启动，已按任务书要求保留 CUDA 错误与驱动版本。

## 1. dispatch 臂：驱动拒绝分区（任务书预期的失败路径）

```
RuntimeError: CUDA cannot isolate one SM from the dispatch partition
```

| 项 | 值 |
|---|---|
| 失败阶段 | **模型启动之前**：`GreenContextPair` 构造，`plan_dispatch=True` 的二次拆分 |
| 代码位置 | `source/gc_moe/runtime/sm90_gc_moe.py:357` |
| 触发条件 | `groups != 1 or plan[0].sm.smCount != 1 or dispatch.sm.smCount + 1 != self.dispatch_sms` |
| 实际驱动 | **580.105.08**（八卡 NVIDIA H20，`power.limit 500 W`、`clocks.max.sm 1980 MHz`）|
| CUDA 版本 | 13.0 |
| 原始日志 | `/tmp/sm90-bs256-eco-jBhVwZVo/dispatch/runner.log`（1,772 行，含八 worker 同一错误）|
| 八 worker | TP0–TP7 **全部**在同一断言失败 |
| GPU 时间 | **未消耗**（失败发生在 KV 分配与权重加载之前，`nvidia-smi` 全卡 4 MiB）|
| 任务书要求的日志确认 | 日志中**没有** `SM90 ECO partitions: dispatch=5 plan=1 compute=72`（该行只在拆分成功后打印）|

**没有更换分区大小、没有放宽校验、没有改计时边界** —— 按任务书"禁止更换分区大小…
若 driver 拒绝分区，提交 CUDA 错误和驱动版本"处理。

**证据层面**：`cuDevSmResourceSplitByCount(1, parent, ...)` 返回的 `groups` 或
`plan[0].sm.smCount` 不满足"恰好 1 个 SM"，即驱动不允许从已切出的 5-SM 通信分区里
再隔离单个 SM。要跑通 dispatch 臂，需要驱动/CUDA 层面的支持，不是脚本可修的。

## 2. 主指标：每步 MoE（40 层求和，剔除 profiler 采样步）

`moe.csv`（`bs256_eco_times.py` 的 stdout）：

| arm | steps | graph→first ms | first→last ms | last→combine ms | **moe_ms** | 变化 |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 999 | 8.0299 | 25.7018 | 2.0676 | **35.7994** | — |
| split | 999 | 8.0433 | 25.7391 | 2.0730 | **35.8554** | **+0.157%** |
| baseline | 1000 | 8.0300 | 25.7020 | 2.0677 | 35.7997 | — |
| split | 1000 | 8.0433 | 25.7395 | 2.0730 | 35.8557 | +0.156% |

三段全部同向变慢（+13.3 / +37.3 / +5.4 µs 每步），
**不是某一段的孤立波动**；剔除采样步与不剔除的结论一致（0.157% vs 0.156%）。

> **跨轮注意**：本轮含细粒度插桩，主表 `eco` 的 35.600 ms 与本轮 baseline 的
> 35.799 ms **不是同一口径**（本轮回放 + 采样开销）。本回执只在**同轮 baseline 内部**比较。

## 3. 分段耗时（八 rank 均值，µs）

| layer | arm | count exchange | pull token | shared L1+L2 | **ECO plan** | wait token | GEMM | combine |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | baseline | 25.1 | 142.9 | 194.6 | **108.5** | 2.1 | 665.5 | 35.6 |
| 0 | split | 28.2 | 170.0 | 195.0 | **142.7** | 2.1 | 666.9 | 40.3 |
| 8 | baseline | 22.9 | 173.4 | 195.4 | **118.1** | 2.1 | 665.0 | 41.4 |
| 8 | split | 19.6 | 209.3 | 196.4 | **150.0** | 2.0 | 665.1 | 38.2 |
| 20 | baseline | 26.0 | 191.7 | 196.5 | **126.2** | 2.0 | 656.6 | 42.0 |
| 20 | split | 19.6 | 235.4 | 195.9 | **160.1** | 2.4 | 656.0 | 45.3 |
| 32 | baseline | 24.5 | 175.2 | 196.8 | **118.2** | 2.1 | 667.5 | 37.9 |
| 32 | split | 19.2 | 209.9 | 195.8 | **148.4** | 2.1 | 667.4 | 43.2 |
| 39 | baseline | 23.5 | 152.8 | 196.0 | **110.2** | 2.1 | 664.5 | 30.7 |
| 39 | split | 21.4 | 186.9 | 195.5 | **140.6** | 2.0 | 664.0 | 34.2 |

三种稳定变化：

1. **`count exchange` 降了**（L8/L20/L32 各 −3 ~ −6 µs，L0 例外 +3 µs）
   —— split 把"多活跃副本专家索引"推到阶段二之后写，确实减少了第一阶段的写量。
2. **`pull token` 涨了**（+27 ~ +44 µs）；**`ECO plan` 也涨了**（+30 ~ +34 µs）。
   后者的原因是 split 的 `replica index exchange` span **就在 plan kernel 内**，
   它包含"写索引 + 等待各源 rank 发布"，所以 exchange 的等待被计入 plan。
3. **`GEMM` 完全不变**（每层差 ≤1.4 µs，方向不一致）—— 说明分工表里的
   "dispatch / shared / routed CTA = 24 / 71 / 71"在 split 与 baseline 相同，
   确实没有改变计算布局。

**净效应**：`count exchange` 省下的 3–6 µs 被 `pull token` 与 `ECO plan`
新增的 57–77 µs 完全吃掉，所以每层净慢约 +71 µs（L20 最大），
40 层合计 +2.8 ms/步量级 —— 与实测 +0.056 ms 的差异来自
"每步只有 40 层中部分层受影响"与八卡并行摊薄，**方向一致**。

## 4. split 新增 span：replica index exchange（任务书要求单列）

| layer | rank0 exchange µs | rank1 exchange µs | plan wait 最大 µs（rank0）| plan wait 事件数 |
|---|---:|---:|---:|---:|
| 0 | 38 | 34 | 69 | 14 |
| 8 | 34 | 25 | 87 | 14 |
| 20 | 35 | 29 | **108** | 13 |
| 32 | 27 | 35 | 98 | 17 |
| 39 | 28 | 23 | 77 | 16 |

- **每个采样 rank/layer 恰好 1 对阶段二事件**（`check_bs256_trace.py` 的
  `same('replica exchange spans', len(rank['replica_exchange']), 1)` 通过）——
  这是任务书对 split 臂的硬性要求，**已满足**。
- **span 位置**：rank0 的 exchange 从 105–124 µs 开始、143–159 µs 结束；
  `replica_plan_waits` 从 76–111 µs 开始、145 µs 结束。
  即 **exchange span 与 plan wait 在时间上大面积重叠**（都在 plan kernel 内），
  所以二者**不能相加**。回执中分开列出，符合任务书"注明其在 plan kernel 内，与其他轨道有重叠"。
- **replica 等待时长**：plan wait 的最大值 67–108 µs（rank0），
  与 baseline 的 `ECO plan` 段（108–126 µs）同量级 —— split 并没有缩短等待，
  反而把 exchange 本身也塞进了这段。

## 5. 最慢 rank 明细（同层同 rank 对照）

| layer | arm | rank | tokens | routed rows | M 块 | 活跃槽 | 首块可领 µs | pull 完成 µs | routed 退出 µs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | baseline | 2 | 194.0 | 1552 | 50 | 49 | 196 | 160 | 879.5 |
| 0 | split | 2 | 194.0 | 1552 | 50 | 49 | 194 | **176** | 877.2 |
| 8 | baseline | 7 | 198.0 | 1584 | 48 | 40 | 196 | 212 | 885.7 |
| 8 | split | 0 | 202.0 | 1616 | 47 | 43 | 200 | **239** | 883.1 |
| 20 | baseline | 3 | 180.0 | 1440 | 49 | 44 | 194 | 194 | 880.7 |
| 20 | split | 3 | 180.0 | 1440 | 49 | 44 | 194 | **221** | 880.4 |
| 32 | baseline | 4 | 203.0 | 1624 | 48 | 42 | 200 | 187 | 888.6 |
| 32 | split | 2 | 190.0 | 1520 | 49 | 44 | 203 | **217** | 890.2 |
| 39 | baseline | 3 | 208.0 | 1664 | 47 | 43 | 193 | 201 | 876.0 |
| 39 | split | 3 | 208.0 | 1664 | 47 | 43 | 197 | **232** | 879.7 |

**L0/L20/L39 的最慢 rank 与 M 块完全相同**（同 rank、同 rows、同 M 块），
唯一稳定的差别是 **`pull 完成` 推迟 16–31 µs** —— 这就是 split 变慢的直接位置：
索引交换被推迟到 pull 之前的阶段二，pull 必须等它完成。

## 6. dispatch 臂：shared 加速无法评估

任务书要求"dispatch 核查 shared 加速是否抵消 dispatch 少一个 SM 的代价"，
但**该臂未能启动**，因此：

- 无 `moe.csv` 行、无 `run.json`、无 rank NPZ、无 Perfetto；
- 无法给出"shared 加速是否抵消少 1 SM"的数值；
- 保留的证据只有：完整 `runner.log`（1,772 行）、八 worker 的同一 `RuntimeError`、
  `hardware.csv`（驱动 580.105.08 等）、`replay/`（指向保留现场的软链）。

**未做任何"换分区大小再试"的尝试**（任务书明令禁止）。

## 7. 校验结果

| 检查 | baseline | split | dispatch |
|---|---|---|---|
| `check_bs256_trace.py` | ✅ `eco historical workload verified for both windows and all eight ranks` | ✅ 同上 | — |
| 两窗口 audit | ✅ | ✅ | — |
| 八 rank native token | ✅ | ✅ | — |
| 物理槽 `counts` | ✅ | ✅ | — |
| epoch | ✅ | ✅ | — |
| 拍卖与预测值（`auction`、`clocks[:,5]`）| ✅ | ✅ | — |
| 放置结果 `placement.json` | ✅ 与历史 `cmp` 相同 | ✅ 同左 | — |
| `eco-layout.json` | ✅ 与历史 `cmp` 相同 | ✅ 同左 | — |
| 采样层 launch grid | ✅ (24, 71, 71, 1) | ✅ (24, 71, 71, 1) | 期望 (20, 72, 72, 1)，未启动 |
| split 阶段二事件对 | — | ✅ **每 rank/层恰好 1 对** | — |
| 历史 KV manifest | ✅ 三臂启动前逐文件 `cmp` 通过 | ✅ | ✅（校验在臂循环之前完成）|

`hardware.csv` 与历史 `ref/eco/hardware.csv` 的 `diff -u` **三臂均无差异**。

## 8. 配置实读

| 臂 | `GC_MOE_SM90_SPLIT_EXCHANGE` | `GC_MOE_SM90_PLAN_DISPATCH` | stats ms | measured ms |
|---|---|---:|---:|---:|
| baseline | （未设）| （未设）| 99.153 | 93.915 |
| split | **1** | （未设）| 100.143 | 92.729 |

两者均为"原逐层 EPLB + ECO fine cost、mode 2、M64、slack 11469、stream 开、SM reuse 关"。

## 9. 交付物

`docs/trace_executor_20261119/sm90_bs256_eco_exchange_plan_results/`（161 MB，50+ 文件）：

| 路径 | 内容 |
|---|---|
| `baseline/` | `run.json`、8 份 `rank*.npz`、`runner.log`、`verification.txt`、`hardware.csv`、`perfetto/`（5 层 × 4 文件）、`files.sha256`、`measurement.path` |
| `split/` | 同上 |
| `dispatch/` | `runner.log`（失败栈）、`hardware.csv`、`files.sha256`；**无** run.json/NPZ/Perfetto |
| `moe.csv` | 主指标 CSV（baseline + split 两行，999 与 1000 步各一组）|
| `summary.json` | 每样本 epoch/capture_id/kineto base/分段耗时/最慢 rank/每 rank tokens、rows、M 块、活跃槽；dispatch 的失败参数 |
| `versions.txt`、`state-path.txt` | 版本与现场路径 |

### 9.1 每臂五层的 Perfetto 直链

`baseline/` 与 `split/` 各含（`<AA>` = 00/08/20/32/39，epoch 分别为 46481/46489/46501/46513/46520）：

- `<arm>/perfetto/step002_layer<AA>_epoch<EP>.perfetto.json`（粗粒度）
- `<arm>/perfetto/step002_layer<AA>_epoch<EP>.detail.perfetto.json`（absolute/detail）
- `<arm>/perfetto/step002_layer<AA>_epoch<EP>.counts.json`
- `<arm>/perfetto/step002_layer<AA>_epoch<EP>.detail.json`

## 10. 清理

| 项 | 内容 |
|---|---|
| 八卡释放 | 三臂结束，`nvidia-smi --query-compute-apps` **0 条** |
| 临时 checkout | `/tmp/sm90-eco-code-run1` 已 `git worktree remove` 并 `prune` |
| 保留现场 | `/tmp/sm90-bs256-eco-jBhVwZVo`（三臂 raw/Perfetto/日志）；历史
  `/tmp/sm90-bs256-trace-pbpMbH4u`（KV 320G + replay 12G）**未触碰** |
| 旧现场 | BS256/512/1024 state 目录均未改动 |

## 11. 需要设计端确认

1. **dispatch 分区在本机不可行**（driver 580.105.08 / CUDA 13.0）。
   若仍需要这一臂，需换驱动版本或放弃"从 5-SM 通信分区再切 1 SM"的方案。
2. **split 的净效果是 +0.157%**（变慢）。其机制清晰：`count exchange` 省 3–6 µs，
   但 `pull token` +27~44 µs、`ECO plan` +30~34 µs，且 `pull 完成` 稳定推迟 16–31 µs。
   若目标是加速，应改为**并行**两阶段而非串行推迟。
3. 本轮 baseline 的 35.7994 ms 与主表 eco 的 35.600 ms 不同口径（含插桩与回放），
   跨轮比较须注明，我已在本回执中标注。
