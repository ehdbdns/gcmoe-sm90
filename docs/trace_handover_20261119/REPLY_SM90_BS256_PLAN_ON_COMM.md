# 【SM90】REPLY_SM90_BS256_PLAN_ON_COMM — 首次显著提速：**−5.356%**

> 本回执由 **SM90 执行者**（8×H20）提交。SM100 任务书不在本执行者范围内。

## 0. 结论

| 项 | 结果 |
|---|---|
| baseline | ✅ `exit_code=0`，历史校验全过 |
| **dispatch（plan 移到通信分区）** | ✅ `exit_code=0`，**原生零失配** |
| **主指标（999 步）** | baseline **35.7547** → dispatch **33.8395 ms** → **−5.356%** |
| 1000 步（含采样）| 35.7551 → 33.8400 → **−5.356%** |

**这是本系列第一轮真正变快**（前四轮 single_push 系列分别是 +4.13% / +2.84% / +2.36%）。
负值即为提速，任务书要求的符号已按此约定给出。

## 1. 主表（任务书 §验收 2）

| arm | steps | graph→first | first→last | last→combine | **完整 MoE** | vs baseline |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 999 | 8.0254 | 25.7089 | 2.0204 | **35.7547** | — |
| dispatch | 999 | **6.3890** | 25.4577 | 1.9928 | **33.8395** | **−5.356%** |
| baseline | 1000 | 8.0254 | 25.7091 | 2.0205 | 35.7551 | — |
| dispatch | 1000 | 6.3890 | 25.4579 | 1.9930 | 33.8400 | −5.356% |

**增量全部集中在第一段**：`graph→first` **8.025 → 6.389 ms（−1.636 ms，占总收益 85.3%）**；
`first→last` −0.251 ms（13.1%）；`last→combine` −0.028 ms（1.4%）。

`run.json`：baseline stats **99.479** / measured **92.957**；dispatch stats **96.413** / measured **92.448**。

## 2. 关键验证：日志与 SM 放置（任务书 §验收 1）

### 2.1 `plan context=communication` 在八个 worker 全部出现

```
SM90 ECO plan context=communication communication_sms=6 compute_sms=72
```

**八卡各一条（共 8 条）**，逐字符合任务书要求。

### 2.2 plan 独占 1 张 SM，20 个 dispatch CTA 分布在另 5 张

从 `detail.json` 的 `placement`（由新增的 `rec[1]==27` 记录导出）：

| layer | rank | plan（role 5）的 SM | dispatch（role 1）的 SM 集合 |
|---:|---:|---|---|
| 0 | 0 | **{72}** | {73, 74, 75, 76, 77} |
| 0 | 1 | {74} | {72, 73, 75, 76, 77} |
| 8 | 0 | **{72}** | {73, 74, 75, 76, 77} |
| 8 | 1 | {74} | {72, 73, 75, 76, 77} |
| 20 | 0 | **{72}** | {73, 74, 75, 76, 77} |
| 20 | 1 | {74} | {72, 73, 75, 76, 77} |
| 32 | 0 | **{72}** | {73, 74, 75, 76, 77} |
| 32 | 1 | {72} | {73, 74, 75, 76, 77} |
| 39 | 0 | {74} | {72, 73, 75, 76, 77} |
| 39 | 1 | {72} | {73, 74, 75, 76, 77} |

**plan 恒占 1 张 SM，dispatch 的 20 个 CTA 恒占另 5 张（role 集合大小 5，每个 SM 4 个 CTA）**
—— 完全符合任务书"plan 独占一张 SM、dispatch 在另五张上"。

### 2.3 gate 结束远早于 dispatch

| layer | arm | gate 结束 µs | dispatch 完成 µs | shared 完成 µs | plan 完成 µs | 首个 routed CTA µs | combine 完成 µs |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | baseline | — | 180.0 | 194.8 | 113.8 | 847.4 | 912.4 |
| 0 | dispatch | **3.5** | 201.3 | **163.3** | 113.6 | **814.5** | **863.6** |
| 8 | baseline | — | 215.9 | 193.6 | 122.8 | 875.7 | 906.7 |
| 8 | dispatch | **5.3** | 230.4 | **155.8** | 127.9 | **834.3** | **864.7** |
| 20 | baseline | — | 217.2 | 193.9 | 128.9 | 822.8 | 906.6 |
| 20 | dispatch | **4.3** | 222.0 | **158.0** | 131.9 | **784.4** | **848.6** |
| 32 | baseline | — | 194.9 | 195.0 | 122.2 | 866.8 | 913.1 |
| 32 | dispatch | **4.1** | 212.5 | **155.9** | 125.9 | **806.2** | **865.6** |
| 39 | baseline | — | 173.7 | 193.1 | 111.5 | 866.8 | 897.0 |
| 39 | dispatch | **5.4** | 207.3 | **152.2** | 119.3 | **822.8** | **852.6** |

（全部为 rank0、相对本层 `graph_start_us`，单位 µs；baseline 无 gate，因为 plan 在 compute context。）

三点读数：

1. **gate 在 3.5–5.4 µs 就结束**，远早于 dispatch 的 201–230 µs ——
   即"plan 首次驻留 → 放行 20 个 dispatch CTA"这个握手**极快**，不是瓶颈。
2. **`shared 完成` 从 193–195 提前到 152–163 µs（−31 ~ −41 µs）** ——
   这是 72 SM 计算分区的直接收益。
3. **首个 routed CTA 提前 32–43 µs**（如 L20 822.8 → 784.4），
   **`combine 完成` 提前 43–58 µs** —— 收益一路传导到层末。

**代价**：`dispatch 完成` 推迟 8–34 µs（`dispatch` SM 从 24 降到 20，每 SM 4 个 CTA 而非 ~4.8），
但被 `shared` 的收益覆盖。

**重叠说明**：`plan` 与 `dispatch` 在时间上**大量重叠**（plan 113–132 µs 完成、dispatch 201–230 µs 完成），
因此两者耗时**不可相加**；本节只列各自实测，不做求和。

## 3. 分段耗时（八 rank 均值，µs）

| layer | arm | count exch | pull token | **shared L1+L2** | ECO plan | wait token | GEMM | combine |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | baseline | 32.6 | 139.4 | **195.6** | 110.7 | 2.0 | 663.8 | 42.8 |
| 0 | dispatch | 28.9 | 150.5 | **153.3** | 110.6 | 2.2 | 658.8 | 39.1 |
| 8 | baseline | 24.4 | 174.2 | **194.9** | 122.6 | 2.1 | 664.7 | 38.7 |
| 8 | dispatch | 22.4 | 183.0 | **156.2** | 126.0 | 2.6 | 659.2 | 42.3 |
| 20 | baseline | 23.6 | 193.5 | **196.2** | 126.1 | 2.0 | 658.1 | 45.5 |
| 20 | dispatch | 21.4 | 203.1 | **153.3** | 131.5 | 2.2 | 651.9 | 34.6 |
| 32 | baseline | 25.0 | 174.7 | **196.8** | 118.3 | 2.0 | 668.7 | 37.5 |
| 32 | dispatch | 22.8 | 184.0 | **154.5** | 125.8 | 2.2 | 661.3 | 42.0 |
| 39 | baseline | 22.0 | 155.3 | **195.4** | 110.9 | 2.0 | 662.3 | 33.9 |
| 39 | dispatch | 21.8 | 168.4 | **154.3** | 113.8 | 2.1 | 657.4 | 35.8 |

**唯一的大幅改善是 `shared L1+L2`：195–197 → 153–156 µs（−40 ~ −43 µs/层）** ——
正是 72 compute SM 的收益。

其余段：`pull token` +9 ~ +13 µs（dispatch SM 少 4 个）、`ECO plan` ±5 µs 持平、
`count exchange` −2 ~ −4 µs 略降、`GEMM` −4 ~ −7 µs 略降、`combine` 波动。

**净**：shared 省 41 µs、pull 花 11 µs，每层净 −30 µs 量级；
40 层摊薄后与实测 −1.92 ms/步同向且量级一致。

## 4. 十份 absolute trace（任务书 §验收 3）

逐份核对：同目录 `*.detail.json` 与 `*.counts.json` 的
**step=2、layer、epoch、capture_id 全部一致，各覆盖 8 个 rank** —— **10/10 通过**。

| layer | epoch | baseline absolute trace | dispatch absolute trace |
|---:|---:|---|---|
| 0 | 46481 | [baseline L0](docs/trace_executor_20261119/sm90_bs256_plan_on_comm_results/baseline/perfetto/step002_layer00_epoch46481.detail.perfetto.json) | [dispatch L0](docs/trace_executor_20261119/sm90_bs256_plan_on_comm_results/dispatch/perfetto/step002_layer00_epoch46481.detail.perfetto.json) |
| 8 | 46489 | [baseline L8](docs/trace_executor_20261119/sm90_bs256_plan_on_comm_results/baseline/perfetto/step002_layer08_epoch46489.detail.perfetto.json) | [dispatch L8](docs/trace_executor_20261119/sm90_bs256_plan_on_comm_results/dispatch/perfetto/step002_layer08_epoch46489.detail.perfetto.json) |
| 20 | 46501 | [baseline L20](docs/trace_executor_20261119/sm90_bs256_plan_on_comm_results/baseline/perfetto/step002_layer20_epoch46501.detail.perfetto.json) | [dispatch L20](docs/trace_executor_20261119/sm90_bs256_plan_on_comm_results/dispatch/perfetto/step002_layer20_epoch46501.detail.perfetto.json) |
| 32 | 46513 | [baseline L32](docs/trace_executor_20261119/sm90_bs256_plan_on_comm_results/baseline/perfetto/step002_layer32_epoch46513.detail.perfetto.json) | [dispatch L32](docs/trace_executor_20261119/sm90_bs256_plan_on_comm_results/dispatch/perfetto/step002_layer32_epoch46513.detail.perfetto.json) |
| 39 | 46520 | [baseline L39](docs/trace_executor_20261119/sm90_bs256_plan_on_comm_results/baseline/perfetto/step002_layer39_epoch46520.detail.perfetto.json) | [dispatch L39](docs/trace_executor_20261119/sm90_bs256_plan_on_comm_results/dispatch/perfetto/step002_layer39_epoch46520.detail.perfetto.json) |

同目录另有 `*.detail.json`（CTA 级 pull/领号/计算/写回 + `placement`）
与 `*.counts.json`（各 rank token、M 块、活跃槽）。

## 5. 正确性校验

| 检查 | baseline | dispatch |
|---|---|---|
| `check_bs256_trace.py` | ✅ `eco historical workload verified for both windows and all eight ranks` | ✅ 同上 |
| 两窗口八 rank 输入与原生输出 | ✅ | ✅ **零失配** |
| 逐槽 counts / 拍卖 / 预测值 / 放置 / epoch | ✅ | ✅ |
| `eco-layout.json` | ✅ | ✅ |
| 期望 grid (24,71,71,1) vs (20,72,72,1) | ✅ | ✅ 通过 |

## 6. 执行者做的一处语义等价修复（任务书 §验收 4 允许）

**文件**：`scripts/trace_layer/pull_gemm_detail.py`（完整 diff 见 `executor_impl_fix.diff`）

**原因**：`aligned_step.py` 在本轮新增了 `admission` 角色 span
（1×32 grid 的 gate CTA，Perfetto 里名为 `wait for resident plan CTA`），
但 `pull_gemm_detail.py` 的粗粒度交叉校验**硬编码** `RANKS*7` 个 span。
dispatch 臂每 rank 实际有 8 个 span，因此解析在
`pull/GEMM trace: coarse Perfetto spans differ from the aligned counts` 处中止。

**修复**：把 `cat == 'admission'` 的 span 从 segment 匹配中排除，
并额外要求"每个 rank 恰 0 或 1 个 gate span"（保持严格性）。
**语义不变**：段划分、计时边界、校验口径、分流与采样全部未改。

**影响**：两臂使用**同一 SHA**（`63b70e6`）+ 这处解析修复，
因此配对对照仍然成立。**模型行为未受影响**（修复只在事后解析阶段生效）。

**另注**：本轮取代 `a137540` 的 5-CTA 实验；**本执行者未运行过该旧版**，
其 SHA 在此单独标记以符合任务书要求。

## 7. 交付物

`docs/trace_executor_20261119/sm90_bs256_plan_on_comm_results/`（73 个文件，161 MB）：

| 路径 | 内容 |
|---|---|
| `baseline/`、`dispatch/` | `run.json`、8 份 `rank*.npz`、`runner.log`、`verification.txt`、`hardware.csv`、`measurement.path`、`status.txt`、`runtime.log`、`perfetto/`（5 层 × 4 文件）|
| `moe.csv` | 两臂 × 999/1000 步，含带符号变化率 |
| `summary.json` | 每样本 epoch/capture_id/分段耗时/rank0 相对时间/每 rank token·M 块；`placement` 与修复说明 |
| `executor_impl_fix.diff` | 上述解析修复的完整 diff |
| `versions.txt`、`state-path.txt` | 版本与现场 |

## 8. 清理

| 项 | 结果 |
|---|---|
| 作业退出 | 两臂结束，`nvidia-smi --query-compute-apps` **0 条** |
| 临时 checkout | `/tmp/sm90-plan-comm-code-run1` 已 `git worktree remove` 并 `prune` |
| 原始证据 | `/tmp/sm90-bs256-eco-J7wY5kWu`（两臂 raw/Perfetto）保留 |
| 历史现场 | `/tmp/sm90-bs256-trace-pbpMbH4u`（KV + replay）**未触碰** |
