# 【SM90】REPLY_SM90_BS256_SOURCE_INBOX — 首块**没有提前**（40 配对全变晚），MoE +139.33%

> 本回执由 **SM90 执行者**（8×H20）提交。SM100 任务书不在本执行者范围内。

## 0. 结论：**不满足本轮目标**

任务书 §判读 3 明确定义了通过条件：

> 首块发布提前且完整 MoE 持平或降低，才支持本轮目标。

本轮两条都**不成立**：

| 判据 | 结果 |
|---|---|
| 首块发布提前 | ❌ **0/40 提前**（80 配对中全为变晚），最早可领取块晚 43.5–63.3 µs，首个**满块**晚 396.7–444.5 µs |
| 完整 MoE 持平或降低 | ❌ **+139.33%**（33.8576 → 81.0303 ms/步）|

两臂均 `exit_code=0`、原生零失配、历史校验全过 —— **正确性没问题，纯性能反向**。

## 1. 主表（`moe.csv`，999 步/1000 步）

| arm | steps | graph→first | first→last | last→combine | **完整 MoE** | vs baseline |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 999 | 6.3946 | 25.4511 | 2.0119 | **33.8576** | — |
| source_inbox | 999 | 6.4376 | **53.3767** | **21.2160** | **81.0303** | **+139.326%** |
| baseline | 1000 | 6.3947 | 25.4517 | 2.0116 | 33.8580 | — |
| source_inbox | 1000 | 6.4377 | 53.3678 | 21.2100 | 81.0155 | +139.280% |

`run.json`：baseline stats **96.743** / measured **92.549**；
source_inbox stats **182.549** / measured **139.439**。

**增量分布完全不同于历史各轮**：`first→last` +27.93 ms（翻倍），
`last→combine` **+19.20 ms（约 10.5 倍）**。首段几乎不变（+0.043 ms）。

## 2. `first_m_block.csv`（任务书 §判读 2，**核心判据**）

每个采样层、每 rank、两种块类型，均以"本地 MoE graph 起点"为零点（µs）。

### 2.1 首次发布**任意**可领取 L1 块

| layer | baseline 均值 | inbox 均值 | Δ 均值 | 有效配对 | 提前/持平/变晚 | Δ 中位 |
|---:|---:|---:|---:|---:|---|---:|
| 0 | 64.0 | 114.8 | **+50.8** | 8 | **0 / 0 / 8** | +39.4 |
| 8 | 54.1 | 117.0 | **+62.9** | 8 | **0 / 0 / 8** | +66.4 |
| 20 | 49.6 | 112.9 | **+63.3** | 8 | **0 / 0 / 8** | +55.8 |
| 32 | 53.9 | 97.4 | **+43.5** | 8 | **0 / 0 / 8** | +41.3 |
| 39 | 51.2 | 97.5 | **+46.3** | 8 | **0 / 0 / 8** | +36.5 |

### 2.2 首次发布**满** M 块

| layer | baseline 均值 | inbox 均值 | Δ 均值 | 有效配对 | 提前/持平/变晚 | Δ 中位 |
|---:|---:|---:|---:|---:|---|---:|
| 0 | 104.0 | 545.3 | **+441.3** | 8 | **0 / 0 / 8** | +406.1 |
| 8 | 76.6 | 489.5 | **+412.9** | 8 | **0 / 0 / 8** | +412.8 |
| 20 | 83.5 | 528.0 | **+444.5** | 8 | **0 / 0 / 8** | +410.4 |
| 32 | 95.5 | 524.5 | **+429.0** | 8 | **0 / 0 / 8** | +403.1 |
| 39 | 83.3 | 480.0 | **+396.7** | 8 | **0 / 0 / 8** | +365.7 |

**80 个配对（5 层 × 8 rank × 2 种块）中没有一个是提前的**，
也没有持平。**任务书要求的"提前"方向完全没有出现。**

## 3. 分段耗时（八 rank 均值，µs，L20 为代表）

| arm | local route prep | source push + dest completion<br>（baseline = count exch + pull token）| shared L1+L2 | ECO plan | wait token | **GEMM** | **combine** |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 21.5（count exch）| 205.2（pull token）| 153.2 | 133.0 | 2.5 | **651.3** | **39.2** |
| source_inbox | 13.8 | **1439.1** | 154.5 | 131.7 | 3.8 | **1345.9** | **360.0** |

**三个量级跳变**：

1. `source push and destination completion` **205 → 1439 µs（7.0×）**；
2. **`GEMM` 651 → 1346 µs（2.07×）** —— **计算本身翻倍，这是最严重的异常**；
3. **`combine` 39 → 360 µs（9.2×）**。

`local route preparation` 反而从 21.5 降到 13.8 µs（inbox 省掉了原 count exchange），
`shared L1+L2`、`ECO plan`、`wait token` 基本不变。

**关于"块变碎"的推断已被实测否定**：两臂的 `m_blocks` 与 `routed_rows`
**逐层完全相同**（见 §3.1），所以 GEMM 翻倍**不是**块数或块大小变化引起的，
而是 **GEMM 阶段本身在等数据**（inbox 的行到达时机改变，使 CTA 在块内等待）。
这与 §2 的"满块晚 400+ µs"是同一现象的两个侧面：满块难凑齐，GEMM 就空等。

### 3.1 两臂 M 块与行数完全一致（八 rank 合计）

| layer | arm | routed rows | M 块 | 行/块 |
|---:|---|---:|---:|---:|
| 0 | baseline | 12,288 | 393 | 31.3 |
| 0 | source_inbox | 12,288 | **393** | 31.3 |
| 8 | baseline | 12,288 | 377 | 32.6 |
| 8 | source_inbox | 12,288 | **377** | 32.6 |
| 20 | baseline | 12,288 | 362 | 33.9 |
| 20 | source_inbox | 12,288 | **362** | 33.9 |
| 32 | baseline | 12,288 | 375 | 32.8 |
| 32 | source_inbox | 12,288 | **375** | 32.8 |
| 39 | baseline | 12,288 | 385 | 31.9 |
| 39 | source_inbox | 12,288 | **385** | 31.9 |

**五层的行数与 M 块数在两臂之间逐值相同**，因此 GEMM 的 2.07× 与 combine 的 9.2×
**不能**用"块变碎/块变小"解释；它们是同一批块在**等待数据**。

## 4. 正确性校验（两臂）

| 检查 | baseline | source_inbox |
|---|---|---|
| `check_bs256_trace.py` | ✅ `eco historical workload verified for both windows and all eight ranks` | ✅ 同上 |
| 两窗口八 rank 原生输出 | ✅ | ✅ **零失配** |
| 逐层逐槽 token 数 / 放置 / 拍卖 | ✅ | ✅ |
| 实际 grid (20, 72, 72, 1) | ✅ | ✅ |
| 退出码 | 0 | 0 |

**两臂配置实读**：baseline `GC_MOE_SM90_SOURCE_INBOX` 未设；
source_inbox **=1**；两臂 `GC_MOE_SM90_PLAN_DISPATCH` **均为 1**（plan 都在通信分区），
符合任务书"两臂 plan 均在通信分区、CTA 数均为 20/72/72/1、均关闭 SM 复用"。

## 5. 五层 absolute Perfetto 与 counts/detail（任务书 §判读 2 末段）

| layer | epoch | baseline | source_inbox |
|---:|---:|---|---|
| 0 | 46481 | [baseline L0](docs/trace_executor_20261119/sm90_bs256_source_inbox_results/baseline/perfetto/step002_layer00_epoch46481.detail.perfetto.json) | [inbox L0](docs/trace_executor_20261119/sm90_bs256_source_inbox_results/source_inbox/perfetto/step002_layer00_epoch46481.detail.perfetto.json) |
| 8 | 46489 | [baseline L8](docs/trace_executor_20261119/sm90_bs256_source_inbox_results/baseline/perfetto/step002_layer08_epoch46489.detail.perfetto.json) | [inbox L8](docs/trace_executor_20261119/sm90_bs256_source_inbox_results/source_inbox/perfetto/step002_layer08_epoch46489.detail.perfetto.json) |
| 20 | 46501 | [baseline L20](docs/trace_executor_20261119/sm90_bs256_source_inbox_results/baseline/perfetto/step002_layer20_epoch46501.detail.perfetto.json) | [inbox L20](docs/trace_executor_20261119/sm90_bs256_source_inbox_results/source_inbox/perfetto/step002_layer20_epoch46501.detail.perfetto.json) |
| 32 | 46513 | [baseline L32](docs/trace_executor_20261119/sm90_bs256_source_inbox_results/baseline/perfetto/step002_layer32_epoch46513.detail.perfetto.json) | [inbox L32](docs/trace_executor_20261119/sm90_bs256_source_inbox_results/source_inbox/perfetto/step002_layer32_epoch46513.detail.perfetto.json) |
| 39 | 46520 | [baseline L39](docs/trace_executor_20261119/sm90_bs256_source_inbox_results/baseline/perfetto/step002_layer39_epoch46520.detail.perfetto.json) | [inbox L39](docs/trace_executor_20261119/sm90_bs256_source_inbox_results/source_inbox/perfetto/step002_layer39_epoch46520.detail.perfetto.json) |

同目录另有对应的 `*.detail.json`（CTA 级 pull/领号/计算/写回）
与 `*.counts.json`（**每 rank 的 token 与 M 块数**）。

## 6. 不确定性说明（任务书 §判读 3 要求）

**本轮不是"接近计时波动"的情况，不需要依赖波动判读**：

- MoE 变化 **+139.33%**（+47.17 ms），是任务书关注的量级的**几十倍**；
- 首块 80/80 配对全部变晚，方向单一，无任何提前样本；
- 三段耗时（first→last、last→combine）均翻倍以上，不是单段噪声；
- GEMM 本身 2.07×，不可能由计时误差产生。

**因此结论是确定的：source_inbox 在本任务书的配置下显著劣化。**

## 7. 交付物

`docs/trace_executor_20261119/sm90_bs256_source_inbox_results/`（168 MB）：

| 路径 | 内容 |
|---|---|
| **`first_m_block.csv`** | 80 行：step/layer/rank/block_kind/两臂原值/差值 |
| **`moe.csv`** | 两臂 × 999/1000 步 + 带符号变化率 |
| `baseline/`、`source_inbox/` | `run.json`、8 份 `rank*.npz`、`runner.log`、`verification.txt`、`hardware.csv`、`measurement.path`、`status.txt`、`runtime.log`、`perfetto/`（5 层 × 4 文件）|
| `versions.txt`、`state-path.txt` | 版本与现场 |

## 8. 清理

| 项 | 结果 |
|---|---|
| 作业退出 | 两臂结束，`nvidia-smi --query-compute-apps` **0 条** |
| 临时 checkout | `/tmp/sm90-source-inbox-code-run1` 已 `git worktree remove` 并 `prune` |
| 原始证据 | `/tmp/sm90-bs256-eco-wPicRRss`（两臂 raw/Perfetto）保留 |
| 历史现场 | `/tmp/sm90-bs256-trace-pbpMbH4u`（KV + replay）**未触碰** |

## 9. 给设计端的机制提示

1. **GEMM 翻倍（2.07×）是最大异常**，且 combine 同时 9.2×。
   这与"块的有效行数变小"一致：inbox 改变了 token 到达顺序/时机，
   使 M 块凑不满 64 行就进入 GEMM。
2. **满块首发布晚 400+ µs** 直接印证：inbox 让"凑满一个 64 行块"变慢了。
3. 若目标是"首块更早可领"，inbox 的当前实现**方向相反**：
   它让块更碎，但每个碎块都要走完整 GEMM/combine 固定开销。
4. **每层 M 块数与行数两臂完全相同**（§3.1），因此问题不在分块策略，
   而在 inbox 改变了行的**到达时序**，让 GEMM/combine 在块内空等。
   建议下一步测量"块内首个行到达 → 块凑满"的分布，而不是继续调整块大小。
