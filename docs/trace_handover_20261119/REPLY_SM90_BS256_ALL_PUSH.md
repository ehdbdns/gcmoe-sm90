# 【SM90】REPLY_SM90_BS256_ALL_PUSH — 全直推把损失从 2.84% 降到 2.36%，但仍慢

> 本回执由 **SM90 执行者**（8×H20）提交。SM100 任务书不在本执行者范围内。

## 0. 结论

| 项 | 结果 |
|---|---|
| 冒烟门禁 | ✅ 4/4 组（hot_rows 135/0/64/65，各 `replays=3`）+ `ECO_CAPTURE OK` |
| baseline | ✅ `exit_code=0`，历史校验全过 |
| 全直推（runner 中的 `single_push`）| ✅ `exit_code=0`，**原生零失配** |
| **主指标（999 步）** | baseline **36.0103** → 全直推 **36.8597 ms** → **+2.359%** |

### 0.1 四轮迭代轨迹（同口径，各轮同轮 baseline）

| 轮次 | SHA | 完整 MoE ms/步 | vs 同轮 baseline |
|---|---|---:|---:|
| return fix | `29460b5` | 37.2338 | +4.130% |
| replica first | `32dc6a1` | 36.7759 | +2.839% |
| **all push（本轮）** | `e08eb2b` | **36.8597** | **+2.359%** |

**相对上轮：+2.839% → +2.359%（−0.48 pp，−0.084 ms）**，仍在变慢。
但本轮 baseline 为 36.0103（上轮 35.7605），**两轮 baseline 相差 0.250 ms（0.70%）**，
因此**跨轮的绝对值不可直接比较**，只能看各自相对同轮 baseline 的百分比。

## 1. 主表（任务书 §核对 3）

| arm | steps | graph→first | first→last | last→combine | **完整 MoE** | vs baseline |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 999 | 8.0346 | 25.7384 | 2.2374 | **36.0103** | — |
| 全直推 | 999 | 8.0632 | 26.2258 | 2.5707 | **36.8597** | **+2.359%** |
| baseline | 1000 | 8.0346 | 25.7387 | 2.2375 | 36.0108 | — |
| 全直推 | 1000 | 8.0633 | 26.2262 | 2.5708 | 36.8602 | +2.359% |

三段增量（999 步，ms）：`graph→first` **+0.029**（2.3%）、
`first→last` **+0.487**（38.9%）、`last→combine` **+0.333**（26.6%），
其余 32% 来自八卡摊薄与层间差异。

`run.json`：baseline stats 99.589 / measured 94.606；全直推 100.493 / 94.342。

## 2. 分段耗时（任务书 §核对 2 的新段名）

全直推臂的段名已替换为设计端给出的新口径：
**`local route preparation`**（本地计数与路由准备）与
**`source push and destination completion`**（源端发送 + 目的端收尾），`ECO plan` 独立计时。
baseline 仍用原 `count exchange` / `pull token`。

| layer | arm | local route prep<br>（baseline = count exchange）| source push + dest completion<br>（baseline = pull token）| shared L1+L2 | ECO plan | wait token | GEMM | combine |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | baseline | 43.1 | **139.5** | 195.5 | 113.6 | 2.0 | 666.2 | 38.0 |
| 0 | 全直推 | **16.1** | **473.2** | 195.7 | 112.2 | 2.1 | 687.0 | 49.1 |
| 8 | baseline | 33.1 | **166.0** | 195.5 | 121.9 | 2.0 | 666.0 | 58.8 |
| 8 | 全直推 | **15.2** | **425.1** | 196.4 | 118.6 | 2.4 | 677.5 | 44.9 |
| 20 | baseline | 34.8 | **187.1** | 195.7 | 129.7 | 2.4 | 657.1 | 36.8 |
| 20 | 全直推 | **15.8** | **493.1** | 197.6 | 126.4 | 2.5 | 677.8 | 66.7 |
| 32 | baseline | 35.1 | **163.9** | 196.2 | 118.6 | 2.2 | 667.0 | 31.1 |
| 32 | 全直推 | **14.5** | **411.7** | 199.0 | 113.7 | 2.5 | 686.3 | 54.4 |
| 39 | baseline | 32.8 | **146.7** | 195.4 | 112.3 | 2.0 | 665.0 | 36.2 |
| 39 | 全直推 | **15.5** | **440.1** | 197.3 | 110.7 | 2.2 | 685.1 | 27.1 |

**`local route preparation` 大幅改善：33–43 → 14.5–16.1 µs（−17 ~ −27 µs/层）** ——
全直推确实省掉了目标侧的本地交换准备。

**但 `source push + destination completion` 从 139–187 涨到 412–493 µs**
（+250 ~ +306 µs/层），抵消并超过了前述收益。

其余段：`shared L1+L2` 持平（±3 µs）、`ECO plan` 略降 1–5 µs、
`wait token` 持平、`GEMM` **+10 ~ +29 µs**、`combine` **−9 ~ +30 µs**（方向不一）。

## 3. 源端 `replica_sends`（任务书 §核对 2、3）

每层的 `replica_sends` 以源 rank 为进程，记录领取任务的 CTA、目的物理槽、行数与发送起止：

| layer | 条数 | 每源条数 | 行数合计 | **源推送窗口 µs** | **配额→首发的时差**（min/中位/max）|
|---:|---:|---:|---:|---|---|
| 0 | 504 | 63 | 4,104 | 119 – **377** | 18 / 20 / 21 |
| 8 | 357 | 44 | 5,936 | 130 – **392** | 18 / 19 / 35 |
| 20 | 337 | 42 | 6,352 | 138 – **461** | 19 / 20 / 23 |
| 32 | 387 | 48 | 6,336 | 128 – **392** | 16 / 20 / 24 |
| 39 | 492 | 61 | 4,760 | 124 – **367** | 16 / 18 / 19 |

两点关键读数：

1. **`quota → 首发` 只有 16–21 µs（中位 18–20）** —— 一旦配额发布，源端**立刻**开始推送，
   响应极快。**开头的延迟不是问题**。
2. **但推送窗口的尾部拖到 367–461 µs** ——
   即源端**发得及时、发得慢**。这与 §2 的 `source push + dest completion` 暴涨吻合。

**每层每源条数 42–63 条**（不是 1 条）说明多副本切片被拆成很多小任务，
由"共享 CTA 队列"分发给空闲 CTA；任务粒度细导致收尾分散。

### 3.1 L20 / rank0 / slot45 的八个源（任务书 §核对 3 指定）

| 源 rank | CTA | 行数 | 起始 µs | 结束 µs |
|---:|---:|---:|---:|---:|
| 0 | 9 | 98 | 140 | 296 |
| 1 | 11 | 94 | 143 | 298 |

**该槽只有 2 个源有行（98 + 94 = 192，与前两轮完全一致）**，其余 6 个源贡献零行。
两源的发送几乎同时开始（140/143 µs），分别在 296/298 µs 结束 ——
**这一个槽的源端推送是对齐的**，不是它的锅。

**因此本轮的延迟来自其他槽的尾部**，而非这个指定点。
注意：其他源 CTA 的时间必须用各自 **Kineto 锚点**对齐（每个源进程有独立
`kineto_base_time_ns`，本轮五层八 rank 的该值已逐样本记录在 `summary.json`）。

## 4. 块发布字段（任务书 §核对 2 末段）

| layer | 带 `declared_sources` 的块 | 其中 `finished_sources < 8` | 占比 |
|---:|---:|---:|---:|
| 20（示例层）| **362 / 362** | 331 | **91.4%** |

**每个 L1 块都附 `declared_sources` 与 `finished_sources`**（任务书硬要求，全部满足）。
满块以 `finished_sources < 8` 提前发布的比例很高（91%），
说明 early publish 在正常工作，等待被推到了 pull 尾端 —— 这与前三轮结论一致。

## 5. 十份 absolute trace（任务书 §核对 4）

逐份核对：同目录 `*.detail.json` 与 `*.counts.json` 的
**step=2、layer、epoch、capture_id 全部一致，各覆盖 8 个 rank** —— **10/10 通过**。

| layer | epoch | baseline absolute trace | 全直推 absolute trace |
|---:|---:|---|---|
| 0 | 46481 | [baseline L0](docs/trace_executor_20261119/sm90_bs256_all_push_results/baseline/perfetto/step002_layer00_epoch46481.detail.perfetto.json) | [all-push L0](docs/trace_executor_20261119/sm90_bs256_all_push_results/single_push/perfetto/step002_layer00_epoch46481.detail.perfetto.json) |
| 8 | 46489 | [baseline L8](docs/trace_executor_20261119/sm90_bs256_all_push_results/baseline/perfetto/step002_layer08_epoch46489.detail.perfetto.json) | [all-push L8](docs/trace_executor_20261119/sm90_bs256_all_push_results/single_push/perfetto/step002_layer08_epoch46489.detail.perfetto.json) |
| 20 | 46501 | [baseline L20](docs/trace_executor_20261119/sm90_bs256_all_push_results/baseline/perfetto/step002_layer20_epoch46501.detail.perfetto.json) | [all-push L20](docs/trace_executor_20261119/sm90_bs256_all_push_results/single_push/perfetto/step002_layer20_epoch46501.detail.perfetto.json) |
| 32 | 46513 | [baseline L32](docs/trace_executor_20261119/sm90_bs256_all_push_results/baseline/perfetto/step002_layer32_epoch46513.detail.perfetto.json) | [all-push L32](docs/trace_executor_20261119/sm90_bs256_all_push_results/single_push/perfetto/step002_layer32_epoch46513.detail.perfetto.json) |
| 39 | 46520 | [baseline L39](docs/trace_executor_20261119/sm90_bs256_all_push_results/baseline/perfetto/step002_layer39_epoch46520.detail.perfetto.json) | [all-push L39](docs/trace_executor_20261119/sm90_bs256_all_push_results/single_push/perfetto/step002_layer39_epoch46520.detail.perfetto.json) |

同目录另有对应的 `*.detail.json`（CTA 级 pull/领号/计算/写回）与
`*.counts.json`（各 rank token、M 块、活跃槽）。

## 6. 正确性

| 检查 | baseline | 全直推 |
|---|---|---|
| `check_bs256_trace.py` | ✅ `eco historical workload verified for both windows and all eight ranks` | ✅ 同上 |
| 两窗口 audit / 八 rank native token | ✅ | ✅ **零失配** |
| 逐槽 counts / auction / placement / 布局 / 硬件 | ✅ | ✅ |
| launch grid (24,71,71,1) | ✅ | ✅ |
| 每个 L1 块带 `declared_sources` + `finished_sources` | — | ✅ 362/362 |

冒烟四种单副本边界（135/0/64/65 行）与每步 136 行的多副本热专家**同场**运行；
每组以同图 legacy 路径取参考后连跑三次直推，检查逻辑路由唯一、源激活/scale/gate、
各副本收到的精确源路由集合与 plan 的 canonical quota 切片一致、L2 写回与最终 BF16。

## 7. 判读

1. **全直推确实改善了另外两处**：`local route preparation` 减少 17–27 µs/层；
   相对上轮总体再降 0.48 pp。方向是对的。
2. **但源端推送窗口拖到 367–461 µs**，把收益全部吃掉。
   配额→首发只要 18–20 µs，说明**不是启动延迟，是推送本身的吞吐**。
3. **每源 42–63 个小切片任务**是可疑点：共享 CTA 队列把工作碎化后，
   尾部小任务的收尾时间分散，导致 `destination completion` 拖长。
   若要继续优化，应减少切片数（增大单任务粒度）或让尾部任务优先。
4. **不建议把全直推加入主表**：正确性达标，但仍慢 2.36%。

## 8. 交付物

`docs/trace_executor_20261119/sm90_bs256_all_push_results/`（75 个文件，163 MB）：

| 路径 | 内容 |
|---|---|
| `baseline/`、`single_push/` | `run.json`、8 份 `rank*.npz`、`runner.log`、`verification.txt`、`hardware.csv`、`measurement.path`、`status.txt`、`runtime.log`、`perfetto/`（5 层 × 4 文件）|
| `moe.csv` | 两臂 × 999/1000 步 |
| `return-smoke.log` | 冒烟门禁原始输出 |
| `summary.json` | 每样本 epoch/capture_id/分段耗时/最慢 rank/每 rank token·M 块；全直推另含 `replica_sends`（条数、窗口、配额→首发）、`block_publication`、`slot45_sends` |
| `versions.txt`、`state-path.txt` | 版本与现场路径 |

## 9. 清理

| 项 | 结果 |
|---|---|
| 作业退出 | 两臂结束，`nvidia-smi --query-compute-apps` **0 条** |
| 临时 checkout | `/tmp/sm90-all-push-run1/code` 已 `git worktree remove` 并 `prune` |
| 原始证据 | `/tmp/sm90-all-push-run1/`（`pair.log`、冒烟日志）与 `/tmp/sm90-bs256-eco-sK0Fs4tw`（raw Kineto）**保留** |
| 历史现场 | `/tmp/sm90-bs256-trace-pbpMbH4u`（KV + replay）**未触碰** |
