# 【SM90】REPLY_SM90_BS256_PUSH_REPLICA_FIRST — 副本等待从 250 µs 降到 4 µs，但 MoE 仍 +2.84%

> 本回执由 **SM90 执行者**（8×H20）提交。SM100 任务书不在本执行者范围内。

## 0. 结论

| 项 | 结果 |
|---|---|
| 冒烟门禁 | ✅ 4/4 组（hot_rows 135/0/64/65，各 `replays=3`）+ `ECO_CAPTURE OK` |
| baseline | ✅ `exit_code=0`，历史校验全过 |
| single_push（新版）| ✅ `exit_code=0`，**原生零失配** |
| **主指标（999 步）** | baseline **35.7605** → single_push **36.7759 ms** → **+2.839%** |
| 旧版 single_push（上轮）| 37.2338 ms，**+4.130%** |

**目标点完全命中**：`L20 / rank0 / slot45` 的副本等待从旧版 **250 µs** 降到新版 **4 µs**
（baseline 3 µs），新版所有副本槽的 `prior_single_slots` **全为 0**。

**但整体只从 +4.13% 改善到 +2.84%，仍然更慢**。原因已定位：
瓶颈从"副本等待"转移到了**单副本推送的串行窗口**（§4）。

## 1. 主表（任务书 §验收 2）

| arm | steps | graph→first | first→last | last→combine | **完整 MoE** | vs 同轮 baseline |
|---|---:|---:|---:|---:|---:|---:|
| baseline（本轮）| 999 | 8.0186 | 25.7008 | 2.0412 | **35.7605** | — |
| **single_push（新）** | 999 | 8.0498 | 26.2221 | 2.5040 | **36.7759** | **+2.839%** |
| baseline（本轮）| 1000 | 8.0187 | 25.7010 | 2.0413 | 35.7610 | — |
| single_push（新）| 1000 | 8.0499 | 26.2226 | 2.5041 | 36.7765 | +2.840% |

### 1.1 跨轮对照

| 指标 | 旧 single_push `29460b5` | 新 single_push `32dc6a1` | 改善 |
|---|---:|---:|---:|
| 完整 MoE ms/步 | 37.2338 | **36.7759** | **−0.458（−1.23%）** |
| vs 同轮 baseline | +4.130% | **+2.839%** | −1.29 pp |

| 指标 | 上轮 baseline | 本轮 baseline | 漂移 |
|---|---:|---:|---:|
| 完整 MoE ms/步 | 35.7569 | 35.7605 | **+0.0036（0.010%）** |

**两轮 baseline 漂移仅 0.010%**，远小于本轮要测的 1.02 ms 差额 ——
**本轮 +2.84% 是可复现的真实差异**，不是运行噪声。

### 1.2 三段拆解

| 段 | baseline | single_push | Δ ms | 占增量 |
|---|---:|---:|---:|---:|
| graph→first | 8.0186 | 8.0498 | +0.031 | 3.1% |
| first→last | 25.7008 | 26.2221 | **+0.521** | **51.5%** |
| last→combine | 2.0412 | 2.5040 | **+0.463** | **45.5%** |
| 合计 | 35.7605 | 36.7759 | **+1.015** | 100% |

与上轮（+0.767 / +0.687）相比，中段与末段各降约 0.2 ms —— 副本等待的修复确实见效，
但 **+1.0 ms 的主体仍在**。

## 2. 副本等待（`replica_stalls.csv`，任务书 §验收 3）

五层八 rank 全部**非空副本槽**，`quota_to_issue_us` 统计：

| arm | layer | n | **中位数** | P95 | 最大值 |
|---|---:|---:|---:|---:|---:|
| **baseline** | 0 | 77 | **5.0** | 32.0 | 39.0 |
| baseline | 8 | 78 | 4.0 | 5.0 | 9.0 |
| baseline | 20 | 81 | 4.0 | 5.0 | 16.0 |
| baseline | 32 | 81 | 4.0 | 5.0 | 16.0 |
| baseline | 39 | 82 | 4.0 | 11.0 | 25.0 |
| **old_push** | 0 | 77 | **349.0** | 391.0 | 399.0 |
| old_push | 8 | 78 | 292.5 | 327.0 | 330.0 |
| old_push | 20 | 81 | 263.0 | 334.0 | 351.0 |
| old_push | 32 | 81 | 249.0 | 293.0 | 299.0 |
| old_push | 39 | 82 | 277.0 | 321.0 | 332.0 |
| **candidate（新）** | 0 | 77 | **226.0** | 334.0 | 354.0 |
| candidate | 8 | 78 | **195.0** | 275.0 | 291.0 |
| candidate | 20 | 81 | **181.0** | 307.0 | 318.0 |
| candidate | 32 | 81 | **194.0** | 269.0 | 295.0 |
| candidate | 39 | 82 | **205.5** | 309.0 | 324.0 |

**中位数改善 55–75 µs（−20 ~ −27%）**，但仍比 baseline 的 4–5 µs 高 **40–50 倍**。

### 2.1 指定对照点：L20 / rank0 / CTA21 / slot45（192 行）

| arm | quota_us | first_issue_us | **quota→issue** | prior_single_slots | last_ready_us |
|---|---:|---:|---:|---:|---:|
| old_push | 121 | 371 | **250** | 1 | 449 |
| baseline | 124 | 127 | **3** | 1 | 205 |
| **candidate** | 124 | 128 | **4** | **0** | 216 |

**完全命中任务书描述**：旧版该槽在 121 µs 已可用、371 µs 才首次 issue，
其中 **229 µs 与同 CTA 处理槽 21 重叠**（`prior_single_after_quota_us=229`）。
新版把副本处理挪到单副本之前，等待降到 **4 µs**，`prior_single_slots` 归零。
**这个具体阻塞被彻底消除**。

新版该槽 `last_ready_us` 216 vs baseline 205（+11 µs），
说明副本拉取本身仍略慢，但不是主因（11 µs vs 需解释的 1015 µs）。

## 3. 单副本发送窗口（`singleton_senders.csv`，任务书 §验收 4）

| arm | 发送 CTA 数 | 起始窗口 | 结束窗口 | **跨度** | 总行数 |
|---|---:|---|---|---:|---:|
| old_push | 320（40 层×8 rank）| 24–35 µs | 217–431 µs | **407 µs** | 33,952 |
| **candidate（新）** | 320 | 24–36 µs | 225–419 µs | **395 µs** | 33,952 |

**总行数完全相同（33,952）**，`check_bs256_trace.py` 的
`singleton pushed versus owned rows across all ranks` 与
`singleton sender CTAs == [0..7]` 均通过。

**但跨度几乎没变（407 → 395 µs，仅 −3%）** —— 这就是本轮的关键：
副本优先只解决了**副本槽**的排队，**8 个源的单副本推送仍然串行铺开**。

### 3.1 单副本满块的源完成度（任务书 §验收 4 要求）

| layer | 满块数 | `finished_sources < 8` | **占比** | 尾块数 |
|---:|---:|---:|---:|---:|
| 0 | 15 | 15 | **100%** | 288 |
| 8 | 17 | 17 | **100%** | 249 |
| 20 | 14 | 14 | **100%** | 232 |
| 32 | 13 | 13 | **100%** | 243 |
| 39 | 16 | 16 | **100%** | 270 |

**满块 100% 在源未收齐时发布**（early publish 生效），尾块远多于满块
（232–288 vs 13–17），符合"尾块另需收齐 8 份声明"的规则。

## 4. 分段耗时：瓶颈已转移（任务书 §验收 4 末段）

| layer | arm | count exch | **pull token** | shared L1+L2 | ECO plan | wait token | GEMM | combine |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | baseline | 33.0 | **140.6** | 196.0 | 110.3 | 2.1 | 666.7 | 32.3 |
| 0 | single_push | 20.2 | **476.4** | 196.1 | 113.9 | 2.2 | 689.9 | 58.1 |
| 8 | baseline | 25.8 | **176.8** | 195.2 | 122.6 | 2.1 | 665.7 | 41.6 |
| 8 | single_push | 20.8 | **433.3** | 195.7 | 120.0 | 2.2 | 678.5 | 48.6 |
| 20 | baseline | 26.1 | **193.0** | 195.6 | 127.1 | 2.1 | 657.1 | 38.9 |
| 20 | single_push | 20.8 | **480.7** | 197.0 | 127.9 | 2.4 | 669.3 | 61.0 |
| 32 | baseline | 24.1 | **176.7** | 196.5 | 120.2 | 2.2 | 668.2 | 46.2 |
| 32 | single_push | 21.1 | **430.9** | 198.5 | 118.0 | 2.4 | 674.0 | 36.2 |
| 39 | baseline | 22.4 | **154.7** | 194.9 | 110.9 | 2.1 | 665.7 | 35.7 |
| 39 | single_push | 19.1 | **452.0** | 196.3 | 110.9 | 2.5 | 683.5 | 49.3 |

**`pull token` 仍是唯一的大项：141–193 → 431–481 µs（+254 ~ +336 µs/层）**。
其余段：`count exchange` −3 ~ −13 µs（改善）、`ECO plan` 基本持平、
`GEMM` +7 ~ +23 µs、`combine` +3 ~ +26 µs。

**上一轮同样如此（433–504 µs）**，本轮只把 L0 从 504 降到 476、L39 从 444 升到 452 ——
**`pull token` 基本没动**，这与 §3 的"发送跨度只降 3%"完全一致。

### 4.1 最慢 rank 时间线

| layer | arm | rank | tokens | M 块 | 活跃槽 | 首块可领 | pull 完成 | routed 退出 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | baseline | 2 | 194.0 | 50 | 49 | 198 | 166 | 879.6 |
| 0 | single_push | 4 | 190.0 | 52 | 47 | 194 | **496** | 918.7 |
| 8 | baseline | 7 | 198.0 | 48 | 40 | 201 | 220 | 888.6 |
| 8 | single_push | 2 | 194.0 | 49 | 43 | 203 | **459** | 893.1 |
| 20 | baseline | 3 | 180.0 | 49 | 44 | 193 | 189 | 877.6 |
| 20 | single_push | 4 | 202.0 | 45 | 40 | 196 | **482** | 911.2 |
| 32 | baseline | 4 | 203.0 | 48 | 42 | 200 | 187 | 901.3 |
| 32 | single_push | 4 | 203.0 | 48 | 42 | 198 | **466** | 886.9 |
| 39 | baseline | 3 | 208.0 | 47 | 43 | 194 | 201 | 881.3 |
| 39 | single_push | 3 | 208.0 | 47 | 43 | 202 | **478** | 918.3 |

**首块可领时刻几乎不变（194–203 vs 193–201）**，`pull 完成` 推迟 **239–330 µs**。
即：CTA **按时开工**，但**要等更久才能收齐所有行** —— 串行推送让 pull 尾端拖长。

## 5. 归因结论（任务书 §验收 4 末段要求的"候选原因 vs 实测归因"）

**实测归因（有直接证据）**：
- 副本等待：`quota→issue` 中位数 249–349 → **181–226 µs**；指定点 L20/r0/s45 **250 → 4 µs**；
  `prior_single_slots` **全部归零**。→ **已修复，贡献约 −0.46 ms**（+4.13% → +2.84%）。
- 剩余 +1.02 ms：**`pull token` +254~336 µs/层**，与发送跨度 395 µs 对应。

**候选原因（无法分割估算，如实说明）**：
现有 `singleton_sends` 记录把**路由扫描、逐行 TMA、fence、远端原子更新**都包在
发送窗口内，**无法分别给出百分比**。因此本回执**不**声称"TMA 占 x%"这类结论。

**可以确定的**：8 个源的发送窗口起点几乎同时（24–36 µs），但结束分散在 225–419 µs，
即 **8 个源在时间上互相排队**（每个源只有 1 个 sender CTA，但跨源没有并发协调）。
这正是下一版应当并行化的对象。

## 6. 十份 absolute trace（任务书 §验收 5）

每层给出本轮 **baseline** 与 **single_push** 的 `*.detail.perfetto.json`。
逐份核对：同目录 `*.detail.json` 与 `*.counts.json` 的
**step=2、layer、epoch、capture_id 全部一致，且各覆盖 8 个 rank**（已逐项验证通过）。

| layer | epoch | baseline absolute trace | single_push absolute trace |
|---:|---:|---|---|
| 0 | 46481 | [baseline L0](docs/trace_executor_20261119/sm90_bs256_push_replica_first_results/baseline/perfetto/step002_layer00_epoch46481.detail.perfetto.json) | [single_push L0](docs/trace_executor_20261119/sm90_bs256_push_replica_first_results/single_push/perfetto/step002_layer00_epoch46481.detail.perfetto.json) |
| 8 | 46489 | [baseline L8](docs/trace_executor_20261119/sm90_bs256_push_replica_first_results/baseline/perfetto/step002_layer08_epoch46489.detail.perfetto.json) | [single_push L8](docs/trace_executor_20261119/sm90_bs256_push_replica_first_results/single_push/perfetto/step002_layer08_epoch46489.detail.perfetto.json) |
| 20 | 46501 | [baseline L20](docs/trace_executor_20261119/sm90_bs256_push_replica_first_results/baseline/perfetto/step002_layer20_epoch46501.detail.perfetto.json) | [single_push L20](docs/trace_executor_20261119/sm90_bs256_push_replica_first_results/single_push/perfetto/step002_layer20_epoch46501.detail.perfetto.json) |
| 32 | 46513 | [baseline L32](docs/trace_executor_20261119/sm90_bs256_push_replica_first_results/baseline/perfetto/step002_layer32_epoch46513.detail.perfetto.json) | [single_push L32](docs/trace_executor_20261119/sm90_bs256_push_replica_first_results/single_push/perfetto/step002_layer32_epoch46513.detail.perfetto.json) |
| 39 | 46520 | [baseline L39](docs/trace_executor_20261119/sm90_bs256_push_replica_first_results/baseline/perfetto/step002_layer39_epoch46520.detail.perfetto.json) | [single_push L39](docs/trace_executor_20261119/sm90_bs256_push_replica_first_results/single_push/perfetto/step002_layer39_epoch46520.detail.perfetto.json) |

同目录另有每份对应的 `*.detail.json`（CTA 级 pull/领号/计算/写回）与
`*.counts.json`（各 rank token、M 块、活跃槽）。**同一臂内各层共用 Kineto 公共时间基准**
（baseline 全部 `capture_id=8f783aaf…`，single_push 全部 `598848a9…`），
因此可跨层对齐、保留 rank 间真实先后。

## 7. 正确性（两臂）

| 检查 | baseline | single_push |
|---|---|---|
| `check_bs256_trace.py` | ✅ `eco historical workload verified for both windows and all eight ranks` | ✅ 同上 |
| 两窗口 audit / 八 rank native token | ✅ | ✅ **零失配** |
| 物理槽 counts / epoch / 拍卖 / 预测值 / placement | ✅ | ✅ |
| `eco-layout.json` `cmp` | ✅ | ✅ |
| launch grid (24,71,71,1) | ✅ | ✅ |
| 推送行数 vs 拥有的单副本行 | — | ✅ 相等（33,952）|
| 八个 sender CTA | — | ✅ 恰 `[0..7]` |
| 副本槽 `prior_single_slots` | — | ✅ **新版全为 0** |

## 8. 交付物

`docs/trace_executor_20261119/sm90_bs256_push_replica_first_results/`（77 个文件，162 MB）：

| 路径 | 内容 |
|---|---|
| `baseline/`、`single_push/` | `run.json`、8 份 `rank*.npz`、`runner.log`、`verification.txt`、`hardware.csv`、`measurement.path`、`status.txt`、`runtime.log`、`perfetto/`（5 层 × 4 文件）|
| **`replica_stalls.csv`** | 五层八 rank 全部非空副本槽的 quota/issue/ready 与 `prior_single_*` |
| **`singleton_senders.csv`** | 分 rank/CTA 的发送行数、起止与持续时间（含 `old_push`/`baseline`/`candidate` 三臂）|
| `moe.csv` | 两臂 × 999/1000 步 |
| `return-smoke.log` | 冒烟门禁原始输出 |
| `summary.json` | 每样本 epoch/capture_id/分段耗时/最慢 rank/每 rank token·M 块；`replica_stalls_by_arm_layer`、`singleton_senders_by_arm`、`l20_r0_slot45` |
| `versions.txt`、`state-path.txt` | 版本与现场 |

## 9. 清理

| 项 | 结果 |
|---|---|
| 作业退出 | 两臂结束，`nvidia-smi --query-compute-apps` **0 条** |
| 临时 checkout | `/tmp/sm90-replica-first-run1/code` 已 `git worktree remove` 并 `prune` |
| 原始证据 | `/tmp/sm90-replica-first-run1/`（`diagnosis/`、`pair.log`、`return-smoke.log`）与 `/tmp/sm90-bs256-eco-fbog0Vym`（raw Kineto）**保留** |
| 历史现场 | `/tmp/sm90-bs256-trace-pbpMbH4u`（KV + replay）**未触碰** |

## 10. 建议

1. **副本优先是成功的局部优化**（250 → 4 µs，−1.29 pp），应保留。
2. **剩下的 +2.84% 全在 `pull token`**，而它的成因是 8 个源的发送窗口跨度 395 µs。
   下一步应让**跨源的推送并行**（当前每个源只有 1 个 CTA，但彼此串行排队）。
3. 在单副本推送并行化完成前，**single_push 不应进入主表** —— 正确性已达标，但慢 2.84%。
