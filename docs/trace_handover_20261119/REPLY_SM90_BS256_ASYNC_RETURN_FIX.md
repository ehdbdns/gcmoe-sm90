# 【SM90】REPLY_SM90_BS256_ASYNC_RETURN_FIX — 回址修复**正确性通过**（零失配），但计时 +4.13%

> 本回执由 **SM90 执行者**（8×H20）提交。SM100 任务书不在本执行者范围内。

## 0. 结论

| 项 | 结果 |
|---|---|
| **八卡回址冒烟门禁** | ✅ **4/4 组通过**（hot_rows 135 / 0 / 64 / 65，各 `replays=3`）+ `ECO_CAPTURE OK` |
| **baseline** | ✅ `exit_code=0`，历史校验全过 |
| **single_push** | ✅ **`exit_code=0`，原生零失配**（上轮首个 decode token 即 59.6% 失配）|
| **主指标（999 步）** | baseline **35.7569** → single_push **37.2338 ms** → **+4.130%** |
| 含采样 1000 步 | 35.7573 → 37.2345 → **+4.131%** |

**回址修复解决了正确性问题**（这是上轮 `7d225d3` 失败的直接原因），
但引入 **+1.48 ms/步（+4.13%）的显著变慢**，来源已定位到 `pull token` 段。

## 1. 冒烟门禁明细（任务书 §执行要求，先于模型启动）

`return-smoke.log`（原样归档）：

```
PUSH_PAYLOAD_OK hot_rows=135 replays=3 rank0_sha256=a9b71607...
PUSH_PAYLOAD_OK hot_rows=0   replays=3 rank0_sha256=1aed8113...
PUSH_PAYLOAD_OK hot_rows=64  replays=3 rank0_sha256=57ff3dd1...
PUSH_PAYLOAD_OK hot_rows=65  replays=3 rank0_sha256=e8968361...
ECO_CAPTURE OK
```

四组比任务书要求的 `135 / 0 / 64 / 65` 逐项命中，每组三次图重放，
八卡全部物理槽的逐行激活、scale、gate、回址与逻辑专家逐项比对通过。

## 2. 主指标与三段拆分（`moe.csv`）

| arm | steps | graph→first ms | first→last ms | last→combine ms | **moe_ms** | 变化 |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 999 | 8.0163 | 25.6912 | 2.0495 | **35.7569** | — |
| single_push | 999 | 8.0391 | 26.4578 | **2.7369** | **37.2338** | **+4.130%** |
| baseline | 1000 | 8.0164 | 25.6916 | 2.0494 | 35.7573 | — |
| single_push | 1000 | 8.0392 | 26.4582 | 2.7371 | 37.2345 | +4.131% |

**拆解（999 步）**：

| 段 | baseline | single_push | Δ | 占总增量 |
|---|---:|---:|---:|---:|
| graph→first | 8.0163 | 8.0391 | **+0.0228** | 1.5% |
| first→last | 25.6912 | 26.4578 | **+0.7666** | **51.8%** |
| last→combine | 2.0495 | 2.7369 | **+0.6874** | **46.5%** |
| 合计 | 35.7569 | 37.2338 | **+1.4769** | 100% |

**增量几乎均分在"中段计算"与"末段收尾"**，首段很小。
这与上轮的串行 split / 源端索引（都在首末段）**完全不同的分布**。

## 3. 分段耗时（八 rank 均值，µs）—— 根因定位

| layer | arm | count exch | **pull token** | shared L1+L2 | ECO plan | wait token | GEMM | combine |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | baseline | 34.8 | **140.8** | 195.5 | 111.3 | 2.1 | 666.2 | 30.8 |
| 0 | single_push | 20.5 | **504.0** | 195.9 | 112.3 | 2.5 | 681.3 | 55.8 |
| 8 | baseline | 26.0 | **171.4** | 195.2 | 117.2 | 2.0 | 666.1 | 37.2 |
| 8 | single_push | 20.9 | **478.6** | 195.8 | 119.3 | 2.1 | 686.5 | 45.0 |
| 20 | baseline | 23.2 | **195.9** | 196.7 | 128.6 | 2.1 | 657.3 | 42.4 |
| 20 | single_push | 20.6 | **495.7** | 196.2 | 124.4 | 2.4 | 687.3 | 69.8 |
| 32 | baseline | 24.1 | **176.8** | 196.2 | 121.1 | 2.0 | 667.5 | 34.0 |
| 32 | single_push | 20.4 | **432.9** | 195.3 | 121.1 | 2.5 | 681.6 | 39.3 |
| 39 | baseline | 22.0 | **157.0** | 195.3 | 111.7 | 2.0 | 663.7 | 33.4 |
| 39 | single_push | 19.1 | **443.6** | 196.1 | 112.0 | 2.2 | 677.5 | 59.1 |

**唯一的大幅变化是 `pull token`：141–196 → 433–504 µs（+237 ~ +363 µs/层）**。

其他段的读数：

- `count exchange` **降了** 3–14 µs（单副本不再走原交换路径）；
- `shared L1+L2`、`ECO plan`、`wait token` **基本不变**；
- `GEMM` **+14 ~ +30 µs**（单副本行被推入规范 GEMM 行后，块形状变化）；
- `combine` **+5 ~ +27 µs**。

**+1.48 ms/步 ≈ 40 层 ×（pull +0.3 ms 量级 + GEMM/combine 小幅）/ 八卡并行摊薄**，方向一致。

## 4. 单副本直推的发送窗口（`singleton_sends`）

| layer | 发送 CTA 数 | 8 个源的发送窗口 | 窗口跨度 |
|---|---:|---|---:|
| 0 | 8 | 27 → 425 µs | **398 µs** |
| 8 | 8 | 30 → 352 µs | 322 µs |
| 20 | 8 | 30 → 405 µs | 375 µs |
| 32 | 8 | 28 → 343 µs | 315 µs |
| 39 | 8 | 24 → 384 µs | 360 µs |

**每个源 rank 恰好 1 个发送 CTA（共 8 个，编号 0–7）**，
`check_bs256_trace.py` 的 `same('singleton sender CTAs', sorted(...), list(range(8)))` 通过。
但 8 个源**串行铺开**：最早的 27–34 µs 开始，最晚的 425 µs 才结束。
**这 300–400 µs 的跨度直接体现为 `pull token` 段从 141 涨到 504 µs**：
原路径的 pull 是所有 rank 并行拉取，新路径下目标 CTA 必须等**每个源**都把自己的
单副本行推完才能凑齐满块。

## 5. 逐层单副本块快照（任务书 §交付 2 的四个字段）

对每个采样层的**单副本槽**，统计 phase-1 块发布时的源状态快照
（字段由目标 CTA 在检查该块前最近一次读到）：

| layer | 满块总数 | **declared_sources < 8** | **finished_sources < 8** | 尾块数 | 单副本槽数 |
|---:|---:|---:|---:|---:|---:|
| 0 | 15 | **0** | **15** | 288 | 323 |
| 8 | 17 | **0** | **17** | 249 | 344 |
| 20 | 14 | **0** | **14** | 232 | 347 |
| 32 | 13 | **0** | **13** | 243 | 340 |
| 39 | 16 | **0** | **16** | 270 | 325 |

**关键读数**：

1. **`declared_sources < 8` 全为 0** —— 满块**从不**在"声明源不足 8"的情况下发布，
   这正是本轮修复要达到的性质（上轮在此处出错）。
2. **`finished_sources < 8` 全部命中（15/15、17/17、…）** ——
   满块发布时**仍有源未完成**，即发布是"按 M 行凑满即走"，不等待全部源收尾。
   这是设计意图（early publish），也正是 §4 里 pull 段被拉长的机制：
   早发布省了等待，但目标侧要花更长时间凑齐其余行。
3. 尾块远多于满块（232–288 vs 13–17），符合"尾块另需收齐 8 份声明"的规则。

## 6. 正确性校验（两臂）

| 检查 | baseline | single_push |
|---|---|---|
| `check_bs256_trace.py` | ✅ `eco historical workload verified for both windows and all eight ranks` | ✅ 同上 |
| 两窗口 audit / 八 rank native token | ✅ | ✅ **零失配** |
| 物理槽 counts / epoch / 拍卖 / 预测值 | ✅ | ✅ |
| `eco-layout.json` | ✅ `cmp` 相同 | ✅ 同左 |
| launch grid (24, 71, 71, 1) | ✅ | ✅ |
| 单副本推送行数 vs 拥有的单副本行 | — | ✅ **相等**（跨全部 rank）|
| 单副本发送 CTA | — | ✅ 恰为 `[0..7]` |
| `declared_sources` 字段出现在 phase-1 单副本块 | — | ✅ 全部存在 |

**上轮的 59.6% 失配完全消失**：single_push 两窗口 2000 步全部 256 token/步，原生逐位一致。

## 7. 判读

1. **回址修复是有效的**：冒烟 4/4 + 正式两窗口零失配，证明
   `route_meta` 三元组分开存储（`c6bb760`）与 dispatch staging 修正（`29460b5`）解决了上轮的根本问题。
2. **但性能代价很大**：+4.13%，且是**可复现的**（999 与 1000 步同号、同量级，
   三段拆分同向）。这不是噪声 —— 与 `REPLY_SM90_BS256_SOURCE_INDEX_PULL` 的 0.09%
   不同，1.48 ms 远超两轮 baseline 漂移（0.04%）。
3. **根因是串行的源端推送**：8 个源 CTA 的发送窗口跨度 315–398 µs，
   目标侧 `pull token` 因此从 141–196 涨到 433–504 µs。
   若要保留 early-publish 的正确性，需要让 8 个源**并行**推送（而非当前的顺序铺开）。
4. **`finished_sources < 8` 的满块占 100%** 说明早发布确实在生效，
   但它省下的等待被更长的 pull 抵消 —— 这是本轮最重要的机制结论。

## 8. 交付物

`docs/trace_executor_20261119/sm90_bs256_async_return_fix_results/`（75 个文件，162 MB）：

| 路径 | 内容 |
|---|---|
| `return-smoke.log` | 冒烟门禁原始输出（4 组 `PUSH_PAYLOAD_OK` + `ECO_CAPTURE OK`）|
| `baseline/`、`single_push/` | `run.json`、8 份 `rank*.npz`、`runner.log`、`verification.txt`、`hardware.csv`、`measurement.path`、`status.txt`、`runtime.log`、`perfetto/`（5 层 × 4 文件）|
| `moe.csv` | 两臂 × 999/1000 步 |
| `summary.json` | 每样本 epoch/capture_id/分段耗时/最慢 rank/每 rank token·M 块·活跃槽；single_push 另含 `singleton_blocks`（四个字段）与 `singleton_sends` |
| `versions.txt`、`state-path.txt` | 版本与现场 |

**Perfetto 直链**（`<arm>` ∈ {baseline, single_push}，`<AA>`/`<EP>` =
00/46481、08/46489、20/46501、32/46513、39/46520）：

- `<arm>/perfetto/step002_layer<AA>_epoch<EP>.perfetto.json`（coarse）
- `<arm>/perfetto/step002_layer<AA>_epoch<EP>.detail.perfetto.json`（absolute/detail）
- `<arm>/perfetto/step002_layer<AA>_epoch<EP>.counts.json`
- `<arm>/perfetto/step002_layer<AA>_epoch<EP>.detail.json`

## 9. 清理

| 项 | 结果 |
|---|---|
| 作业退出 | 两臂结束，`nvidia-smi --query-compute-apps` **0 条** |
| 临时 checkout | `/tmp/sm90-return-fix-run2` 已 `git worktree remove` 并 `prune` |
| 临时输出 | `/tmp/sm90-bs256-eco-PMshKpNh` 保留（raw Kineto 等）|
| 历史现场 | `/tmp/sm90-bs256-trace-pbpMbH4u`（KV + replay）**未触碰** |

## 10. 一次未计入的启动（如实记录）

首次执行时我把冒烟日志 `return-smoke.log` 写在了 worktree 内，
导致 `run_aligned_ep8.sh` 的 `test -z "$(git status --porcelain)"` 检查失败
（`status.txt` = `step=1: checkout and environment`，`exit_code=1`）。
**该次未启动模型**（失败在 checkout 检查，早于任何 GPU 分配），
随后把日志改到 `/tmp/rt/retfix_logs/` 并重跑，即为本回执的数据。

## 11. 建议

1. **8 个源的推送并行化**是下一步的关键：当前窗口跨度 315–398 µs 是 pull 暴涨的直接原因。
2. 若并行化后 pull 能回到 200 µs 量级，本轮 +1.48 ms 有望转为接近持平甚至收益。
3. 在性能修好之前，**不应把 single_push 加入主表** —— 它的正确性已达标，但慢 4.13%。
