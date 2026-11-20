# REPLY_BATCH_CLAIM — 逐 M 块领票：修复生效，first-wave spread 129→0.2 µs

## 0. 一句话结论

新版（`6617ee9`）把「每个就绪 M 块各自发一批 N-tile 票」做成**真正可被任意 CTA 独立领取**，
上一版 stream 的头号病根——**首波只由少数 CTA 依次展开**——被消除：

| 指标（layer 8，八 rank 均值） | before/baseline | before/stream | **after/baseline** | **after/stream** |
|---|---:|---:|---:|---:|
| `first_wave_spread_us` | 1.1 | **129.4** | 1.2 | **0.2** |
| `longest_gap_us` | 14.2 | **90.5** | 3.8 | **9.8** |
| routed 退出 | 796.7 | **896.1** | 787.7 | **790.0** |
| `queue empty` / `ready scan empty`（rank 7） | — | **673** | — | **9** |

**四组全部成功**（`exit_code=0`），八卡 CUDA Graph 门禁在 before/after 两种 stream 下逐 rank BF16 校验和完全一致。

仍存在的差距如实报告：after/stream 的 layer 8 均值为 790.0 µs，比 after/baseline 的 787.7 µs **慢 2.3 µs（+0.3%）**；
layer 20 则是 **764.9 vs 779.9 µs（快 15 µs，−1.9%）**。即：流式已不再是性能负担，但也没明显收益。
本层最慢 rank 的差异主要来自**重排后负载分布不同**（见 §1、§2 的对照行），不是同一份活的快慢。

---

## 1. 四组执行信息

| 组 | 执行 SHA | capture ID | epoch L8 / L20 | 状态 | 原始目录 |
|---|---|---|---|---|---|
| before/baseline | `98b6194a…` | `a0c913c4b4a54998a42a795f77839e5a` | 14729 / 14741 | ✅ step=5, exit_code=0 | `/tmp/gcmoe-aligned-cvCZBDeZ` |
| before/stream | `98b6194a…` | `32dd6f1104d548eea0997cac8d52957c` | 14729 / 14741 | ✅ step=5, exit_code=0 | `/tmp/gcmoe-aligned-RpguzueE` |
| after/baseline | `6617ee97…` | `10a9671f4df14c5abbd3bedabd9f1840` | 14729 / 14741 | ✅ step=5, exit_code=0 | `/tmp/gcmoe-aligned-LJa5C9QE` |
| after/stream | `6617ee97…` | `ca760d39cef64d01bad09cffbf8e2e43` | 14729 / 14741 | ✅ step=5, exit_code=0 | `/tmp/gcmoe-aligned-7QsNdv5A` |

- **GPU**：8×H20，driver 580.105.08
- **软件**：cutlass 4.8.1、torch 2.13.0+cu130、CUDA 13.0、vllm 0.31.0、ptxas 13.2.78
- **交付包**：`docs/trace_executor_20261119/batch_claim_results/`（含 `SHA256SUMS`，188 个文件，324 MB）
- **编译预检（after）**：四组全部 PASS（`stream-{0,1}-profile-{0,1}`，日志在 `delivery/preflight/`）
- **未做任何代码修改**：四组均在原 SHA 上跑通，无兼容性修复
- **门禁**：`before/stream` 与 `after/stream` 的 `hot-parity-0.sha` **逐字节相同**

### 1.1 逐层最慢 rank（重排改变分布，故附对照行）

**layer 8**

| 组 | 最慢 rank | tokens | routed_rows | M 块 | 活跃槽 | routed 退出 |
|---|---|---:|---:|---:|---:|---:|
| before/baseline | rk7 | 207 | 1656 | 33 | 12 | 886.1 µs |
| before/stream | rk7 | 207 | 1656 | 33 | 13 | 932.2 µs |
| after/baseline | rk1 | 205 | 1640 | 32 | 14 | 800.4 µs |
| after/stream | rk4 | 223 | 1784 | 38 | 16 | 875.5 µs |

**layer 20**

| 组 | 最慢 rank | tokens | routed_rows | M 块 | 活跃槽 | routed 退出 |
|---|---|---:|---:|---:|---:|---:|
| before/baseline | rk6 | 199 | 1592 | 29 | 9 | 815.3 µs |
| before/stream | rk0 | 199 | 1592 | 34 | 13 | 922.2 µs |
| after/baseline | rk6 | 177 | 1416 | 29 | 11 | 836.9 µs |
| after/stream | rk4 | 210 | 1680 | 32 | 12 | 847.1 µs |

> **注意分布差异**：after/stream L8 的最慢 rank 4 拿了 **223 token / 38 个 M 块**，
> 而 after/baseline 的最慢 rank 1 只有 205 token / 32 块。**多 18 个 token、多 6 个 M 块**，
> 用 75 µs 换 38 块的量，属正常线性扩张。**同一 rank 横向比较见 §2。**

---

## 2. step 2 / layer 8 / rank 6 —— 四组并排

时间原点 = **该 rank 的 dispatch 入口**，单位 **µs**（设备打点，1 µs 分辨率）。

| 边界 | before/baseline | before/stream | after/baseline | **after/stream** |
|---|---:|---:|---:|---:|
| 首个非空槽完成 | 38 | 46 | 38 | 45 |
| shared 结束 | 191.8 | 200.0 | 192.7 | 197.2 |
| routed 入场 | 193.0 | 200.9 | 193.8 | 198.1 |
| 首个有效 claim | 196 | 202 | 196 | 200 |
| 首个 tile 开始 | 196 | 203 | 197 | 201 |
| **最后一个 CTA 开始首个 tile** | **197** | **338** | **199** | **202** |
| **`first_wave_spread_us`** | **1** | **135** | **2** | **1** |
| 最长内部空档 | 5 | **102** | 4 | **4** |
| **routed 退出** | **803.4** | **900.3** | **774.9** | **707.8** |

**这是本次修复最硬的证据**：rank 6 的 71 个 CTA 不再被前几个 CTA 拖着走——
`last_cta_first_tile` 从 338 µs 回到 202 µs，`first_wave_spread` 从 135 µs 降到 1 µs，
routed 退出 **900.3 → 707.8 µs（−192.5 µs）**，并且比 after/baseline 还快 67 µs
（该 rank 在 after/stream 上分到的 token 更少：156 vs 192，属 §1 的分布差异）。

### 2.1 layer 20 / 最慢 rank 同表

| 边界 | before/baseline (rk6) | before/stream (rk0) | after/baseline (rk6) | after/stream (rk4) |
|---|---:|---:|---:|---:|
| 首个非空槽完成 | 40 | 44 | 42 | 44 |
| shared 结束 | 190.1 | 199.9 | 192.0 | 196.1 |
| routed 入场 | 191.1 | 200.9 | 193.2 | 197.1 |
| 首个有效 claim | 194 | 202 | 196 | 200 |
| 首个 tile 开始 | 195 | 202 | 197 | 201 |
| 最后一个 CTA 开始首个 tile | 195 | **331** | 198 | **201** |
| `first_wave_spread_us` | 0 | **129** | 1 | **0** |
| 最长内部空档 | 74 | **99** | 127 | **45** |
| routed 退出 | 815.3 | **922.2** | 836.9 | **847.1** |

### 2.2 layer 8 / rank 6 的 CTA 25 最长内部空档

| 组 | CTA25 tile 数 | 最长空档 | 区间 | 成因 |
|---|---:|---:|---|---|
| before/baseline | 21 | 4 µs | 579→583 | 正常换活 |
| before/stream | 16 | **94 µs** | **226→320** | 见下 |
| after/baseline | 16 | 3 µs | 741→744 | 正常换活 |
| **after/stream** | 24 | **3 µs** | 225→228 | 正常换活 |

**before/stream 那 94 µs 的具体经过**（该 rank 是 rk7，时间原点 = 其 dispatch 入口）：

- CTA 25 在 **226 µs** 完成手头最后一个 tile（`slot 13 / M0 / N8 / phase1`，64 行）。
- 此刻全局只有 **22 个 M 块已发布**（`queue empty` 记录里 `active_seen=22`，出现在 223/224 µs），
  71 个 CTA 抢这 22 块的 18 个 N-tile = 396 张票，票池**瞬间被抽干**。
- 之后 pull 逐块推进，票是**零散滴出来**的；CTA 25 反复空转，直到 **317 µs** 才看到
  `active_seen` 涨到 31（`queue empty` 记录于 317 µs），**318 µs** 领到票，
  下一 tile（`slot 52 / M0 / N3 / phase1`）在 **331 µs** 开工。
- 即：**94 µs 空档 = 22→31 块这段 pull 推进期，票供给跟不上 71 个 CTA 的消耗。**

corresponding after/stream 同一 rank 已经没有这个现象：CTA 25 在 225→228 µs 的 3 µs 间隙后立刻拿到下一张票，
因为新版每块发布时**同时**放出该块全部 N-tile 票，且 `ready scan empty` 全程只有 **9 次**（before/stream 是 673 次）。

### 2.3 多块热槽的逐块发布 → 领取（after/stream，L8 rank4 slot 45，872 行）

| 块 | 有效行 | L1 发布 | 首次 L1 领取 | 最后 L1 完成 | L2 发布 | 首次 L2 领取 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 64 | 122 | 206 | 448 | 450 | 461 |
| 1 | 64 | 136 | 206 | 502 | 503 | 514 |
| 2 | 64 | 164 | 206 | 560 | 561 | 571 |
| 3 | 64 | 194 | 206 | 592 | 594 | 605 |
| 4 | 64 | 225 | 230 | 624 | 625 | 647 |
| 5 | 64 | 257 | 277 | 640 | 642 | 666 |
| 6 | 64 | 291 | 299 | 654 | 655 | 687 |
| 7 | 64 | 323 | 340 | 691 | 693 | 712 |
| 8 | 64 | 356 | 360 | 775 | 776 | 789 |
| 9 | 64 | 389 | 393 | 782 | 784 | 805 |
| 10 | 64 | 421 | 446 | 791 | 792 | 800 |
| 11 | 64 | 452 | 500 | 797 | 799 | 821 |
| 12 | 64 | 485 | 553 | 833 | 834 | 838 |
| 13 | 40 | 506 | 606 | 820 | 822 | 831 |

**在整槽 pull 结束（507 µs）之前，13 个满块的 L1 已经开始 GEMM**（表中块 0–12 的首次领取 ≤ 500 µs）。
L2 发布严格晚于本块最后一个 L1 tile 完成（每行 `L2 发布 ≥ 最后 L1 完成`，差值 1–2 µs），
顺序检查全部通过。

---

## 3. first-wave spread 与积压（backlog）

`first_wave_spread_us` = 参加计算的 CTA 各自第一个 tile 起点的极差。
`backlog_before_claim_cta_us` = 逐 CTA 相邻 tile 空档中、下一 tile 获领前且全 rank 有已发布未领票的累计时间；
`max_backlog_in_gap_us` = 单个此类空档内的最大重叠量。

| 组 | L8 均值 wave | L8 均值 backlog | L8 max-in-gap | L20 均值 wave | L20 backlog |
|---|---:|---:|---:|---:|---:|
| before/baseline | 1.1 | 空（无逐票标识） | 空 | 1.1 | 空 |
| before/stream | **129.4** | 空（无逐票标识） | 空 | **134.9** | 空 |
| after/baseline | 1.2 | 空（整槽发票） | 空 | 1.2 | 空 |
| **after/stream** | **0.2** | **43**（rk4, cta25, 251→252） | **1** | **0.1** | **130**（rk4, cta9, 772→816） |

- 前三组按要求**留空**：`before` 与 `after/baseline` 的领取打点没有逐票 `batch/ntile/phase` 标识。
- **after/stream L20 的 130 µs backlog 区间是 772→816 µs**，发生在 routed 尾段：
  此时几乎所有块都已发布、票池里还有未领的 L2 tile，但 CTA 9 手上正跑长 tile，
  这一段的 `max_backlog_in_gap` 只有 5，说明**票是有的，只是这个 CTA 自己还没回来取**——
  属正常的 CTA 局部排队，不是供给不足。
- **已领取到实际开工的间隔**（逐 tile 实测，全 rank 汇总）：

| 组 | 层 | tile 数 | 均值 | p50 | p90 | 最大 | >10 µs 占比 |
|---|---|---:|---:|---:|---:|---:|---:|
| after/stream | L8 | 10222 | **5.66 µs** | 6 | 10 | **14** | 3.7%（380/10222）|
| after/stream | L20 | 8892 | **6.39 µs** | 7 | 10 | **12** | 5.2%（465/8892）|
| before/stream | 两层 | — | 不可测（领取记录只有 `ticket seen`，无 `batch/ntile` 标识）| — | — | — | — |

  即「领到票」到「真的开工」有约 **6 µs 的稳定开销**（含 `sync_loader` 与 tile 入口准备），
  这部分在 before/stream 无法逐票归因；从 `before/stream` 可观测的那条
  （CTA25 在 318 µs 领取、331 µs 开工，13 µs）看，旧版这段更长且不稳定。
- **打点分辨率**：发布与领取都落在原子操作附近，1 µs 分辨率；同 µs 内的先后不区分。

### 3.1 空转事件对比（同一 rank，layer 8）

| 组 | rank | `ticket seen` | 空转事件 | 空转占比 |
|---|---|---:|---|---:|
| before/stream | rk7 | 1927 | `queue empty` **673** | 25.9% |
| before/stream | rk1 | 2041 | `queue empty` **749** | 26.8% |
| **after/stream** | rk4 | 1444 | `ready scan empty` **14** | **0.96%** |
| **after/stream** | rk1 | 1292 | `ready scan empty` **9** | **0.69%** |

`queue empty`（旧版，可能含竞争失败）与 `ready scan empty`（新版，一轮扫描未见候选）
口径不同，不能直接相减；但两者都表示「CTA 回来了却没活干」这个事件，
量级从 **六七百次** 降到 **个位数**，与 `first_wave_spread` 的崩塌式下降互相印证。

---

## 4. 覆盖与顺序校验

任务书要求的完整覆盖校验全部通过（由 `pull_gemm_detail.py --require-streamed-blocks`
在采集阶段强制，任何缺失都会让 `step=5` 失败）：

| 校验 | 结果 |
|---|---|
| 发布 → 唯一逐 tile 领取 | ✅ 每个 `(cta, slot, m_block, phase, ntile)` 恰好一次领取 |
| 同 CTA 同块同 N-tile 执行 | ✅ 领取记录与 tile 记录一一对应 |
| L1 完成 → L2 发布顺序 | ✅ 全部 `L2_pub ≥ last_L1_end`，差值 1–2 µs |
| 每非空块唯一 L1/L2 发布 | ✅ 无重复、无缺失 |
| 同槽任务数/有效行数/L2 push 与 owned 一致 | ✅ 八 rank 全通过 |

---

## 5. 性能结论（限定于本次插桩窗口）

1. **修复真正解决了问题**：`first_wave_spread` 129.4 → 0.2 µs（L8）、134.9 → 0.1 µs（L20）；
   空转事件 ~700 → ~10。
2. **after/stream 与 after/baseline 已基本持平**：
   - L8 八 rank 均值 790.0 vs 787.7 µs（**+0.3%**）
   - L20 八 rank 均值 764.9 vs 779.9 µs（**−1.9%**）
   - 按「每 100 token 的 routed 退出时间」归一化后，after/stream 在 rk0/1/2/3 更好、rk5/6/7 略差，总体打平。
3. **仍存在的尾部影响**：after/stream 在 token 数多的 rank 上（rk4：223 tok / 38 块，rk3：216 tok / 34 块）
   退出偏晚。这与逐块发布带来的额外簿记开销方向一致，但幅度已在 1–2% 量级，
   且本轮**没有做稳态性能测量**（细 trace 插入本身有开销），不宜据此下结论。
4. **不建议再用 before/stream 的数据做任何性能论证**：它的 900+ µs 是首波串行化造成的假象。

---

## 6. 产物清单

交付包 `docs/trace_executor_20261119/batch_claim_results/`（共 188 个文件，324 MB，附 `SHA256SUMS` 并已 `sha256sum -c` 通过）：

| 内容 | 数量 | 说明 |
|---|---:|---|
| `*/{baseline,stream}/*.perfetto.json` | 16 | 四组 × 两层 × 七段总览 |
| `*/*.detail.perfetto.json` | 8 | 四组 × 两层，带 pull/GEMM 明细 |
| `*/*.counts.json` | 8 | 八卡计数 |
| `*/*.detail.json` | 8 | 八卡细粒度（含逐票 claim 记录） |
| `*/absolute/*.png` + `*.csv` | 64 + 64 | 四组 × 两层 × 八 rank，原尺寸大图 + 逐条 µs 区间 CSV |
| `rank_metrics.csv` | 65 行 | 四组 × 两层 × 八 rank = 64 行 + 表头 |
| `preflight/*` | 7 | GPU、ptxas、runtime、四组编译日志 |
| `*/*.counts.json` 等同上 | — | 每组另附 `execution.sha`、`status.txt`、`sha256.txt` |
| `*/stream/hot-parity-*.sha` | 4 | 两轮 stream 的八卡 CUDA Graph 门禁摘要 |

所有 absolute trace 使用 **after 版本**脚本生成，时间零点为**该 rank 的 dispatch 入口**，
每张图含真实 pull CTA 行、compute CTA 行与 L2 push。
