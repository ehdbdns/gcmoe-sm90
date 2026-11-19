# REPLY_FINE_TRACE — 细粒度 pull / routed GEMM 分析

## 0. 结论摘要

**两层都由「某个热点槽」决定 routed 退出时间，且拖尾**不是**等 token、也不是 push。**
- **layer 8**：慢 rank = **rank 6**（exit 805.3 µs）。热点槽 **slot 48（576 行）** 是最晚就绪槽，
  比次晚槽晚 65 µs；但它 pull 完成（319 µs）到首个 L1（434 µs）之间 **没有任何 CTA 报空队列**。
  最后 471 µs 全花在这个槽自己的 L1→L2 串行链上，最后 5.3 µs 才是 kernel 退出。
- **layer 20**：慢 rank = **rank 2**（exit 918.7 µs）。热点槽 **slot 48（576 行）**，
  token/M块最多的 rank 0（224 tok / 37 块）反而更早退出（865.9 µs）。

上一轮 layer 8 rank 6 的现象**重现了**：它仍是 token 中等（189）、M 块最多（36）、活跃槽最多（21）的一档，
且它的热点槽仍是最后完成 pull 的槽。

## 1. 执行信息

| 项 | 值 |
|---|---|
| 起始 SHA | `0eb86e4c702b400513e3f07de995d04e0f767b34` |
| 执行 SHA | 同上（**仅改解析器一处字段下标，见 §6**）|
| capture ID | `c412fb057f4e4a298678e62204216f72` |
| step | 2 |
| 层 / epoch | layer 8 → epoch 14729；layer 20 → epoch 14741 |
| 入口 | `run_aligned_ep8.sh <sha> --fine-trace` |

机器：8 × H20 / 580.105.08 / 97871 MiB；torch 2.13.0+cu130 / vLLM 0.31.0。

步骤 2 编译门禁：
```
compiled roles [1, 2, 5, 3]      (profile=0)   9.0s
compiled roles [1, 4, 2, 5, 3]   (profile=1)  14.2s
ALIGNED_STEP_CPU PASS / PULL_GEMM_DETAIL_CPU PASS
```

采样：prefill 256 → 200 warmup → 逐层 EPLB 重排 → 2 未采样步 → 1 采样步。
`fine.json`：8 rank，两个 epoch（14729 / 14741），每 rank `count` 105–122 万 / `capacity` 4,194,304，**无回绕**。

## 2. 八卡总表

### layer 8（epoch 14729）

| rank | tokens | routed_rows | M块 | 活跃槽 | 热点槽 | 热点行数 | 热点CTA | 首个claim | 最后pull | routed exit |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 184 | 1472 | 30 | 15 | 44 | 312 | 20 | 194 | 205 | 741.2 |
| 1 | 185 | 1480 | 33 | 18 | 43 | 512 | 19 | 201 | 320 | 761.6 |
| 2 | 198 | 1584 | 33 | 14 | 46 | 640 | 22 | 200 | 408 | 789.6 |
| 3 | 194 | 1552 | 31 | 13 | 49 | 576 | 1 | 199 | 345 | 780.4 |
| 4 | 192 | 1536 | 33 | 20 | 41 | 512 | 17 | 199 | 346 | 778.9 |
| 5 | 199 | 1592 | 32 | 13 | 44 | 712 | 20 | 199 | 424 | 792.7 |
| 6 | 189 | 1512 | 36 | 21 | 48 | 576 | 0 | 196 | 329 | 805.3 |
| 7 | 195 | 1560 | 32 | 14 | 46 | 448 | 22 | 199 | 287 | 772.6 |

### layer 20（epoch 14741）

| rank | tokens | routed_rows | M块 | 活跃槽 | 热点槽 | 热点行数 | 热点CTA | 首个claim | 最后pull | routed exit |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 224 | 1792 | 37 | 14 | 47 | 520 | 23 | 200 | 333 | 865.9 |
| 1 | 213 | 1704 | 31 | 11 | 45 | 576 | 21 | 201 | 388 | 839.2 |
| 2 | 240 | 1920 | 42 | 19 | 48 | 576 | 0 | 192 | 378 | 918.7 |
| 3 | 210 | 1680 | 29 | 10 | 44 | 384 | 20 | 195 | 294 | 781.2 |
| 4 | 220 | 1760 | 30 | 7 | 49 | 896 | 1 | 204 | 488 | 897.9 |
| 5 | 206 | 1648 | 33 | 14 | 0 | 808 | 0 | 199 | 556 | 853.0 |
| 6 | 107 | 856 | 18 | 7 | 53 | 696 | 5 | 197 | 427 | 707.0 |
| 7 | 116 | 928 | 20 | 10 | 54 | 544 | 6 | 197 | 344 | 592.7 |

## 3. 慢 rank 为什么慢

### layer 8：主卡 rank 6（exit 805.3 µs），对照卡 rank 2（exit 789.6 µs，33 块 / 198 tok，与之接近）

**问题 1：拉取是否拖后**

最后三个完成 pull 的槽：

| 槽 | 行数 | CTA | pull 起-止 (µs) | 首个 L1 | pull→L1 (µs) |
|---:|---:|---:|---|---:|---:|
| 48 | 576 | 0 | 74–319 | 434 | 115 |
| 50 | 320 | 2 | 64–254 | 352 | 98 |
| 52 | 192 | 4 | 71–177 | 321 | 144 |

- 最重槽 slot 48（576 行，CTA 0）**正是最后就绪槽**。
- 它比次晚槽（slot 50，320 行，254 µs）晚 **65 µs**。
- 该 CTA 先前处理过别的槽；plan wait 共 9 对，首次 54–109 µs。

> 注意：该层**最后 pull 完成是 424 µs（rank 5）**，但 rank 5 的 exit 只有 792.7 µs；
> rank 6 的 pull 在 329 µs 就结束，却最晚退出。**pull 慢 ≠ 该 rank 退出慢。**

**问题 2：后续 CTA 有无工作**

区间 **319–434 µs**（115 µs，热点槽 pull 完成 → 该槽首个 L1）：
- 活跃 compute CTA 采样 32 点，范围 **35–71**（共 71 个 compute CTA）
- 该区间内 **queue-empty 事件 0 个**（全层共 71 个）
- 结论：热点槽等 L1 的 115 µs 里，**没有 CTA 报告空队列**，GPU 仍在跑别的槽 → 这段等待**不是** compute 无活可干，而是该槽自己的 L1 还没开始。

**问题 3：计算尾部拖在哪**

| 阶段 | 槽 | CTA | M块 | ntile | valid | 起-止 (µs) |
|---|---:|---:|---:|---:|---:|---|
| L2 | 48 | 44 | 8 | 19 | 64 | 775–800 |
| L2 | 48 | 6 | 8 | 17 | 64 | 775–800 |
| L2 | 48 | 50 | 8 | 18 | 64 | 776–800 |
| L2 | 48 | 7 | 8 | 15 | 64 | 774–799 |
| L2 | 48 | 12 | 8 | 10 | 64 | 774–799 |

- 最后 L2 tile: slot 48 CTA 44 M8 ntile 19 valid 64，775–800 µs
- 最后 push  : slot 48 CTA 50 M8 ntile 18，797–800 µs

- 最后 pull → 最后 tile：**471 µs**（占 exit 的 58.5%）
- 最后 tile → routed exit：**5.3 µs**
- 最后 push 与最后 tile 几乎同时（差 0 µs）→ **push 不独立拖尾**

**问题 4：跨卡等待**

rank 6 的 routed exit 805.3 µs，其它 rank 的 exit 在 741–793 µs；combine 区间见 `*.detail.perfetto.json`。
本 rank 的拖尾（471 µs 的 L1→L2 链）与 peer 等待可以分开：它在 800 µs 完成最后 L2 tile，
而最慢的 peer 在 793 µs 已退出，所以 rank 6 **不是**在等 peer。

### layer 20：主卡 rank 2（exit 918.7 µs）

最后三个完成 pull 的槽：

| 槽 | 行数 | CTA | pull 起-止 (µs) | 首个 L1 | pull→L1 (µs) |
|---:|---:|---:|---|---:|---:|
| 48 | 576 | 0 | 110–368 | 504 | 136 |
| 47 | 448 | 23 | 42–317 | 404 | 87 |
| 50 | 432 | 2 | 64–308 | 319 | 11 |

- 最重槽 slot 48（576 行，CTA 0）**正是最后就绪槽**。
- 它比次晚槽（slot 47，448 行，317 µs）晚 **51 µs**。
- 该 CTA 先前处理过别的槽；plan wait 共 10 对，首次 50–119 µs。

区间 **368–504 µs**（136 µs，热点槽 pull 完成 → 该槽首个 L1）：
- 活跃 compute CTA 采样 43 点，范围 **52–71**（共 71 个 compute CTA）
- 该区间内 **queue-empty 事件 0 个**（全层共 80 个）
- 结论：热点槽等 L1 的 136 µs 里，**没有 CTA 报告空队列**，GPU 仍在跑别的槽 → 这段等待**不是** compute 无活可干，而是该槽自己的 L1 还没开始。

| 阶段 | 槽 | CTA | M块 | ntile | valid | 起-止 (µs) |
|---|---:|---:|---:|---:|---:|---|
| L2 | 48 | 63 | 8 | 16 | 64 | 889–913 |
| L2 | 48 | 47 | 8 | 18 | 64 | 889–913 |
| L2 | 48 | 9 | 8 | 17 | 64 | 889–913 |
| L2 | 48 | 41 | 8 | 19 | 64 | 889–913 |
| L2 | 48 | 69 | 8 | 8 | 64 | 887–912 |

- 最后 L2 tile: slot 48 CTA 63 M8 ntile 16 valid 64，889–913 µs
- 最后 push  : slot 48 CTA 47 M8 ntile 18，910–913 µs

- 最后 pull → 最后 tile：**535 µs**（占 exit 的 58.2%）
- 最后 tile → routed exit：**5.7 µs**
- 最后 push 与最后 tile 几乎同时（差 0 µs）→ **push 不独立拖尾**

## 4. 上一轮现象的复核

| 上一轮（epoch 14729 旧采样）| 本轮（epoch 14729 新采样）|
|---|---|
| layer 8 rank 6：189 tok / 31 块 / 11 活跃槽，热点 local slot 47 有 1040 行 | layer 8 rank 6：**189 tok / 36 块 / 21 活跃槽**，热点 **slot 48 有 576 行** |

- **慢 rank 仍是 rank 6**，token 数完全一致（189），说明这两轮的负载分布稳定；
- 热点槽编号与行数变了（47/1040 → 48/576），但**「热点槽 = 最后就绪槽」这一机制重现**；
- 上一轮「热点槽拖后任务发布」的推断，本轮得到**否定证据**：热点槽等待期间没有空队列（§3 问题 2）。

## 5. 产物

`docs/trace_executor_20261119/fine_results/`：
```
step002_layer08_epoch14729.counts.json
step002_layer08_epoch14729.perfetto.json
step002_layer08_epoch14729.detail.json
step002_layer08_epoch14729.detail.perfetto.json
step002_layer20_epoch14741.counts.json
step002_layer20_epoch14741.perfetto.json
step002_layer20_epoch14741.detail.json
step002_layer20_epoch14741.detail.perfetto.json
```

原始数据（未提交 Git，保留在采集目录 `/tmp/gcmoe-aligned-C6S6odEg`）：
`raw/rank0..7.json`（torch trace）、`raw/capture.json`、`raw/fine.json`。

校验值：
```
e3810089eb537099189d7e47e0ccd5b4bad7fd65ecfc88ca29a9981f5d99e42e  deliver/step002_layer08_epoch14729.counts.json
fa7e752424ff9f4c29a392428aea2b6fece119eecafd4fc2790e4883640753d8  deliver/step002_layer08_epoch14729.detail.json
b5c08af7e8ac5e81504df8bd50e658b27b6bab1bfca1b4a7addd8402bcab0704  deliver/step002_layer08_epoch14729.detail.perfetto.json
dd180909c4519c7f5f3a347c2505850537daea52e5b231b4d1b9ff53b67b9afe  deliver/step002_layer08_epoch14729.perfetto.json
22d7c5794d3a7d848fca9f80c0ae173726c38db29d348378a322a2928540b468  deliver/step002_layer20_epoch14741.counts.json
7eff6da7f7d21b1fce5b8f580eb3e9960608e6a15d0ead92af49c5e84a8badb1  deliver/step002_layer20_epoch14741.detail.json
266a23402ce2ce068c71d46b663afbf231e51471978ed95150bc628c6c6591ce  deliver/step002_layer20_epoch14741.detail.perfetto.json
e6abff6af5bd1a821a1aeb8fbb557bd297ebc8abd6578d99c44bd65805ebc519  deliver/step002_layer20_epoch14741.perfetto.json
78a1a41b17e6afdc26e2fdf6f870910bdde89e2c1bd37ee439efd220136b230d  raw/fine.json
17994646d344f59c489e552329cecc60c74c54ad598186f930e39369ef3b9f2c  raw/capture.json
7977b93745d4aa7dd97e3d5f20c80343d56d3c2a1d0204c17564180d150a2a19  raw/rank0.json
2bc2dc5d42e0407057029683a7762fd8af6a45cf75a254878d32e6511c176e3e  raw/rank1.json
889751e5acf59ebde12115b1aea949771241d24c40ea6bed0ce921744406c65b  raw/rank2.json
1756f533fbf8421350824b9afc4580e110446d66365115cd5029808c89aa4adc  raw/rank3.json
639cd2b3584f1b16d77c53002cf29df256d892c7ecab6188499353fc36ad52fa  raw/rank4.json
d10b03b51b520f663454971865d0863e4f2dda02cf85df31df3caf183b88e984  raw/rank5.json
64a5f98c39ee183e397509966cd8cd4aaf107852f4c52be0b453c773297b70cc  raw/rank6.json
ca7e6d8559a829b4b8e1f155758e6d504fb2e3e0ceb674d2e1010101b82450db  raw/rank7.json
```

## 6. 本轮唯一的代码修改（解析器字段下标）

**现象**：入口在步骤 5 失败，`detail.log` 报 `epoch 14729, rank 0: unpaired replica plan waits`。

**定位**：`scripts/trace_layer/pull_gemm_detail.py:92-93` 用 `rec[4]` 判断 type 3 的
begin/released。但记录布局是 `[ts, type, id, a0, a1, a2, a3, …]`，该标志在 **`a0`（`rec[3]`）**，
`a1` 对这条记录恒为 0。实测：

```
rank0 / epoch 14729 / type3:  a0 分布 {0: 12, 1: 12}   ← 完美配对
                              a1 分布 {0: 24}          ← 解析器读的这列
```

**修改**：`rec[4]` → `rec[3]`（两行），不改变任何分段语义、阈值或配对规则。
**验证**：用同一份已采 `fine.json` 重跑解析器，退出码 0，四份产物全部生成，**未重跑 GPU**。

## 7. 仍需额外打点的环节

- **热点槽自己 L1 启动慢的根因**：从 pull 完成到首个 L1 的 115 µs，现有打点只能证明
  「CTA 没有空队列」，但无法区分「等待权重 TMA」「等待该槽被某个 CTA 领到」。
  需要 type 4/5 的票号与 `active_ready` 序列做联合分析，或新增 CTA 领任务的细打点。
- **layer 20 rank 7 只有 116 tok**（其余 206–240），放置极不均衡，与 layer 8 的机理不同，
  需要单看该层的 EPLB 输出。

---

## 8. 图（本轮补充）

任务书 §4 未要求 PNG，按用户要求补充。四张图与绘图脚本都在
`docs/trace_executor_20261119/fine_results/`：

| 文件 | 内容 |
|---|---|
| `step002_layer08_epoch14729.detail.png` | layer 8：全部 56 槽 pull 跨度 + 活跃 CTA + 全部 tile + 热点槽放大 |
| `step002_layer20_epoch14741.detail.png` | layer 20：同上 |
| `step002_layer08_epoch14729.tail.png` | layer 8 主卡热点的**尾部拆解条**（pull / 等L1 / L1 / 空档 / L2 / push）|
| `step002_layer20_epoch14741.tail.png` | layer 20 同上 |
| `plot/plot_detail.py`、`plot/plot_tail.py` | 生成脚本（只读已验收的 detail.json）|

### 图里发现的新证据：L1 与 L2 之间有 96 / 157 µs 的空档

尾部拆解把热点的 486 µs（layer 8）/ 545 µs（layer 20）分成五段：

| 层 | pull→L1 | L1 (162 tiles) | **L1→L2 空档** | L2 (180 tiles) | L2→exit |
|---|---:|---:|---:|---:|---:|
| layer 8 (rank 6) | 115 | 184 | **96** | 86 | 5 |
| layer 20 (rank 2) | 136 | 165 | **157** | 87 | 6 |

关键点：**这段空档不是在等别的槽的 L1**。该层**其它槽的最后 L1 分别在 487 µs（layer 8）/
557 µs（layer 20）就结束了**，而热点槽的 L2 要到 714 µs / 826 µs 才开始 —— 空档远大于
「等其它 L1 结束」所需的时间。

因此 layer 8 / layer 20 的尾部结构是：

```
pull ──(等L1)── L1(162 tiles) ──(空档 96/157 µs)── L2(180 tiles, 86/87 µs) ── push ── exit
                                                                    ↑
                                                    这一段才是真正的拖尾主因
```

这修正了 §3 问题 3 里「L1→L2 是连续串行链」的表述：**L2 的启动被推迟了约 100–160 µs，
而推迟的原因不在已观测的 L1 完成时刻上**。可能来源（现有打点无法区分）：

1. L2 需要等该专家**全部 M 块**的 L1 完成（含跨 rank 的副本），而 detail 只记录了本 rank 的 L1；
2. L2 的 activation 要等上一层 L2 的 push 落地，形成跨层依赖；
3. L2 tile 的领取排队（票号/`active_ready`）。

**这是本轮最值得继续追的一条线索**，需要跨 rank 的 L1 完成时刻，或 L2 tile 的领取打点。
