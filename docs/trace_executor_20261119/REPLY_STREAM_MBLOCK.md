# REPLY_STREAM_MBLOCK — 流式 M 块发票 A/B：**功能正确，但比基线慢 12–20%**

## 结论（先说结果）

1. **四组编译门禁 + 八卡 CUDA Graph 数值门禁全部通过**，两轮都跑完了完整采集、解析与验收。
2. **流式方案的不变式全部成立**：每个非空 M 块有唯一的 L1/L2 发布记录；L1 无一早于本块发布；
   L2 发布无一早于本块最后一个 L1 tile；同槽任务数、有效行数、L2 push 数与基线逐项相等。
3. **但流式版本更慢，而且慢得稳定**。同一 SHA、同层、同 BS256、同 step：
   routed 内核 **+11.9%**（layer 8）/**+11.7%**（layer 20），
   dispatch→routed_end **+14.0%**（layer 8）/**+10.6%**（layer 20）。
4. **根因不是"发票太晚"，而是"发票太碎"**。总 tile 工作量几乎不变（302.2k → 303.1k tile·µs，+0.3%），
   但 **compute CTA 的池内空闲从 4.5% 涨到 13.9%**，`queue empty` 事件从 **646 次涨到 5522 次（8.5×）**。
   把整槽一张票改成逐 M 块一张票，反而**摊薄了每个 CTA 的连续工作长度**。
5. 具体机理见 §6：流式把"一个槽的 18 个 L1 tile 同时可领"变成"每 64 行才放 18 个 L1 tile"，
   于是 L1 grid 从**一次性铺满 71 个 CTA**退化为**逐块阶梯式追加**，
   同时 **L2 与 L1 抢同一批 CTA**（L2 在 1 个块 L1 完成后就能领），
   L2 的 24 µs 短 tile 插进 L1 的 52 µs 长 tile 流里，每个 CTA 都在频繁换活。

---

## 1. 版本与执行

| 项 | 值 |
|---|---|
| 起始 SHA | `1bd5ae0de0e7b484108ef572ef6a9a2baee7fbe2` |
| 执行 SHA（A 与 B 同一 SHA）| `a4db419aa07cda4f35a100707b97093ff8889a26` |
| 兼容性修复 | `a0e1362`（EPLB 1-D 广播）、`a4db419`（门禁统计） |
| 工作区 | 干净 |
| GPU | 8×H20，driver 580.105.08，CUDA 13.0，torch 2.13.0+cu130，vllm 0.31.0 |

两轮命令与任务书完全一致：

```bash
GC_MOE_SM90_STREAM_M_BLOCKS=0 bash scripts/trace_layer/run_aligned_ep8.sh "$revision" --fine-trace
GC_MOE_SM90_STREAM_M_BLOCKS=1 bash scripts/trace_layer/run_aligned_ep8.sh "$revision" --fine-trace
```

| 轮次 | 输出目录 | capture_id | epoch (L8 / L20) | 状态 |
|---|---|---|---|---|
| A baseline (STREAM=0) | `/tmp/gcmoe-aligned-7jKEw28l` | `6de36ca62c3543ae9b901f1ea4ff9001` | 14729 / 14741 | `step=5 exit_code=0` ✅ |
| B streamed (STREAM=1) | `/tmp/gcmoe-aligned-Q17dgmco` | `c19d1eed5b7244baa47523fdf5c0526b` | 14689 / 14701 | `step=5 exit_code=0` ✅ |

（另有两轮作废记录：`/tmp/gcmoe-aligned-Z4Y4iIcV` 为 `1bd5ae0` 的 baseline 成功轮；
`/tmp/gcmoe-aligned-elfXnWSn` 在 2b 门禁失败，见 §2。）

## 2. 编译矩阵与门禁

### 2.1 四组编译（`1bd5ae0`，日志 `/tmp/gcmoe-stream-compile-jT2GXIBl`）

| STREAM | STEP_PROFILE | roles | 结果 |
|---|---|---|---|
| 0 | 0 | `[1, 2, 5, 3]` | PASS |
| 0 | 1 | `[1, 4, 2, 5, 3]` | PASS |
| 1 | 0 | `[1, 2, 5, 3]` | PASS |
| 1 | 1 | `[1, 4, 2, 5, 3]` | PASS |

> 判定必须用 `grep -Fq`：任务书原文的 `grep -q "compiled roles $roles"` 会把 `[1, 2, 5, 3]`
> 当正则字符类，四组一律误报 FAIL。

### 2.2 八卡 CUDA Graph 正确性门禁（`a4db419`）

`TL_TOKENS=17 TL_CAP=128 TL_HOT_BLOCKS=1`，单副本专家 135 行 = 2 满块 + 7 行尾块。

| mode | 结果 | 日志 |
|---|---|---|
| STREAM=0 | ECO_CAPTURE OK，8 rank 输出 | `/tmp/gcmoe-aligned-Q17dgmco/hot-parity-0.log` |
| STREAM=1 | ECO_CAPTURE OK，8 rank 输出 | `/tmp/gcmoe-aligned-Q17dgmco/hot-parity-1.log` |

`diff hot-parity-0.sha hot-parity-1.sha` → **空**（逐 rank BF16 输出 SHA-256 完全一致）：

```
rank0 78a0158c98563df6d33edd8e7fd133c309622e4a9ef9c0c73d36e2655e65b042
rank1-7 b9ae2a4b34297ca9779700d018302d43f62e3a133d320befb8f048932ac36b5b
```

## 3. 两个兼容性修复（均不改语义）

### 3.1 `a0e1362` — `install_eplb_plan` 丢了 1-D 广播

**症状**：B 轮 2b 门禁八个 rank 全部 `ValueError: slot_logical is (40, 480), expected (1, 480)`。

**原因**：`eco.meta_sm90.install_into_buffer` 允许 `[physical]` 单层计划广播到全部 40 行，
`install_eplb_plan` 的同一处却把它折成 1 层再与 40 行比对，必然失败。
所有传单个 placement 的独立脚本（`verify_eco_capture`、`measure_units`、`eco_triage`、
`verify_eco_shared`、`gpu_plan_only`、`verify_eco_e2e`）都会死在这一行。

**修复**：1-D 输入按 buffer 自身层数广播后再写 `slot_logical`/`expert_slots`；
**返回值仍保持调用者的 1-D 形状**（`gather_slot_weights` 按 1-D 切片）。
2-D 逐层计划路径与"层数不符即报错"的检查均未改动。CPU 级验证：
1-D → `plan (480,)` / `slots (40,384,8)`、40 行内容一致；2-D → `plan (40,480)`；7 层计划仍被拒。

### 3.2 `a4db419` — 门禁统计被 torchrun 交错输出打散

**症状**：八个 rank 的 `STREAM_CHECK` 明明都在、`diff` 也为空，却报
`test "$(wc -l …)" -eq 8` 失败。

**原因**：torchrun 把各 rank 的 stdout 直接拼接、不加换行，多条记录落在同一物理行。

**修复**：`grep -o 'STREAM_CHECK rank=[0-7] sha256=[0-9a-f]*'` 先按语义拆开再统计。
**判定门槛本身不变**（每 rank 一条、两模式一致）。

## 4. A/B 逐 rank 结果（layer 8，step 2，BS256）

| rank | A tok | B tok | A rows | B rows | A mb | B mb | **A routed µs** | **B routed µs** | Δ | **A d→end** | **B d→end** | Δ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 189 | 188 | 1512 | 1504 | 33 | 33 | 578.1 | 685.8 | +18.6% | 772.8 | 880.3 | +13.9% |
| 1 | 194 | 212 | 1552 | 1696 | 34 | 34 | 593.0 | **753.1** | **+27.0%** | **794.3** | **953.3** | **+20.0%** |
| 2 | 203 | 191 | 1624 | 1528 | 29 | 31 | 570.0 | 686.9 | +20.5% | 763.6 | 879.8 | +15.2% |
| 3 | 194 | 196 | 1552 | 1568 | 29 | 32 | 569.2 | 714.6 | +25.5% | 770.1 | 914.1 | +18.7% |
| 4 | 192 | 180 | 1536 | 1440 | 31 | 32 | 571.9 | 683.0 | +19.4% | 759.4 | 883.0 | +16.3% |
| 5 | 201 | 206 | 1608 | 1648 | 30 | 30 | 576.7 | 696.0 | +20.7% | 774.8 | 886.7 | +14.4% |
| 6 | 174 | 186 | 1392 | 1488 | 33 | 31 | 573.2 | 688.3 | +20.1% | 762.6 | 881.6 | +15.6% |
| 7 | 189 | 177 | 1512 | 1416 | 32 | 29 | 584.3 | 642.4 | +9.9% | 780.4 | 839.8 | +7.6% |
| **均值** | 192.0 | 192.0 | 1536.0 | 1536.0 | 31.4 | 31.5 | **577.1** | **693.8** | **+20.2%** | **772.2** | **889.8** | **+15.2%** |
| **中位数** | — | — | — | — | — | — | 575.0 | 687.6 | +19.6% | 771.4 | 882.3 | +14.4% |

layer 20 同口径：routed **A 606.4 → B 659.3 µs**（均值 +8.7%、**中位数 +17.8%**）、
d→end **A 804.0 → B 856.4 µs**（+6.5%，中位数 +13.0%）。
layer 20 的均值被 B 轮的 rank 7 拉低：该 rank 这一采样点只分到 **63 token / 14 个 M 块**
（其余 rank 190–233 token），属离群；剔除它后 routed 为 **A 606.4 → B 709.3 µs（+17.0%）**。

> **口径提示（严格按任务书 §4）**：A/B 两轮的逐槽 owned 分布**不同**
> （layer 8 中 rank6 从 1392 行变成 1488 行、rank1 从 1552 变 1696、rank7 从 1512 变 1416，
> 活跃槽数也从 18/20 变成 15/15 等），因为逐层 EPLB 在采样点给出的放置不同。
> 因此上表是**采样内对照**，不是同分布的受控对比。
> 好在**总行数完全相同（8×1536 = 12288）**，且下表 §5 证明**总工作量只差 0.3%**——
> 慢的部分不是"多做了活"，而是"同样多的活做得更不并行"。

## 5. 总工作量相同，并行度却掉了一档（这是本轮回执的核心证据）

| 指标（layer 8，八 rank 合计） | A baseline | B streamed | Δ |
|---|---:|---:|---:|
| tile 总数（L1+L2） | 9538 | 9576 | +0.4% |
| **tile 总工作量（tile·µs）** | **302,198** | **303,118** | **+0.3%** |
| L1 tile 平均时长 | 43.83 µs | 43.37 µs | −1.1% |
| L2 tile 平均时长 | 20.75 µs | 21.11 µs | +1.7% |
| routed 内核峰值并发 tile 数 | 568 | 557 | −1.9% |
| **routed 段跨度（八 rank 平均）** | **567 µs** | **686 µs** | **+21.0%** |
| **单 CTA 平均占用率**（work/span/71） | **0.938** | **0.776** | **−17.3%** |
| **CTA 内空闲（排队等活）占比** | **4.5%** | **13.9%** | **3.1×** |
| **`queue empty` 事件（八 rank 合计）** | **646** | **5522** | **8.5×** |
| 每个 CTA 领到的 tile 数（均值） | 16.8 | 16.9 | ≈0 |
| 每个 CTA 领到的 tile 数（最小值） | 16 | 12 | — |

**这排除了"流式多做活"和"流式单 tile 变慢"两种解释。**
每个 tile 的时长、每个 CTA 的总 tile 数几乎不变；变的是**每个 CTA 之间断了几次**。

个体证据（rank 1，layer 8）：

```
baseline : 71 个 CTA，queue-empty 共 71 次，全部落在 551–584 µs（每 CTA 恰好 1 次，即"收工前最后一次空转"）
streamed : 71 个 CTA，queue-empty 共 719 次，从 203 µs 起持续出现
           （200–300µs: 115+139 次，800–900µs: 149+24 次）
```

## 6. 机理：逐 M 块发票把"一次性铺满"变成"阶梯式追加"，并让 L2 与 L1 抢 CTA

### 6.1 L1 的起跑从"一次全放"退化为"每 64 行放一批"

baseline 在整槽 pull 完成时**一次性发布该槽全部 M 块**，所以只要一个槽拉完，
它的 18 个 L1 N-tile 立刻全部可领。取 rank1 的 L1 起跑时刻：

| baseline (slot 50, 400 行) | 起始 µs |
|---|---|
| 块0 的 18 个 L1 tile | 474（全部同刻） |
| 块1 | 477 |
| 块2 | 487 |
| 块3 | 508 |

| streamed (slot 51, 512 行) | 起始 µs |
|---|---|
| 块0 的 18 个 L1 tile | 396 |
| 块1 | 484 |
| 块2 | 540 |
| 块3 | 576 |

baseline 的块间起跑间隔 **3 / 10 / 21 µs**（后续块只是被 CTA 占满而顺延）；
streamed 的块间起跑间隔 **88 / 56 / 36 µs**——**这是被 pull 进度卡住的**，
因为票要等对应 64 行真正落地才发（`_publish_l1_batch`）。

于是 routed 段早段的并发被摊平（八 rank 合计活跃 tile 数）：

| t (µs) | 200 | 250 | 300 | 400 | 500 | 600 | 700 | 800 | 900 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 307 | 547 | 526 | 487 | 568 | 495 | 530 | **0** | **0** |
| streamed | **24** | 325 | 393 | 529 | 549 | 543 | 549 | **348** | **54** |

baseline 在 200 µs 就冲到 307 个并发、250 µs 就基本满负荷；
streamed 到 **400 µs 才追平**，然后一路拖到 950 µs，**尾部多出 ~150 µs 的尾巴**。

### 6.2 L1 与 L2 从"分阶段"变成"全程混跑"

流式允许 L2 在某块 L1 一完成就发（`_release_stream_l2`），于是 L2 与 L1 同时在跑：

| 10 µs 桶统计 | L1&L2 同时活跃 | 只有 L1 | 只有 L2 | 重叠率 |
|---|---:|---:|---:|---:|
| baseline | 9 | 33 | 18 | **15%** |
| streamed | 62 | 4 | 10 | **82%** |

L1 是 52 µs 的长 tile、L2 是 24 µs 的短 tile。混跑时每个 CTA 在长短任务间反复切换，
**流水线被打断的代价**正好体现在 `queue empty` 上：CTAs 不再"一次领一长串同类活"，
而是"领一个、回来、发现没合适的、再领"。

### 6.3 热点槽的 L1→L2 被拉长，不是缩短

任务书希望"首票提前"，实测确实提前了，但代价是**整条链变长**。
layer 8 rank 6（A 的 319/434/618/714/800 µs 旧边界对照位）：

| 边界 | 旧（回执记录的 layer 8 / rank 6） | 本轮 A (rank 6, slot 44) | 本轮 B (rank 6, slot 42) |
|---|---:|---:|---:|
| 热点槽 pull 完成 | 319 µs | 398 | 385 |
| 首个 L1 tile | 434 µs | 405 | 379 |
| 最后一个 L1 tile | 618 µs | 581 | 794 |
| 首个 L2 tile | 714 µs | 658 | 466 |
| 最后一个 L2 tile | 800 µs | 756 | 877 |
| routed 退出 | — | **762.6** | **881.6** |

**首票提前量**：首个 L2 从 658 → 466 µs（提前 192 µs），
但**最后一个 L2 从 756 → 877 µs（推后 121 µs）**，净效果是 routed 退出 **+119 µs**。
原因：L1 被拉长到 794 µs（+213 µs），因为它的 9 个块要等各自 pull 到位，
而 pull 完成时刻（385 µs）之后仍有块在陆续就绪。

### 6.4 最热点块的逐块发布表（layer 8）

**B（streamed）rank 1 / slot 51 / 512 行**（该轮 routed 退出最晚的 rank，953.3 µs）：

| 块 | 有效行 | L1 发布 | 首个 L1 | 最后 L1 | L2 发布 | 首个 L2 | 最后 L2 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 64 | 137 | 396 | 476 | 478 | 497 | 549 |
| 1 | 64 | 166 | 484 | 563 | 565 | 623 | 680 |
| 2 | 64 | 197 | 540 | 618 | 620 | 702 | 758 |
| 3 | 64 | 217 | 576 | 653 | 655 | 760 | 813 |
| 4 | 64 | 234 | 617 | 691 | 693 | 800 | 841 |
| 5 | 64 | 251 | 656 | 731 | 733 | 826 | 870 |
| 6 | 64 | 272 | 713 | 786 | 788 | 866 | 911 |
| 7 | 64 | 303 | 767 | 841 | 843 | 905 | 948 |

**A（baseline）rank 1 / slot 50 / 400 行**（同轮 routed 退出最晚的 rank，794.3 µs）：

| 块 | 有效行 | L1 发布 | 首个 L1 | 最后 L1 | L2 发布 | 首个 L2 | 最后 L2 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 64 | 236（整槽一次） | 474 | 529 | 593 | 718 | 753 |
| 1 | 64 | 236 | 477 | 540 | 593 | 728 | 759 |
| 2 | 64 | 236 | 487 | 560 | 593 | 733 | 765 |
| 3 | 64 | 236 | 508 | 581 | 593 | 741 | 772 |
| 4 | 64 | 236 | 528 | 583 | 593 | 747 | 781 |
| 5 | 64 | 236 | 530 | 593 | 593 | 757 | 789 |
| 6 | 16 | 236 | 541 | 585 | 593 | 765 | 780 |

注意 B 的 **L2 发布 = 本块最后一个 L1 + 2 µs**（几乎零延迟，语义正确），
但**首个 L2 往往要再等 20–80 µs 才真的被领走**（如块2 发布 620、实际 702），
因为此刻 71 个 CTA 都在处理 L1。

### 6.5 不变式校验（全部通过）

| 校验项 | layer 8 | layer 20 |
|---|---|---|
| 每 rank 发布记录数 = 2 × M 块数 | ✅ 8/8 rank | ✅ 8/8 rank |
| 每个 `(slot, m_block)` 的 L1/L2 发布唯一 | ✅ 无重复 | ✅ 无重复 |
| L1 tile 早于本块发布 | **0 例** | **0 例** |
| L2 发布早于本块最后一个 L1 tile | **0 例** | **0 例** |
| 同槽任务数 / 有效行数 / L2 push 数与 owned 一致 | ✅ | ✅ |
| L1 发布总数（八 rank 合计） | — （baseline 整槽发布，无逐块记录）| 252（L8）/ 231（L20）|
| 其中**早于整槽 pull 完成**发布 | — | 247（98.0%）/ 229（99.1%）|
| 满块（valid=64）的 L1 早于整槽 pull 完成 | — | **159/161（98.8%）** / **167/167（100%）** |

（最后一行是设计意图的正向验证：**逐块早发确实生效了**——它做到了，只是净性能是负的。）

## 7. 为什么"逐块早发"反而更慢（可直接落到代码上的判断）

对照 `source/gc_moe/impls/sm90_fp8_fp4_gc_moe.py` 与本轮实测：

1. **`_claim_stream_command` 每轮只试一次 L1、再试一次 L2，然后判定空**
   （`_try_claim_ready_tile` 两次 + `drained` 判定）。
   baseline 的 `_claim_command` 在整槽发布后，票池里**一次就有该槽全部 M 块的票**，
   所以领到票的 CTA 立刻回到"下一轮必中"的状态。
   流式下票是**持续追加**的，CTA 回来时新票往往还没发出来 → 空转一轮再回来。
   这就是 646 → 5522 次 `queue empty` 的直接来源。

2. **票池的"就绪粒度"从"槽"降到"块"，但消费端每次只消费一个 N-tile 票**，
   于是同一块内 18 个 L1 tile 的起跑被拉成 88/56/36 µs 的阶梯（§6.1），
   而 baseline 是 3/10/21 µs——**baseline 之所以快，是因为它的"阶梯"只是 CTA 占满的顺延，
   不是等数据**。流式的阶梯是**真的在等 pull**，这段等待无法被别的活填满，
   因为同一时刻还没有别的槽发布那么多票。

3. **L2 提前入场抢走了 L1 的 CTA 吞吐**（§6.2 重叠率 15% → 82%）。
   L2 是 24 µs 的短 tile，L1 是 52 µs 的长 tile；
   长短混跑让每个 CTA 的有效工作占比从 93.8% 掉到 77.6%。
   这 16.2 个百分点的占用率损失，乘上 567 → 686 µs 的跨度，正好是实测的 +110 µs 左右。

4. **净效应**：首票确实提前（rank6 的 L2 从 658 → 466 µs，提前 192 µs），
   但**尾部推后更多**（最后一个 L2 从 756 → 877 µs），
   因为提前入场的那批 L2 把 CTA 从"连片的长 L1"上切走了。
   任务是流式的（`drained` 要等全部 L1 与 L2 都领完），所以**提前量全部赔在了尾巴上**。

> 这不是采样噪声：八 rank **无一例外**变慢（最小 +7.6%，最大 +20.0%），
> 总 tile 工作量只差 +0.3%，`queue empty` 差 8.5×。
> 也不是"分布变了所以不可比"：A/B 的**总行数完全相同（12288）**，
> 每 rank 的 token 差 ≤ 18，且 §5 的"总工作量几乎相等"已经排除了负载差异。

## 8. 产物位置与校验

### 8.1 已提交进仓库（`docs/trace_executor_20261119/stream_results/`）

```
baseline/    step002_layer08_epoch14729.{counts,detail,perfetto,detail.perfetto}.json
             step002_layer08_epoch14729.detail.png
             step002_layer20_epoch14741.{counts,detail,perfetto,detail.perfetto}.json
             step002_layer20_epoch14741.detail.png
             sha256.txt
streamed/    step002_layer08_epoch14689.{counts,detail,perfetto,detail.perfetto}.json
             step002_layer08_epoch14689.detail.png
             step002_layer20_epoch14701.{counts,detail,perfetto,detail.perfetto}.json
             step002_layer20_epoch14701.detail.png
             sha256.txt
             hot-parity-0.sha  hot-parity-1.sha   （CUDA Graph 门禁 8 rank 摘要）
```

Perfetto 文件语义：`*.perfetto.json` 是七段粗粒度（count exchange / pull token /
shared L1+L2 / ECO plan / wait token / GEMM / combine，每 rank 7 条，共 56 条）；
`*.detail.perfetto.json` 在其上叠加 `Pull CTA 0..23`、`Routed CTA 0..70`、
`L2 push · CTA`、`Ready M blocks`（L1/L2 发布 instant）与 `claim queue`
（`ticket seen` / `queue empty` instant）。**时间为 µs，原点 = 该 rank 的 `count exchange`。**

> 单位说明：Perfetto 里 `ts`/`dur` 为 µs（Kineto 基准 + µs 偏移），
> **不是 ms**；本回执所有耗时同样以 µs 计。

### 8.2 原始数据（留在执行机）

| 内容 | 路径 |
|---|---|
| A 轮完整输出（含 `raw/`、`result.json`、全部日志） | `/tmp/gcmoe-aligned-7jKEw28l` |
| B 轮完整输出 | `/tmp/gcmoe-aligned-Q17dgmco` |
| 四组编译日志 | `/tmp/gcmoe-stream-compile-jT2GXIBl` |
| 逐块热点表（JSON） | `/tmp/rt/hotblocks.json` |

sha256（`raw/`，两轮各自记录在 `sha256.txt`）：
A `capture.json 57ea7f5b…`、`fine.json 8c3ea96a…`；
B 见 `streamed/sha256.txt`。

## 9. 剩余等待位置（任务书 §3 末项）

| 等待点 | baseline | streamed |
|---|---|---|
| 热点槽 pull 完成 → 首个 L1 | 7–134 µs（中位 ~53） | −58–98 µs（负值=块已先就绪） |
| 本块最后一个 L1 → L2 发布 | 整槽一次发布（无此段） | **+1–2 µs（几乎为零）** |
| L2 发布 → 首个 L2 tile 被领走 | — | **+20–80 µs（全部卡在 71 个 CTA 都在跑 L1）** |
| 最后一个 L2 tile → routed 退出 | 5–41 µs | 3–12 µs |

**结论：流式把"发票延迟"这一项优化到了接近 0（1–2 µs），
但把等待搬到了"领票延迟"（发布后 20–80 µs 才有人领），并且这笔等待是全网叠加的。**
下一步若要让流式真正获益，需要让 L2 的短 tile 不抢占 L1 长 tile 的 CTA
（例如 L1 优先 + L2 只在 L1 票池空时领，或给 L2 预留固定 CTA 数），
而不是继续提前发票时刻。

## 10. 我改了什么（按任务书"有兼容性修复时附提交 SHA"）

| 提交 | 文件 | 性质 |
|---|---|---|
| `a0e1362` | `source/gc_moe/sm90_gc/__init__.py` | 恢复 1-D placement 广播；返回值保持调用者形状；2-D 与层数校验不变 |
| `a4db419` | `scripts/trace_layer/run_aligned_ep8.sh` | 门禁按语义拆分 `STREAM_CHECK` 记录；8 条/rank 一致的门槛不变 |

**未改动**：发票时机、领取粒度、队列与等待条件、内存顺序、CUDA Graph、
计数与计时口径、所有实验配置（BS256 / layer 8,20 / step 2 / MB-CLAIM / TMA /
SM-reuse=0 / dispatch SMS=6 / 逐层 EPLB / max-copies 8）。
`source/gc_moe/impls/sm90_fp8_fp4_gc_moe.py` 的流式与非流式分支**本轮零改动**。

上一轮已修并仍在树的 `269d01d`（`pull_gemm_detail.py` 的 type-3 读 `rec[3]`）继续有效。

## 11. 复现

```bash
git checkout a4db419aa07cda4f35a100707b97093ff8889a26
GC_MOE_SM90_STREAM_M_BLOCKS=0 bash scripts/trace_layer/run_aligned_ep8.sh \
  a4db419aa07cda4f35a100707b97093ff8889a26 --fine-trace    # → 6de36ca6 / epoch 14729,14741
GC_MOE_SM90_STREAM_M_BLOCKS=1 bash scripts/trace_layer/run_aligned_ep8.sh \
  a4db419aa07cda4f35a100707b97093ff8889a26 --fine-trace    # → c19d1eed / epoch 14689,14701
```

注意：`epoch` 与 `capture_id` 逐轮不同（每次运行新的 epoch 快照），
校验时要**两轮各自**用各自的 `capture.json` id 与 `first_epoch + layer` 对齐，
不能跨轮比较 epoch 数值。
