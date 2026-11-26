# 回执 · SM90 H20 EP8 · BS256 Mega/GC 八臂（4 个冗余槽）

执行者：SM90 八卡执行者。任务书 `docs/trace_executor_20261119/SM90_H20_EPLB_EXTRA_SWEEP.md`，测量 SHA `47e9b06328336e10584ef942a710aa332538327b`。

| 项 | 值 |
|---|---|
| 模型启动 | **9 次**（arm0 发布者 + 八臂）|
| 完成臂 | **8/8 `complete`** |
| KV 预算 | 10 GiB/rank，1000+1000 步 |
| 冗余槽 | **k=4（52 物理槽/rank）** |
| 是否触发容量回退 | **否**，4 槽一次通过 |
| `run_tuned_eight.sh` 退出 | 1（`pairs` 阶段判定跨后端配对不可严格比较，见下）|
| `verification.json` failures | **{}**（空）|

## 4 槽配置无需回退

任务书要求「任一放置臂因显存或 KV 容量不足失败时，五个放置臂统一切换到 1 槽」。实际：

| 臂 | `post-warmup free` | 结果 |
|---|---|---|
| 4 | 14.74 GiB | 通过 |
| 5 | 8.15 GiB | 通过 |
| 6 | 14.79 GiB | 通过 |
| 7 | 14.74 GiB | 通过 |
| 8 | 14.78 GiB | 通过 |

最紧的是 arm5（Mega + EPLB，94.8 GiB 峰值），仍余 8.15 GiB。此前六臂任务里 Mega EPLB 用 56 槽时只余 4.41 GiB 就 OOM —— **52 槽是可行的**。

## 八臂主表

| 臂 | 后端 | shared | 放置 | step ms | FFN ms/步 | FFN µs/层 | token IR | 总 M 块 | 峰值 rank M 块 | 活跃槽 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Mega | vLLM | 原生 | 93.36 | 49.821 | **1245.5** | 1.368 | 376.1 | 56.4 | 300.0 |
| 2 | GC | fused split | 原生 | 90.43 | 46.358 | **1159.0** | 1.368 | 376.1 | 56.4 | 300.0 |
| 3 | GC | vLLM unfused | 原生 | 88.47 | 44.626 | **1115.6** | 1.368 | 376.1 | 56.4 | 300.0 |
| 4 | GC | fused split | EPLB k=4 + RR | 87.36 | 42.962 | **1074.0** | 1.089 | 391.4 | 52.8 | 332.0 |
| 5 | Mega | vLLM unfused | EPLB k=4 + RR | 93.03 | 49.630 | **1240.8** | 1.089 | 391.4 | 52.8 | 332.0 |
| 6 | GC | vLLM unfused | EPLB k=4 + RR | 83.69 | 40.535 | **1013.4** | 1.089 | 391.4 | 52.8 | 332.0 |
| 7 | GC | fused split | EPLB k=4 + ECO | 84.69 | 40.642 | **1016.0** | 1.057 | 376.1 | 49.6 | 315.1 |
| 8 | GC | vLLM unfused | EPLB k=4 + ECO | 82.05 | 38.706 | **967.6** | 1.057 | 376.1 | 49.6 | 315.1 |

### 三个维度的排序（FFN µs/层，越小越好）

**原生三臂（同 token IR 1.368）**：GC unfused 1115.6 < GC fused 1159.0 < Mega 1245.5

**EPLB+RR 三臂（同 token IR 1.089）**：GC unfused 1013.4 < GC fused 1074.0 < Mega 1240.8

**ECO 两臂（同 token IR 1.057）**：GC unfused **967.6** < GC fused 1016.0

**全八臂最快：arm8（GC unfused + EPLB k=4 + ECO）967.6 µs/层**。

## 28 对对照：7 对严格可比，21 对因后端差异不可比

`pairs.csv` 的 `strict_pair_comparable` 只有 7 对为 True：

| 配对 | native_diff | FFN 变化 | 说明 |
|---|---|---|---|
| 1→5 | 0 | **-0.38%** | Mega 内 |
| 2→4 | 0 | **-7.33%** | GC unfused 内 |
| 2→7 | 0 | **-12.33%** | GC unfused 内 |
| 3→6 | 0 | **-9.17%** | GC unfused 内 |
| 3→8 | 0 | **-13.27%** | GC unfused 内 |
| 4→7 | 0 | **-5.40%** | GC fused 内 |
| 6→8 | 0 | **-4.51%** | GC unfused 内 |

### 21 对不可比的根因

`native_diff` 只取决于**是否跨后端或跨 shared 实现**：

| native_diff | 出现条件 | 对数 |
|---|---|---|
| **0** | 同后端 + 同 shared 实现 | 7 |
| **711** | GC fused ↔ GC unfused（shared 实现不同）| 10 |
| **1160** | GC unfused ↔ Mega（后端不同）| 6 |
| **1184** | GC fused ↔ Mega（两者都不同）| 5 |

256,000 个 token 中出现 711–1184 个 top-1 差异（0.28%–0.46%）。这是**任务书明确要求的记录项**：

> 记录后端原生 M 块尺寸、权重精度差异及所有配对的原生输出差异。

Mega 用 FP8×FP8 routed，GC 用 FP8×FP4；fused 与 unfused 的 shared 执行方也不同。因此这三类跨实现配对**按定义不可等输出**。

`report_eight.py` 把它们记为 `strict_pair_comparable=False` 并让脚本返回 1，但 `verification.json` 的 `failures` 为空 —— 说明数据本身没有校验失败，只有跨实现配对被标记为不可比。

## 七对严格可比的性能结论

| 配对 | 变化 | 含义 |
|---|---|---|
| 2→4 | **−7.33%** | GC fused：EPLB k=4 相对原生 |
| 3→6 | **−9.17%** | GC unfused：EPLB k=4 相对原生 |
| 2→7 | **−12.33%** | GC fused：EPLB + ECO 相对原生 |
| 3→8 | **−13.27%** | GC unfused：EPLB + ECO 相对原生 |
| 4→7 | **−5.40%** | GC fused：ECO 相对 RR（均 k=4）|
| 6→8 | **−4.51%** | GC unfused：ECO 相对 RR（均 k=4）|
| 1→5 | **−0.38%** | Mega：EPLB k=4 相对原生 |

**核心结论**：

1. **EPLB k=4 在 GC 上收益显著**：fused −7.33%、unfused −9.17%；但 **Mega 上几乎没有（−0.38%）**。
2. **ECO 相对 RR 再快 4.5–5.4%**（GC 两个 shared 实现上一致）。
3. **GC unfused 始终优于 GC fused**（原生 1115.6 vs 1159.0；EPLB 1013.4 vs 1074.0）。
4. **Mega 后端在本配置下明显落后**：原生 1245.5 vs GC unfused 1115.6（+11.6%），且 EPLB 对它几乎无效。

## token IR 与冗余的关系

三组放置的 token IR 完全一致（同组内），说明 k=4 的 EPLB 放置对三个后端是同一套：

| 组 | token IR | 相对原生降幅 |
|---|---|---|
| 原生（1/2/3）| 1.3679 | — |
| EPLB + RR（4/5/6）| 1.0890 | **−20.4%** |
| EPLB + ECO（7/8）| 1.0569 | **−22.7%** |

## 归档

- https://github.com/ehdbdns/gcmoe-sm90/tree/main/docs/trace_executor_20261119/sm90_h20_token_ir_eight_results（225 MB）
- 含 `kv-10/`：八臂完整目录（各臂 `run.json`、8 rank NPZ、`installed-layout.json`、`artifacts.sha256`、`complete`、截断到 2 MB 的 `runner.log`）、`summary.csv`、`pairs.csv`、`verification.json`、`gpus.csv`、`processes.csv`。
- 大型 KV 与 route bank 保留在现场，未提交。
- 现场：`/mnt/fuse/h20-ir-eight-4Thp8jCg`。

## 退卡与清理

- 最后一臂（arm8）16:43:47 完成，`complete` 落盘。
- 八卡已释放，无残留 worker。
- 未修改任何代码（diff 为空）。

## 需要设计端确认

`report_eight.py` 把跨后端/跨 shared 的配对标为 `strict_pair_comparable=False` 并返回 1。任务书本身要求「记录后端原生输出差异」，说明这类差异是预期的。若希望脚本在这种情形下返回 0（只要求同实现内配对严格），需要设计端修改判定口径 —— 超出执行者权限，我未改。
