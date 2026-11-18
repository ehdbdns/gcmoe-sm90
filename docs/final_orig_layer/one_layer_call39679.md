# 单层时间线（FINAL_V2，8 rank × 7 段）

## 0. 一句话

用 **trace 行号** 取出**同一次 forward** 的 5 个 gc_moe kernel（行 198395–198399），
把 `wait token` 从 routed kernel 里单独切出来，得到 8 rank × 7 段的时间线。
全部 8 个 rank 通过交接文档列出的 5 条物理约束。

## 1. 数据源（全部同一次运行）

| 项 | 值 |
|---|---|
| run label | `FINAL_V2` |
| 结果 JSON | `/home/admin/workspace/aop_lab/app_source/gcmoe_exp/t19_costfn/runs/FINAL_V2.json` |
| trace 目录 | `/home/admin/workspace/aop_lab/app_source/gcmoe_exp/t19_costfn/prof/final_v2/traces` |
| epoch | **87640** |
| trace 行号 | **198395–198399**（8 rank 完全相同）|
| 该层调用序号 | 39679 / 39680 |
| ms/step（整 run）| 132.90 ms |

8 个 rank 的角色序列逐行断言一致：
`dispatch(24×256) → shared(71×256) → routed(71×384) → plan(1×512) → combine(288×384)`

## 2. 硬约束校验（8/8 通过）

| rank | shared−dispatch | routed−shared_end | combine−routed_end | wait+GEMM−routed | pull 越界(clamp) |
|---:|---:|---:|---:|---:|---:|
| 0 | -0.90 | +0.93 | +2.78 | +0.000 | 40.2 |
| 1 | -1.06 | +0.86 | +2.66 | +0.000 | 10.1 |
| 2 | -0.83 | +1.28 | +2.24 | +0.000 | 0.0 |
| 3 | +1.02 | +1.06 | +3.52 | +0.000 | 0.0 |
| 4 | +1.79 | +0.93 | +2.66 | +0.000 | 0.0 |
| 5 | +0.74 | +0.90 | +2.62 | +0.000 | 0.0 |
| 6 | -2.43 | +1.18 | +2.94 | +0.000 | 0.0 |
| 7 | -2.56 | +0.96 | +2.75 | +0.000 | 0.0 |

对照交接文档给出的预期：
- `shared − dispatch`：预期 −2…+3 µs → 实测 **−1.06…+1.79 µs** ✅
- `routed − shared_end`：预期 +0.9…+1.2 µs，必须 >0 → 实测 **+0.86…+1.28 µs**，8/8 全正 ✅
- `combine − routed_end`：预期 +2.6…+4.5 µs → 实测 **+2.24…+3.52 µs** ✅
- `count exchange` 末 == `pull token` 起：**残差 0.000000 µs**（同一定义，恒等）✅
- `wait token + GEMM == routed kernel 总时长`：**残差 0.000000 µs** ✅

## 3. 七段耗时（µs）

| rank | count exchange | pull token | shared L1+L2 | ECO plan | wait token | GEMM | combine |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 20.0 | 525.8 | 196.1 | 103.5 | 349.7 | 436.6 | 170.1 |
| 1 | 19.0 | 263.9 | 198.3 | 106.6 | 84.8 | 503.6 | 364.9 |
| 2 | 20.0 | 335.0 | 196.3 | 105.4 | 158.2 | 448.2 | 349.9 |
| 3 | 20.0 | 301.0 | 192.4 | 113.2 | 126.5 | 497.8 | 333.1 |
| 4 | 22.0 | 575.0 | 191.5 | 112.0 | 402.8 | 542.0 | 15.0 |
| 5 | 20.0 | 234.0 | 192.9 | 107.8 | 59.4 | 496.1 | 400.5 |
| 6 | 19.0 | 406.0 | 193.0 | 108.7 | 233.3 | 408.7 | 315.2 |
| 7 | 19.0 | 315.0 | 191.2 | 112.2 | 144.4 | 426.8 | 389.1 |
| **mean** | 19.9 | 369.5 | 194.0 | 108.7 | 194.9 | 469.9 | 292.2 |

## 4. kernel 原始区间（µs，同一原点 = 8 rank 中最早的 dispatch 起点）

| rank | dispatch | shared | routed | plan | combine |
|---:|---|---|---|---|---|
| 0 | 1.8→547.6 | 0.9→197.0 | 197.9→984.3 | 0.3→103.8 | 987.0→1157.2 |
| 1 | 2.1→285.0 | 1.0→199.4 | 200.2→788.6 | 0.5→107.1 | 791.2→1156.1 |
| 2 | 1.7→406.3 | 0.8→197.2 | 198.4→804.8 | 0.2→105.6 | 807.1→1156.9 |
| 3 | 1.2→363.9 | 2.2→194.6 | 195.6→819.9 | 3.5→116.7 | 823.4→1156.6 |
| 4 | 0.0→652.1 | 1.8→193.2 | 194.2→1139.0 | 0.0→112.0 | 1141.6→1156.6 |
| 5 | 1.9→281.4 | 2.6→195.5 | 196.4→752.0 | 1.0→108.8 | 754.6→1155.2 |
| 6 | 3.5→448.0 | 1.1→194.0 | 195.2→837.2 | 2.5→111.2 | 840.1→1155.4 |
| 7 | 2.9→350.5 | 0.4→191.5 | 192.5→763.7 | 1.8→113.9 | 766.4→1155.6 |

⚠️ 注意：`dispatch`、`shared`、`plan`、`routed` 是**并发**的四个 kernel（同一 call 内起点相差 <2 µs）。
图中把它们画成三条泳道就是为了不掩盖这个重叠。
## 5. plan 预测 vs 实测（同一层、同一 epoch）

plan 在 epoch 87640 导出的 `T[r]`（`pull_tl[epoch][24..31]`），
与同一 call 的 routed kernel 实测跨度比较（各自按均值归一化，只比形状）：

| rank | 实测 routed (µs) | 实测比例 | 预测比例 | 差 |
|---:|---:|---:|---:|---:|
| 0 | 786.3 | 1.1827 | 0.9958 | +0.1869 |
| 1 | 588.4 | 0.8849 | 0.9731 | -0.0882 |
| 2 | 606.4 | 0.9121 | 1.0300 | -0.1179 |
| 3 | 624.3 | 0.9389 | 0.9918 | -0.0529 |
| 4 | 944.8 | 1.4210 | 1.0145 | +0.4065 |
| 5 | 555.6 | 0.8356 | 0.9812 | -0.1456 |
| 6 | 642.0 | 0.9656 | 1.0129 | -0.0473 |
| 7 | 571.2 | 0.8591 | 1.0007 | -0.1416 |

- **L1 偏差 = 1.1867**，最大偏差 = 0.4065
- 预测跨度 **5.7%**，实测跨度 **58.5%**

**结论：这一层的 plan 严重低估了 rank 间差异。**
plan 认为 8 个 rank 只差 5.7%，实测差 58.5%；L1=1.19。
与 `PROGRESS_20261117.md` 的统计结论同向（plan 系统性高估/低估），
但**单层样本不能代表整体**——整 run 的 `[plan-pred]` 是 L1=0.8356（见 §7）。

单层出现这么大偏差的原因（可从本层数据直接读出）：
- rank 4 的 routed 用了 944.8 µs，rank 5 只用 555.6 µs（1.70×），
  而 plan 预测两者只差 2.4%；
- 但 rank 4 的 `combine` 只有 **15.0 µs**，rank 5 是 **400.5 µs** ⇒
  rank 4 把时间花在 GEMM 上、combine 极短，rank 5 正好相反。
  plan 的 cost 只覆盖 GEMM+push+gap，**不含 combine**，所以它看不到这个交换。

## 6. 本层 token / M 块数

⚠️ **这一项在 FINAL_V2 里无法与上面的 call 对齐**，原因见 §7。
下面给出的是**本 run 最后一个有数据的 (layer, step)**，标注清楚以免误读：

| rank | tokens | M 块数 | 激活槽 |
|---:|---:|---:|---:|
| 0 | 196 | 21 | 19 |
| 1 | 188 | 21 | 21 |
| 2 | 188 | 25 | 24 |
| 3 | 192 | 26 | 25 |
| 4 | 193 | 21 | 19 |
| 5 | 182 | 24 | 24 |
| 6 | 204 | 16 | 15 |
| 7 | 193 | 15 | 15 |

（来自 `ir_step.tables` 第 39998 行 = layer 39 / step 998）

**不要**把这张表和图里那一层并列 —— 它们不是同一层。

## 6b. 与 `ONE_CALL.md`（行 39605–39609）的关系

两者都按 trace 行号取同一次 forward，**方法一致**，但取了**不同的调用**：

| | 行号 | call | epoch | 是否有 `pull_tl` 打点 |
|---|---:|---:|---|---|
| `ONE_CALL.md` | 39605–39609 | 7921 | — | ❌ 不在 256 环内 |
| **本文** | **198395–198399** | **39679** | **87640** | ✅ 在环内 |

`pull_tl` 只保留**最后 256 个 call**（epoch 87385–87640），而 call 7921 远在窗口前段
——它那一层的 `pull token` / `wait token` 不可能由 `pull_tl` 直接给出。
本文选择的 call 39679 是环内最后一个，因此 **`count exchange` / `pull token` /
`wait token` 三段都有内核真实打点**，不需要任何插值。

`ONE_CALL.md` 里 8 个 rank 的 pull/waittok 数值因此不应与本文并列比较。

## 7. ⚠️ 三种数据源覆盖的时间窗不同（必须知道）

这是本 run 的硬事实，不是脚本 bug。三个缓冲区各有自己的窗口：

| 数据源 | 覆盖内容 | 对应 step 区间 |
|---|---|---|
| torch trace | 39680 个 call（**992 步 × 40 层**）| 大致 step 0–991 |
| `pull_tl`（256 环）| **最后 256 个 call** = epoch 87385–87640 | step 1007–1013 |
| `ir_step.tables` | 1000 步 × 40 层，39960 个有效行 | step 0–998 |
| `fine_rows`（524288 环）| 82 个 epoch，47000–47081 = 测量窗口**开头** | step −3–0 |

由此产生三个后果：

1. **`wait token` 只能用 `pull_tl` 算**，所以只能用最后 256 个 call（本图取第 256 个）✅
2. **`ir_step` 的 token / M 块数无法与这个 call 对齐**（窗口不重叠）——
   §6 给的是 step 998，**不是**本图那一层。这是刻意标注，不是拼数据。
3. **`fine_rows` 与 `pull_tl` 完全不重叠**（47000–47081 vs 87385–87640），
   所以不能用 fine 的 GEMM 明细去替换 trace 的 routed 段。

另外两个已量化的口径问题：

- **trace 的 `ts` 和 `dur` 同为 µs**。证据：同一 call 内 `routed.ts − shared.ts`
  与 `shared.dur` 的比值恒为 **1.005**（8 个 call 全部如此）。
  一开始误当作 ns，导致物理约束全为负；改回 µs 后 5 条约束全部落进预期区间。
- **`pull_tl` 的 pull 窗口与 dispatch kernel 时长统计一致，但不恒等**：
  差的中位 −7.5 µs，p10/p90 = −56.7/+36.2 µs（8 rank × 256 call 统计）。
  原因是 CUPTI 的 launch 边界与 `globaltimer` 是两个时钟域。
  因此 `pull_done` 若越过 `dispatch.end` 会被 **clamp 到 kernel 边界**，
  越界量在本图中最多 **40.2 µs**（rank 0），已在 `checks.pull_overshoot_clamped_us` 里逐 rank 记录。

## 8. 复现

```bash
cd /home/admin/workspace/aop_lab/app_source/gcmoe_exp/t19_costfn
P=/home/admin/workspace/aop_lab/app_source/gc_moe_rtp_ep8_20261031/env3/bin/python

# 1) 抽取 trace 尾部（每个 rank 约 28 s；结果落在 /tmp/rt/tail{r}.json）
for r in 0 1 2 3 4 5 6 7; do $P scripts/trace_layer/extract_tail.py $r; done

# 2) 生成分段 + perfetto + 校验
$P scripts/trace_layer/build_layer.py

# 3) 出图（全层 + 前 260 µs 放大）
$P scripts/trace_layer/plot_layer.py \
     --segments docs/final_orig_layer/one_layer_call39679.segments.json \
     --out docs/final_orig_layer/one_layer_call39679.png
$P scripts/trace_layer/plot_layer.py \
     --segments docs/final_orig_layer/one_layer_call39679.segments.json \
     --out docs/final_orig_layer/one_layer_call39679_zoom.png --xmax 260
```

`build_layer.py` 会打印 8 个 rank 的全部校验值；任何一项出界都会在输出的
`checks` 里直接看到。

## 9. 文件清单

| 文件 | 内容 |
|---|---|
| `one_layer_call39679.perfetto.json` | 8 rank × 7 段的 perfetto（可直接拖进 ui.perfetto.dev）|
| `one_layer_call39679.png` | 全层时间线（3 条并发泳道 × 8 rank）|
| `one_layer_call39679_zoom.png` | 0–260 µs 放大（看清 pull/plan/shared 的并行）|
| `one_layer_call39679.segments.json` | 全部原始数值 + 逐 rank 校验 |
| `scripts/trace_layer/extract_tail.py` | 按行号抽 trace 尾部的 5-role 记录 |
| `scripts/trace_layer/build_layer.py` | 分段、校验、出 perfetto |
| `scripts/trace_layer/plot_layer.py` | 出 PNG |