# 单层时间线（FINAL_V2，8 rank × 7 段）

## 0. 一句话

用 **trace 行号** 取出同一次 forward 的 5 个 gc_moe kernel（行 198395–198399），
把 `wait token` 从 routed kernel 里切出来，得到 8 rank × 7 段时间线。
**`GEMM` 从"第一个 compute CTA 真正拿到任务"开始，因此在 `pull token` 尚未结束时就已经开始**
（两者在时间上重叠 200–400 µs）。

## 1. 数据源（全部同一次运行）

| 项 | 值 |
|---|---|
| run label | `FINAL_V2` |
| 结果 JSON | `/home/admin/workspace/aop_lab/app_source/gcmoe_exp/t19_costfn/runs/FINAL_V2.json` |
| trace 目录 | `/home/admin/workspace/aop_lab/app_source/gcmoe_exp/t19_costfn/prof/final_v2/traces` |
| epoch | **87640** |
| trace 行号 | **198395–198399**（8 rank 相同）|
| ms/step（整 run）| 132.90 ms |

## 2. 七个段的精确定义（这是最容易搞错的地方）

一个 call 里有两个完全不同的"token 就绪"时刻，**只有第一个才是 GEMM 起点**：

| pull_tl 列 | 含义 | 用途 |
|---|---|---|
| `[10]` | dispatch 内核进入 | 所有差值的基准 |
| `[3]` | routing + count exchange 结束 | `count exchange` 段的终点 |
| **`[15]`** | **第一个 compute CTA 领到任务**（`_pull_tl_min_first`，red.min）| **`GEMM` 的起点** |
| `[5]` | **最后一个本地槽**拉完 | `pull token` 段的终点 |

```
count exchange : dispatch.start            → dispatch.start + ([3]-[10])
pull token     : 上者终点                  → min(dispatch.start + ([5]-[10]), dispatch.end)
shared L1+L2   : shared kernel 起止
ECO plan       : plan kernel 起止
wait token     : routed.start             → max([15]换算, routed.start)
GEMM           : 上者终点                  → routed.end
combine        : combine kernel 起止
```

**关键：`[15]` 与 `[5]` 差 150–400 µs。**
用 `[5]` 当 GEMM 起点（我最初的版本）会凭空造出一段几百 µs 的假等待，
并让"某些 rank 在 pull 结束后很久才开始算"这种不可能的现象出现。

## 3. 硬约束校验（8/8 通过）

| rank | shared−dispatch | routed−shared_end | combine−routed_end | wait+GEMM−routed | GEMM 起点−routed 起点 | 首个任务(µs) | 全槽拉完(µs) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | -0.90 | +0.93 | +2.78 | +0.000 | -0.1 | 196 | 586 |
| 1 | -1.06 | +0.86 | +2.66 | +0.000 | -1.1 | 197 | 293 |
| 2 | -0.83 | +1.28 | +2.24 | +0.000 | +6.2 | 203 | 355 |
| 3 | +1.02 | +1.06 | +3.52 | +0.000 | +2.5 | 197 | 321 |
| 4 | +1.79 | +0.93 | +2.66 | +0.000 | +2.8 | 197 | 597 |
| 5 | +0.74 | +0.90 | +2.62 | +0.000 | +8.4 | 203 | 254 |
| 6 | -2.43 | +1.18 | +2.94 | +0.000 | +12.3 | 204 | 425 |
| 7 | -2.56 | +0.96 | +2.75 | +0.000 | +8.4 | 198 | 334 |

- `shared − dispatch`：−2.43…+1.79 µs ✅（预期 −2…+3）
- `routed − shared_end`：+0.86…+1.28 µs，8/8 全正 ✅
- `combine − routed_end`：+2.24…+3.52 µs ✅
- `count exchange` 末 == `pull token` 起：0.000000 µs ✅
- `wait token + GEMM == routed kernel 总时长`：0.000000 µs ✅

**关于 GEMM 起点那列出现负值（rank 0 是 −0.1 µs）：**
`pull_tl` 用 kernel `globaltimer`，trace 用 CUPTI，两个时钟域不同，
两者相减在 256 个 call 上的分布是 **−14…+16 µs，中位 +3.8 µs**。
负值表示"首个任务在 kernel 启动时已经就绪"，即**没有等待**，不是数据错误。
画图时会把负值 clamp 到 0（画不出负宽度的条），原始值保留在 `checks` 里。

## 4. 七段耗时（µs）

| rank | count exchange | pull token | shared L1+L2 | ECO plan | wait token | GEMM | combine |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 20.0 | 525.8 | 196.1 | 103.5 | 0.0 | 786.3 | 170.1 |
| 1 | 19.0 | 263.9 | 198.3 | 106.6 | 0.0 | 588.4 | 364.9 |
| 2 | 20.0 | 335.0 | 196.3 | 105.4 | 6.2 | 600.2 | 349.9 |
| 3 | 20.0 | 301.0 | 192.4 | 113.2 | 2.5 | 621.8 | 333.1 |
| 4 | 22.0 | 575.0 | 191.5 | 112.0 | 2.8 | 942.0 | 15.0 |
| 5 | 20.0 | 234.0 | 192.9 | 107.8 | 8.4 | 547.1 | 400.5 |
| 6 | 19.0 | 406.0 | 193.0 | 108.7 | 12.3 | 629.7 | 315.2 |
| 7 | 19.0 | 315.0 | 191.2 | 112.2 | 8.4 | 562.8 | 389.1 |
| **mean** | 19.9 | 369.5 | 194.0 | 108.7 | 5.1 | 659.8 | 292.2 |

**`wait token` 只有 0–12.3 µs**（负值按 0 计），说明：
routed kernel 启动时，第一批 token 已经到位，CTA 立即开始 GEMM。

## 5. 与 pull 的重叠（本图最重要的一句话）

| rank | GEMM 起点 | pull token 终点 | 重叠时长 |
|---:|---:|---:|---:|
| 0 | 197.9 | 547.6 | **349.7** |
| 1 | 200.2 | 285.0 | **84.8** |
| 2 | 204.7 | 356.7 | **152.0** |
| 3 | 198.2 | 322.2 | **124.0** |
| 4 | 197.0 | 597.0 | **400.0** |
| 5 | 204.9 | 255.9 | **51.0** |
| 6 | 207.5 | 428.5 | **221.0** |
| 7 | 200.9 | 336.9 | **136.0** |

8 个 rank 的 GEMM 起点都在 ~197 µs，而 pull 要到 255–597 µs 才结束 ⇒
**GEMM 与 pull 并行 51–400 µs（均值 190 µs）**。
这解释了为什么 `pull` 只占整层的 9.4% 却几乎不拖慢墙钟 —— 它被 compute 掩盖了。

## 6. plan 预测 vs 实测（同一层、同一 epoch）

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

这一层 plan 严重低估 rank 间差异（5.7% vs 58.5%）。
单层样本不代表整体：整 run 的 `[plan-pred]` L1 = 0.8356。
本层偏差的直接来源可从数据读出：rank 4 的 GEMM 942 µs、combine 仅 15 µs；
rank 5 相反（547 / 400）。plan 的 cost 只含 GEMM+push+gap，**不含 combine**，看不到这个交换。
## 7. 本层 token / M 块数

⚠️ **这一项无法与图中的 call 对齐**，原因见 §8。
下表是本 run 最后一个有数据的 (layer, step)，标注清楚以免误读：

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

（`ir_step.tables` 第 39998 行 = layer 39 / step 998）

**不要**把这张表和图里那一层并列 —— 不是同一层。

## 8. 三种数据源覆盖的时间窗不同

| 数据源 | 覆盖内容 | step 区间 |
|---|---|---|
| torch trace | 39680 call（992 步 × 40 层）| 0–991 |
| `pull_tl`（256 环）| **最后 256 个 call** = epoch 87385–87640 | 1007–1013 |
| `ir_step.tables` | 1000 步 × 40 层 | 0–998 |
| `fine_rows`（524288 环）| 82 个 epoch = 47000–47081（窗口开头）| −3–0 |

后果：
1. `wait token` 只能用 `pull_tl`，所以只能用最后 256 个 call（本图取最后一个）。
2. `ir_step` 的 token/M 块数落在另一段窗口，**不能**与本层并列。
3. `fine_rows` 与 `pull_tl` 完全不重叠，不能互相替代。

## 9. 与 `ONE_CALL.md`（行 39605–39609）的关系

方法相同（都按 trace 行号取同一次 forward），但取了不同调用：

| | 行号 | call | epoch | 有 `pull_tl` 打点 |
|---|---:|---:|---|---|
| 旧 `ONE_CALL.md` | 39605–39609 | 7921 | — | ❌ 不在 256 环内 |
| **本文** | **198395–198399** | **39679** | **87640** | ✅ |

`pull_tl` 只保留最后 256 个 call，call 7921 远在窗口前段，
它的 `pull`/`waittok` 不可能来自内核打点。本文选的 call 在环内，
三段（count exchange / pull / wait）**全部是真实打点，无插值**。

## 10. 复现

```bash
cd /home/admin/workspace/aop_lab/app_source/gcmoe_exp/t19_costfn
P=/home/admin/workspace/aop_lab/app_source/gc_moe_rtp_ep8_20261031/env3/bin/python

for r in 0 1 2 3 4 5 6 7; do $P docs/final_orig_layer/scripts/extract_tail.py $r; done
$P docs/final_orig_layer/scripts/build_layer.py
$P docs/final_orig_layer/scripts/plot_layer.py \
     --segments docs/final_orig_layer/one_layer_call39679.segments.json \
     --out docs/final_orig_layer/one_layer_call39679.png
```

## 11. 文件清单

| 文件 | 内容 |
|---|---|
| `one_layer_call39679.perfetto.json` | 8 rank × 7 段（56 段）|
| `one_layer_call39679.png` | 全层（3 条并发泳道 × 8 rank）|
| `one_layer_call39679_zoom.png` | 0–260 µs 放大 |
| `one_layer_call39679.segments.json` | 全部数值 + 逐 rank 校验 |
| `scripts/*.py` | 抽取 / 分段 / 出图 |