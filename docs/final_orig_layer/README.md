# 原始耗时函数 · 单个 measure 层 · 8 rank 全量明细

**同一次运行**（`FINAL_ORIG_B256`）同时采集了三套数据，均为同一 epoch `87200`：

| 数据 | 来源 | 说明 |
|---|---|---|
| token / 激活专家 / M块数 | `ir_step.layer_detail` | dispatch kernel 设备侧按 (layer,step) 记录 |
| 每 M 块的 valid | `eco_trace.fine_rows` type=6 | 61k 条记录，覆盖整层 |
| 预测 T[r] | `pull_tl[epoch][24+r]` | plan 自己的 cost |
| 实测耗时 | `pull_tl[epoch][10] → [11]` | dispatch 入口 → compute 退出（同一 rank 的设备时间戳，均来自 `pull_tl`，有效） |

## 配置

```
GC_MOE_ECO_COSTFN=0   # 原始耗时函数: wb=26, wa=0, 末项 +token
GC_MOE_EPLB=1 GC_MOE_ECO=1 GC_MOE_EPLB_EXTRA_PER_RANK=8
GC_MOE_MB_CLAIM=1     # 按 M 块领任务
GC_MOE_SHARED_IN_GC=1 # shared expert 由 gc_moe role 4 计算
ENFORCE_EAGER=0       # CUDA graph
BS256 (32 seq/rank), 1000 warm + 1000 measure
```

实测 `ms/step = 53.21`（measure 段稳态）。

> 本目录已被一次完整重跑（`FINAL_V2`）替换：那次运行同时开了 torch trace、
> 逐 (layer,step) 表（`ir_step.tables`，全 40×1000）和 FINE 每块 valid 日志，
> 因此下面的每个数字都来自**同一层**。旧的 `one_layer_epoch87200.*` 已删除。

## 该层（step 170, layer 13, trace idx 6493）

| rank | tokens | 激活专家 | M块数 | valid均值 | valid中位 | valid总和 | role1 dispatch µs | role4 shared µs | role2 routed µs | role3 combine µs | r2起−r4止 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 197 | 20 | 21 | 42.3 | 56 | 61104 | 318 | 194 | 610 | 142 | 1.0 |
| 1 | 197 | 23 | 23 | 36.2 | 28 | 55024 | 237 | 193 | 609 | 141 | 1.2 |
| 2 | 189 | 19 | 19 | 40.0 | 32 | 59280 | 307 | 192 | 591 | 159 | 1.0 |
| 3 | 190 | 20 | 20 | 35.2 | 24 | 57456 | 240 | 193 | 589 | 163 | 0.9 |
| 4 | 189 | 21 | 21 | 39.8 | 48 | 58976 | 285 | 194 | 594 | 158 | 1.1 |
| 5 | 192 | 18 | 18 | 38.0 | 40 | 57760 | 379 | 193 | 610 | 139 | 1.0 |
| 6 | 196 | 18 | 18 | 39.2 | 40 | 59584 | 290 | 193 | 586 | 154 | 1.0 |
| 7 | 186 | 14 | 15 | 39.0 | 40 | 57760 | 525 | 192 | 737 | 16 | 1.0 |

- token IR = 1.0260   M块 IR = 1.1871
- M块 max/min = 1.533   token max/min = 1.059

## 三个关键结论

**1. `M块数 == 激活专家数`**

`block_m=64`，每 rank 每层 186–197 token，摊到 14–23 个激活槽，每槽 8–14 个 token，
全部 < 64，所以每个激活槽恰好产生 1 个 M 块。**块数不均衡 = 激活槽数不均衡。**

**2. token 几乎均衡，块数差 1.53 倍**

| 量 | max/min |
|---|---:|
| token | **1.059** |
| M 块数 | **1.533** |

rank 1 和 rank 0 的 token 完全相同（197），但 M 块数是 23 vs 21；
rank 7 的 token 最少（186）却只有 15 块，rank 2 用 189 个 token 摊到 19 块。
**token 只差 6%，块数差 53%。** 这就是“按 token 建模”失效的根源。

**3. valid 方向相反，部分抵消了块数差异**

| rank | M块 | valid均值 | valid总和 | routed µs |
|---:|---:|---:|---:|---:|
| 1 | 23 | **36.2** | 55 024 | 609 |
| 0 | 21 | **42.3** | 61 104 | 610 |

rank 1 块更多但每块 valid 更小（36.2 vs 42.3），valid 总和反而更低（55k vs 61k），
最终 routed 耗时几乎相同（609 vs 610 µs）。**块数 × 单块 valid 才是耗时驱动量。**

## 时序图（单层）请看 GROUPED.md

本目录的 `one_layer.perfetto.json` 是早期版本；当前正确的单层时序图是
`one_layer_grouped.png` / `one_layer_grouped.perfetto.json`，说明见 **`GROUPED.md`**。
其中记录了旧版 wait 切分的错误（把"内核还没启动"的时间算进了 role2 内部等待）及修正。

## 文件

- `one_layer.perfetto.json` — 8 rank × 5 role，拖进 <https://ui.perfetto.dev>
  每 rank 一个 process，role 一条 thread，条内写着阶段名和耗时。
  箭头是 `wait_for(role4)`，8 个 rank 上 `role2起 − role4止` 均为 0.9–1.2 µs。
- `one_layer_ir.json` — 该层的 8 rank token / M块 / 激活专家

## 复现要点（这次踩过的坑）

| 坑 | 症状 | 修法 |
|---|---|---|
| `fine_floor` 用 `header[0]` | 记录全是 prefill（epoch 46919 vs 窗口 87240） | 改用 `eco_tl_step` |
| `_irstep_reduce` 丢原始表 | 只能拿 layer 0/step 0 | dump `tables`（40×1000×8） |
| `FINE_RECS` 不够 | 单层 19.8 万条 > 26.2 万容量 | 提到 524288 |
| trace 与 ir_step 无共同锚点 | perfetto 层和 token 表不同层 | `step=(i-start)//40, layer=i%40` |
