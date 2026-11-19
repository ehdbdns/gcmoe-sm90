# Absolute Trace：单层全过程时间线（pull CTA + compute CTA 同框）

把「谁在什么时候干了什么」画到一张图上：横轴是层内绝对时间（µs），
纵轴是 CTA 行。可以直接看出 **pull 与 GEMM 的重叠关系**、
**每个 M 块算了多久**、**combine push 何时开始**。

- 环境目录：`/home/admin/workspace/aop_lab/app_source/gcmoe_exp/t18_abstrace`
- 已有样例：`docs/abstrace/AT256_eco_rank0_full.png` 等
- 实现脚本：`scripts/plot_absolute_trace.py`（绘图）、
  `scripts/dump_absolute_timeline.py`（导出 CSV）

---

## 一、这套 trace 的数据从哪来

内核在 `fine_tl` 里**逐事件**写 10 字记录（`source/gc_moe/impls/sm90_fp8_fp4_gc_moe.py`）：

```
[t_us, type, id, a0, a1, a2, a3, a4, epoch, -]
```

| type | 含义 | 字段 |
|---:|---|---|
| 1 / 2 | `slot_begin` / `slot_end` | `id`=全局槽号，`a0`=逻辑专家 |
| **6 / 7** | **`mblock_begin` / `mblock_end`** | `id`=CTA，`a0`=expert，`a1`=start_row，`a2`=valid，`a3`=ntile，`a4`=phase(1=L1,2=L2) |
| 8 / 9 | `push_begin` / `push_end` | 同上字段 |
| 10 / 11 | `shared_start` / `shared_end` | 后期加的（本文档环境不一定有）|

三个关键设计（决定了这套 trace 可用）：

1. **`fine_tl` 是环形缓冲**：`slot = counter % GC_MOE_FINE_RECS`。
   一个 BS256 层就有 ~59k 条记录，前缀缓冲只会留下 prefill 的尾巴。
2. **时间基准用 64 位原点**：`globaltimer//1000` 超过 int32 会溢出
   （实测解出来是 −7.2e8），所以第一条记录把原点 latch 到 `fine_t0`，
   之后只存「相对原点的微秒数」。
3. **配对只在同一个 epoch 内做**：环形回绕会残留上一层的记录，
   跨 epoch 配对会得到 ±5e7 µs 的荒谬跨度。

---

## 二、怎么开启（完整可复现命令）

```bash
cd /home/admin/workspace/aop_lab/app_source/gcmoe_exp/t18_abstrace
source scripts/env.sh

# ---- 必开：ECO + EPLB + 三类 trace ----
export GC_MOE_EPLB=1
export GC_MOE_ECO=1
export GC_MOE_ECO_WB=26
export GC_MOE_ECO_WA=0
export GC_MOE_EPLB_EXTRA_PER_RANK=8
export GC_MOE_ECOPLB_PER_LAYER=1

export GC_MOE_ECO_TRACE=1        # eco_tl：plan 的预测 T[r] 等
export GC_MOE_PULL_TRACE=1       # pull_tl：dispatch 入口/退出等标量打点
export GC_MOE_FINE_TRACE=1       # fine_tl：逐事件日志（本图的数据源）

export GC_MOE_FINE_EPOCHS=1      # 只记录 1 个 epoch（= 1 层），避免环形覆盖
export GC_MOE_FINE_RECS=524288   # 一层 ~59k 条，给足空间
export GC_MOE_FINE_MOD=0         # 不按 epoch 取模过滤

# ---- 模型权重预热：不做这一步，权重加载会从 3 秒变成 10 分钟 ----
bash scripts/warm_page_cache.sh

# ---- 跑实验 ----
$GC_ENV/bin/python scripts/real_decode_100_100.py \
  --label AT256_eco --output runs/AT256_eco.json \
  --prefill-lo 256 --prefill-hi 2048 \
  --seqs-per-rank 32 --warm-steps 1000 --measure-steps 1000
```

### 环境变量速查

| 变量 | 作用 | 不开会怎样 |
|---|---|---|
| `GC_MOE_FINE_TRACE=1` | 开启 `fine_tl` 逐事件日志 | **没有数据，画不出图** |
| `GC_MOE_FINE_EPOCHS=1` | 只记 1 个 epoch | 记满环形缓冲，只剩最后几层 |
| `GC_MOE_FINE_RECS=524288` | 环形大小 | 默认 262144，一层勉强够 |
| `GC_MOE_ECO_TRACE=1` | `eco_tl`（plan 预测）| 拿不到 T[r] |
| `GC_MOE_PULL_TRACE=1` | `pull_tl` 标量打点 | 拿不到 dispatch 入口等 |
| `GC_MOE_SHARED_IN_GC=1` | role4 计算 shared | **shared 段完全不存在** |

> **注意**：`GC_MOE_FINE_EPOCHS` 的语义是「从 `fine_floor` 起记录 N 个 epoch」。
> N=1 时只留一层，对单层分析最省空间；要对比多层就调大，但必须同时调大 `GC_MOE_FINE_RECS`。

### 数据落点

跑完后 `runs/<label>.json` 里：

```
eco_trace.ranks[0..7].fine_rows   ← 环形数组，每条 10 个字
eco_trace.ranks[r].fine_ctr       ← 累计写入条数（判断是否回绕）
eco_trace.ranks[r].pull_rows      ← 每次调用一行标量打点
```

---

## 三、怎么画图

```bash
$GC_ENV/bin/python scripts/plot_absolute_trace.py runs/AT256_eco.json 0 out.png
#                                                  ^run.json      ^rank ^输出
```

参数：`<run.json> [rank=0] [out.png]`

### 图的读法

```
  行 0  ─┬─  compute CTA 0        ███ 橙色 = 一个 M 块的 GEMM
         │                        ███ 绿色 = 该 M 块的 combine push
         ├─  compute CTA 1
         │        ...
         ├─  compute CTA 70
   ──────┼───────────────────────────────────────── 分隔线
         ├─  pull CTA 0          ███ 蓝色 = 这个 CTA 正在拉一个槽
         ├─  pull CTA 1
         │        ...
         └─  pull CTA 55
```

- **蓝色**：`type 1 → 2`，标签 `E<逻辑专家> s<全局槽号>`
- **橙色**：`type 6 → 7`，标签 `E<专家>T<ntile>M<M块序号>v<valid行数>`；
  `M` = `start_row / block_m`；L2 阶段额外加 `+`
- **绿色**：`type 8 → 9`，combine push，**不打文字标签**（1–3 µs 宽放不下字，
  且会遮住不同 valid 之间的长度差异）

### 三个容易踩的画图坑（脚本里已处理）

1. **橙条在 push 起点截断** —— 橙色是纯计算，不含搬运。
2. **绿条右端对齐到 GEMM 结束点** —— 原始 `push_end` 受整 µs 截断，
   画出来会在绿条后残留 1 µs 橙条，在放大图里会被误读成「push 结束后还在算」。
   实测：390/1000 个 push 与 GEMM 结束同 µs，610/1000 短 1 µs，**没有短超过 1 µs 的**。
3. **配对必须在同一 epoch 内** —— 跨 epoch 会混入环形残留。

### 怎么选「哪一层」

`pick_call()` 自动挑**第一个完整的 decode 调用**：
该 epoch 内同时含 `slot_begin/slot_end/mblock_begin/mblock_end`，
且时间跨度 < 2000 µs（用于排除 prefill，prefill 单次要几 ms）。

要指定层，改 `pick_call()` 或在 `main()` 里传入 epoch。
**注意**：`t18` 这套 trace 不记录 `step/layer` 编号，只按 epoch 区分。
如果需要精确对到「第几步第几层」，要另外从 `ir_step` 表取
（`epoch = epoch0 + step*40 + layer`），或用 `pull_rows[r][34]/[35]`（若该版本已支持）。

---

## 四、导出 CSV（便于自己核对）

```bash
$GC_ENV/bin/python scripts/dump_absolute_timeline.py runs/AT256_eco.json 0 0 --csv out.csv
#                                                       ^run   ^rank ^epoch
```

输出每行一条事件，字段：`t_us, kind, id, a0, a1, a2`。

---

## 五、样例产物与实测规模

BS256 / rank 0 / 单层：

| 项 | 数量 |
|---|---|
| pull 片段 | 56（56 个 dispatch CTA，每 CTA 一个槽）|
| GEMM 片段 | 1900（71 个 compute CTA）|
| combine push 片段 | 1000（全部属 L2；L1 无 push）|
| 配对失败 | 0 / 0 / 0 |
| pull 全部拉完 | 149.0 µs |
| 层结束 | 746.6 µs |

push 时长随 valid 单调增长：

| valid | n | 原始事件 (µs) | 图上绿条宽度 (µs) |
|---:|---:|---:|---:|
| 8 | 180 | 1.12 | 1.78 |
| 16 | 220 | 1.31 | 2.06 |
| 32 | 100 | 1.88 | 2.57 |
| 40 | 120 | 1.82 | 2.30 |
| 64 | 120 | 2.46 | 2.94 |

---

## 六、这套 trace 与 torch profile 的关系

| | Absolute trace（本目录） | torch profile trace |
|---|---|---|
| 时间来源 | 内核 `globaltimer` 打点 | CUPTI kernel 起止 |
| 粒度 | **每个 M 块 / 每个槽** | 每个 **kernel** |
| 能否看到 CTA 内部 | **能**（哪块、多久、valid 多少）| 不能 |
| 能否看到 kernel 边界 | 不能（要另外取）| **能** |
| 适用 | 分析尾部塌陷、MT 块粒度、valid 与耗时关系 | 分析 role 间依赖、等 token |

两者互补：**role 间关系用 torch profile，role 内部用量化用 absolute trace。**
