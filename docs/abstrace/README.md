# Absolute trace：单层全过程时间线（pull CTA + compute CTA 同框）

一个实测 decode 层，**rank 0、BS256、ECO 臂、1000+1000 步**，横轴是层内绝对时间（µs），
纵轴是 CTA 行。这是把「谁在什么时候干了什么」画到一张图上的版本。

## 文件

| 文件 | 内容 |
|---|---|
| `AT256_eco_rank0_full.png` | 整层（0–747 µs），全部 1900 个 GEMM 片段 + 1000 个 push 片段的真实位置 |
| `AT256_eco_rank0_zoom23-48us.png` | 起始窗口 23–48 µs：pull 与第一批 L1 GEMM 同框，**每片段带编号** |
| `AT256_eco_rank0_zoom606-636us_L2push.png` | L2 窗口 606–636 µs：橙色 compute 与绿色 push **首尾相接**，valid 覆盖 8/16/32/40 |
| `AT256_eco_rank0_slices.csv` | 全量切片表，任何片段都能按 (kind, cta, begin_us) 查到它的编号与 valid |

## 图的读法

```
  行 0  ─┬─  compute CTA 0        ███ 橙色 = 一个 M 块的 GEMM
         │                        ███ 绿色 = 该 M 块的 combine push（独立阶段，接在橙后面）
         ├─  compute CTA 1
         │        ...
         ├─  compute CTA 70
   ──────┼───────────────────────────────────────────── 分隔线
         ├─  pull CTA 0          ███ 蓝色 = 这个 CTA 正在拉一个槽
         ├─  pull CTA 1
         │        ...
         └─  pull CTA 55
```

* **蓝色**：`type 1 → 2`（slot_begin → slot_end），标签 `E<逻辑专家> s<全局槽号>`。
  一个 CTA 会串行拉多个槽，所以同一行上有多个蓝色片段。
* **橙色**：`type 6 → 7`（task_begin → task_end），标签
  `E<专家>T<ntile>M<M块序号>v<valid行数>`，第二遍 GEMM（L2 输入）额外加 `+`。
  `M` 就是 `start_row / block_m`。
  **橙条在 push 起点处截断**，所以橙色是纯计算，不含搬运。
* **绿色**：`type 8 → 9`（combine push 起止），标签 `p<ntile>v<valid行数>`。
  这是 GEMM 算完后把结果写回各 owner rank 的那段。

### 这一轮的两个改动

1. **每个 M 块标出 valid**（`v8` … `v64`）。BS256 下单槽 token 数远小于
   `block_m=64`，所以 94% 的块是 `M0`，但 valid 从 8 到 64 变化很大，
   计算量与 push 量都随之变化 —— 这正是要看的变量。
2. **push 从「叠一个绿标签」变成「独立时间段」。** 之前绿色是画在橙条
   下方的 0.55 高细条，视觉上像是计算的一部分；现在绿色是**全高独立条**，
   橙色在 push 起点 `min(gemm_end, push_begin)` 处截断，两个阶段首尾相接。
   交叉校验：1000 个 push 全部落在各自 GEMM 区间内，push 起点比 GEMM 结束
   早 1–4 µs（内核在最后几个 n-iter 就开始搬），0 个配对失败。

push 时长确实随 valid 单调增长：

| valid | n | push 均值 (µs) |
|---|---|---|
| 8 | 180 | 1.12 |
| 16 | 220 | 1.31 |
| 24 | 140 | 1.71 |
| 32 | 100 | 1.88 |
| 40 | 120 | 1.82 |
| 48 | 60 | 2.32 |
| 56 | 60 | 2.38 |
| 64 | 120 | 2.46 |

L2 放大图（606–636 µs）专门选了一个 valid 分布最宽的窗口，
在同一屏里能直接比出 `p*v8` 比 `p*v40` 短。

## 数据规模（BS256，rank 0，单层）

| 项 | 数量 |
|---|---|
| pull 片段 | 56（56 个 dispatch CTA，每 CTA 一个槽）|
| GEMM 片段 | 1900（71 个 compute CTA）|
| combine push 片段 | 1000（全部属 L2，L1 无 push）|
| 配对失败 | 0 / 0 / 0 |
| pull 全部拉完 | 149.0 µs |
| 层结束 | 746.6 µs |

## 怎么产生的

1. 内核在 `fine_tl` 里逐事件写 10 字记录：
   `[t_us, type, id, a0, a1, a2, a3, a4, epoch, -]`
   * pull：`1/2`，`id=全局槽`，`a0=逻辑专家`
   * GEMM：`6/7`，`id=CTA`，`a0=expert a1=start_row a2=valid a3=ntile a4=phase`
   * push：`8/9`，`id=CTA`，同 GEMM 字段
2. `fine_tl` 是**环形**缓冲（`slot = counter % GC_MOE_FINE_RECS`），
   因为一层就有 ~59k 条，前缀缓冲会只留下 prefill 的尾巴。
3. dump 出来后按**时间戳排序**重建次序：`irstep_begin` 会清零 `fine_ctr`，
   所以不能靠计数器判断环形回绕点（实测 count=7901 但有 467k 条有效记录）。
4. 配对只在**同一个 epoch 内**做，且丢弃 `end < begin` 或跨度 >100 ms 的异常对
   （环形回绕残留）。

## 复现

```bash
cd /home/admin/workspace/aop_lab/app_source/gcmoe_exp/t18_abstrace
source scripts/env.sh
export GC_MOE_EPLB=1 GC_MOE_ECO=1 GC_MOE_ECO_WB=26 GC_MOE_ECO_WA=0
export GC_MOE_EPLB_EXTRA_PER_RANK=8 GC_MOE_ECOPLB_PER_LAYER=1
export GC_MOE_ECO_TRACE=1 GC_MOE_PULL_TRACE=1 GC_MOE_FINE_TRACE=1
export GC_MOE_FINE_EPOCHS=1 GC_MOE_FINE_RECS=524288
$GC_ENV/bin/python scripts/real_decode_100_100.py --label AT256_eco \
  --output runs/AT256_eco.json --prefill-lo 256 --prefill-hi 2048 \
  --seqs-per-rank 32 --warm-steps 1000 --measure-steps 1000
$GC_ENV/bin/python scripts/plot_absolute_trace.py runs/AT256_eco.json 0 out.png
```
