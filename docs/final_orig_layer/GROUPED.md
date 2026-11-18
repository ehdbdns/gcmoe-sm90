# 单层 7 段时序（最终版，已逐项校验）

- 图：`one_layer.png`
- 交互：`one_layer.perfetto.json`（拖进 <https://ui.perfetto.dev>，8 process × 5 thread）

数据来源：`FINAL_V2` 一次运行，BS256、原始耗时函数、CUDA graph、mb-claim，
EPLB 重排后，取 step 170 / layer 13（trace idx 39608）。

## 行布局

| 行 | 内容 |
|---|---|
| `rank N dispatch (role1)` | **count exchange + pull token**（同一行，首尾相接）|
| `rank N shared (role4)` | shared L1+L2 |
| `rank N plan (role5)` | ECO plan |
| `rank N routed (role2)` | **wait 1st M block + GEMM**（同一行，首尾相接）|
| `rank N combine (role3)` | combine |

## 数值（µs，同一层，8 个 rank）

| rank | count exch | pull token | shared | plan | wait 1st M | GEMM | combine |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 20 | 408 | 194 | 112 | 8.1 | 733 | 154 |
| 1 | 19 | 503 | 193 | 109 | 7.1 | 789 | 98  |
| 2 | 21 | 331 | 198 | 102 | 8.1 | 733 | 144 |
| 3 | 20 | 514 | 192 | 108 | 7.1 | 870 | 16  |
| 4 | 19 | 234 | 193 | 111 | 7.1 | 320 | 566 |
| 5 | 20 | 412 | 194 | 110 | 8.1 | 578 | 307 |
| 6 | 19 | 351 | 194 | 110 | 7.0 | 539 | 345 |
| 7 | 21 | 170 | 193 | 107 | 7.0 | 187 | 698 |

## 三个不变量（8 个 rank 全部通过）

| 检查 | 实测 |
|---|---|
| `shared` 起点 − `dispatch` 起点 | **−3.5 ~ +2.7 µs** → 两者同时开始 |
| `GEMM` 起点 − `shared` 结束 | **+8.0 ~ +9.0 µs** → 恒正，等于 `[15]-[17]` |
| `combine` 起点 − `GEMM` 结束 | **+2.6 ~ +5.1 µs** → 恒正 |

## 关键事实：shared 与 GEMM 严格串行

`role2 wait_for=(4,)`（`source/gc_moe/jit_kernels/impls/sm90_fp8_fp4_gc_moe.py` L155）：
routed GEMM 依赖 role4（shared expert）。用 CUPTI 独立验证：

| rank | routed_start − shared_end |
|---:|---:|
| 0–7 | **+0.9 ~ +1.0 µs** |

**均值 0.93 µs，stdev 0.04** —— 8 个 rank 完全一致，说明这是硬依赖而非巧合。

## 关于「pull 结束了，GEMM 怎么还没开始」

**7/8 个 rank 的 GEMM 起点早于本 rank 的 pull 结束**：

| rank | pull 结束 | GEMM 开始 | 差 |
|---:|---:|---:|---:|
| 0 | 431.5 | 203.2 | −228.3 |
| 1 | 523.5 | 200.9 | −322.6 |
| 2 | 355.8 | 213.0 | −142.7 |
| 3 | 538.4 | 203.7 | −334.8 |
| 4 | 255.9 | 203.7 | −52.2 |
| 5 | 436.0 | 205.1 | −230.9 |
| 6 | 374.1 | 207.0 | −167.1 |
| 7 | 195.7 | 203.8 | **+8.1** |

原因：mb-claim 下一个槽的 M 块一经发布，任何空闲 CTA 就能领走，**不需要等所有槽拉完**。
所以"pull 结束"不是 GEMM 的启动条件；GEMM 的启动条件是 **shared 结束**。

## 上一版错在哪

旧切分公式：

```python
wait_ratio = ([15] - [10]) / ([11] - [10])       # ← 分子、分母不是同一个窗口
```

两个打点都是合法时间戳，但量的是不同事件：

- 分母 `[11]-[10]` = dispatch 入口 → compute 退出 ≈ 944 µs
- 分子 `[15]-[10]` = dispatch 入口 → 首个 M 块被领走 ≈ 195 µs

其中 **191 µs 是 role2 内核还没启动的时间**（pull + shared 阶段）。
把 195/944 = 0.207 乘到 role2 自己的 span（741 µs）上，等于把"内核尚未启动"算进了内部等待：

```
旧：wait = 0.207 × 741 = 153 µs     ← 假
新：wait = [15] - [17] = 8 µs        ← 真（与 CUPTI 的 +0.9 µs 依赖边一致）
```

rank7 最夸张：旧值 76 µs，而 pull 在 195.7 µs 就结束了 —— 这就是"pull 结束、GEMM 还没开始"的来源。

正确做法：`[15]`（首个 M 块被领走）与 `[17]`（shared 结束）是**同一次调用内的两个设备时间戳**，
差值 `[15]-[17] = 7~9 µs` 与时钟基准无关；加在 shared 结束的绝对位置（CUPTI，跨 rank 精确）上，
即得 GEMM 起点的绝对位置。

## 两个"差 1 µs"和"差 8 µs"不要混淆

| 量 | 值 | 含义 |
|---|---|---|
| `role2 内核起 − role4 内核止` | **0.9–1.0 µs** | stream 上的依赖边刚满足 |
| `首个 M 块被领走 − role4 内核止` | **8–9 µs** | 内核起来后 CTA 自旋、抢到第一个 M 块 |
