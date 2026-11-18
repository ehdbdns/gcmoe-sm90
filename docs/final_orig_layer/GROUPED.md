# 单层 7 段时序（grouped 版，**已修正 wait 切分**）

- 图：`one_layer_grouped.png`
- 交互：`one_layer_grouped.perfetto.json`（拖进 <https://ui.perfetto.dev>，8 process × 5 thread）

行布局：同类阶段放在同一行，8 个 rank 的行顺序完全一致。

| 行 | 内容 |
|---|---|
| `rank N dispatch (role1)` | **count exchange + pull token**（同一行，首尾相接）|
| `rank N shared (role4)` | shared L1+L2 |
| `rank N plan (role5)` | ECO plan |
| `rank N routed (role2)` | **wait 1st M block + GEMM**（同一行，首尾相接）|
| `rank N combine (role3)` | combine |

## 关键时序（µs，同一层，trace idx 39608）

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

## 三个不变量（本次逐项校验，8 rank 全部通过）

| 检查 | 实测 |
|---|---|
| shared 起点 − dispatch 起点 | **−3.5 ~ +2.7 µs** → 两者同时开始 |
| GEMM 起点 − shared 结束 | **+8.0 ~ +9.0 µs** → 恒为正，符合 `role2 wait_for=(4,)` |
| combine 起点 − GEMM 结束 | **+2.6 ~ +5.1 µs** → 恒为正 |

## 为什么 GEMM 一定在 shared 之后才开

`source/gc_moe/jit_kernels/impls/sm90_fp8_fp4_gc_moe.py` L155：

```python
wait_for = (4,) if config.shared_experts and role == 2 and schedule.tile_m >= 32 else ()
```

routed GEMM（role 2）在 stream 上**依赖 role 4（shared expert）**。所以 GEMM 的
最早可能起点就是 shared 结束时刻，8 个 rank 上实测只晚了 7–9 µs。这与 pull 何时结束无关。

## pull 与 GEMM 的先后（用户提出的疑问，已澄清）

| rank | pull token 结束 | GEMM 开始 | GEMM 开始 − pull 结束 |
|---:|---:|---:|---:|
| 0 | 431.5 | 203.2 | −228.3 |
| 1 | 523.5 | 200.9 | −322.6 |
| 2 | 355.8 | 213.0 | −142.7 |
| 3 | 538.4 | 203.7 | −334.8 |
| 4 | 255.9 | 203.7 | −52.2 |
| 5 | 436.0 | 205.1 | −230.9 |
| 6 | 374.1 | 207.0 | −167.1 |
| 7 | 195.7 | 203.8 | **+8.1** |

**7/8 个 rank 的 GEMM 起点早于本 rank 的 pull 结束**，因为 routed 计算不需要等所有槽拉完
（mb-claim 下，某个槽的 M 块一被发布就能被任意空闲 CTA 领走）。不要误读成"pull 结束才开算"。

## 上一版错在哪（本次修正的就是这个）

旧切分公式：

```python
wait_ratio = ([15] - [10]) / ([11] - [10])       # ← 分子、分母不是同一个窗口
wsplit     = r2_start + (r2_end - r2_start) * wait_ratio
```

两个打点本身都是合法时间戳（`[10]` = CTA0 进入 dispatch 内核，`[11]` = compute 内核退出），
但**分子和分母量的是两段不同的东西**：

- 分母 `[11]-[10]` = **dispatch 入口 → compute 退出**，约 944 µs（rank0）
- 分子 `[15]-[10]` = dispatch 入口 → 首个 M 块被领走，约 195 µs

这 195 µs 里有 **191 µs 是 role2 内核根本还没启动的时间**（dispatch 的 pull 阶段 + shared 阶段）。
把 195/944 = 0.207 这个比例乘到 **role2 自己的 span**（741 µs）上，等于把"内核还没启动"的
那段时间也算进了 role2 内部等待：

```
旧：wait = 0.207 × 741 = 153 µs     ← 假
新：wait = [15] - [17] = 8 µs        ← 真
```

rank7 最夸张：旧值 `197/486 × 187 = 76 µs`，而 pull 在 195.7 µs 就结束了，
图上于是出现"pull 已经结束、GEMM 76 µs 后才开始"——这就是用户看到的那个错误。
真实值只有 7 µs，且原因明确是 `wait_for=(4,)`。

正确做法：`[15]`（首个 M 块被领走）与 `[17]`（shared 结束）是**同一次调用内的两个设备时间戳**，
差值 `[15]-[17] = 7~9 µs` 与时钟基准无关；直接把它加在 shared 结束时刻（trace 的绝对位置）上，
就得到 GEMM 起点的绝对位置。

## 两个"差 1 µs"和"差 8 µs"不要混淆

| 量 | 值 | 含义 |
|---|---|---|
| `role2 内核起 − role4 内核止` | **0.9–1.0 µs** | stream 上的依赖边（`wait_for=(4,)`）刚满足 |
| `首个 M 块被领走 − role4 内核止` | **8–9 µs** | 内核起来后，CTA 自旋、抢到第一个 M 块还需约 7~8 µs |

图里的 `wait 1st M blk` 段是**后者**（对 role2 内核起点的偏移是 7~8 µs），
它与 README 上表里的 `r2起−r4止 ≈ 1 µs` 不矛盾，量的是不同事件。

## 仍然成立的两条限制

- `[5]`（pull end）比 dispatch kernel 退出还晚（rank0 471 vs 428 µs），因为它是"最后一个完成的
  CTA"写的，那个 CTA 可能已属于下一次调用。所以 pull 段的**终点用 kernel span 的末尾**，
  不用 `[5]` 的绝对值。上表的 "pull token" 就是 `dispatch kernel span − count exchange`。
- `pull_tl` 是每个 rank 独立的 uint32 µs 计数器且会回绕，8 个 rank 基准最多差 16.8 s。
  **绝对时间只允许用 torch profile 的 kernel 起止**（同一 profiler 会话、同一设备时钟）；
  设备打点只能用来算**同一次调用内的差值**。
