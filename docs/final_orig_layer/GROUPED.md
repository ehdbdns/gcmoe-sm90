# 单层 7 段时序（grouped，最新）

- 图：`one_layer_grouped.png`
- 交互：`one_layer_grouped.perfetto.json`（拖进 https://ui.perfetto.dev，8 process × 5 thread）

行布局按你的要求分组，同类阶段放同一行：

| 行 | 内容 |
|---|---|
| `rank N dispatch (role1)` | **count exchange + pull token**（同一行）|
| `rank N shared (role4)` | shared L1+L2 |
| `rank N plan (role5)` | ECO plan |
| `rank N routed (role2)` | **wait 1st M block + GEMM**（同一行）|
| `rank N combine (role3)` | combine |

## 数值（µs，同一层，trace idx 39608）

| rank | count exch | pull token | shared | plan | wait 1st M | GEMM | combine |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 18 | 410 | 194 | 112 | 153 | 588 | 154 |
| 1 | 21 | 501 | 193 | 109 | 161 | 634 | 98 |
| 2 | 19 | 333 | 198 | 102 | 157 | 584 | 144 |
| 3 | 21 | 513 | 192 | 108 | 160 | 716 | 16 |
| 4 | 17 | 236 | 193 | 111 | 109 | 218 | 566 |
| 5 | 25 | 407 | 194 | 110 | 162 | 424 | 307 |
| 6 | 17 | 353 | 194 | 110 | 153 | 393 | 345 |
| 7 | 19 | 173 | 193 | 107 | 79 | 115 | 698 |

## 时序正确性（本次已逐项校验）

| 检查 | 实测 |
|---|---|
| dispatch 起点 | 1.9–4.3 µs |
| shared 起点 | 0.0–6.5 µs → **与 dispatch 同时开始** |
| GEMM 结束 | 390.5–1073.2 µs |
| combine 开始 | 394.1–1077.3 µs → **恒晚于 GEMM 结束 2–5 µs** |

## 上一版错在哪

之前把 `pull_tl` 的设备打点当成绝对时刻。`pull_tl` 是**每 rank 独立的 uint32 µs 计数器且会回绕**，
8 个 rank 基准不同（实测相差最多 16.8 s），混进 trace 时间轴后：
role4 被画到 dispatch 之后很久，routed GEMM 被画到 combine 之后。

现在：
- **绝对定位只用 trace 的 kernel 起止**（同一 profiler 会话、同一设备时钟，跨 rank 精确）
- `pull_tl` 只用来算**同一调用内**的比例：
  `[10]→[3]` 占 dispatch 多少（count exchange）、`[10]→[15]` 占 compute 多少（wait）

另外一个已确认的限制：`[5]`（pull end）**晚于 dispatch kernel 退出**（rank0 是 471 vs 251 µs），
因为它由"最后一个完成的 CTA"写，那个 CTA 可能已属于下一次调用。所以 pull 段的终点用
kernel span 比例得出，不直接采用 `[5]` 的绝对值。
