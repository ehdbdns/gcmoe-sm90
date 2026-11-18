# 单层 8-rank torch profile 切片（epoch 14673）

从 BS256 + `GC_MOE_ECO_COSTFN=1` + EPLB 重排 + `ENFORCE_EAGER=1` 的
torch profile 中，切出**同一层**（epoch tag `14673`，对应 trace 内的
call #128）在 8 个 rank 上的全部事件。

- 源目录：`prof/prof_eco_bs256_v3/traces/`（每 rank 1.18 GB 全量 chrome trace）
- 切片窗口：`4077691964997 .. 4077691966987`（绝对 µs，长约 **1990 µs**，含 20 µs 上下文）
- 每文件：~62–68 KB，含该层该 rank 的全部 `cpu_op` / `kernel` / `cuda_runtime` / `ac2g`

## 怎么打开

把任意 `rankN.json` 拖进
[`chrome://tracing`](chrome://tracing) 或 <https://ui.perfetto.dev> 即可。

- 先按 `name` 过滤 `gc_moe`，可以看到该层的 4 个 role：
  `grid=24` dispatch / `grid=71` routed compute / `grid=1` ECO plan / `grid=288` combine
- 过滤 `Marlin` 或 `nvjet` 可以看到 shared expert（vLLM aux stream，**不在 gc_moe 内**）
- `ac2g` 事件可看出 kernel 落在哪条 stream（gc_moe 的 role 用独立 green-context stream）

## 这一层的时间结构（单位 µs，相对各 rank 自己的 dispatch 起点）

| 阶段 | 起 | 止 | 说明 |
|---|---|---|---|
| count exchange | `[10]` | `[21]` | 各 rank 都 ~16 µs，但**结束时刻不齐**（散 438 µs） |
| wait for peers | `[21]` | `[4]` | 各 rank 等待量不同（1–439 µs），**吸收掉 launch 错峰** |
| token pull | `[4]` | `[5]` | 跨 rank 同步点 `[4]` 对齐在 **1 µs** |
| shared L1+L2 | — | — | 跑在 aux stream，比 dispatch 早 ~1.2 ms 结束 |
| wait token | compute kernel launch | `[15]` | **parking 时间**，中位 285 µs，81% 的调用 >100 µs |
| routed GEMM | `[15]` | compute exit | 真正在算 |
| combine | `[12]` | `[13]` | role 3 |

见 `layer_timeline_8rank.png`。

## 注意

`[10]`（dispatch kernel 第一个 CTA 落地）与 `[21]`（本 rank 24 个 CTA 都到齐）
之间隔着 `_dispatch_grid` 的 CTA 内 rendezvous；而 `[3]`/`[21]` 之后还有一次
**跨 rank** 的 `wait_epoch(peer_table[tid]+1)`，其后的第一个时刻才是 `[4]`。
所以「count exchange 的结尾」在 rank 间是不对齐的，**同步点在 `[4]`**。
