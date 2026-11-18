# 单次调用 perfetto（直接从 torch profile trace 抽取，带真实行号）

- 交互文件：`one_call_from_torch.perfetto.json`（拖进 <https://ui.perfetto.dev>）
- 原始行号数据：`one_call_raw_rows.json`

## 为什么要按「行号」而不是按「数组下标」

每个 forward call 在 torch trace 里恰好产生 **5 个 gc_moe kernel**，顺序固定：

```
dispatch(24x256) → shared(71x256) → routed(71x384) → plan(1x512) → combine(288x384)
```

即一次调用占据 **5 个连续 trace 行**，所以 `call_index = row // 5`。
之前把每个 role 各自排序后取「第 39608 项」，五个数组的第 39608 项并不是同一次调用
（实测相差 53 秒），因此旧图的数字全部作废。

本次直接抽取 **trace 行 39605–39609**，8 个 rank 的角色序列完全一致：

| 行号 | role | grid × block |
|---:|---|---|
| 39605 | dispatch (role1) | 24 × 256 |
| 39606 | shared (role4) | 71 × 256 |
| 39607 | routed (role2) | 71 × 384 |
| 39608 | plan (role5) | 1 × 512 |
| 39609 | combine (role3) | 288 × 384 |

## 分段定义

| 段 | 区间 | 来源 |
|---|---|---|
| `cnt` count exchange | dispatch kernel 起 → `[3]` | kernel span + 同调用设备差值 |
| `pull` pull token | `[3]` → dispatch kernel 止 | 同上 |
| `shared` | shared kernel 起 → 止 | kernel span |
| `plan` | plan kernel 起 → 止 | kernel span |
| **`waittok` 等 token** | **routed kernel 起 → 本 rank 最后一个本地槽 pull 完成** | kernel 起 + 同调用 `[5]` |
| `gemm` | 上述时刻 → routed kernel 止 | 同一 kernel 的剩余部分 |
| `comb` | combine kernel 起 → 止 | kernel span |

**`waittok` 是此前被漏掉的一段**：routed kernel 已经启动，但输入 token 还没拉完，
CTA 只能等。它与 `gemm` 之和恒等于 routed kernel 的总时长。

## 数值（µs）

| rank | cnt | pull | shared | plan | **wait_token** | **gemm** |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 20 | 236 | 192 | 113 | **391** | 237 |
| 1 | 19 | 394 | 194 | 112 | **100** | 541 |
| 2 | 20 | 298 | 192 | 111 | **162** | 489 |
| 3 | 20 | 254 | 192 | 114 | **125** | 498 |
| 4 | 22 | 238 | 192 | 109 | **402** | 241 |
| 5 | 20 | 335 | 193 | 111 | **62** | 565 |
| 6 | 19 | 257 | 193 | 113 | **230** | 420 |
| 7 | 19 | 267 | 192 | 113 | **139** | 484 |

## 校验（8/8 rank 通过）

| 检查 | 实测 |
|---|---|
| 角色序列 | 全部为 dispatch→shared→routed→plan→combine |
| shared 起点 − dispatch 起点 | −1.9 ~ +2.7 µs（同时启动）|
| routed 起点 − shared 结束 | **+0.9 ~ +1.2 µs**（`role2 wait_for=(4,)`）|
| combine 起点 − routed 结束 | +2.6 ~ +4.5 µs |

## 读图要点

- **shared 与 dispatch 并排起步**，说明共享专家与通信重叠；
- **routed 严格在 shared 结束后 ~1 µs 才启动**，这是硬依赖，不是调度抖动；
- **routed 启动后有一段 `waittok`（62–402 µs）**，长短取决于该 rank 的副本槽何时拉完；
- `waittok` 长的 rank（0、4）恰好 `gemm` 短，两者互补，routed 总时长并没有变长。
