# 逐步 × 每 rank 原始时间轴（`eco_tl`）

与 30-slot 视图（`docs/TRACE_30SLOTS.md`）不同，**这里不做任何归约**：每次调用留下
自己的一行，每个 rank 写自己的 buffer。min/max/sum 会把「哪个 rank 慢」这个信息
直接丢掉，而那正是要看的。

存储：`eco_tl[256][16]` int32 **微秒**（= 16 KB/rank）。用 µs 不用 ns：
`globaltimer` 的 ns 值在设备启动 ~2.1 s 后溢出 int32。

## 16 个 slot

| slot | 名字 | 写者 | 含义 |
|---:|---|---|---|
| 0 | `step_tag` | plan | 本轮的 epoch（排序/校验） |
| 1 | `plan_wake` | dispatch block0/tid0 | ★ release `eco_ready[0]` 的时刻 |
| 2 | `plan_start` | plan | plan CTA 进入 `_eco_plan` |
| 3 | `plan_counts_seen` | plan | `wait_epoch(counts)` 返回 |
| 4 | `plan_static_done` | plan | 静态表 + D_e + k/rem + 单副本定死 |
| 5 | `plan_coldwarm_done` | plan | cold / warm 完 |
| 6 | `plan_auction_done` | plan | 拍卖轮完 |
| 7 | `plan_end` | plan | quota / quota_start / warm state 发布完 |
| 8 | `push_begin` | dispatch | push 循环开始 |
| 9 | `push_end` | dispatch | push 的 grid 会合通过 |
| 10 | `pull_norep_begin` | dispatch | 第一个单副本槽开始拉 |
| 11 | `pull_norep_end` | dispatch | 最后一个单副本槽发布完 |
| 12 | `pull_rep_begin` | dispatch | 第一个副本槽拿到 plan |
| 13 | `pull_rep_end` | dispatch | 最后一个副本槽发布完 |
| 14 | `plan_ready_after_norep` | dispatch | 单副本段已完成后，plan 才就绪的时刻 |
| 15 | `role2_first_task` | role 2 | compute 第一次真的拿到任务 |
| **16** | `role2_first_spin` | role 2 | ★ **消费者第一次空转**（找不到已发布的工作） |
| **17** | `first_replica_pull_done` | dispatch | ★ **第一个副本专家 pull 完成**（首个 replica 的 `expert_ready`） |

10–13 / 15 用 **first / last-writer**（`eco_tl_ctr` 里每类一个计数器 + 一次普通 store），
不加 barrier，所以不扰动被测的调度。

★ **行号不是每个角色自己算的**：`eco_tl_step` 由 dispatch 在唤醒 plan **之前**写入，
所有角色都读它。原因：plan 提前唤醒后，plan 自己读 `header[0]` 可能仍拿到上一轮的
epoch，会写错行（实测：rank 1-7 的 `plan_wait_counts` 变成 31 ms）。

## 任务一：唤醒点搬家

`st_release(eco_ready[0])` 从 push **之后**移到 push **之前**（`header[24]` 之后）。
安全性三条：

1. plan 只读 `eco_counts`，而 `header[24]` 跨 rank 栅栏已保证全局可见；
2. plan 不读 `source_index` / `send_count` / `topk_idx`（它的直方图是 `eco_hist`）；
3. `eco_ready[0]` 仍是 `epoch + 1` 语义，与原来一致。

### 验收（56 行 = 8 rank × 7 step，warmup≤1 已丢弃）

```
plan_wake before push_end : 56/56 rows (OK)     ← 搬家成功
plan_wake - push_end      : -6.4 us (8 rank 全部为负)
```

## 实测（µs，每 rank 平均）

| 指标 | r0 | r1 | r2 | r3 | r4 | r5 | r6 | r7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| plan_exec (wake→end) | 93.4 | 94.0 | 92.3 | 93.1 | 94.3 | 93.7 | 93.3 | 94.0 |
| ├ wait_counts | 1.3 | 1.4 | 1.9 | 1.3 | 1.7 | 1.7 | 1.6 | 1.6 |
| ├ static + D_e | 2.0 | 2.0 | 1.7 | 2.0 | 2.0 | 1.9 | 2.0 | 1.7 |
| ├ cold/warm | 4.0 | 4.0 | 4.3 | 4.0 | 4.0 | 4.1 | 4.1 | 4.3 |
| ├ **auction** | 84.4 | 85.0 | 83.1 | 84.6 | 85.1 | 84.7 | 84.0 | 85.0 |
| └ publish | 1.7 | 1.6 | 1.3 | 1.3 | 1.4 | 1.3 | 1.6 | 1.4 |
| push | 5.6 | 5.0 | 5.4 | 5.3 | 5.3 | 5.3 | 5.3 | 5.1 |
| pull_norep | 139.6 | 114.0 | 146.9 | 113.0 | 116.9 | 116.0 | 123.0 | 114.1 |
| pull_rep | 117.7 | 140.9 | 152.7 | 130.9 | 122.4 | 133.0 | 128.3 | 132.1 |
| plan_end − push_end | 87.0 | 87.6 | 85.7 | 86.9 | 87.6 | 87.0 | 86.7 | 87.7 |
| **plan_wake − push_end** | -6.4 | -6.4 | -6.6 | -6.3 | -6.7 | -6.7 | -6.6 | -6.3 |
| starvation (rep − norep_end) | -139.7 | -114.0 | -147.0 | -113.0 | -117.0 | -116.0 | -123.0 | -114.3 |
| role2_first − push_end | 16.9 | 17.6 | 17.4 | 17.4 | 18.1 | 17.0 | 17.1 | 17.1 |

全局平均：`plan_exec 93.5` / `auction 84.6` / `push 5.2` / `pull_norep 123.6` /
`pull_rep 132.1` / `role2_first 17.3` µs。

## 三个判定式

$$
\begin{aligned}
\text{plan 是否被 push 盖住} &= \texttt{plan\_end} - \texttt{push\_end} = +87.0\ \mu s \quad(\text{>0 ⇒ 没盖住})\\
\text{唤醒是否早于 push 结束} &= \texttt{plan\_wake} - \texttt{push\_end} = -6.4\ \mu s \quad(\text{<0 ⇒ 是})\\
\text{真正的饥饿} &= \texttt{pull\_rep\_begin} - \texttt{pull\_norep\_end} = -115\ \mu s \quad(\text{<0 ⇒ 无饥饿})
\end{aligned}
$$

## 结论

0. **★ 真实饥饿窗口只有 24 µs**（`slot17 - slot16`）：
   消费者在 wake 后 71.5 µs 第一次空转，首个副本在 wake 后 95.6 µs 就 pull 完了。
   最差 rank 47 µs，最好 4.6 µs。**原始假设「24 个副本槽串行等 plan ≈ 1.46 ms」不成立。**

1. **唤醒搬家成功**：plan 在 push 结束前 6.4 µs 就被唤醒，56/56 行成立。
2. **但 plan 仍然没有藏在 push 里**：`push` 只有 5.2 µs，而 plan 要 93.5 µs，
   所以 plan 必然溢出到 push 之后（`plan_end - push_end = +87 µs`）。
   **push 太短，藏不住 plan。**
3. **没有饥饿**：副本段在单副本段结束前 115 µs 就开始（`starvation < 0`），
   且 `plan_ready_after_norep ≈ 4 µs`——单副本段做完时 plan 已基本就绪。
4. **plan 里 90% 是拍卖（84.6 / 93.5）**，这与 30-slot 视图一致。

## ⚠️ 常驻 role 的行号污染（踩过的坑）

`role 2` 是常驻 CTA：它在**上一轮还没结束时**就开始为下一轮空转。所以
first-writer 记录会落到**上一轮**（实测 rank 1-7 报出 31 ms 的假饥饿）。

两个门控对比：

| 门控 | 结果 |
|---|---|
| `header[0] == epoch` | ❌ 无效 —— 常驻 role 在 kernel 入口读的 epoch 一直是旧值 |
| `eco_ready[0] == epoch + 1` | ✅ 有效 —— 本轮 plan 已被唤醒 |

同理，`slot 2 (plan_start)` 也会落在上一轮（plan 也是常驻），所以
`plan_exec` / `plan_wait_counts` 一律以 `plan_wake` 为基准，不用 `plan_start`。

## 复现

```bash
export GC_MOE_ECO_TRACE=1 ECO_TL_CSV=/tmp/eco_tl.csv ECO_REPEAT=8
torchrun --nproc_per_node 8 scripts/verify_eco_ep8.py
python scripts/agg_eco_timeline.py /tmp/eco_tl.csv --warmup 1
```
