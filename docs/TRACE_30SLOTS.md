# ECO 30-slot 时间轴埋点：写法与实测

`GC_MOE_ECO_TRACE=1` 编译期打开（关闭时零开销）。时钟统一用 `%globaltimer`（ns），
**跨 rank / 跨 CTA 可比**；不使用 `clock64()`（各 SM 时钟不同步）。

## 一、谁写 / 什么事件 / 为什么这个写法

### 组 1：role 1（dispatch，24 CTA）

| slot | 名字 | 写者 | 事件 | 写法 | 为什么 |
|---:|---|---|---|---|---|
| 0 | role1_start | 每个 role-1 CTA | 通过会合 #1（清零已被排序） | **min** | 要「最早进入」的那个 |
| 1 | role1_barrier_done | block 0 / tid 0 | 本地会合 #1 完成 | store | 单写者 |
| 2 | role1_counts_ready | block 0 / tid 0 | counts 广播 + grid 会合完成 | store | 单写者 |
| 6 | plan_wait_first | 副本槽 CTA | 开始等 plan | **min** | 最早开始等的 |
| 7 | plan_seen_last | 副本槽 CTA | 最后拿到 plan | **max** | ★ 尾部 |
| 8 | role1_end | 每个 role-1 CTA | CTA 退出 | **max** | 最晚退出的 |

★ `7 - 6` = 从「有人开始等」到「所有人都拿到」的跨度（尾部效应）。

### 组 2：role 5（plan，1 CTA，全 store）

| slot | 名字 | 事件 |
|---:|---|---|
| 3 | plan_start | 进入 plan（`wait_epoch` 之前） |
| 4 | plan_counts_seen | `wait_epoch(counts_ready)` 返回 |
| 17 | plan_start_done | 静态表 + D_e + k/rem + 单副本定死 完成 |
| 18 | plan_rounds_done | cold/warm 完成 |
| 19 | plan_index_done | 拍卖轮完成 |
| 5 | plan_published | quota / quota_start / warm state 发布完 |

推导：`4-3`=等 counts，`17-4`=静态表+D_e，`18-17`=cold/warm，`19-18`=**拍卖**，`5-19`=发布，`5-3`=plan 总计。

### 组 3：role 2（routed compute，71 CTA）

| slot | 名字 | 写者 | 事件 | 写法 |
|---:|---|---|---|---|
| 9 | role2_sched_start | 每个 role-2 CTA | 进入调度循环 | min |
| 10 | role2_counts_seen | role-2 | 首次轮询生产者计数器 | min |
| 11 | role2_block_first | role-2 | 首次阻塞在生产者计数器 | min |
| 12 | role2_block_last | role-2 | 最后一个拿到工作的 | max |
| 13 | role2_sched_end | role-2 | 退出调度循环 | max |
| 16 | role2_end | role-2 | kernel 退出 | max |

⚠️ **与 SM100 的差异**：SM90 的 role 2 **不等 plan**，它消费 `active_ready`/`expert_ready`，
而这两个由 dispatch 在 pull 之后发布，**pull 才是等 plan 的那一环**。
所以 slot 10/11/12 测的是「等生产者计数器」，plan 相关的等待看 slot 27。

### 组 4：role 3（combine）

| slot | 名字 | 事件 | 写法 |
|---:|---|---|---|
| 14 | role3_start | 进入 combine | min |
| 15 | role3_end | 完成（epoch bump） | max |

### 辅助 slot

| slot | 用途 |
|---:|---|
| 24 | anchor：本轮的 role-1 起点（每轮 store） |
| 27 | 累计「副本槽等 plan」的 ns（累加，需除以调用次数） |

### 清零协议（关键）

SM100 的 min/max 槽**跨调用累积**且不清零（只有单次调用场景可用）。本任务需要每轮值，所以：

1. block 0 在 dispatch 入口**并行**清零 32 个槽（`tid < 32` 一字一线程）
2. min 槽预置 `UINT64_MAX`（**若预置 0，min 永远取到 0**，这是我踩过的坑）
3. 之后的会合 #1 排序「清零」在「任何 min/max 写」之前

## 二、实测（8×H20，EP8，32 token，wb/wa=0/0，稳态 call1-7 中位数）

```
role1 (dispatch)  start -> end      171.1 us
  ├ rndv#1                            1.2
  ├ counts 广播 + grid 会合           15.1
  └ plan-wait 尾部 (slot 7-6)        121.2   ★★ 占 71%

role5 (plan)      total             110.9 us
  ├ wait-counts (4-3)                 18.4
  ├ static + D_e (17-4)                1.8
  ├ cold/warm (18-17)                  4.1
  ├ AUCTION (19-18)                   85.6   ★ 占 77%
  └ publish (5-19)                     1.3

role2 (compute)   sched             245.1 us
  ├ first-poll (10-9)                  0.4
  ├ first-block (11-10)                0.3
  └ teardown (16-13)                   3.4

role3 (combine)   total              40.6 us
replica-wait (slot27) / call        137.1 us
```

## 三、结论

1. **plan 本身只有 111 µs，其中拍卖 86 µs（77%）**。即使把拍卖完全删掉，也只省 86 µs。
2. **副本槽实测等 137 µs / 次调用**，与 slot 7-6 的尾部 121 µs 吻合——
   这 121 µs 的量级是 plan 的 1.1 倍，但**远小于** ms/step 里 1.74 ms 的差距
   （1.74 ms/step ÷ 40 层 ≈ 43 µs/层，所以每层 121 µs 的尾部是无法直接叠加的，
   它们在不同层之间互相重叠）。
3. **最主要的可优化项是拍卖（86 µs）**，其次是把 plan 的启动提前（目前被
   `header[24]` 之后的 push 串住）。

## 四、本轮修掉的 3 个真 bug（都是靠静态审计/确定性对照定位，不是靠重跑试）

| # | 问题 | 症状 | 定位方式 |
|---|---|---|---|
| 1 | `eco_hist` 清零是全 grid 分片的，但只有 block 级 `sync_threads()` | `eco_counts` 求和 1533 而非 1536 → `quota` 少 3-38 | 加 `eco_counts_sum` 对拍，发现它在 1533/1536 抖动 |
| 2 | `header[24]` 之前缺 grid 会合（广播由 24 个 CTA 完成） | `quota_sum` 1517/1522 | 逐轮 `quota_sum` 打印 |
| 3 | min 槽预置 0 | 所有 min 槽读回 0 | 读源码发现 `min(0, ts) == 0` |

★ 三者共同点：**`pulled` 始终正确，只有 `quota` 错**。这个不对称是定位的突破口
（pull 用 clamp 掩盖了部分错误）。
