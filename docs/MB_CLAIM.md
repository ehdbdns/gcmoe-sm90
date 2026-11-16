# 尾部并行度：把领任务粒度从 (expert, ntile) 改为 (expert, ntile, mblock)

## 问题（由 absolute trace 直接定位）

`docs/abstrace/` 的图里，尾部塌陷的根因不是"任务总量不均衡"，
而是**同一个 ntile 的多个 M 块被同一个 CTA 串行包办**：

| 基线 trace（BS256/ECO/rank0） | 值 |
|---|---|
| 一个 ntile 含多个 M 块的 tile 数 | 40 / 940 |
| 其中 **全部落在同一个 CTA** 上的 | **40 / 40** |
| 最长串行 | **E46T10：3 个 M 块连续 76 µs 同一 CTA** |

E46 有 184 token → 3 个 M 块；这 3 块的 `valid` 都很大（64/64/56），
该 CTA 从 655 µs 一直算到 729 µs，同期其余 51 个 CTA 空转。

## 根因

任务领取有两级：

1. `_claim_command` 用单调票号 `head+7` 取一个 **active 条目**；
2. 该条目原本是 `(expert, ntile)`，`_produce_tasks` 领到后
   `for start in range(0, count, max_m)` **遍历该 tile 的全部 M 块**。

`_publish_task` / `_load_persistent_tile` 是配对的、逐块串行，
所以同一 tile 的块只能由这一个 CTA 依次 load，consumer 半边也就只能依次算。

## 改法

**让票号本身就是 `(expert, ntile, mblock)`。**

| 位置 | 改动 |
|---|---|
| `layout/sm90_gc_moe.py` | `active_expert/active_ready` 由 `local` 扩到 `local*64`，新增 `active_mb`；`fine_lay` 同步扩容 |
| dispatch（两处发布点） | `owned>0` 时按 `ceil(owned/max_m)` 发布多条 active 票号，每条带自己的 `mblk` |
| `_claim_command` | 直接读 `active_mb[active_slot]`，不再用 per-tile 游标 |
| `_produce_tasks` | 只 load `command[3]` 指定的那**一个** M 块，然后立刻再领 |
| command 张量 | 3 字 → 4 字（新增 mblock） |

为什么不用 per-`(expert,ntile)` 游标：`head+7` 是**单调**票号，
一个票号一旦发出就不能再回到同一个 tile，所以"同一个 tile 领多次"
必须靠 dispatch 侧多发票号，而不是 consumer 侧重试。

## 验证

### CPU（跑 GPU 之前）

* 模拟票号分配：`(expert,ntile,mblock)` 集合与旧 producer **完全一致**，
  1000 个块、1000 个票号、无丢失无重复。
* makespan 模拟（用实测 trace 的每个块的真实时长）：730 µs → 703 µs。

### GPU（BS256 / ECO / 8×H20，1000+1000）

| 指标 | 基线 | MB-CLAIM | 变化 |
|---|---|---|---|
| ms/step | 62.39 | **58.59** | **−6.1%** |
| 生成 token 总数 | 513280 | 513280 | 一致 |
| 每步 token | 256 | 256 | 一致 |
| 层结束（trace） | 746.6 µs | **687.5 µs** | −59 µs |
| M 块总数/step | 309.7 | 309.5 | 一致（→ 无丢块/重复）|
| **多块 tile 全压一个 CTA** | **40/40** | **0/60** | 修复 |
| **最长同 tile 串行** | **3 块** | **0** | 修复 |
| 多块 tile 墙钟跨度 | 57 µs（max 76） | 26 µs（max 34） | 摊开 |

`gen_tokens_total` 与每步 token 完全一致，是"没有算错/漏算"的关键证据；
`mb_total_mean` 只差 0.2/step（0.06%），属正常放置抖动。

## 复现

```bash
cd .../t18_abstrace
cp /tmp/at256.sh /tmp/run_mbclaim.sh   # 必须带 PYTHONPATH=$HERE/source 等
sed -i 's|--label AT256_eco |--label AT256_eco_mbclaim |' /tmp/run_mbclaim.sh
bash /tmp/run_mbclaim.sh
```

注意：脚本必须 `source config/experiment.env` 且设
`PYTHONPATH=$HERE/source:$HERE/patches:$HERE/integration`，
否则跑的是库里未修改的 kernel（本机踩过：漏了 PYTHONPATH 会静默卡在
`wait_count` 上）。
