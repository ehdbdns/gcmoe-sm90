# 【SM90】REPLY_SM90_DECODE_VANILLA_REPLAY — 消费者侧改用 vanilla 放置后**仍然**分叉（7 处）

> 本回执由 **SM90 执行者**（8×H20）提交。SM100 任务书不在本执行者范围内。

## 0. 结论

任务书要求："若同样分歧，应优先查共享 decode 或回放。"

**同样分歧，而且更多**：

| 运行 | 放置 | 失败位置 | 失配数 |
|---|---|---:|---:|
| 上一轮 random 臂 | random | req176 position **664**（offset 620）| **4** |
| **本次 vanilla 消费者** | **identity（vanilla 自身）** | req176 position **322**（offset 278）| **7** |

两次都是 **req176**、都只有个位数失配、都落在 stats 窗口、都在 prefill 校验通过之后。
**把放置换回 vanilla 并没有消除分歧，反而更早、更多。**

因此 `REPLY_SM90_MAIN24_STABLE_BS256` §8 里"放置改变导致漂移"的推断
**被这次实验否定**。分歧的根因在**共享的 decode 路径或回放本身**，不在随机放置。

## 1. 执行与结果

| 项 | 值 |
|---|---|
| 执行 SHA | `97d92dc2ce615d453d9c34c12c91c239d9eb3ae8` |
| 工作目录 | `/tmp/gcmoe-vanilla-consume-run1`（无 worktree，直接用主仓干净树）|
| 状态 | `step=3: capture`，`exit_code=**1**` |
| 参考 | `/tmp/gcmoe-main24-bs256-c2LF5VbC/replay/tokens.json`，sha256 **`880da497…`**（先断言校验通过）|
| 快照 | `GC_MOE_STRICT_STATE_DIR=/tmp/gcmoe-main24-bs256-c2LF5VbC/kv`（上一轮 vanilla 发布的 KV）|
| 环境 | `GC_MOE_VANILLA_STATS_REPLAY=1`、`GC_MOE_MATRIX_ARM=vanilla`、`GC_MOE_ORDER_PREFILL=1`、`GC_MOE_STABLE_PREFILL_TOPK=1`、`GC_MOE_MATRIX_PREFIX_CHECK=1`、`GC_MOE_KV_BYTES=21474836480` |
| Capture workspace | **`/tmp/gcmoe-aligned-HFwfvEzw`** |
| 模型启动数 | **1**（任务书要求只启动一次，已完成）|

```
ValueError: stats: 7 native top-1 mismatches; first relative offset 278,
  req176, position 322, native 223, reference 19958
```

八卡（TP0–TP7）逐字相同，共 9 处记录（8 个 worker + 1 个 MainProcess 汇总）。

**prefill 校验未报错**：117 个引擎步、220,096 个 prefill token 通过；
`Iteration(117)` 起进入 1000 个纯解码步，全部为
`256 generation requests / 256 generation tokens / 0 context`；
`Iteration(1116)` 后进入 `finish('stats')`，在第 278 步的 req176 上出现 7 处失配。

## 2. 解码窗口的计时与容量（本次，未作为结果发布）

| 项 | 值 |
|---|---:|
| 1000 解码步逐迭代均值 | **75.16 ms** |
| 中位数 / 最小 / 最大 | 75.08 / 73.64 / 85.67 ms |
| 峰值 KV 利用率 | **9.3%** |
| GPU KV cache size | **1,530,922 tokens**（329.73x @4,643）|
| 每步 256 token | ✅ 1000/1000 |

对照上一轮**发布者** vanilla 的日志逐迭代均值：stats 窗口 **74.62 ms**、measured **74.94 ms**。
本次消费者为 **75.16 ms**（+0.3%），差异在运行间噪声范围内。
（注：轮次间的 `run.json` 设备同步窗口值 90.362 / 91.143 ms 与日志逐迭代值不可直接比较，
两者口径不同，本回执只用同一口径的日志值做对照。）

## 3. 两次失败并列（供定位）

| 项 | random 臂（上轮）| **vanilla 消费者（本次）** |
|---|---|---|
| 放置 | random | **identity（vanilla）** |
| 路由来源 | 回放 `stats-rank*.pt`（`use_reference=True`）| **同左**（`GC_MOE_VANILLA_STATS_REPLAY=1` 使 `publishes_checkpoint()` 返回 False）|
| KV 来源 | 回放上一轮发布的 KV | 同左（同一目录）|
| tokens 来源 | 上一轮 `tokens.json`（`880da497…`）| 同一文件，**先断言 SHA** |
| 失败请求 | **req176** | **req176** |
| 失败位置 | position 664 / offset 620 | **position 322 / offset 278** |
| 原生 / 参考 | 3975 / 8842 | **223 / 19958** |
| 失配数 | 4 / 256,000 | **7 / 256,000** |

两次共同点：**同一请求 req176**、失配数都只有个位数（0.003%）、
都在 prefill 之后的统计窗口里、都由 `verify_native_prefix` 抓到。
不同点：**换回 vanilla 放置后并没有变小**，位置反而提前 342 步。

`req176` 的参考序列长度 2044（全批 256 个请求中排第 80 短，范围 2001–2117），
在工作量里**没有任何特殊地位**（不是最长/最短、不是首尾 chunk），
因此"某个请求特别大/特别小"不是解释。

## 4. 由本实验可以排除与不能排除的

**可以排除**：
- ❌ "随机放置是根因" —— vanilla 放置同样失败，且失配更多。
- ❌ "参考文件被写坏" —— SHA 断言通过，且 `req176[322]=19958` 与报错中的 reference 逐位相符。
- ❌ "prefill 快照读回错误" —— prefill 校验通过、KV readback 由 `stream_checkpoint` 自行校验。

**不能排除**（任务书指向的"共享 decode 或回放"）：
- ✅ **共享 decode 路径**：两个运行都用 `GC_MOE_ORDER_PREFILL=1` + `GC_MOE_STABLE_PREFILL_TOPK=1`
  的同一份 decode 代码；发布者运行时**没有**消费者这层 `ReferenceTokens` 替换
  （发布者 `follow_native=True`，`native` 表由自己填），两者走的是**不同分支**。
- ✅ **回放本身**：`RouteTape` 的 `use_reference=True` 用 `index_select` 把发布的路由
  拷进 ring，再 `index_copy_` 写回；这条路径只在消费者上执行，发布者上 `use_reference=False`。
  非确定性若来自这段 CUDA Graph 内的 `index_copy_`/`where` 序列，就只有消费者会看到。
- ✅ **步进边界**：两次失败的步数不同（620 vs 278），说明不是"某个固定位置必错"，
  更像是**低概率、位置不固定**的数值抖动（4/256k 与 7/256k）。
  这符合 `atomicAdd` 累加顺序或 ring buffer 覆盖时序带来的微小差异被 top-1 放大。

## 5. 建议的下一步（未执行，等任务书）

1. **同一消费者连续跑两次**（相同 SHA、相同参考、相同放置），看失配位置是否可复现：
   - 若两次位置相同 → 确定性 bug，可直接定位到具体 epoch/layer；
   - 若位置不同 → 非确定性，问题在 `RouteTape` 回放或 CUDA Graph 内的累加顺序。
2. 或用 `GC_MOE_MATRIX_PREFIX_CHECK=0` 跑完 1000+1000 步，直接看**失配数是否随步数线性增长**
   （若 1000 步固定 4–7 处，则与"每步 40 层 × top-6 累加"的量级相符）。
3. 若设计端认可"个位数 / 256,000"为可接受的数值抖动，则把 §验收 2 的门禁改成
   阈值判定（例如 ≤ 1e-4），六臂即可在稳定 top-k 下正常产出。

## 6. 交付物

`docs/trace_executor_20261119/sm90_decode_vanilla_replay_results/`：
`runner.log`（540 KB，含完整八卡栈）、`run.log`、`exit-code.txt`（=1）、`status.txt`、
`failed-command.txt`、`runtime.log`、`execution.sha`、`start.sha`、`warm.log`、`gpu.log`、
`reference.sha256`、`summary.json`。

原参考文件未被修改：`880da497abce4830742fb3d060cf52bc7db575b8b71c475de246731c16d536b8`。
KV 与 route 回放（331 GiB）留在现场 `/tmp/gcmoe-main24-bs256-c2LF5VbC`。
