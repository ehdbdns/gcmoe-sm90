# 【SM90】REPLY_SM90_STABLE_DECODE_AA — 稳定 decode top-k 后 **A/A 通过（原生差异 0）**

> 本回执由 **SM90 执行者**（8×H20）提交。SM100 任务书不在本执行者范围内。

## 0. 结论

启用 `GC_MOE_STABLE_DECODE_TOPK=1`（纯 decode 的 sparse indexer top-k 改为
按原分数降序、同分按逻辑位置升序）后，在**全新目录**发布 vanilla 参考、
再用**同一 vanilla 以消费者身份回放**，两段都跑通：

| 阶段 | 状态 | stats 窗口 ms/步 | 引擎步数 | 每步 256 token |
|---|---|---:|---:|---|
| **publisher** | ✅ `exit_code=0` | **93.344** | 117 prefill + 1000 decode | ✅ 1000/1000 |
| **consumer** | ✅ `exit_code=0` | **94.193** | 117 prefill + 1000 decode | ✅ 1000/1000 |

**逐位一致性全部通过**：

| 项 | 结果 |
|---|---|
| `prompt_hash` | 两段相同（`254af9d8…`）|
| `prefill_steps` | 117 = 117 |
| 八卡路由 `ids`/`weights` 摘要 | **8/8 rank 完全相同**（`ids` 前缀 `ccfe108dc331…`）|
| `total_rows_by_layer`（40 层）| **完全相同** |
| `demand` 摘要 | **完全相同**（`29369f0551e6…`）|
| `ordinal_calls_verified` | 40,000 = 40,000（逐调用行号全部核对）|
| prefill KV 快照 source/readback SHA | 完全相同（`bed4d76a…`）|
| **原生 top-1 失配** | **0** |

这是这一系列诊断中**第一次让"发布者 + 消费者"这条完整链路走到 1000 步且原生差异为零**。
此前失败的两条路径（`REPLY_SM90_MAIN24_STABLE_BS256` 的 random 臂 4 处、
`REPLY_SM90_DECODE_VANILLA_REPLAY` 的 vanilla 消费者 7 处）都由 `stable_prefill_topk` 的
**prefill 分支**覆盖，而 decode 的 sparse indexer top-k 是**另一个入口**
（`SparseIndexerTopk` / `get_indexer_topk`），此前**没有**被稳定化。

这解释了 `REPLY_SM90_DECODE_VANILLA_REPLAY` §4 的推断：分歧不在放置，也不在参考文件，
而在**共享 decode 路径里未被稳定化的 decode top-k**。req176 是该路径上平局命中的请求，
稳定化后不再分叉。

## 1. 环境与执行

| 项 | 值 |
|---|---|
| 执行 SHA | `0dd9579b0abbe8aefab8bfaa57d68defdd2cba93` |
| 工作目录 | `/tmp/gcmoe-sdaa-run1`（主仓干净树直接运行，`git status` 为空）|
| 现场 state | `/tmp/gcmoe-decode-aa-NuU4vuVb`（**166 GiB**，全新目录）|
| 环境 | `GC_MOE_STABLE_PREFILL_TOPK=1`、**`GC_MOE_STABLE_DECODE_TOPK=1`**、`GC_MOE_DECODE_STATS_AA=1`、`GC_MOE_MATRIX_ARM=vanilla`、`GC_MOE_MATRIX_BS=256`、`GC_MOE_MATRIX_PREFIX_CHECK=1`、`GC_MOE_KV_BYTES=21474836480` |
| 两段差异 | 只有 `GC_MOE_VANILLA_STATS_REPLAY`：publisher 未设，consumer 设为 `1` |
| CPU 门禁 | `stable prefill top-k CPU PASS`（含新增的 decode 选择器用例）|
| 代码修改 | **无**（任务书自带的 `0dd9579` 即包含 `stable_prefill_topk.install_decode`）|
| GPU | 8×H20，driver 580.105.08 |
| 软件 | cutlass 4.8.1 / torch 2.13.0+cu130 / CUDA 13.0 / vllm 0.31.0 |
| 参考 SHA256 | `2e997e79a77a63e9e8db36a79e03b0056d0eadb0fd2d38e08826cd5be8bd120e` |

## 2. 资源与容量（两段相同）

| 项 | 值 |
|---|---:|
| 每卡总显存 | 102,085,492,736 B = **95.07 GiB** |
| 最低剩余显存 | 24,582,094,848 B = **22.89 GiB** |
| 最高 PyTorch reserved | 66,339,209,216 B = **61.78 GiB** |
| vLLM GPU KV cache size | **1,530,922 tokens**（329.73x @ 4,643）|
| 峰值 KV 利用率 | **9.3%** |

> `free` 是任务书要求的设备剩余显存读数，**包含 CUDA allocator 之外的占用**
> （NCCL/NVSHMEM 对称堆、CUDA context、驱动预留），故不等于 allocator 可用量。

## 3. 计时（仅作诊断，不进主表）

日志逐迭代口径（GPU KV cache usage 行）：

| 阶段 | 1000 步均值 | 中位数 | 最小 | 最大 |
|---|---:|---:|---:|---:|
| publisher | 77.53 ms | 77.45 | 76.02 | 110.13 |
| consumer | 78.26 ms | 78.22 | 76.75 | 83.79 |

consumer 比 publisher 慢 **+0.94%**，属运行间噪声量级（同机历史 run-to-run 差异 0.3–1.5%）。
设备同步窗口值分别为 93.344 / 94.193 ms（口径与日志逐迭代不同，不可直接比较）。

**注意**：稳定的 decode top-k 用 `argsort(stable=True)` 替换了原来的 top-k 内核，
这**改变了 decode 的实际执行路径**，因此上面这两个数字**不能**与
`REPLY_MAIN24_COST` 的 90.231 ms 直接比较；本诊断按要求不把诊断选择器计入性能结论。

## 4. 与旧参考的关系

新参考（本次发布）与上一轮 `sm90_main24_stable_bs256` 的参考对比：

| 项 | 值 |
|---|---|
| `prompt_hash` | 相同 |
| 请求数 | 256 = 256 |
| 逐位相同的请求流 | **0 / 256** |
| 前缀可比位置 | 274,511 |
| 不同位置 | **857（0.312%）** |
| 差异集中 | `req176` 721 处、`req180` 124 处、`req159` 12 处；**其余 253 个请求逐位相同** |
| `req176` 首处差异 | position **322**：新 **223** / 旧 **19958**（与 `REPLY_SM90_DECODE_VANILLA_REPLAY` 报错完全一致）|

关键点：**恰好是上一轮失败的那三个请求**，且 `req176` 的首处差异位置精确等于
consumer 报错的 position 322。也就是说，稳定 decode top-k 只改变了个别平局请求，
**其余 253 个请求的 token 流逐位不变** —— 这正是"平局打破"应有的最小影响面。

同时，本次参考的请求序列长度变为 1001–1117（上一轮为 2001–2117），
因为 `GC_MOE_DECODE_STATS_AA=1` 只跑统计窗口 1000 步即退出，未做第二轮 1000 步。

## 5. 建议

1. **decode top-k 稳定化应并入主表路径**：`GC_MOE_STABLE_DECODE_TOPK=1` 是
   `REPLY_SM90_MAIN24_STABLE_BS256` 六臂全部卡在 `GC_MOE_MATRIX_PREFIX_CHECK` 的**唯一缺口**。
   建议在六臂主表中与 `GC_MOE_STABLE_PREFILL_TOPK=1` 一起开启，即 `REPLY_SM90_MAIN24_STABLE_BS256`
   §7 的"判定口径"问题可能直接消失（原生差异从 4–7 归零）。
2. 若开启，需重新确认 E2E 计时：本次 publisher 93.344 / consumer 94.193 ms（stats 窗口）
   高于稳定 top-k 前的 90.362 ms，**约 +3.3%**，其中包含 `argsort` 选择器的开销。
   该开销是否会覆盖掉 eco_smr 相对 vanilla 的 −6.8% 收益，需要专门测量。
   建议做一个 A/B：同样六臂，分别带/不带 `GC_MOE_STABLE_DECODE_TOPK`。
3. `GC_MOE_DECODE_STATS_AA=1` 只跑 stats 窗口，无法覆盖 measured 窗口的 1000 步。
   若需要完整 1000+1000 的 A/A，应改用 `GC_MOE_VANILLA_STATS_REPLAY` 的两段式（本次即如此）。

## 6. 交付物

`docs/trace_executor_20261119/sm90_stable_decode_aa_results/`：

- `publisher/`：`runner.log`、`run.log`、`run.json`（stats 窗口）、`status.txt`（`exit_code=0`）、
  `exit-code.txt`（=0）、`runtime.log`、`execution.sha`、`start.sha`、`warm.log`、`gpu.log`
- `consumer/`：同上，另含 `failed-command.txt`（空/无失败）
- `tokens.json`（本次发布的参考，2.4 MB）+ `reference.sha256`
- `summary.json`、`cpu.log`、`df.log`、`state-path.txt`
- `SHA256SUMS`（已自检）

**未提交**：KV 与 route 回放（`/tmp/gcmoe-decode-aa-NuU4vuVb`，166 GiB）按要求留在现场。
