# 【SM90】REPLY_SM90_SPARSE_PREFILL_AA — 稀疏索引顺序不同，多重集与 KV 完全相同

> 本回执由 **SM90 执行者**（8×H20）提交。

## 0. 核心结论

**step 13 的 FlashMLA 输入中，只有 `ordered`（索引排列顺序）不同；
`query`、`lengths`、`sorted_per_row`、`slots`、`selected_kv`、`sink`、`scale` 全部相同。**

这意味着：
- **有效索引的集合（多重集）完全一致** —— `sorted_per_row` 相同；
- **实际读取的 KV 内容完全一致** —— `slots` 与 `selected_kv` 相同；
- **只是这些索引在行内的排列顺序不同**。

这与 R7 在 layer 2 观测到的 `extra_order` 差异属于**同一类现象**，
现在已在 layer 20 的 prefill 路径上独立复现并精确定位到
`flash_mla_sparse_fwd` 的 `indices` 实参。

## 1. 覆盖校验（任务书要求先确认）

| 项 | 结果 |
|---|---|
| 每 rank sparse 记录数 | **4**（step 12/13 × in/out），八 rank 一致 |
| 两臂记录数 | 各 **32**（8 rank × 4）|
| 键集合 `(rank, step, edge)` | **两臂完全相同** |
| `prompt_hash` / `positions` / `batch_order` / `prefill_steps` / `samples` | **全部相同** |
| 零记录（未覆盖）的 rank | **无** |

**任务书要求「零记录回报为未覆盖」——本轮没有未覆盖的 rank。**

## 2. 逐字段比较（step 13，八个 rank 同形）

### 2.1 进入边界（`in`）—— 首个差异字段

| 字段 | 形状 / 类型 | 两次是否相同 |
|---|---|---|
| `query` | `[1976, 8, 512]` bf16 | ✅ 相同 |
| `lengths` | `[1976]` int32 | ✅ 相同 |
| **`ordered`** | **`[1976, 640]` int32** | **❌ 不同（首个差异）** |
| `sorted_per_row` | `[1976, 640]` int32 | ✅ **相同** |
| `slots` | `[4200]` int64 | ✅ 相同 |
| `selected_kv` | `[4200, 1, 512]` bf16 | ✅ 相同 |
| `sink` | `[8]` float32 | ✅ 相同 |
| `scale` | `0.04419417382415922` | ✅ 相同 |

**八个 rank 的差异形态完全一致**（每个 rank 都只有 `ordered` 不同）。

### 2.2 返回边界（`out`）

| 字段 | 两次是否相同 |
|---|---|
| `shape` | ✅ 相同 |
| `dtype` | ✅ 相同 |
| **`digest`** | **❌ 不同** |

即 `flash_mla_sparse_fwd` **写出的 attention 结果不同**。

### 2.3 对照组 step 12

| 边界 | 差异字段 |
|---|---|
| `step 12 in` | **无**（全部相同）|
| `step 12 out` | **无**（全部相同）|

**step 12 完全一致**，与 R8 的结论一致（step 0–12 无差异）。

## 3. 这个结果把因果链钉死了

结合前三轮的边界定位：

| 轮次 | 观测 |
|---|---|
| R7 | layer 2 `extra_order` 不同，`mla_output` 随之不同 |
| R8 | `attn.indexer out` 相同，`attn.wo_b in` 不同 |
| 源码采集 | `indexer` 的输出是 `topk_indices`；`wo_b` 的输入是 `flash_mla_sparse_fwd` 的 `out` |
| **本轮** | **`flash_mla_sparse_fwd` 的 `indices.ordered` 不同，其余输入全同，输出不同** |

**完整因果链**：

```
indexer 产出 topk_indices（相同）
   ↓ combine_topk_swa_indices 合并 SWA 索引
indices.ordered 不同  ← 唯一差异
   ↓ （sorted_per_row / slots / selected_kv 全同）
flash_mla_sparse_fwd 输出不同  ← attention 结果分叉
   ↓
wo_b 输入不同 → 层输出不同 → 后续层传播
```

### 3.1 为什么"集合相同、顺序不同"会导致输出不同

`flash_mla_sparse_fwd` 是**稀疏 attention**，它对每个 query 行
按 `indices` 给出的顺序取出 `topk_length` 个 KV 做加权求和。

- 数学上，求和顺序不影响精确结果；
- 但 **BF16/FP32 累加的舍入误差依赖顺序**，
  顺序不同 → 舍入路径不同 → 输出最低位不同 → 后续层放大。

这正是"位置稳定但数值不稳定"的解释：
**非确定性来自 `combine_topk_swa_indices` 的输出顺序**，
而该顺序又依赖于上游 `topk_indices` 的排列（R7 已证明 `extra_order` 会变）。

## 4. 与 wo_b 输入的对应（任务书要求）

`flash_mla_sparse_fwd` 的 `out` 是 `output[query_start:query_end]` 的切片，
而 `output` 就是 `_alloc_attn_out` 分配的 `[num_tokens, padded_heads, head_dim]`
缓冲区 —— 也即 `wo_b` 接收的 `z`。

因此本轮结果与上一轮完全自洽：

| 上一轮边界 | 本轮对应 |
|---|---|
| `attn.indexer out` ✅ 相同 | `topk_indices` 相同（`query`/`lengths` 同）|
| `attn.wo_b in` ❌ 不同 | `flash_mla_sparse_fwd.out.digest` 不同 |

**`wo_b` 输入的分叉已被本轮直接证实来自 `flash_mla_sparse_fwd` 的输出，
而该输出的唯一输入差异是 `indices` 的排列顺序。**

## 5. 两次相对参考的失配

| 回放 | mismatches / samples |
|---|---:|
| first | **180 / 18,511** |
| second | **196 / 18,511** |

与此前几轮（174–207）同量级。

## 6. 结论与建议

1. **唯一输入差异是 `indices` 的排列顺序**，不是集合、长度、KV 内容或 scale。
2. **输出随之不同**，说明稀疏 attention 的归约对索引顺序敏感。
3. **step 12 完全一致**，说明这不是固定输入的确定性缺陷。
4. 要彻底消除这种非确定性，需要在 `combine_topk_swa_indices`
   输出端固定排列（例如按索引值升序规范化），或在 attention 归约中
   使用顺序无关的累加。**两者都会改变被测内核语义**，超出执行者权限。

### 下一步建议（供设计端）

- 若要验证"规范化顺序即可消除分歧"，可在 `combine_topk_swa_indices`
  之后、`flash_mla_sparse_fwd` 之前插入一次 `sort`，
  用同一探针复测 `ordered` 与 `out.digest` 是否都变为一致。
- 但该改动会改变 attention 的数值路径，属于算法变更，需设计端决定。

## 7. 版本与环境

| 项 | 值 |
|---|---|
| 测量 SHA（两次同一 SHA）| `f8bf779b93ab55e163746f66fc26a2e1a7e312b1` |
| 工作目录 | `/tmp/gcmoe-prefix-aa-rbyTYUMM` |
| 代码工作树（固定 SHA）| `/tmp/rt/sm90_sparse_code` |
| 参考轨迹 sha256 | `ad68503c07f9c213c7ddeda92593f54b65fe65515b10dd8247ae7f1c714b1f15` |
| GPU | 8×H20，driver 580.105.08 |
| 软件 | cutlass 4.8.1 / torch 2.13.0+cu130 / CUDA 13.0 / vllm 0.31.0 |
| 模型启动数 | **2** |
| 代码修改 | **无** |

**环境变量**：`GC_MOE_PREFIX_SPARSE_AA=1`、`GC_MOE_PREFIX_HIDDEN_AA=1`、
`GC_MOE_PREFIX_LAYER_AA=1`、`GC_MOE_PREFIX_BLOCK_AA=1`、`GC_MOE_PREFIX_DIAG=1`、
`GC_MOE_MATRIX_ARM=vanilla`、`GC_MOE_MATRIX_BS=256`、`GC_MOE_MATRIX_SIZE=24`、
`GC_MOE_KV_BYTES=21474836480`、`GC_MOE_LONG_COST=1`、`GC_MOE_SM90_STREAM_M_BLOCKS=1`。

**探针副作用**：每层同步使 prefill 约 1990 ms/步，本任务无性能验收。

## 8. 交付物

`docs/trace_executor_20261119/sm90_sparse_prefill_aa_results/`：
- `first.json`、`second.json`（含 `block_by_rank` 内的 `L20.flash_mla_sparse_fwd` 记录，
  以及 `moe_by_rank` / `hidden_by_rank`）
- `reference.sha`
- `first.log`、`second.log`
- `SHA256SUMS`
