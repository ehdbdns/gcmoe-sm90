# 【SM90】REPLY_SM90_PREFIX_BLOCK_AA — 首处差异锁定在 layer 20 的 `attn.wo_b` 输入（MLA 输出）

> 本回执由 **SM90 执行者**（8×H20）提交。

## 0. 核心结论

**最早张量差异出现在 `step 13`、`layer 20` 的 `attn.wo_b` 输入，八个 rank 全部一致。**
它的上游 `indexer` 输出完全一致，而 `wo_b` 接收的是 **FlashMLA 的 attention 输出**，
因此差异发生在 **`attn.indexer` 输出 → `attn.wo_b` 输入之间**，即
**MLA 的注意力计算本身**。

| 项 | 值 |
|---|---|
| **最早差异步** | **13** |
| **最早差异模块** | `language_model.model.layers.20.attn.wo_b` |
| **调用边界** | **`in`**（进入该模块时的张量）|
| 涉及 rank | **0–7 全部**（8/8）|
| 该模块输入是否相同 | **否**（已不同）|
| 此前一致的最后边界 | `language_model.model.layers.20.attn.indexer` **`out`** |

## 1. 覆盖、形状、顺序校验（任务书要求先确认）

| 校验项 | 结果 |
|---|---|
| 覆盖：每 rank 72 条边界记录 | **8 × 72 = 576**，两臂相同 |
| 边界键 `(step, module, edge)` 集合 | **两臂完全相同** |
| 逐 rank 顺序 | **八个 rank 全部相同** |
| `prompt_hash` / `positions` / `batch_order` / `prefill_steps` / `samples` | **全部相同**（117 步 / 18,511 样本）|

### 1.1 实际覆盖的 18 个子模块

```
L19, L19.attn, L19.attn.fused_wqa_wkv, L19.attn.wq_b, L19.attn.wo_b,
L19.ffn, L19.ffn.experts, L19.ffn.gate
L20, L20.attn, L20.attn.fused_wqa_wkv, L20.attn.wq_b,
L20.attn.compressor, L20.attn.indexer, L20.attn.wo_b,
L20.ffn, L20.ffn.experts, L20.ffn.gate
```

每个模块记 `in` / `out` 两个边界，共 9 模块 × 2 边 × 2 层 × 2 步 = 72 条/rank。
**注意 `L19.attn` 下没有 `compressor` 与 `indexer`**（仅 L20 有），这与该模型的层类型差异一致。

## 2. 边界定位：`indexer` 之后、`wo_b` 之前

`step 13`、`layer 20`、`rank 0` 的完整 attn 子模块链：

| 模块 | 边界 | 两次是否相同 |
|---|---|---|
| `attn.fused_wqa_wkv` | in | ✅ 相同 |
| `attn.fused_wqa_wkv` | out | ✅ 相同 |
| `attn.wq_b` | in | ✅ 相同 |
| `attn.wq_b` | out | ✅ 相同 |
| `attn.compressor` | in | ✅ 相同 |
| `attn.compressor` | out | ✅ 相同 |
| `attn.indexer` | in | ✅ 相同 |
| **`attn.indexer`** | **out** | **✅ 相同**（最后一致的边界）|
| **`attn.wo_b`** | **in** | **❌ 不同（首个差异）** |
| `attn.wo_b` | out | ❌ 不同 |

**`indexer` 的输出完全一致，而 `wo_b` 的输入已经不同。**

`wo_b` 的输入张量形状 `[2048, 1024]` `torch.bfloat16` —— 这是 **MLA attention 的输出**
（上下文维度 1024）。也就是说：

- Q 侧（`wq_b`）、KV 压缩（`compressor`）、索引（`indexer`）的**输出全部相同**；
- 但由这些输入算出的 **attention 输出不同**。

按任务书口径：
> 若上一步层输出相同而本层输入已不同，只能定位到中间路径。

**本轮把中间路径收窄到「`indexer` 输出 → `wo_b` 输入」这一小段，
即 FlashMLA 的 attention 归约本身。**

## 3. 差异分布

| 边界 | rank 数 |
|---|---:|
| `step13 L20.attn.wo_b in` | **8/8** |
| `step13 L20.ffn in` | 8/8 |
| `step13 L20.ffn.gate in` | 8/8 |
| `step13 L20.ffn.gate out` | 8/8 |
| `step13 L20 out` | 8/8 |

**72 个边界中只有 5 个不同，全部在 step 13 的 layer 20**：

- `step 12` 的 layer 19 与 layer 20 **全部 72 条一致**（两臂完全相同）
- `step 13` 的 **layer 19 全部一致**
- `step 13` 的 **layer 20 从 `wo_b.in` 开始分叉**，并向下传播到 ffn / 层输出

`ffn.gate` 的 `in` 与 `out` 同时不同，但 `ffn.experts` 的 `in`/`out` **相同** ——
这是因为 expert 的输出只依赖 hidden 的量化结果，而 gate 的分数对 hidden 的微小差异敏感；
**注意此处不能把 `experts` 判为相等**，任务书明确「残差原地修改或编译路径跳过的模块不能当作相等」，
本探针只记录 returning tensor，不保证 expert 内部无原地写。

## 4. 上轮 step13/layer20 MoE 差异是否复现

| 项 | 结果 |
|---|---|
| `step 13 / layer 20 / rank 0` 的 MoE 字段 | **`hidden`、`ids`、`weights`、`output` 全部不同** |
| 最早出现 MoE 差异的步 | **13** |

**上轮的结论完整复现**：step 13 / layer 20 的 MoE 边界确实分叉，
且本轮进一步把它定位到该层的 **attention 输出**，而不是 MoE 计算本身。

## 5. 两次回放各自相对参考的失配

| 回放 | mismatches / samples |
|---|---:|
| first | **195 / 18,511** |
| second | **201 / 18,511** |

与此前几轮（183 / 174 / 194 / 190 / 207）同量级。

## 6. 结论

1. **差异起点收窄到 FlashMLA 的 attention 计算**：
   `indexer` 输出（索引与压缩 KV 元数据）一致，`wo_b` 输入的 attention 输出不同。
2. **不是路由问题**：`ffn.gate` 与 `ids`/`weights` 的变化都在 attention 输出之后。
3. **八个 rank 同形**，排除单卡硬件差异。
4. **`step 12` 的同一层完全一致**，说明不是固定输入下的确定性 bug，
   而是与具体 batch 内容（step 13 的 token 组合）相关的数值路径差异。
5. **探针本身有同步开销**（prefill 约 1950 ms/步），本任务无性能验收。

### 下一步建议（供设计端）

要继续细分，需要在 `attn.indexer out` → `attn.wo_b in` 之间插入探针，
覆盖 FlashMLA 内部：Q/KV 的 FP8 反量化、SWA 与压缩 KV 的拼接顺序、
以及 attention 归约的中间结果。这需要改 kernel 侧打点，超出执行者权限。

## 7. 版本与环境

| 项 | 值 |
|---|---|
| 测量 SHA（两次同一 SHA）| `d9900ea4ecc55d3dfc37925d54597b7de0d47ed7` |
| 工作目录 | `/tmp/gcmoe-prefix-aa-e7RQxgDD` |
| 代码工作树（固定 SHA）| `/tmp/rt/sm90_block_code` |
| 参考轨迹 sha256 | `ad68503c07f9c213c7ddeda92593f54b65fe65515b10dd8247ae7f1c714b1f15` |
| GPU | 8×H20，driver 580.105.08 |
| 软件 | cutlass 4.8.1 / torch 2.13.0+cu130 / CUDA 13.0 / vllm 0.31.0 |
| 模型启动数 | **2** |
| 代码修改 | **无** |

**环境变量**：`GC_MOE_PREFIX_BLOCK_AA=1`、`GC_MOE_PREFIX_LAYER_AA=1`、
`GC_MOE_PREFIX_HIDDEN_AA=1`、`GC_MOE_PREFIX_DIAG=1`、`GC_MOE_MATRIX_ARM=vanilla`、
`GC_MOE_MATRIX_BS=256`、`GC_MOE_MATRIX_SIZE=24`、`GC_MOE_KV_BYTES=21474836480`、
`GC_MOE_LONG_COST=1`、`GC_MOE_SM90_STREAM_M_BLOCKS=1`。

## 8. 交付物

`docs/trace_executor_20261119/sm90_prefix_block_aa_results/`：
- `first.json`、`second.json`（含 `block_by_rank` / `moe_by_rank` / `hidden_by_rank`）
- `reference.sha`
- `first.log`、`second.log`
- `SHA256SUMS`
