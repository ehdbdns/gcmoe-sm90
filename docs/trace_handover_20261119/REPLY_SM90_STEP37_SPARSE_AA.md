# 【SM90】REPLY_SM90_STEP37_SPARSE_AA — 4 个 rank 索引集合不同；另 4 个 rank 输入相同但输出不同

> 本回执由 **SM90 执行者**（8×H20）提交。

## 0. 核心结论（两个独立机制）

step 37 的 layer 20 存在**两种不同的分歧机制**，按 rank 分开：

| rank 组 | `flash_mla_sparse_fwd` 输入 | `wo_b` 的 `in` | `wo_b` 的 `out` | 机制 |
|---|---|---|---|---|
| **rank 2, 4, 6, 7** | **`ordered` + `sorted_per_row` 不同**（索引**集合**变了）| 不同 | 不同 | **索引选择** |
| **rank 0, 1, 3, 5** | **完全一致** | **✅ 相同** | **❌ 不同** | **attention 算术** |

**这是本轮最重要的发现**：`wo_b` 的输入相同、输出不同，
说明差异来自 **`wo_b` 自身的 GEMM 归约**（或 `_wo_b_proj` 的三条路径之一），
**而不是稀疏 attention**。

## 1. 覆盖校验

| 项 | 结果 |
|---|---|
| sparse 记录 | 各 **32**（2 步 × 2 边 × 8 rank）|
| block 边界 | 各 **608** |
| MoE 条目 | 各 **640**（2 步 × 40 层 × 8 rank）|
| 键集合 | **两臂完全相同**，无未覆盖 |
| `prompt_hash`/`positions`/`batch_order`/`prefill_steps`/`samples` | **全部相同** |

## 2. 稀疏 attention 探针（step 37）

| step | edge | 字段 | 不同 rank 数 |
|---:|---|---|---:|
| 36 | in / out | — | **0**（全部一致）|
| **37** | **in** | **`ordered`** | **4/8** |
| **37** | **in** | **`sorted_per_row`** | **4/8** |
| 37 | in | `query` / `lengths` / `slots` / `selected_kv` / `sink` / `scale` | **0** |
| **37** | **out** | **`digest`** | **4/8** |

### 2.1 逐 rank 的差异字段

```
rank0: []                                  ← 完全一致
rank1: []                                  ← 完全一致
rank2: ['ordered', 'sorted_per_row']       ← 索引集合变了
rank3: []                                  ← 完全一致
rank4: ['ordered', 'sorted_per_row']
rank5: []
rank6: ['ordered', 'sorted_per_row']
rank7: ['ordered', 'sorted_per_row']
```

**`sorted_per_row` 也不同** —— 这与上一轮（step 13）**只差顺序、集合相同**有本质区别：
本轮是 **实际选中的索引集合本身不同**。

**但 `slots` 与 `selected_kv` 仍然相同** —— 说明选中的**物理 KV 槽位集合**没变，
只是每个 query 行内**保留哪些**索引不同。

## 3. 模块边界（step 37 layer 20）

| 边界 | 不同 rank 数 |
|---|---:|
| `attn.wo_b` **`in`** | **4**（仅 2/4/6/7）|
| `attn.wo_b` `out` | **8** |
| `attn` `out` | **8** |
| `ffn` / `ffn.experts` / `ffn.gate` 的 in/out | 8 |
| `L20.flash_mla_sparse_fwd` in/out | 4 |
| `layers.20` `out` | 8 |
| **step 36 的全部边界** | **0** |

### 3.1 关键对照：rank 0 与 rank 2

| rank | `wo_b.in` | `wo_b.out` | 层输出 |
|---|---|---|---|
| **rank 0** | **✅ 相同** | **❌ 不同** | ❌ 不同 |
| **rank 2** | ❌ 不同 | ❌ 不同 | ❌ 不同 |

**rank 0 的输入完全相同、输出却不同**，这条证据直接排除了"上游索引差异导致"的解释 ——
对 rank 0 而言，**分歧发生在 `wo_b` 这个 GEMM 内部**。

## 4. MoE 边界

| step | 不同条目 | 形态 |
|---:|---:|---|
| 36 | **0** | — |
| **37** | **640/640**（八 rank × 40 层全部）| layer 20 起 `hidden`/`weights`/`output` 不同；`ids` 在 layer 20 仍相同，layer 21 起才变 |

**`ids` 在 layer 20 保持一致**，说明该层的路由选择没有分歧，
MoE 输出差异来自 hidden 本身（与 §3 的 `wo_b.out` 不同一致）。

## 5. sampler 与原生 token

| 项 | 值 |
|---|---|
| sampler hidden 最早差异 | **engine step 38，rank 0** |
| 两次原生 token 差异 | **2** 处 |

| 请求 | 位置 | first | second |
|---|---:|---:|---:|
| **req134** | 62 | 14729 | **6669** |
| **req139** | 17 | 70272 | **26947** |

任务书提到「req134」—— 本轮 `req134` 确实出现在原生差异列表中，**位置是 62**。

## 6. 相对于旧参考的失配

| 回放 | mismatches / samples |
|---|---:|
| first | **200 / 18,511** |
| second | **201 / 18,511** |

与此前各轮同量级。

## 7. 结论

1. **step 36 完全一致**，step 37 是分叉点 —— 与上一轮的 MoE 探针结论一致。
2. **存在两种独立机制**：
   - **rank 2/4/6/7**：稀疏索引的**集合**变了（`sorted_per_row` 不同），
     但物理槽位集合（`slots`/`selected_kv`）仍相同；
   - **rank 0/1/3/5**：稀疏 attention 的输入**完全一致**，
     但 `wo_b` 的输出不同 —— **分歧在 GEMM 归约内部**。
3. **第二种机制是全新的**：此前的所有结论都指向 attention，
   本轮证明**至少一半的 rank 其分歧来自 `wo_b` 自身**。
4. `_wo_b_proj` 有三条路径（直接 `wo_b`、GEMM-RS 融合、`sp_reduce_scatter`），
   其中后两者涉及**reduce-scatter 的归约顺序**，是这次非确定性的合理来源。
5. `ids` 在 layer 20 相同、layer 21 才变，与"路由跟随 hidden"一致。

### 建议下一步（供设计端）

- 对 rank 0/1/3/5 这类"输入相同、输出不同"的情况，
  需要探测 `_wo_b_proj` 走了哪条路径（`gemm_rs is None` / `should_run` / `sp_reduce_scatter`），
  以及 `gemm_rs.apply` 内部的归约分块方式。
- 对 rank 2/4/6/7，需要检查 `combine_topk_swa_indices` 在这 4 个 rank 上
  为何产生**不同的索引集合**（`slots` 相同而 `sorted_per_row` 不同是很强的线索）。

## 8. 版本与环境

| 项 | 值 |
|---|---|
| 测量 SHA（两次同一 SHA）| `82df425638de51a377a4bdff27554f06b8340e6b` |
| 工作目录 | `/tmp/gcmoe-prefix-aa-PnebWzR6` |
| 代码工作树（固定 SHA）| `/tmp/rt/sm90_s37_code` |
| 参考轨迹 sha256 | `ad68503c07f9c213c7ddeda92593f54b65fe65515b10dd8247ae7f1c714b1f15` |
| GPU | 8×H20，driver 580.105.08 |
| 软件 | cutlass 4.8.1 / torch 2.13.0+cu130 / CUDA 13.0 / vllm 0.31.0 |
| 模型启动数 | **2** |
| 代码修改 | **无** |

**环境变量**：`GC_MOE_ORDER_PREFILL=1`、`GC_MOE_PREFIX_BLOCK_FIRST=36`、
`GC_MOE_PREFIX_BLOCK_STOP=38`、`GC_MOE_PREFIX_LAYER_FIRST=36`、
`GC_MOE_PREFIX_LAYER_STOP=38`、`GC_MOE_PREFIX_SPARSE_AA=1`、
`GC_MOE_PREFIX_BLOCK_AA=1`、`GC_MOE_PREFIX_HIDDEN_AA=1`、
`GC_MOE_PREFIX_LAYER_AA=1`、`GC_MOE_PREFIX_DIAG=1`、`GC_MOE_MATRIX_ARM=vanilla`、
`GC_MOE_MATRIX_BS=256`、`GC_MOE_MATRIX_SIZE=24`、`GC_MOE_KV_BYTES=21474836480`、
`GC_MOE_LONG_COST=1`、`GC_MOE_SM90_STREAM_M_BLOCKS=1`。

## 9. 交付物

`docs/trace_executor_20261119/sm90_step37_sparse_aa_results/`：
- `first.json`、`second.json`（含 `block_by_rank`（608 条）中的 sparse 记录、
  `moe_by_rank`（640 条）、`hidden_by_rank`）
- `reference.sha`
- `first.log`、`second.log`
- `SHA256SUMS`
