# 【SM90】REPLY_SM90_ORDER_PREFILL_AA — 索引排序规范化消除了 MoE 层的全部分歧

> 本回执由 **SM90 执行者**（8×H20）提交。

## 0. 核心结论

打开 `GC_MOE_ORDER_PREFILL=1`（把 `combine_topk_swa_indices` 返回的每行
有效前缀内非负索引升序排列）后，两次 vanilla 的对照结果是：

| 层级 | 规范化前 | **规范化后** |
|---|---:|---:|
| `flash_mla_sparse_fwd` 边界（32 条）| `ordered` 与 `out.digest` 不同 | ✅ **全部一致** |
| block 边界（608 条）| 5 条不同 | ✅ **0 条不同** |
| MoE 条目（6,400 条）| 160 条/步不同 | ✅ **0 条不同** |
| 两次原生 token 差异 | **187** 处 | **1** 处 |

**上一轮定位的因果链被完整证实**：唯一输入差异是索引排列顺序，
排序规范化后 MoE 层的分歧**全部消失**。

## 1. 覆盖校验（任务书要求先确认）

| 项 | 结果 |
|---|---|
| 每 rank sparse 记录 | **4**（step 12/13 × in/out）|
| 两臂 sparse 总数 | 各 **32** |
| 键集合 `(rank, step, edge)` | **两臂完全相同** |
| 零记录（未覆盖）rank | **无** |
| `prompt_hash` / `positions` / `batch_order` / `prefill_steps` / `samples` | **全部相同** |

## 2. Sparse 探针：32 条边界全部一致

逐字段（`query`、`lengths`、`ordered`、`sorted_per_row`、`slots`、
`selected_kv`、`sink`、`scale`、`digest`）× step（12、13）× edge（in、out）
× 8 rank 的完整比对：

```
step  edge  field           ranks_differing
 (none) — all sparse boundaries identical
```

**包括 `ordered` 与 `out.digest` 在内，没有任何字段不同。**
上一轮 step 13 的 `ordered` 差异已被排序消除，输出随之一致。

## 3. 上游 block 边界：608 条全部一致

| 项 | 值 |
|---|---|
| 比较的边界总数 | **608**（8 rank × 76 边界）|
| 不同的边界 | **0** |

包含 `L20.attn.wo_b` 的 `in`/`out`、`L20.ffn`、`L20` 输出等上一轮所有差异点。

## 4. MoE 边界：6,400 条全部一致

| 项 | 值 |
|---|---|
| 比较的 MoE 条目 | **6,400**（8 rank × 20 步 × 40 层）|
| `hidden`/`ids`/`weights`/`output` 任一不同的条目 | **0** |
| 有 MoE 差异的步 | **无**（此前是 step 13–19）|

**上一轮 step 13 / layer 20 的 MoE 分叉已完全消失。**

## 5. 两次运行自身的一致性（任务书要求）

| 项 | 结果 |
|---|---|
| `positions` | ✅ 相同 |
| `batch_order` | ✅ 相同 |
| `prefill_steps` / `samples` | ✅ 相同（117 / 18,511）|
| **逐 request 已采样 token** | **255 / 256 相同**（1 处不同）|
| **sampler hidden（936 个 rank×step 组合）** | 304 相同 / **632 不同** |

### 5.1 剩余的那 1 处原生 token 分歧

| 项 | 值 |
|---|---|
| 请求 | **req139** |
| 请求内位置 | **14** |
| first 原生 token | **27438**（= 参考值）|
| second 原生 token | **6119** |
| 所在引擎步 | **116** |
| 八个 rank 的 sampler hidden | **全部不同** |

即：**MoE 层在 20 步内已完全一致，但最后一步的 sampler hidden 仍然分叉。**

### 5.2 sampler hidden 为何仍有 632 处不同

**这需要谨慎解读**：`hidden_by_rank` 记录的是**每个接受 token 传入 sample 的
最后一行 hidden**，覆盖 117 个引擎步的全部 prefill token。

- MoE 探针只覆盖 **step 12–13 的 layer 19/20**（任务书限定范围）；
- sampler hidden 覆盖 **全部 117 步的全部层**；
- 因此 632 处不同**不代表**排序规范化失效，而是说明
  **step 14–116 之间仍有其它非确定性来源**（MoE 探针没有覆盖那段）。

## 6. 相对于旧参考的失配（任务书要求分开报告）

| 回放 | vs 旧参考的失配 |
|---|---:|
| first | **200 / 18,511** |
| second | **201 / 18,511** |

与此前几轮（174–207）同量级。**排序规范化不改变"与旧参考的失配"**，
因为旧参考本身是在未规范化的路径上生成的 —— 这一项不能用来判断修复效果。

## 7. 结论

1. **排序规范化确实消除了 MoE 层的全部分歧**：
   block 边界 608/608 一致、MoE 条目 6400/6400 一致、sparse 边界 32/32 一致。
2. **上一轮"索引顺序导致注意力输出不同"的因果链得到直接验证**。
3. **剩余 1 处原生 token 分歧发生在 step 116 的 sampler hidden**，
   超出了本轮 MoE 探针的覆盖范围（step 12–13），说明**还有别的非确定性来源**。
4. 排序规范化带来开销：prefill 单步从约 1990 ms 升到约 2070–2850 ms，
   本任务无性能验收。

### 建议下一步（供设计端）

- 把 MoE 探针的覆盖步范围从 12–13 扩到 14–116，定位第二个非确定性来源。
- 或先把 sampler hidden 的比对范围与 MoE 探针范围对齐（只比 step 12–13），
  确认在覆盖范围内 sampler 是否也已一致。

## 8. 版本与环境

| 项 | 值 |
|---|---|
| 测量 SHA（两次同一 SHA）| `cb0226f292d7a1d37392f31196a36fed11d802a1` |
| 工作目录 | `/tmp/gcmoe-prefix-aa-JBmpW1tZ` |
| 代码工作树（固定 SHA）| `/tmp/rt/sm90_order_code` |
| 参考轨迹 sha256 | `ad68503c07f9c213c7ddeda92593f54b65fe65515b10dd8247ae7f1c714b1f15` |
| GPU | 8×H20，driver 580.105.08 |
| 软件 | cutlass 4.8.1 / torch 2.13.0+cu130 / CUDA 13.0 / vllm 0.31.0 |
| 模型启动数 | **2** |
| 代码修改 | **无** |

**环境变量**：`GC_MOE_ORDER_PREFILL=1`、`GC_MOE_PREFIX_SPARSE_AA=1`、
`GC_MOE_PREFIX_HIDDEN_AA=1`、`GC_MOE_PREFIX_LAYER_AA=1`、`GC_MOE_PREFIX_BLOCK_AA=1`、
`GC_MOE_PREFIX_DIAG=1`、`GC_MOE_MATRIX_ARM=vanilla`、`GC_MOE_MATRIX_BS=256`、
`GC_MOE_MATRIX_SIZE=24`、`GC_MOE_KV_BYTES=21474836480`、`GC_MOE_LONG_COST=1`、
`GC_MOE_SM90_STREAM_M_BLOCKS=1`。

## 9. 交付物

`docs/trace_executor_20261119/sm90_order_prefill_aa_results/`：
- `first.json`、`second.json`（含 `block_by_rank` 内的问题 sparse 记录、
  `moe_by_rank`、`hidden_by_rank`）
- `reference.sha`
- `first.log`、`second.log`
- `SHA256SUMS`
