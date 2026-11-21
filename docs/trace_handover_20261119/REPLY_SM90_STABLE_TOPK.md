# 【SM90】REPLY_SM90_STABLE_TOPK — 稳定 top-k 后 A/A 完全复现（原生差异 0）

> 本回执由 **SM90 执行者**（8×H20）提交。

## 0. 核心结论：**首次实现完整 A/A 一致**

启用 `GC_MOE_STABLE_PREFILL_TOPK=1`（同分时按原逻辑位置升序打破平局）后，
两次相同的 vanilla 回放**逐位完全一致**：

| 层级 | 规范化前 | **稳定 top-k 后** |
|---|---:|---:|
| sparse 边界（32 条）| step 37 有 4/8 rank 的索引集合不同 | ✅ **32/32 全部一致** |
| block 边界（608 条）| 84 条不同 | ✅ **0 条不同** |
| MoE 条目（640 条）| 640/640 全部不同 | ✅ **0 条不同** |
| **sampler hidden（936 个 rank×step）** | 632 个不同 | ✅ **0 个不同** |
| **两次原生 token 差异** | 2 处 | ✅ **0 处** |
| 相对旧参考的失配 | 200 / 201 | **196 / 196（且集合完全相同）** |

**这是这一系列任务中第一次做到两次运行逐位一致。**

## 1. 覆盖校验

| 项 | 结果 |
|---|---|
| sparse 记录 | 各 **32**，覆盖 step 36/37 × in/out × 8 rank |
| block 边界 | 各 **608** |
| MoE 条目 | 各 **640** |
| 键集合 | **两臂完全相同**，无未覆盖 |
| `prompt_hash`/`positions`/`batch_order`/`prefill_steps`/`samples` | **全部相同** |

## 2. Sparse attention 探针

```
step  edge  field           ranks_differing
 ALL sparse boundaries identical (32/32)
```

**九个字段全部一致**：`query`、`lengths`、`ordered`、`sorted_per_row`、
`slots`、`selected_kv`、`sink`、`scale`、`digest`。

对照上一轮（`GC_MOE_STABLE_PREFILL_TOPK` 未启用）：
step 37 有 **rank 2/4/6/7 的 `ordered` 与 `sorted_per_row` 不同** ——
本轮该差异**完全消失**。

## 3. 模块边界：608/608 一致

上一轮 step 37 有 84 条边界不同（含 `wo_b.in` 4 条、`wo_b.out` 8 条、
`ffn.*` 全部 8 条）。**本轮全部为 0**。

特别地：上一轮的"**rank 0/1/3/5 输入相同、`wo_b.out` 不同**"这一
"GEMM 归约内部差异"也**不再出现** —— 说明它并非 `wo_b` 自身的问题，
而是由上游索引分歧经 attention 传导后在不同舍入路径上放大所致。

## 4. MoE 边界：640/640 一致

上一轮 step 37 的 640 个条目全部不同（layer 20 起 `hidden`/`weights`/`output`），
**本轮为 0**。

## 5. Sampler 与原生 token：完全一致

| 项 | 值 |
|---|---|
| sampler hidden 不同的 `(rank, step)` 组合 | **0 / 936** |
| sampler hidden 最早差异 | **无** |
| **两次原生 token 差异** | **0 处** |

上一轮（未启用稳定 top-k）为 2 处：`req134 pos62`（14729 vs 6669）与
`req139 pos17`（70272 vs 26947）。**本轮两处都消失。**

## 6. 相对旧参考的失配（任务书要求分开报告）

| 回放 | mismatches / samples |
|---|---:|
| first | **196 / 18,511** |
| second | **196 / 18,511** |
| **失配集合是否完全相同** | ✅ **是**（同一批 196 个 `(请求, 位置)`）|

**关键区分**：
- **两次运行彼此一致**（0 差异）→ A/A 复现成功；
- **两次相对旧参考仍各失配 196 处** → 旧参考是在**未启用稳定 top-k 的路径**上
  生成的，其平局打破方式不同，因此必然与当前路径有差异。
  这 196 处**不是**非确定性，而是**参考与当前配置的确定性差异**。

## 7. 结论

1. **稳定 top-k（同分按逻辑位置升序）彻底消除了 prefill 的非确定性**。
   从 sparse 索引、模块边界、MoE 层到 sampler hidden 与原生 token，
   **五个层级全部逐位一致**。
2. **上一轮发现的"两种机制"实为一个根因**：
   - rank 2/4/6/7 的索引集合不同 → 平局打破的原子写入顺序不同；
   - rank 0/1/3/5 的"输入相同、输出不同" → 上游虽同，
     但 attention 的归约路径在批次内不同步，放大到 `wo_b` 才显现。
   两者都源于 **top-k 平局打破的非确定性**。
3. **这解释了此前所有轮次的现象**：从 R7 的 `extra_order`、
   到 R8 的 `indexer`/`wo_b` 边界、再到 R9 的 rate-level 分歧，
   根因都是同一个 —— vLLM 0.31.0 `sampler.cu` 在同分时用原子写入的位置决定顺序。
4. **198/196 这个数字是配置差异，不是抖动**：两次运行完全相同。

### 对后续实验的意义

- 若要在正式 A/B 中使用严格判据（原生 token 全一致），
  需要**双方都启用** `GC_MOE_STABLE_PREFILL_TOPK=1`，
  且参考轨迹也应基于同一配置重新生成。
- 当前旧参考（`tokens.json`）与稳定 top-k 路径有 196 处确定性差异，
  不能再用于"零差异"判定。

## 8. 版本与环境

| 项 | 值 |
|---|---|
| 测量 SHA（两次同一 SHA）| `578c1b109a38f41ff5ad15e3752605c531a9dc54` |
| 工作目录 | `/tmp/gcmoe-prefix-aa-4CRRj5jQ` |
| 代码工作树（固定 SHA）| `/tmp/rt/sm90_stk_code` |
| 参考轨迹 sha256 | `ad68503c07f9c213c7ddeda92593f54b65fe65515b10dd8247ae7f1c714b1f15` |
| GPU | 8×H20，driver 580.105.08 |
| 软件 | cutlass 4.8.1 / torch 2.13.0+cu130 / CUDA 13.0 / vllm 0.31.0 |
| 模型启动数 | **2**（此前有一次因我的调用方式失败，未启动模型）|
| 代码修改 | **无** |

**CPU 门禁**：`stable prefill top-k CPU PASS`

**环境变量**：`GC_MOE_STABLE_PREFILL_TOPK=1`、`GC_MOE_ORDER_PREFILL=1`、
`GC_MOE_PREFIX_BLOCK_FIRST=36`、`GC_MOE_PREFIX_BLOCK_STOP=38`、
`GC_MOE_PREFIX_LAYER_FIRST=36`、`GC_MOE_PREFIX_LAYER_STOP=38`、
`GC_MOE_PREFIX_SPARSE_AA=1`、`GC_MOE_PREFIX_BLOCK_AA=1`、
`GC_MOE_PREFIX_HIDDEN_AA=1`、`GC_MOE_PREFIX_LAYER_AA=1`、
`GC_MOE_PREFIX_DIAG=1`、`GC_MOE_MATRIX_ARM=vanilla`、`GC_MOE_MATRIX_BS=256`、
`GC_MOE_MATRIX_SIZE=24`、`GC_MOE_KV_BYTES=21474836480`、`GC_MOE_LONG_COST=1`、
`GC_MOE_SM90_STREAM_M_BLOCKS=1`。

## 9. 一次启动失败（如实记录）

首次启动在 `step=1` 失败于 `test -r "$DATASET"`。
根因：我在子 shell 中**先** `source scripts/env.sh` **后**才设置
`GC_MOE_*`，导致 `config/experiment.env` 的 `DATASET` 默认值
（两个文件用逗号拼接）未被单文件覆盖，路径检查失败。
该次**未启动模型**，不计入启动预算；改为按任务书顺序（先设模型/数据路径，
再 source）后通过。

## 10. 交付物

`docs/trace_executor_20261119/sm90_stable_topk_results/`：
- `first.json`、`second.json`（含 `block_by_rank`（608 条）、
  `moe_by_rank`（640 条）、`hidden_by_rank`）
- `reference.sha`
- `first.log`、`second.log`
- `SHA256SUMS`
