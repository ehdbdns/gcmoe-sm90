# 【SM90】REPLY_SM90_ORDER_STEP38 — MoE 边界在 step 37 就已分叉，早于 sampler 的 step 38

> 本回执由 **SM90 执行者**（8×H20）提交。

## 0. 核心结论

固定索引排序后，**MoE 边界的最早分叉在 `step 37` 的 `layer 20`**，
而 **sampler hidden 的最早分叉在 `step 38`**。

**MoE 探针比 sampler 早一步发现了分歧** —— 这与任务书预期的
"首个 sampler hidden 差异在第 38 步，req134" 在**步数上偏移了一步**，
但方向一致：分歧确实出现在 30 多步这个区间。

| 定位方式 | 最早差异 |
|---|---|
| **MoE 探针（step 20–38）** | **step 37 / layer 20** |
| **sampler hidden** | **step 38**（rank 0 最先）|
| **原生 token** | 仅 **1** 处（req139 pos14，engine step 116）|

## 1. 覆盖校验

| 项 | 结果 |
|---|---|
| 每 rank MoE 条目 | **760**（19 步 × 40 层，step 20–38）|
| 两臂覆盖集合 | **完全相同**（6,080 = 8 rank × 760）|
| `prompt_hash` / `positions` / `batch_order` / `prefill_steps` / `samples` | **全部相同** |
| 零记录（未覆盖） | **无** |

## 2. MoE 边界：6080 条中 288 条不同

| step | 不同条目数 | 涉及层 |
|---:|---:|---|
| 20–36 | **0** | — |
| **37** | **160** | **20, 21, …, 39** |
| **38** | **128** | **24, 25, …, 39** |

### 2.1 step 37 的逐层形态（rank 0）

| 层 | tokens | hidden | ids | weights | output |
|---|---|---|---|---|---|
| 18 | ✅ | ✅ | ✅ | ✅ | ✅ |
| **19** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **20** | ✅ | **❌** | **✅** | **❌** | **❌** |
| 21 | ✅ | ❌ | ❌ | ❌ | ❌ |

**这是本轮最重要的新发现**：

- **`ids` 在 layer 20 处仍然相同**，但 `hidden` / `weights` / `output` 不同；
- **八个 rank 的形态完全一致**（0–7 全部是 `['hidden','weights','output']`）；
- 从 **layer 21 起**，`ids` 也开始不同（因为 hidden 已变，路由随之改变）。

对比此前 step 13 的形态：

| | step 13 | **step 37** |
|---|---|---|
| layer 20 的 `ids` | ❌ 不同 | **✅ 相同** |
| layer 20 的 `hidden` | ❌ 不同 | ❌ 不同 |
| 分叉起点 | `attn.wo_b in`（attention 输出）| **尚未定位到模块级** |

**这排除了"路由先行"的可能**：step 37 中路由完全相同，
但 MoE 的输出不同 —— 说明差异来自 **MoE 计算内部**（或进入 MoE 前的 hidden），
**不是路由选择**。

## 3. sampler 侧

| 项 | 值 |
|---|---|
| sampler hidden 最早差异 | **engine step 38**（rank 0 最先）|
| 不同的 `(rank, step)` 组合 | **632 / 936** |
| 两次原生 token 差异 | **1** 处 |

### 3.1 唯一剩下的原生 token 分歧

| 项 | 值 |
|---|---|
| 请求 | **req139** |
| 请求内位置 | **14** |
| first | **6119** |
| second | **27438** |
| 所在引擎步 | **116** |

与上一轮（`GC_MOE_ORDER_PREFILL=1`）**完全一致**：
同一个请求、同一位置、同一对数值。说明排序规范化后剩余的
这 1 处分歧是**稳定的**，不是随机抖动。

## 4. 两次相对于旧参考的失配

| 回放 | mismatches / samples |
|---|---:|
| first | **202 / 18,511** |
| second | **201 / 18,511** |

与此前几轮同量级（174–207）。

## 5. 结论

1. **MoE 层的分叉在 step 37，比 sampler 的 step 38 早一步**。
   任务书根据上一轮推测的"step 38"在 sampler 侧成立，
   但 MoE 侧实际更早。
2. **step 36 及之前完全一致**，说明固定索引排序确实把前面的分歧都消除了。
3. **step 37 的 layer 20 中 `ids` 相同而 `hidden`/`weights`/`output` 不同** ——
   这是一个**新的分歧类型**，与 step 13 的"attention 输出不同"不同。
   它指向 **MoE 计算内部**（或 MoE 入口前的 hidden 累加），而非路由。
4. **八个 rank 同形**，排除单卡问题。
5. **剩余 1 处原生 token 分歧稳定复现**（req139 pos14），
   不是随机噪声。

### 建议下一步（供设计端）

- 在 **step 37 的 layer 20** 上启用 block 级探针（`GC_MOE_PREFIX_BLOCK_AA`），
  定位 hidden 在进入 MoE 前是否已经不同（即 attention 输出侧），
  还是 MoE 计算把相同的输入算出了不同的输出。
- 由于 `ids` 相同，可排除 router 与 gate 的数值路径；
  重点应在 **MoE 的 L1/L2 GEMM 归约顺序** 或 **combine 的归约顺序**。

## 6. 版本与环境

| 项 | 值 |
|---|---|
| 测量 SHA（两次同一 SHA）| `3ddc5ee12f82502f5caacd79c8f4e2125b229400` |
| 工作目录 | `/tmp/gcmoe-prefix-aa-837jAv4C` |
| 代码工作树（固定 SHA）| `/tmp/rt/sm90_s38_code` |
| 参考轨迹 sha256 | `ad68503c07f9c213c7ddeda92593f54b65fe65515b10dd8247ae7f1c714b1f15` |
| GPU | 8×H20，driver 580.105.08 |
| 软件 | cutlass 4.8.1 / torch 2.13.0+cu130 / CUDA 13.0 / vllm 0.31.0 |
| 模型启动数 | **2** |
| 代码修改 | **无** |

**环境变量**：`GC_MOE_ORDER_PREFILL=1`、`GC_MOE_PREFIX_LAYER_FIRST=20`、
`GC_MOE_PREFIX_LAYER_STOP=39`、`GC_MOE_PREFIX_HIDDEN_AA=1`、
`GC_MOE_PREFIX_LAYER_AA=1`、`GC_MOE_PREFIX_DIAG=1`、`GC_MOE_MATRIX_ARM=vanilla`、
`GC_MOE_MATRIX_BS=256`、`GC_MOE_MATRIX_SIZE=24`、`GC_MOE_KV_BYTES=21474836480`、
`GC_MOE_LONG_COST=1`、`GC_MOE_SM90_STREAM_M_BLOCKS=1`。
（按任务书 `unset` 了 `GC_MOE_PREFIX_SPARSE_AA` 与 `GC_MOE_PREFIX_BLOCK_AA`。）

**探针有同步开销**，本任务无性能验收。

## 7. 交付物

`docs/trace_executor_20261119/sm90_order_step38_results/`：
- `first.json`、`second.json`（含 `moe_by_rank`（760 条/rank）与 `hidden_by_rank`）
- `reference.sha`
- `first.log`、`second.log`
- `SHA256SUMS`
