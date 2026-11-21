# 【SM90】REPLY_SM90_PREFIX_LAYER_AA — 首处 MoE 边界差异在 step 13 / layer 20 的输入

> 本回执由 **SM90 执行者**（8×H20）提交。

## 0. 核心结论

**最早的分歧出现在 `step 13` 的 `layer 20`，且发生在本层的 `hidden` 输入本身。**
`step 13` 的 `layer 19` 输出完全一致，所以分歧落在
**layer 19 的输出 → layer 20 的输入之间的中间路径**。

- `step 0–12` 的 20×8 = 160 个 `(step,layer,rank)` 条目**全部一致**
- `step 13` 起，`layer 20–39` 每层每个 rank 全部不同（`layer 0–19` 仍一致）

| 项 | 值 |
|---|---|
| **最早差异步** | **13** |
| **最早差异层** | **20** |
| 涉及 rank | **0–7 全部**（八个 rank 同形）|
| 差异字段 | `hidden`、`ids`、`weights`、`output` 全部不同 |

这与任务书给出的线索「最早 sampler hidden 差异在引擎步 13」**完全吻合**。

## 1. 覆盖、顺序、tokens 校验（任务书要求先确认）

| 校验项 | 结果 |
|---|---|
| 覆盖：20 步 × 40 层 × 8 rank | **6,400 / 6,400**（两臂相同）|
| 条目顺序（按 `(step,layer)` 逐 rank）| **八个 rank 全部相同** |
| `tokens` 字段 | **6,400 条全部相等** |
| `prompt_hash` / `positions` / `batch_order` / `prefill_steps` / `samples` | **全部相同** |

所以下文的差异不是覆盖缺失或顺序错位造成的，可以判为「不相等」。

## 2. 边界定位：layer 19 输出一致，layer 20 输入已不同

`step 13`、`rank 0` 逐层检查：

| 层 | tokens | hidden | ids | weights | output |
|---|---|---|---|---|---|
| 18 | ✅ | ✅ | ✅ | ✅ | ✅ |
| **19** | ✅ | ✅ | ✅ | ✅ | **✅** |
| **20** | ✅ | **❌** | **❌** | **❌** | **❌** |

**layer 19 的输出完全相同，而 layer 20 的 `hidden` 输入已经不同。**

按任务书的口径：
> 若上一步层输出相同而本层输入已不同，只能定位到中间路径。

因此本轮**只能定位到「layer 19 输出 → layer 20 输入」这段中间路径**，
不能进一步区分是 residual 累加、layer norm、还是 hidden 在写入/读取时的差异。

### 2.1 八个 rank 的差异形态一致

| rank | 差异字段 |
|---|---|
| 0–7（全部）| `['hidden', 'ids', 'weights', 'output']` |

八个 rank 在每个差异条目上都**同时**在 `hidden` 处不同，
说明这不是某个 rank 落单，而是**整批 hidden 状态同时分叉**。

## 3. 差异随步数的传播

| step | 有差异的条目数（每步 40 层 × 8 rank = 320）| 涉及层 |
|---:|---:|---|
| 0–12 | **0** | — |
| **13** | **160** | 20–39 |
| 14 | 160 | 20–39 |
| 15 | 160 | 20–39 |
| 16 | 160 | 20–39 |
| 17 | 160 | 20–39 |
| 18 | 160 | 20–39 |
| 19 | 160 | 20–39 |

**从 step 13 起，layer 20–39 全部不同（每步 20 层 × 8 rank = 160），
而 layer 0–19 始终一致。** 说明分歧一旦在 layer 20 入口发生，
就会沿后续 20 层持续传播。

## 4. sampler 侧复核（任务书要求）

| 项 | 数值 |
|---|---|
| 两次原生 token 不同的 `(请求,位置)` | **192** |
| 其中 sampler hidden **不同** | **192 / 192** |
| 其中 sampler hidden **相同** | **0** |
| sampler hidden 最早分歧的引擎步 | **116** |

**MoE 层的分叉在 step 13，而 sampler 可见的分叉要到 step 116** ——
中间相差 103 个引擎步。这说明 sampler 的 hidden 摘要对早期的微小差异
不敏感（或该差异在后续层被吸收/抵消），直到最后一步才在 hidden 上显现。

## 5. 两次回放各自相对参考的失配

| 回放 | mismatches / samples |
|---|---:|
| first | **207 / 18,511** |
| second | **190 / 18,511** |

与上两轮（183 / 174 / 194）同量级，进一步印证参考轨迹不可位复现。

## 6. 结论与建议

1. **分歧起点**：`step 13`，`layer 20` 的输入 hidden。
   `layer 19` 输出一致而 `layer 20` 输入不同 —— 定位到两者之间的中间路径。
2. **不是路由或 MoE 计算本身引起的**：`ids` 与 `weights` 在 layer 20 处
   与 `hidden` 同时变化，说明路由是**跟着 hidden 变的**，不是先变。
3. **八个 rank 同形**，排除单 rank 问题。
4. 下一步若要继续细分，需要在 layer 19 的输出与 layer 20 的输入之间
   加入更细的探针（residual 写入后、norm 之后、MoE 入口之前），
   这部分当前探针没有覆盖。
5. **探针改变了同步时序**：每层都做摘要同步，prefill 单步耗时从约 460 ms
   升到约 2000 ms。因此本轮只用于定位，不能用于性能结论。

## 7. 版本与环境

| 项 | 值 |
|---|---|
| 测量 SHA（两次同一 SHA）| `b0bcfb96b843c865192e2fd585d335a5daa27550` |
| 工作目录 | `/tmp/gcmoe-prefix-aa-jvULFc4t` |
| 代码工作树（固定 SHA）| `/tmp/rt/sm90_layer_code` |
| 参考轨迹 sha256 | `ad68503c07f9c213c7ddeda92593f54b65fe65515b10dd8247ae7f1c714b1f15` |
| GPU | 8×H20，driver 580.105.08 |
| 软件 | cutlass 4.8.1 / torch 2.13.0+cu130 / CUDA 13.0 / vllm 0.31.0 |
| 模型启动数 | **2**（first 1 + second 1）|
| 代码修改 | **无** |

**环境变量**（按任务书）：`GC_MOE_PREFIX_LAYER_AA=1`、`GC_MOE_PREFIX_HIDDEN_AA=1`、
`GC_MOE_PREFIX_DIAG=1`、`GC_MOE_MATRIX_ARM=vanilla`、`GC_MOE_MATRIX_BS=256`、
`GC_MOE_MATRIX_SIZE=24`、`GC_MOE_KV_BYTES=21474836480`、`GC_MOE_LONG_COST=1`、
`GC_MOE_SM90_STREAM_M_BLOCKS=1`。

## 8. 交付物

`docs/trace_executor_20261119/sm90_prefix_layer_aa_results/`：
- `first.json`、`second.json`（含 `moe_by_rank` 与 `hidden_by_rank`）
- `reference.sha`
- `first.log`、`second.log`（完整日志）
- `SHA256SUMS`
