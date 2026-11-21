# 【SM90】REPLY_SM90_PREFIX_HIDDEN_AA — sampler 输入在全部 187 个失配点上已经不同

> 本回执由 **SM90 执行者**（8×H20）提交。

## 0. 核心结论

**两次同配置 vanilla 回放中，187 个原生 token 不同的位置，hidden 摘要（sampler 输入）
在全部八个 rank 上都不同。hidden 相同而 token 不同的情况为 0 个。**

这意味着分歧的源头**在 sampler 之前**，不在 sampler 及其下游。
按任务书的口径，本轮只把问题定位到「sampler 输入已经不同」这一层。

| 分类 | 数量 |
|---|---:|
| 两次原生 token 不同且**至少一个 rank 的 hidden 不同** | **187 / 187** |
| 两次原生 token 不同且**所有 rank 的 hidden 相同** | **0** |
| 无法映射到 hidden 记录 | 0 |

## 1. 两次回放的一致性（任务书要求逐项确认）

| 字段 | 结果 |
|---|---|
| `prompt_hash` | ✅ 相同 |
| `positions` | ✅ 相同 |
| `batch_order` | ✅ 相同（`000697f7…`）|
| `prefill_steps` | ✅ 相同（**117**）|
| `samples` | ✅ 相同（**18,511**）|

两次运行在**请求集合、位置、顺序、引擎步数**上完全一致，唯一的差异是数值。

### 1.1 两次各自的失配数

| 回放 | 相对参考的失配 | 涉及请求数 |
|---|---:|---:|
| first | **183 / 18,511** | 84 |
| second | **194 / 18,511** | — |

上轮（R7 自对照）vanilla 是 183，random 是 174；本轮 first 又是 183、
second 是 194 —— **两次同配置 vanilla 之间就有 183 vs 194 的差异**，
这本身就说明参考轨迹不可复现。

## 2. hidden 摘要与请求、位置的对应（任务书要求）

`hidden_by_rank[rank][engine_step] = [[row_index, sha256], ...]`

- 每个接受 token 记一次，记录的是传给 `sample` 的**最后一行 hidden**。
- `row_index` 即请求序号；各请求的接受次数已与 `positions` 逐项核对一致
  （`positions[req] = 117` 或 `116`，与 hidden 记录数相同）。
- 汇总口径：同一 `(engine_step, rank, request)` 下还有多个 row 时取行集合；
  比较时要求**八个 rank 全部相同**才算「hidden 相同」。

## 3. 最早出现差异的位置

| 项 | 值 |
|---|---|
| **引擎步** | **116**（最后一次 prefill 引擎步，共 117 步，0-based）|
| **请求** | **req73** |
| **请求内位置** | **18** |
| first 原生 token | **372** |
| second 原生 token | **15** |
| 八个 rank 的 hidden | **全部不同**（0–7 逐一为 False）|

即在最后一次 prefill 步上，`req73` 的第 18 个 token 就已分叉，
且八个 rank 的 sampler 输入同时不同。

## 4. 这个结果排除了什么

按任务书的判据：

> 第一种情况只定位到 sampler 输入已经不同，第二种才继续查 sampler 及其下游。

本轮**全部 187 个差异都属于第一种**，因此：

1. **不需要**再查 sampler 的采样逻辑、logits 处理或 token 选择 —— 
   它们接收到的输入就已经不同。
2. 分歧必然产生在 **hidden 状态的计算路径**上：MoE 的 routed 分支、
   shared 分支、attention、或它们的归约顺序。
3. 结合上轮（R7）的结论 —— vanilla 自对照就有 183 处失配、
   且 83/88 的失配位置在两臂间重合 —— 说明这是一条
   **位置稳定但数值不稳定的非确定性路径**，符合浮点归约顺序变化的特征。

## 5. 对判据的建议（供设计端决策）

1. **prefill 阶段的 hidden 本身不可位复现**，因此以 hidden 摘要做「零差异」门禁
   同样会间歇性失败。若要继续用 hidden 定位，建议改成
   **记录 + 允许容差**（例如按相对误差阈值）而不是 SHA 相等于否。
2. **正式 decode 窗口**的 hidden 是否稳定，本轮未测（任务书只跑 prefill）。
   若要判断最终交付能否通过严格判定，需要单独对 1000 步正式窗口做同样对照。
3. 要让 prefill 的 hidden 位复现，需要固定 kernel 的归约顺序或禁用相关
   fast-math 路径 —— 这会改变被测内核语义，超出执行者权限。

## 6. 版本与环境

| 项 | 值 |
|---|---|
| 测量 SHA（两次回放同一 SHA） | `cd9eab73df442b3d3756d79f3b3ed519d3c6998b` |
| 工作目录 | `/tmp/gcmoe-prefix-aa-Rt9SZZDb` |
| 代码工作树（second 重试用固定 SHA） | `/tmp/rt/sm90_code` |
| 参考轨迹 | `replay/tokens.json`，sha256 `ad68503c07f9c213c7ddeda92593f54b65fe65515b10dd8247ae7f1c714b1f15` |
| GPU | 8×H20，driver 580.105.08 |
| 软件 | cutlass 4.8.1 / torch 2.13.0+cu130 / CUDA 13.0 / vllm 0.31.0 |
| 模型启动数 | **3**（first 1 + second 首次失败 0 + second 重试 1）|
| 代码修改 | **无** |

**环境变量**（按任务书）：`GC_MOE_PREFIX_HIDDEN_AA=1`、`GC_MOE_PREFIX_DIAG=1`、
`GC_MOE_MATRIX_ARM=vanilla`、`GC_MOE_MATRIX_BS=256`、`GC_MOE_MATRIX_SIZE=24`、
`GC_MOE_KV_BYTES=21474836480`、`GC_MOE_LONG_COST=1`、`GC_MOE_SM90_STREAM_M_BLOCKS=1`。

## 7. 一次中断说明（如实记录）

second 的**首次**启动在 `step=1: checkout and environment` 即失败，
原因是 `run_aligned_ep8.sh` 的
`test "$(git rev-parse HEAD)" = "$(git rev-parse --verify "$1^{commit}")"` 不成立：
由于期间 SM100 执行者推了两个提交、我的仓库轮询做了 fast-forward，
主工作树的 HEAD 已从 `cd9eab7` 前移到 `e3a8479`。

**处理**：把自动 fast-forward 停掉，另建固定 SHA 的独立工作树
`/tmp/rt/sm90_code`（`cd9eab73df…`）重跑 second，**通过并完成**。
这次中断没有启动模型，不计入启动预算。

## 8. 交付物

`docs/trace_executor_20261119/sm90_prefix_hidden_aa_results/`：
- `first.json`、`second.json`（含 `arm` / `prompt_hash` / `prefill_steps` /
  `positions` / `samples` / `mismatches` / `batch_order` / **`hidden_by_rank`**）
- `reference.sha`
- `first.log`、`second.log`（两臂完整日志）
- `SHA256SUMS`
