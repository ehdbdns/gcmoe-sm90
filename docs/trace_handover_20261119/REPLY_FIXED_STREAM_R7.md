# REPLY_FIXED_STREAM_R7 — **A/A 负对照未通过：baseline 自己都无法复现**

## 0. 判定

```json
{"negative_control_match":false,
 "strict_pair_comparable":false,
 "scheduling_gate_passed":false,
 "fixed_workload_faster":true,
 "lower":{"e2e_ms_per_step":true,"moe_max_rank_sum_ms":true},
 "faster_layers":14,"total_layers":40,
 "checkpoint_match":true,
 "same_hardware":true,"same_run_controls":true,"same_physical_routes":true,
 "first_moe_input_match":true,"first_moe_output_match":true,
 "shared_weights_match":true,"shared_branch_active":true,"first_step_finite":true,
 "native_top1_different_of_256":5,"first_native_difference_step":0}
```

**本轮最重要的结论不是性能，而是负对照失败**：两个进程都跑 baseline、
从同一份 KV 快照恢复、物理路由与逐槽负载全部一致，**但 A/A 在第 0 步就出现分歧**
（8 个 rank 的 `extra_order` 有 107–110 行不同，`mla_output` 有 49–84 行不同）。

按任务书「共同失败的 A/A 说明分歧可在开启 stream 之前发生」，这直接限定了 R6 那些
native top-1 差异的来源：**至少有一部分与 stream 无关**。

## 1. 两项主指标（都下降）

| 指标 | baseline | stream | 差值 | 变化 |
|---|---:|---:|---:|---:|
| **E2E（100 步）ms/step** | **82.416** | **81.112** | **−1.305** | **−1.58%** ✅ |
| **step 2 的 40 层 MoE ms** | **34.176** | **34.112** | **−0.065** | **−0.19%** ✅ |
| plan 占比 MAE（pp） | 0.2453 | 0.2437 | −0.0016 | −0.66% |
| plan 最大绝对误差（pp） | 0.8974 | 0.9061 | +0.0087 | +0.97% |
| step 2 开 profiler 墙钟 ms | 101.629 | 86.217 | −15.412 | −15.16% |
| 逐层（40 层） | — | — | — | **14 快 / 26 慢** |

**但 `fixed_workload_faster = true` 的成立前提是「控制项相同」**，
而 `negative_control_match = false` 说明这个前提在本轮**没有被证实**，
所以不能据此宣称 stream 有收益（见 §5）。

## 2. A/A 负对照：分歧出现在 stream 之前

### 2.1 控制步结束时的 KV 摘要

```
control_end_kv_different_ranks = [0, 1, 2, 3, 4, 5, 6, 7]
```

**两个都跑 baseline 的进程，在控制步结束时八个 rank 的完整 KV backing 摘要全部不同。**
任务书明确要求「第二次恢复不能掩盖这项差异」——这里如实记录：
即使两次都从同一快照恢复并逐 rank 回读校验（`checkpoint_match = true`），
执行完第 0 步之后 KV 内容就已经分化。

### 2.2 A/A 的首个可见分歧（step 0，layer 2，FlashMLA 边界）

`flashmla_rows.csv`（128 行 = 2 步 × 8 rank × 8 字段）显示：

| 字段 | step 0 有差异的 rank 数 | 典型行数 |
|---|---:|---|
| `query` | **0** | 0 |
| `swa_lengths` | **0** | 0 |
| `swa_multiset` | **0** | 0 |
| `swa_order` | **0** | 0 |
| `extra_lengths` | **0** | 0 |
| `extra_multiset` | **0** | 0 |
| **`extra_order`** | **8/8** | 107–110 |
| **`mla_output`** | **8/8** | 49–84 |

**这是本轮最有价值的定位结果**：

- **Q 一致、SWA 的索引长度/集合/顺序全部一致** → 查询侧和 SWA 分支没有分歧；
- **压缩 KV 的 `extra_lengths` 和 `extra_multiset`（含重复项）也一致** → 参与计算的索引**集合**相同；
- **只有 `extra_order`（保留重复项后的排列顺序）不同，且 `mla_output` 随之不同**。

即：**参与压缩注意力的索引集合完全相同，但它们的排列顺序不同，导致 MLA 输出不同。**
按任务书「Q 和索引一致而输出有差异时，还需核实本步写入的 KV 和内部实现，
当前探针无法单独区分二者」——这里 Q 与索引集合一致、仅顺序不同，
所以剩余两种可能就是**该步写入的 KV 内容**或**FlashMLA 内部实现的顺序敏感性**。

各 rank 的差异行数（step 0）：

| rank | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `extra_order` | 108 | 109 | 109 | 110 | 110 | 108 | 107 | 108 |
| `mla_output` | 77 | 74 | 49 | 84 | 65 | 84 | 60 | 71 |

示例请求：`req104–req110` 为主。

### 2.3 A/B 的 treatment 步（step 1）

`treatment_step_layers.csv` 与 `first_step_layers.csv` **完全同形**：
各 2,560 行，各 2,432 行不符，**首个分歧同样在 layer 2**。
`flashmla.treatment_match = false`，step 1 同样是 `extra_order`（104–109 行）
与 `mla_output`（54–87 行）不同。

**A/A 与 A/B 的分歧形态一致**，进一步说明 stream 不是这个分歧的来源。

## 3. 首步层级探针（`first_step_layers.csv`）

| layer | 0 | 1 | **2** | 3–39 |
|---|---|---|---|---|
| 不符行数 | **0** | **0** | **64** | 每层 64 |

与 R6 相同的模式：**layer 0 和 1 完全一致，从 layer 2 起全部不符**。
`shared_weights_match = true`（差异列表空）、`shared_branch_active = true`、
`first_step_finite = true`（无非有限值）。

## 4. 原生预测差异

| 项 | R5 | R6 | **R7** |
|---|---:|---:|---:|
| 采样步 `native_top1_different_of_256` | 41 | 7 | **5** |
| 105 步逐步差异范围 | 16–41 | 0–11 | **2–12** |

本轮的 5 个首个差异：

| request | baseline | stream |
|---|---:|---:|
| req11 | 35 | 36 |
| req44 | 58 | 51 |
| req116 | 4506 | 51 |

（示例 3 条，完整列表见 `comparison.json`。）

**注意**：这些是 A/B 的差异。由于 A/A 本身已有分歧，
不能把这 5 个全部归因于 stream。

## 5. 为什么不能宣称 stream 有收益

| 判据 | 结果 |
|---|---|
| `negative_control_match`（A/A 必须复现）| ❌ **false** |
| `checkpoint_match` | ✅ true |
| `same_hardware` / `same_run_controls` / `same_physical_routes` | ✅ true |
| `first_moe_input_match` / `first_moe_output_match` | ✅ true |
| `shared_weights_match` / `shared_branch_active` / `first_step_finite` | ✅ true |
| 105 步原生预测全一致 | ❌ false（每步 2–12 处）|

任务书 §回执 3 要求「**严格结论要求 A/A、切换前后的既有门禁及 105 步原生输出全通过**，
同时 E2E 和 MoE 都下降」。本轮 E2E 与 MoE 确实都降（−1.58% / −0.19%），
但 A/A 未通过，因此**不能下严格结论**。

E2E 的 −1.58% 在本轮的可信度低于 R5，因为：
- A/A 两个 baseline 进程的 KV 都不一样，说明执行存在**非确定性**；
- 十个十步分段虽然全部为负（−2.098 到 −0.407），
  但这份"一致为负"是在一个本身不可复现的基线上取得的。

## 6. 版本与环境

| 项 | 值 |
|---|---|
| 实测 SHA | `9db2ed969acd14a211d9c453f24a18dcf995f5cd` |
| **实际启动次数** | **2**（baseline 1 + stream 1）|
| baseline capture ID | `ffbd77773c5a481ba8e2333b0c670462` |
| stream capture ID | `20a7cf5cfde94d6b830f80f7f7521413` |
| KV 快照（两份，共 176 GiB，仅留执行机） | `/tmp/gcmoe-branch-J5iv7Nna` |
| baseline 原始目录 | `/tmp/gcmoe-aligned-DHDgVOe3` |
| stream 原始目录 | `/tmp/gcmoe-aligned-8ve807pt` |
| 工作目录 | `/tmp/gcmoe-r7-TyAPaNuT` |
| GPU | 8×H20，driver 580.105.08 |
| 软件 | cutlass 4.8.1 / torch 2.13.0+cu130 / CUDA 13.0 / vllm 0.31.0 |
| 代码修改 | **无** |

**预检**：全部 `test_*_cpu.py` PASS、`STREAM_CHECKPOINT_CUDA PASS`、
`FIRST_STEP_CUDA PASS`、**`ATTENTION_PROBE_CUDA PASS`**（经真实 vLLM
`_forward_decode` 与 CUDA Graph 校验 FlashMLA 调用边界）、
`compiled roles [1, 4, 2, 5, 3] in 19.1s`。

## 7. 交付物

| 项 | 要求 | 实际 |
|---|---:|---:|
| 40 层 counts | 80 | **80** |
| 两类 Perfetto | 80 + 22 | **80 + 22** |
| 逐 rank 行 | 640 | **642** |
| checkpoint 清单 | 16（两组各 8）| ✅ |
| `first_step_layers.csv` | 1 | **1**（2,560 行）|
| `treatment_step_layers.csv` | 1 | **1**（2,560 行）|
| `flashmla_rows.csv` | 1 | **1**（128 行）|
| `SHA256SUMS` | 通过 | **通过** |

KV 二进制与逐行 MoE 参考摘要按任务书留在执行机。
