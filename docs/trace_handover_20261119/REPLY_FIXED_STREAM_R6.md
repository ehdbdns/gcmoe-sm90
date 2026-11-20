# REPLY_FIXED_STREAM_R6 — 共享专家修复生效（原生差异 41→7），首个分歧定位到 layer 2

## 0. 判定

```json
{"study":"scheduling","matched_outputs":true,
 "lower":{"e2e_ms_per_step":true,"moe_max_rank_sum_ms":false},
 "faster_layers":14,"total_layers":40,
 "checkpoint_match":true,"same_hardware":true,
 "first_moe_input_match":true,"first_moe_output_match":true,
 "shared_weights_match":true,"shared_weight_differences":[],
 "shared_branch_active":true,"first_step_finite":true,"nonfinite_examples":[],
 "first_step_chain_match":false,
 "native_top1_different_of_256":7,"first_native_difference_step":0,
 "fixed_workload_faster":false,
 "strict_pair_comparable":false,
 "scheduling_gate_passed":false}
```

**本轮的核心成果是定位，不是性能**：共享专家输入的修复把采样步原生差异从
**41/256 降到 7/256**，并把首个数值分歧精确定位到 **layer 2 的 `hidden_input`**。

## 1. 两项主指标（方向相反）

| 指标 | baseline | stream | 差值 | 变化 |
|---|---:|---:|---:|---:|
| **E2E（100 步）ms/step** | **82.506** | **82.243** | **−0.263** | **−0.32%** ✅ |
| **step 2 的 40 层 MoE ms** | **34.170** | **34.203** | **+0.034** | **+0.10%** ❌ |
| plan 占比 MAE（pp） | 0.2446 | 0.2574 | +0.0128 | +5.24% |
| plan 最大绝对误差（pp） | 0.9162 | 0.9630 | +0.0467 | +5.10% |
| step 2 开 profiler 墙钟 ms | 85.176 | 86.153 | +0.976 | +1.15% |
| 逐层（40 层） | — | — | — | **14 快 / 26 慢** |

MoE 只差 **0.034 ms（+0.10%）**，在噪声内；E2E −0.32% 同样很小。
按任务书「两项分别报告，无需凑正向结果」，如实并列。

## 2. 十个十步的 E2E 分段差

单位 ms，正数 = stream 更慢：

| 段 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Δ ms | +0.180 | +0.104 | −0.172 | **+3.559** | **−3.263** | −0.211 | −0.621 | −1.041 | −0.485 | −0.684 |

合计 **−2.63 ms**，均值 −0.263 ms/段。
**第 4、5 段出现 +3.559 / −3.263 的对冲**，极差 6.82 ms —— 远超整体 0.263 ms 的收益，
说明本轮 E2E 差值完全被单段抖动主导。

## 3. 共享专家修复的效果

| 项 | R5 | **R6** |
|---|---:|---:|
| `shared_weights_match` | — | ✅ **true** |
| `shared_weight_differences` | — | **`[]`**（空）|
| `shared_branch_active` | — | ✅ **true**（各层 `shared_input` / `shared_y` 非零）|
| `first_step_finite` | — | ✅ **true**，`nonfinite_examples = []` |
| 采样步原生差异 | 41/256 | **7/256** |

**修复确实让共享分支从"清零缓冲区"变成"真实激活"**：`shared_branch_active = true`
表示每层每条分支都有非零输入输出，`shared_weights_match` 表示两臂实际共享权重一致。
原生差异从 41 降到 7，说明该修复消除了大部分（83%）分歧来源。

## 4. 首个分歧精确定位：layer 2 的 `hidden_input`

`first_step_layers.csv` 共 2,560 行（40 层 × 8 rank × 8 stage/field），
其中 **2,432 行 `matching = False`**，按层分布：

| layer | 0 | 1 | **2** | 3–39 |
|---|---|---|---|---|
| 不符行数 | **0** | **0** | **64** | 每层 64 |

**layer 0 与 layer 1 完全一致**，从 **layer 2 起**每一层、每一个 rank、每一个阶段全不符。

### 4.1 layer 2 各字段的逐行差异（八个 rank 完全一致）

| 阶段 | 字段 | 不同行数（每 rank）|
|---|---|---:|
| input | **`hidden_input`** | **108** ← 最早分歧 |
| quantization | `quantized` | 108 |
| quantization | `scales` | 72 |
| quantization | `shared_input` | 108 |
| quantization | `shared_scales` | 4 |
| experts | `routed_slots` | 648 |
| experts | `shared_y` | 108 |
| combine | `output` | 108 |

**八个 rank 的数值完全相同**（108/108/108/108/108/108/108/108），
说明这是一个**与 rank 无关的、确定性的传播**，不是某个 rank 掉队。

### 4.2 定位结论

- **边界**：layer 1 的输出 / layer 2 的输入。
- layer 2 的 `hidden_input` 已经不同，而 layer 1 的 `output` 相同，
  所以分歧**发生在 layer 1 的前向到 layer 2 的输入之间**（或 layer 1 自身的
  combine / residual 路径），不在更早的量化或路由环节。
- 受影响行数 108，而 routed 槽位差异为 648（= 108 行 × 6 top-k），
  与「108 个 token 的路由结果不同」自洽 —— 即**分歧先出现于 hidden 状态，
  再通过 router 传导到槽位选择**。

按任务书，这只是**定位线索**；根因仍需相应数据支持，本轮不做推断。

### 4.3 七个原生差异请求

| request | baseline | stream |
|---|---:|---:|
| req44 | 58 | 51 |
| req63 | 18 | 35 |
| req89 | 89 | 73 |
| req135 | 6196 | 35 |
| req166 | 29577 | 18 |
| req217 | 35 | 37 |

（`comparison.json` 列出 6 条示例，第 7 条见原始 `first_step_native_differences` 完整列表。）

## 5. 仍未通过的门禁

| 判据 | 结果 |
|---|---|
| `first_step_chain_match` | **false**（layer 2 起全不符）|
| 105 步原生预测全一致 | **false**（每步 0–11 处，总计 2,782 − 大幅下降但未归零）|
| `last_warm_input_match` / `warm_native_match` | false（重排前预热历史不同，设计固有）|
| 两项主耗时同时下降 | **false**（MoE +0.10%）|

`strict_pair_comparable` 与 `scheduling_gate_passed` 均为 false。

## 6. 版本与环境

| 项 | 值 |
|---|---|
| 实测 SHA | `d8168d4893d90df0df9748b563747f94ce06ad21` |
| **实际启动次数** | **2**（baseline 1 + stream 1）|
| baseline capture ID | `958511b99d12471794fbfce851ec2ac1` |
| stream capture ID | `36dfea8f42904c2fa1a900542bc85374` |
| baseline 原始目录 | `/tmp/gcmoe-aligned-NMq56y4O` |
| stream 原始目录 | `/tmp/gcmoe-aligned-5ZrkPUmz` |
| KV checkpoint（89 GiB，仅留执行机） | `/tmp/gcmoe-branch-Ds1Gi4OC` |
| 工作目录 | `/tmp/gcmoe-r6-3gF32SKv` |
| GPU | 8×H20，driver 580.105.08 |
| 软件 | cutlass 4.8.1 / torch 2.13.0+cu130 / CUDA 13.0 / vllm 0.31.0 |
| 代码修改 | **无** |

**预检**：全部 `test_*_cpu.py` PASS（含新增 `FIRST_STEP_CPU`）、
`STREAM_CHECKPOINT_CUDA PASS`、`FIRST_STEP_CUDA PASS`、
`compiled roles [1, 4, 2, 5, 3] in 19.4s`。

## 7. 交付物

| 项 | 要求 | 实际 |
|---|---:|---:|
| 层总览 Perfetto | 80 | **80** |
| counts | 80 | **80** |
| absolute Perfetto | 22 | **22** |
| 逐 rank 行 | 640 | **642** |
| `first_step_layers.csv` | 1 | **1**（2,560 行）|
| checkpoint 清单 | 8 | **8** |
| `SHA256SUMS` | 通过 | **通过** |

附 `paired.csv`、`layer_pairs.csv`、`comparison.json`、`workload.json`（两臂）、
`placement.json`、`workload.sha256`、`cold-replay-check.log`、`preflight.log`。
逐行哈希与 89 GiB KV 二进制按任务书留在执行机。

## 8. 限制

1. **每臂一次，无重复性误差界**。E2E −0.32% 与 MoE +0.10% 都落在抖动范围内。
2. **本轮的共享专家修复改变了被测模型的计算**，因此跨轮耗时（R5→R6）不可直接归因于 stream。
3. **首个分歧定位到 layer 2 的 hidden_input 是观测边界**，不是已证实的根因。
4. 层与 rank 不是独立重复；结论限于本次回放负载、固定先后顺序、每臂一次。
