# REPLY_FIXED_STREAM_R8 — **严格对照首次全部通过**，但 E2E 略升

## 0. 判定

```json
{"negative_control_match":true,
 "strict_pair_comparable":true,
 "scheduling_gate_passed":false,
 "fixed_workload_faster":false,
 "lower":{"e2e_ms_per_step":false,"moe_max_rank_sum_ms":true},
 "faster_layers":24,"total_layers":40,
 "index_order":{"control_match":true,"treatment_match":true,
                "compressed_layers":[2..39]},
 "native_top1_different_of_256":0,
 "first_native_difference_step":null}
```

**这是 R3 以来第一次 `strict_pair_comparable = true`**：A/A 复现、索引顺序对齐、
原生预测零差异全部达成。**但 E2E 略微上升，所以 `scheduling_gate_passed` 仍为 false。**

按任务书「若严格对照通过但一项或两项耗时未下降，把相应观测作为正式结果交付」。

## 1. 两项主指标（一升一降）

| 指标 | baseline | stream | 差值 | 变化 |
|---|---:|---:|---:|---:|
| **E2E（100 步）ms/step** | **82.615** | **82.723** | **+0.108** | **+0.13%** ❌ |
| **step 2 的 40 层 MoE ms** | **34.178** | **34.050** | **−0.128** | **−0.37%** ✅ |
| plan 占比 MAE（pp） | 0.2453 | 0.2442 | −0.0011 | −0.43% |
| plan 最大绝对误差（pp） | 0.9197 | 0.8917 | −0.0280 | −3.05% |
| step 2 开 profiler 墙钟 ms | 100.627 | 114.297 | +13.670 | +13.58% |
| 逐层（40 层） | — | — | — | **24 快 / 16 慢** |

**MoE 下降 0.37%，E2E 上升 0.13%** —— 两项为本轮单次配对观测。

## 2. 十个十步的 E2E 分段（关键证据）

单位 ms，正数 = stream 更慢：

| 段 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Δ ms | −1.016 | −0.130 | +0.355 | **+4.706** | **+1.769** | −0.052 | −1.626 | +0.058 | −1.768 | −1.211 |

- 各栏为该 10 步窗口的每步均值差。全 100 步累计相差 **+10.847 ms**，折合 **+0.108 ms/步**。
- 第 4 段相差 **+4.706 ms/步**。十段中有六段 stream 更快、四段更慢。
- 段间极差 **6.47 ms**，远大于整体差值 0.108 ms。

第 4 段对全窗口均值影响最大。现有计时仅记录十步边界，无法识别该段是哪一步或哪项工作耗时增加。正式判定使用完整 100 步。

## 3. 严格对照项：全部通过 ← 本轮最大成果

| 判据 | 结果 |
|---|---|
| `checkpoint_match` | ✅ true |
| `same_hardware` | ✅ true |
| `same_run_controls` | ✅ true |
| `same_physical_routes` | ✅ true |
| `matched_outputs` | ✅ true |
| `first_moe_input_match` / `first_moe_output_match` | ✅ true |
| `shared_weights_match` | ✅ true（差异列表空）|
| `shared_branch_active` | ✅ true |
| `first_step_finite` | ✅ true |
| **`first_step_chain_match`** | ✅ **true**（R7 为 false）|
| **`negative_control_match`** | ✅ **true**（R7 为 false）|
| **`index_order.control_match`** | ✅ **true** |
| **`index_order.treatment_match`** | ✅ **true** |
| **原生 top-1 差异** | ✅ **0/256**（R7 为 5）|
| `first_native_difference_step` | **null**（无分歧）|

**R7 的两个失败项（A/A 不可复现、layer 2 分歧）在本轮全部消失。**
索引排序规范化解决了 R7 定位到的 `extra_order` 问题。

## 4. 索引顺序的行为（`compressed_index_order.csv`）

608 行 = 2 步（0、1）× 38 个压缩层（2–39）× 8 rank。

| 字段 | 结果 |
|---|---|
| `native_order_match` | **608/608 为 False** —— 每组至少有一个请求的排序前序列不同|
| **`effective_order_match`** | **608/608 为 True** —— 实际送入 FlashMLA 的序列**完全一致** |
| `lengths_match` | **608/608 为 True** |
| 重排行数 | baseline **153,920** = stream **153,920**（两臂做了完全相同量的重排）|

step 0 / layer 2 示例：八个 rank 各 250 行被重排，两臂数字一致。

**这正是本轮设计的核心**：原生顺序不同（不可控的非确定性），
但规范化后实际消费顺序相同（可控），因此 A/A 与 A/B 都能逐位对齐。

## 5. 各层快慢

**24 层变快 / 16 层变慢**。

| 变化最大 | 层 | baseline µs | stream µs | Δ |
|---|---|---:|---:|---:|
| 改善 | **6** | 942.2 | **835.2** | **−107.1** |
| 改善 | 4 | 873.9 | 853.6 | −20.3 |
| 改善 | 34 | 900.3 | 881.9 | −18.3 |
| 退化 | **8** | 821.2 | **846.5** | **+25.3** |
| 退化 | 11 | 844.0 | 866.5 | +22.5 |
| 退化 | 24 | 821.2 | 842.0 | +20.8 |

layer 6 的最慢 rank 用时下降 107 µs。R4–R7 在这一层也观察到下降，之前各轮的严格可比条件与 R8 有差异。

## 6. 版本与环境

| 项 | 值 |
|---|---|
| 实测 SHA | `4c8b7cab73cf9ac31c66778892dc549cddc7703a` |
| **实际启动次数** | **2**（baseline 1 + stream 1）|
| baseline capture ID | `688ce0bfca9a468f9af27d6db7af6aa5` |
| stream capture ID | `07d59042d158463684522654a667ac37` |
| baseline 原始目录 | `/tmp/gcmoe-aligned-IZ8QSJgs` |
| stream 原始目录 | `/tmp/gcmoe-aligned-BRzEs4nW` |
| KV 快照（170 GiB，仅留执行机） | `/tmp/gcmoe-branch-lCLJj6uC` |
| 工作目录 | `/tmp/gcmoe-r8-aAooUIKU` |
| GPU | 8×H20，driver 580.105.08 |
| 软件 | cutlass 4.8.1 / torch 2.13.0+cu130 / CUDA 13.0 / vllm 0.31.0 |
| 代码修改 | **无** |

**预检**：全部 `test_*_cpu.py` PASS、`STREAM_CHECKPOINT_CUDA PASS`、
`FIRST_STEP_CUDA PASS`、**`ATTENTION_ORDER_CUDA PASS`**、`ATTENTION_PROBE_CUDA PASS`、
`compiled roles [1, 4, 2, 5, 3] in 19.0s`。

## 7. 交付物

| 项 | 要求 | 实际 |
|---|---:|---:|
| counts | 80 | **80** |
| 层总览 Perfetto | 80 | **80** |
| absolute Perfetto | 22 | **22** |
| 逐 rank 数据行 | 640 | **640**（每臂 320 行，两份 CSV 各另有一行表头） |
| checkpoint 清单 | 16（两组各 8）| **16** |
| `compressed_index_order.csv` | 1 | **1**（608 行）|
| `SHA256SUMS` | 通过 | **通过** |

附 `flashmla_rows.csv`、`first_step_layers.csv`、`treatment_step_layers.csv`、
`paired.csv`、`layer_pairs.csv`、`comparison.json`、`workload.json`、`placement.json`。
KV 二进制保留在执行机。

## 8. 结论与限制

1. **严格对照首次全部通过**，说明索引排序规范化成功消除了 R7 的非确定性来源。
2. **MoE 下降 0.37%**（逐层 24 快 16 慢），方向明确。
3. **本次 100 步 E2E 均值上升 0.13%**。每臂一次，差值能否复现仍未知。
4. **`scheduling_gate_passed` 仍为 false**，因为判据要求两项同时下降。
   按任务书，这是正式结果：**严格可比已达成，性能方向未双双为负。**
5. 每臂一次，无重复性误差界。本轮新增的排序开销计入两臂 E2E，
   所以绝对值不能直接和 R3–R7 比较。

## 发布者复核与第 20 层 trace

已校验结果包 SHA256SUMS，并从两臂原始审计记录重新生成比较结果。105 步共 26,880 个原生 top‑1 全部相同。step 2 配对的 320 组（层、rank）逐槽负载一致。逐 rank 将 40 层从 graph 起点到 combine 结束的跨度相加，再取八卡最大值，得到 baseline 34.177632 ms（rank 0）和 stream 34.049579 ms（rank 2）。端到端使用完整的 100 步计时。两项联合提速判据未通过，按任务约定保留本轮结论。[两项耗时](../trace_executor_20261119/fixed_stream_results_r8/paired.csv) · [严格对照](../trace_executor_20261119/fixed_stream_results_r8/comparison.json)

| step 2 / layer 20 | torch profiler Perfetto | absolute Perfetto | 各 rank token 与 M 块 |
|---|---|---|---|
| baseline | [总览](https://github.com/ehdbdns/gcmoe-sm90/blob/9ca7ec9/docs/trace_executor_20261119/fixed_stream_results_r8/step02/baseline/paired/step002_layer20_epoch14501.perfetto.json) | [CTA 细节](https://github.com/ehdbdns/gcmoe-sm90/blob/9ca7ec9/docs/trace_executor_20261119/fixed_stream_results_r8/step02/baseline/paired/step002_layer20_epoch14501.absolute.perfetto.json) | [counts](https://github.com/ehdbdns/gcmoe-sm90/blob/9ca7ec9/docs/trace_executor_20261119/fixed_stream_results_r8/step02/baseline/paired/step002_layer20_epoch14501.counts.json) |
| stream | [总览](https://github.com/ehdbdns/gcmoe-sm90/blob/9ca7ec9/docs/trace_executor_20261119/fixed_stream_results_r8/step02/stream/paired/step002_layer20_epoch14501.perfetto.json) | [CTA 细节](https://github.com/ehdbdns/gcmoe-sm90/blob/9ca7ec9/docs/trace_executor_20261119/fixed_stream_results_r8/step02/stream/paired/step002_layer20_epoch14501.absolute.perfetto.json) | [counts](https://github.com/ehdbdns/gcmoe-sm90/blob/9ca7ec9/docs/trace_executor_20261119/fixed_stream_results_r8/step02/stream/paired/step002_layer20_epoch14501.counts.json) |
