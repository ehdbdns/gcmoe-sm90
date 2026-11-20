# REPLY_VALID_COST — 逐 valid 价格 vs 线性价格（stream 固定开启）

## 0. 判定

```json
{"cost_inputs_match":true,
 "strict_pair_comparable":true,
 "native_top1_different_of_256":0,
 "same_logical_routes":true,"ordered_stream_replay":true,
 "prediction_mae_lower":true,
 "prediction_max_error_lower":false,
 "cost_pair_faster":true,
 "lower":{"e2e_ms_per_step":true,"moe_max_rank_sum_ms":true},
 "faster_layers":21,"total_layers":40,
 "scheduling_gate_passed":false}
```

**两项主耗时都下降、预测 MAE 也下降，但最大误差上升**，所以
`scheduling_gate_passed` 为 false（判据要求 MAE 与最大误差同时改善）。

## 1. 两项主耗时

| 指标 | 线性价格 | 逐 valid 价格 | 差值 | 变化 |
|---|---:|---:|---:|---:|
| **E2E（100 步）ms/step** | **80.950** | **80.855** | **−0.096** | **−0.12%** ✅ |
| **step 2 的 40 层 MoE ms** | **33.977** | **33.961** | **−0.016** | **−0.05%** ✅ |
| step 2 开 profiler 墙钟 ms | 84.593 | 85.241 | +0.648 | +0.77% |
| 逐层（40 层） | — | — | — | **21 快 / 19 慢** |

两项都下降，但幅度极小（0.12% / 0.05%），与单个 10 步窗口的抖动同量级。

## 2. 预测精度

| 指标 | 线性价格配额 | valid 价格配额 | 变化 |
|---|---:|---:|---:|
| plan 占比 **MAE（pp）** | 0.2445 | **0.2235** | **−8.59%** ✅ |
| plan **最大误差（pp）** | 0.8915 | **1.2601** | **+41.35%** ❌ |

**MAE 明显改善 8.59%，但最大误差恶化 41%** —— 这是本轮的核心矛盾。

### 2.1 固定配额下的复算（`same_quota_scores`）

在**同一份实际配额**上用两种价格分别打分（消除拍卖影响）：

| 使用的配额 | 线性打分 | valid 打分 | 更优 |
|---|---:|---:|---|
| 线性价格产生的配额 | MAE 0.24455 / max 0.89146 | MAE 0.27154 / max 1.01720 | 线性更好 |
| valid 价格产生的配额 | MAE 0.22832 / max 1.29457 | **MAE 0.22353** / max 1.26005 | valid 更好 |

**结论清晰**：
- 在**线性配额**上，线性价格更强（0.2446 vs 0.2715）—— 符合「价格与配额匹配」的直觉；
- 在 **valid 配额**上，valid 价格更强（0.2235 vs 0.2283）—— **每种价格都最擅长评价自己产生的分配**。

这说明 valid 价格并非普遍更准，而是**与它自己产生的配额自洽**。
本轮 MAE 的改善部分来自「换了配额」，不只是「换了打分函数」。

## 3. 预测偏差变化最大的两层

### 3.1 layer 23（MAE 0.475 → 0.259，改善最大）

| rank | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 线性 tok | 199 | 201 | 190 | 189 | 197 | 184 | 195 | 179 |
| valid tok | 184 | 193 | 190 | 195 | 181 | 200 | 195 | 196 |
| 线性 M 块 | 39 | 39 | 39 | 36 | 37 | 32 | 35 | 34 |
| valid M 块 | 36 | 39 | 39 | 36 | 35 | 34 | 35 | 37 |
| **线性 err（pp）** | **+0.62** | **+0.72** | +0.28 | −0.15 | +0.29 | **−0.87** | −0.08 | **−0.80** |
| **valid err（pp）** | −0.38 | +0.41 | +0.16 | −0.03 | −0.46 | +0.08 | −0.18 | +0.38 |

线性配额下 rank 0/1 高估（+0.62/+0.72）、rank 5/7 低估（−0.87/−0.80），
极差 1.59 pp；valid 配额把这四项压到 −0.38/+0.41/+0.08/+0.38，极差降到 0.87 pp。

### 3.2 layer 17（MAE 0.402 → 0.216，改善第二）

| rank | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 线性 tok | 182 | 181 | 189 | 197 | 191 | 200 | 199 | 193 |
| valid tok | 188 | 191 | 196 | 197 | 196 | 185 | 187 | 193 |
| 线性 M 块 | 32 | 28 | 32 | 32 | 28 | 32 | 34 | 31 |
| valid M 块 | 33 | 29 | 32 | 32 | 29 | 31 | 32 | 31 |
| **线性 err（pp）** | −0.26 | **−0.89** | 0.00 | +0.37 | −0.46 | +0.48 | **+0.73** | +0.03 |
| **valid err（pp）** | +0.15 | −0.43 | +0.29 | +0.41 | −0.08 | −0.29 | +0.02 | −0.05 |

同样是把两端（rank 1 的 −0.89、rank 6 的 +0.73）收敛到 −0.43 / +0.02。

**两层共性与最大误差的来源**：valid 价格把分布抹得更平，
但个别层出现更大的单点偏差（见 §2 的最大误差 +41%）。

## 4. 严格对照项

| 判据 | 结果 |
|---|---|
| `cost_inputs_match` | ✅ true |
| `same_logical_routes` | ✅ true |
| `ordered_stream_replay` | ✅ true |
| `same_run_controls` / `same_physical_routes` | ✅ true |
| `checkpoint_match` | ✅ true |
| `first_step_chain_match` | ✅ true |
| `shared_weights_match` / `shared_branch_active` / `first_step_finite` | ✅ true |
| **`strict_pair_comparable`** | ✅ **true** |
| **原生 top-1 差异** | ✅ **0/256** |
| `last_warm_input_match` / `warm_native_match` | false（重排前预热历史不同，设计固有）|

**R8 的索引排序规范化继续有效**：本轮两臂都是 stream 开启，原生预测完全一致。

## 5. 版本与环境

| 项 | 值 |
|---|---|
| 实测 SHA | `6ed27c2fc25cf92e5d59dca32e061a07f2d8ba10` |
| **实际启动次数** | **2**（线性 1 + valid 1）|
| baseline（= stream + 线性价格）capture ID | `b960778788e745eaa1243540a9a1358f` |
| stream（= stream + valid 价格）capture ID | `eacb27e8eb2946c18603aa50f0a5cfe7` |
| baseline 原始目录 | `/tmp/gcmoe-aligned-fotyYeyl` |
| stream 原始目录 | `/tmp/gcmoe-aligned-5NL3wHaw` |
| 回放目录（R8 冻结） | `/tmp/gcmoe-route-replay-yLz4DvSN`（`workload.sha256` 逐项校验通过）|
| 工作目录 | `/tmp/gcmoe-valid-cost-sR8XkGiA` |
| GPU | 8×H20，driver 580.105.08 |
| 软件 | cutlass 4.8.1 / torch 2.13.0+cu130 / CUDA 13.0 / vllm 0.31.0 |
| 代码修改 | **无** |

**预检**：回放校验 `REPLAY_OK`、全部 `test_*_cpu.py` PASS、
`STREAM_CHECKPOINT_CUDA` / `FIRST_STEP_CUDA` / `ATTENTION_ORDER_CUDA` /
`ATTENTION_PROBE_CUDA` 全 PASS、**cost=0 与 cost=2 两种编译都通过**。

## 6. 交付物

| 项 | 要求 | 实际 |
|---|---:|---:|
| counts | 80 | **80** |
| 层总览 Perfetto | 80 | **80** |
| absolute Perfetto | 22 | **22** |
| 逐 rank 行 | 640 | **642** |
| `fixed_quota_scores.csv` | 两臂各一份 | ✅ |
| `SHA256SUMS` | 通过 | **通过** |

附 `paired.csv`、`layer_pairs.csv`、`comparison.json`、`first_step_layers.csv`。
KV 二进制与 route tape 按任务书留在执行机。

## 7. 结论

1. **两项主耗时都下降**（−0.12% / −0.05%），但幅度极小。
2. **预测 MAE 改善 8.59%**，是 valid 价格的主要收益。
3. **最大误差恶化 41%**，是主要代价；`scheduling_gate_passed = false`。
4. 固定配额复算显示：**每种价格都最擅长评价自己产生的配额**，
   说明 MAE 改善部分来自配额变化而非打分函数本身更优。
5. 每臂一次，无重复性误差界；幅度接近抖动，不宜据此宣称稳健收益。
