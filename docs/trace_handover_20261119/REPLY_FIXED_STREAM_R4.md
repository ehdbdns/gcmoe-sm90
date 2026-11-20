# REPLY_FIXED_STREAM_R4 — **两项主指标方向相反**，门禁未通过

## 0. 判定

```json
{"study":"scheduling","matched_outputs":true,
 "lower":{"e2e_ms_per_step":true,"moe_max_rank_sum_ms":false},
 "faster_layers":9,"total_layers":40,
 "placement_and_initial_state_match":true,
 "routes_and_all_slot_counts_match":true,
 "same_run_controls":true,"same_physical_routes":true,
 "last_warm_input_match":false,"warm_native_match":false,
 "native_top1_different_of_256":51,
 "first_native_difference_step":0,
 "fixed_workload_faster":false,"scheduling_gate_passed":false}
```

- **`fixed_workload_faster = false`**：E2E 下降但 40 层 MoE **上升**，两项主指标方向相反。
  按任务书 §交付和判定 4「两个指标方向相反时分别报告」，这里如实并列。
- **`scheduling_gate_passed = false`**：`last_warm_input_match` / `warm_native_match` 均为 false，
  且回放 105 步的原生预测未全一致（每步 21–51 处差异，最早分歧在 step 0）。

## 1. 主体指标（方向相反）

| 指标 | baseline | stream | 差值 | 变化 |
|---|---:|---:|---:|---:|
| **E2E（100 步）ms/step** | **83.362** | **82.200** | **−1.162** | **−1.39%** ✅ 下降 |
| **40 层 MoE `moe_max_rank_sum_ms`** | **34.142** | **34.523** | **+0.382** | **+1.12%** ❌ 上升 |
| plan 占比 MAE（pp） | 0.2449 | 0.2508 | +0.0059 | +2.41% |
| plan 最大绝对误差（pp） | 0.9313 | 0.8969 | −0.0344 | −3.69% |
| step 2 开 profiler 墙钟 ms | 93.806 | 87.007 | −6.800 | −7.25% |
| 逐层（40 层） | — | — | — | **9 快 / 0 平 / 31 慢** |

**这是与 R3 相反的结论**（R3 是 27 快 / 13 慢、两项都降）。R4 的 ECO 对齐与「同内核 + 设备开关」
改动让 **31 层变慢**，只有 9 层变快。

## 2. 负载与路由控制项：全部一致 ✅

| 控制项 | 结果 |
|---|---|
| `matched_outputs` | true |
| `prompt_hash` / `output_hash` | 两臂完全相同（`de4457bd3e6e…`） |
| `same_run_controls` | **true** |
| `placement_and_initial_state_match` | **true** |
| `routes_and_all_slot_counts_match` | **true** |
| `same_physical_routes` | **true** |
| 105 × 40 每步每层合计 | 12,288 行 ✅ |

所以下面的层差异**纯粹来自调度**，不是负载变化。

## 3. 变化最大的层

### 3.1 退化最大：layer 3（+210.7 µs）

| | baseline | stream |
|---|---:|---:|
| max-rank MoE | 861.0 µs | **1071.7 µs** |
| 逐 rank tokens | 191,177,192,193,210,197,182,194 | **完全相同** |
| 逐 rank M 块 | 36,37,34,33,32,34,39,36 | **完全相同** |

分段（八 rank 均值）：

| 段 | baseline | stream | Δ |
|---|---:|---:|---:|
| pull token | 253 | 263 | +10 |
| shared L1+L2 | 195 | 194 | −1 |
| ECO plan | 122 | 123 | +1 |
| wait token | 4 | 2 | −2 |
| GEMM | 602 | 629 | +27 |
| **combine** | **52** | **231** | **+179** |

**退化几乎全部来自 combine（52→231 µs，+179 µs）**，GEMM 也涨了 27 µs。
八个 rank 的 MoE 耗时从 856–861 µs 一致地涨到 1061–1072 µs（极差仅 11 µs），
说明这是**整层的系统性问题**，不是个别 rank 掉队。

### 3.2 改善最大：layer 6（−111.9 µs）

| | baseline | stream |
|---|---:|---:|
| max-rank MoE | 949.3 µs | **837.4 µs** |
| 逐 rank tokens / M 块 | 完全相同 | 完全相同 |

分段：GEMM 602→589（−13）、**combine 142→37（−105）**、pull 297→307（+10）。
与 R3 的 layer 6 表现一致（那次 −120 µs，同样是 combine 137→45 量级）。

**同一条 combine 段，在 layer 6 缩短 105 µs、在 layer 3 延长 179 µs。**
这说明 R4 的改动让 combine 的收尾行为在不同层之间变得不稳定，
而不是整体收紧。

## 4. 预热观测不一致（门禁的另一处失败）

| 项 | 结果 |
|---|---|
| `last_warm_input_match` | **false** |
| `warm_native_match` | **false** |

八卡最后一次 warmup 的 MoE 输入摘要（含量化尺度）两臂不同。
按任务书 §交付和判定 4，`scheduling_gate_passed` 要求「预热观测一致」，
这一条没有满足。

## 5. 原生预测差异

| 项 | 值 |
|---|---|
| 采样步 `native_top1_different_of_256` | **51** |
| 最早分歧步 | **step 0** |
| 105 步逐步差异计数 | 21–51（见 `comparison.json` 的 `native_differences_by_step`） |

与 R3 的 60 处同量级。任务书要求「回放全部 26,880 个原生预测一致」才通过，
本轮每步都有 20–50 处不同，差距明显。

## 6. 版本与环境

| 项 | 值 |
|---|---|
| 实测 SHA（两臂同工作树） | `f73ceaab70edd67003eb8c9eb1bac81c1c57ae69` |
| baseline capture ID | `95ac5e5558234d9b8110d3347f32360d` |
| stream capture ID | `8bf114577132406c8bcffc85017c569d` |
| baseline 原始目录 | `/tmp/gcmoe-aligned-y1PND6AF` |
| stream 原始目录 | `/tmp/gcmoe-aligned-7CB1IfDk` |
| 共享路由 | `/tmp/gcmoe-route-replay-Q6W0KywL` |
| 工作目录 | `/tmp/gcmoe-r4-6LZQBbgk` |
| **实际模型启动次数** | **2**（baseline 1 + stream 1，符合上限） |
| GPU | 8×H20，driver 580.105.08 |
| 软件 | cutlass 4.8.1 / torch 2.13.0+cu130 / CUDA 13.0 / vllm 0.31.0 / ptxas 13.2.78 |
| 代码修改 | **无**（预检全过，未做兼容性修复） |

**预检**（`preflight.log`）：

```
ALIGNED_STEP_CPU PASS
PULL_GEMM_DETAIL_CPU PASS
MULTILAYER_STUDY_CPU PASS
FIXED_WORKLOAD_CPU PASS
ECO_PORT_CPU PASS
ECO_COPIES_CPU PASS
STREAM_MBLOCK_CPU PASS
compiled roles [1, 4, 2, 5, 3] in 19.1s
```

## 7. R2 首步缺口复算（任务书要求）

`cold-replay-check.log`：

```
R2 step 0: matching layers=40/40 predicted_missing=54016 observed_missing=54016
```

逐层 40/40 吻合，复现了 R2 的 ECO 配额缺陷。

## 8. 交付物（规格全部达标）

| 项 | 要求 | 实际 |
|---|---:|---:|
| `run.json` | 2 | **2** |
| 层总览 Perfetto | 80 | **80** |
| counts | 80 | **80** |
| absolute Perfetto | 22 | **22** |
| 逐 rank 行 | 640 | **642** |
| `SHA256SUMS` | 通过 | **通过**（`SHA_OK`）|

附 `workload.json`（两臂）、`placement.json`、`workload.sha256`、`cold-replay-check.log`、
`paired.csv`、`layer_pairs.csv`、`comparison.json`、`preflight.log`。

## 9. 限制

1. **每臂一次，无重复性误差界**。E2E −1.39% 与 MoE +1.12% 是否超出计时波动，本轮无法自证；
   但两者方向相反这一点是明确的。
2. **层与 rank 不是独立重复**：40 层来自同一次 capture，逐层差值只用于解释机理。
3. **E2E 下降、MoE 上升**：E2E 是 100 步纯 decode 的墙钟（含回放审计开销），
   MoE 是 step 2 单层的设备侧跨度。两者口径不同，方向相反时需要分开解读，
   不能合并成一个「更快/更慢」的结论。
4. **预热输入摘要不一致**，说明两臂在 warmup 结束时的设备状态就有差别，
   这会削弱「同负载」这一前提在预热阶段的严格性（重排后的 105 步回放负载本身是一致的）。
