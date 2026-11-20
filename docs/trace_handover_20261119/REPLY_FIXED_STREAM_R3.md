# REPLY_FIXED_STREAM_R3 — 固定负载回放：`fixed_workload_faster = true`，门禁因原生 top-1 差异为 false

## 0. 判定结果

```json
{"study":"scheduling",
 "matched_outputs":true,
 "lower":{"e2e_ms_per_step":true,"moe_max_rank_sum_ms":true},
 "faster_layers":27,"total_layers":40,
 "placement_and_initial_state_match":true,
 "routes_and_all_slot_counts_match":true,
 "native_top1_different_of_256":60,
 "fixed_workload_faster":true,
 "scheduling_gate_passed":false}
```

- **`fixed_workload_faster = true`** —— 反馈序列、路由、布局、ECO 初始状态与实际逐槽工作量**全部一致**，
  且 E2E 与 40 层 MoE **均下降**。
- **`scheduling_gate_passed = false`** —— 唯一不满足的是 `native_top1_different_of_256 = 60`
  （要求在采样步上为 0）。其余五项判据全部通过。

按任务书 §回执 4「其余结果也应原样提交」，如实回执并停在此处。

## 1. 主体指标

| 指标 | baseline | stream | 差值 | 变化 |
|---|---:|---:|---:|---:|
| **E2E（100 步固定负载回放）ms/step** | **83.124** | **81.403** | **−1.722** | **−2.07%** |
| **40 层 MoE `moe_max_rank_sum_ms`** | **33.976** | **33.726** | **−0.250** | **−0.73%** |
| plan 占比 MAE（pp） | 0.2460 | 0.2447 | −0.0013 | −0.52% |
| plan 最大绝对误差（pp） | 0.9076 | 0.9005 | −0.0071 | −0.78% |
| step 2 开 profiler 墙钟 ms | 85.302 | 91.889 | **+6.587** | **+7.72%** |
| 逐层（40 层） | — | — | — | **27 快 / 0 平 / 13 慢** |

**注意 `profiled_step_ms` 反向（+7.72%）**：这是**开 profiler 那一步**的墙钟，受插桩扰动，
与主指标（无 profiler 的 100 步窗口）不是同一口径。主指标 E2E 与 40 层 MoE 方向一致向下。

## 2. R2 首步缺口复算（任务书 §如何固定负载 3 要求）

`cold-replay-check.log` 用真实路由重算旧四副本循环对 step 0 各层的预测缺口，
与 R2 实测**逐层 40/40 完全吻合**：

```
R2 step 0: matching layers=40/40 predicted_missing=54016 observed_missing=54016
```

逐层明细（节选，完整 40 行在日志里）：

| layer | 预测缺口 | R2 实测缺口 |
|---:|---:|---:|
| 0 | 0 | 0 |
| 1 | 384 | 384 |
| 2 | 2368 | 2368 |
| 4 | 1856 | 1856 |
| 11 | 1984 | 1984 |
| 19 | 2304 | 2304 |
| 28 | 2368 | 2368 |
| 39 | 1408 | 1408 |

**40 层全部相符，总缺口 54,016 行**。这确证了 R2 的根因：
旧 plan 用实际副本数做除法，但冷启动发放配额只遍历前四个副本——
单专家 8 个活跃副本、1,024 行时只发出 768 行。

R3 修复后该缺口消失：两臂的 `routes_and_all_slot_counts_match = true`，
`placement_and_initial_state_match = true`，105 × 40 步每步每层八卡合计均为 12,288 行。

## 3. 两臂一致性核对

| 核对项 | 结果 |
|---|---|
| `prompt_hash` | 两臂均为 `254af9d8a2752eab68f6419f085c1973…` ✅ |
| `output_hash` | 两臂均为 `de4457bd3e6e405052ddc22fdf5a466a…` ✅ |
| 反馈 token 序列 | 完全一致 ✅ |
| 路由 `digest(ids)` / `digest(weights)` | 等于 R2 参考哈希 ✅ |
| EPLB 布局 | 两臂安装同一布局 ✅ |
| ECO 初始状态 | 两臂一致 ✅ |
| 逐槽 owned 行数（105 步 × 40 层） | 两臂逐格一致 ✅ |

**本轮的两臂生成摘要完全匹配**——R2 那次 `output_hash` 不同的情况已不再出现。

### 3.1 采样步原生 top-1 的 60 处差异

两臂的 `native_top1`（替换前的原生 greedy 输出）逐 rank 逐请求比对，
**256 个请求中有 60 个不同**：

```
baseline: [91, 36, 29577, 18, 18, 29577, 29577, 36, 18, 22, 6196, 38, ...]
stream  : [91, 36,    81, 18, 18, 14283, 29577, 35, 18, 22, 6196, 40, ...]
```

差异索引：`2, 5, 7, 11, 21, 26, 29, 33, 37, 57, 58, 59, 60, 65, 66, …`
（部分请求一致，部分 token id 不同，如 idx2 的 `29577` vs `81`）。

这只影响 **gate 判定**，不影响负载：回放时两臂都被强制反馈同一参考 token，
路由与逐槽行数完全一致，所以 E2E 与 MoE 的对比仍是同一份工作量的对比。

## 4. 变化最大的层：load 完全相同，差异纯来自调度

**关键前提**：本轮两臂的 token 与 M 块**逐 rank 完全相同**（固定负载生效），
所以任何层的差异都不是负载变化造成的。

### 4.1 改善最大：layer 6（−120.4 µs）

| | baseline | stream |
|---|---:|---:|
| max-rank MoE | 952.9 µs | **832.5 µs** |
| 逐 rank tokens | 183,183,186,198,201,193,193,199 | **完全相同** |
| 逐 rank M 块 | 34,34,32,32,32,33,32,32 | **完全相同** |
| 八 rank MoE 极差 | 3 µs | 3 µs |

分段（八 rank 均值）：

| 段 | baseline | stream | Δ |
|---|---:|---:|---:|
| count exchange | 23 | 24 | +1 |
| pull token | 295 | 311 | +16 |
| shared L1+L2 | 194 | 194 | 0 |
| ECO plan | 126 | 126 | 0 |
| wait token | 4 | 2 | −2 |
| GEMM | 597 | 582 | −15 |
| **combine** | **151** | **45** | **−106** |

**收益几乎全部来自 combine 段（151→45 µs）**，其次 GEMM 小幅下降 15 µs。
pull 段反而慢了 16 µs。也就是说：逐块发号让 combine 的收尾等待大幅缩短，
这与「每个 M 块各自就绪、不再等整槽」的预期一致。

### 4.2 退化最大：layer 11（+22.5 µs）

| | baseline | stream |
|---|---:|---:|
| max-rank MoE | 848.4 µs | 870.8 µs |
| 逐 rank tokens | 186,196,189,196,195,188,196,190 | **完全相同** |
| 逐 rank M 块 | 34,36,34,31,36,31,37,36 | **完全相同** |

分段：GEMM 595→594（−1）、ECO plan 128→127（−1）、pull 240→244（+4）、
**combine 47→69（+22）**。同样是 combine 主导，只是方向相反。
相对 83 ms 的 E2E，+22.5 µs 属单层波动；40 层里 27 层变快、13 层变慢，
净效果是 −2.07%。

## 5. 版本与环境

| 项 | 值 |
|---|---|
| 测量代码 / 两臂内核 SHA | `6c87beab382a0befce0c9befa6ce62bddeece473`（同工作树）|
| baseline capture ID | `fe69f9b60a124b8d8df21f38f3b9b115` |
| stream capture ID | `4c83080b4da042cfa4d4aca799bdc8fc` |
| baseline 原始目录 | `/tmp/gcmoe-aligned-k4FgGnDK` |
| stream 原始目录 | `/tmp/gcmoe-aligned-Ey6ls2yY` |
| 共享路由文件 | `/tmp/gcmoe-route-replay-Q6W0KywL`（复用 R2，tokens/placement/routes 全部经 `cmp` 与摘要校验）|
| 工作目录 | `/tmp/gcmoe-two-arm-gRn1CmYC` |
| GPU | 8×H20，driver 580.105.08 |
| 软件 | cutlass 4.8.1 / torch 2.13.0+cu130 / CUDA 13.0 / vllm 0.31.0 / ptxas 13.2.78 |
| 模型 | `/home/admin/local_models/DeepSeek-V4.1-Flash` |
| 数据集 | `/mnt/nas2/nanjun.cp/MoE-Bench-Experiments/datasets/dsv4-flash_online_input_ids_26-10-08_1.jsonl` |
| 模型启动次数 | **2**（baseline 1 + stream 1，符合上限）|
| 代码修改 | **无**（五门禁全过，未做任何兼容性修复）|

**五项 CPU 门禁**（模型启动前）：

```
ALIGNED_STEP_CPU PASS
PULL_GEMM_DETAIL_CPU PASS
MULTILAYER_STUDY_CPU PASS
FIXED_WORKLOAD_CPU PASS
ECO_COPIES_CPU PASS
```

## 6. 交付物（全部达标）

| 项 | 要求 | 实际 |
|---|---:|---:|
| `run.json` | 2 | **2** |
| 层总览 Perfetto | 80 | **80** |
| counts | 80 | **80** |
| absolute Perfetto JSON | 22 | **22** |
| 逐 rank 数据行 | 640 | **642**（640 + 2 表头）|
| `SHA256SUMS` | 通过 | **通过** |

附：`workload.json`（两臂各一份）、`placement.json`、`workload.sha256`、
`cold-replay-check.log`、`paired.csv`、`layer_pairs.csv`、`comparison.json`。

两臂各 11 份 absolute 覆盖层 0/4/8/12/16/20/24/28/32/36/39，
step 2 layer 20 的 absolute JSON 路径见交付目录。

## 7. 限制（任务书 §回执 3 要求写明）

1. **每臂只有一次独立运行**，无重复性误差界。−2.07% 与 −0.73% 是否超出计时波动，本轮无法自证。
2. **层与 rank 不是独立重复**：40 层来自同一次 capture，逐层差值只用于解释机理。
3. **`native_top1_different_of_256 = 60`** 是本轮唯一的门禁障碍；它不改变负载对比的有效性，
   但按判据必须为 0 才算通过。
4. **`profiled_step_ms` 与主指标反向**：它是开 profiler 那一步的墙钟，不宜作为性能结论依据。
5. 本轮 E2E 含两臂**相同的回放与记录开销**，绝对值不能直接和自由生成实验（如 R1 的 84.7 ms）比较。
