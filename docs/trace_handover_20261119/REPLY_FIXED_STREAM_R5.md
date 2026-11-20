# REPLY_FIXED_STREAM_R5 — 共同 checkpoint 下两项均降，但 `strict_pair_comparable = false`

## 0. 判定

```json
{"study":"scheduling","matched_outputs":true,
 "lower":{"e2e_ms_per_step":true,"moe_max_rank_sum_ms":true},
 "faster_layers":25,"total_layers":40,
 "placement_and_initial_state_match":true,"routes_and_all_slot_counts_match":true,
 "same_run_controls":true,"same_physical_routes":true,
 "checkpoint_match":true,"same_hardware":true,
 "first_moe_input_match":true,"first_moe_output_match":true,
 "first_moe_input_different_ranks":[],"first_moe_output_different_ranks":[],
 "last_warm_input_match":false,"warm_native_match":false,
 "native_top1_different_of_256":41,"first_native_difference_step":0,
 "fixed_workload_faster":true,
 "strict_pair_comparable":false,
 "scheduling_gate_passed":false}
```

**两项主指标本轮同时下降**（R4 是相反的），但因
`last_warm_input_match` / `warm_native_match` 为 false 且 105 步原生预测未全一致，
`strict_pair_comparable` 与 `scheduling_gate_passed` 仍为 false。

按任务书「优于、持平、退化和不可比都按实回报」，如实提交并停在复核点。

## 1. 两项主指标

| 指标 | baseline | stream | 差值 | 变化 |
|---|---:|---:|---:|---:|
| **E2E（100 步纯 decode）ms/step** | **81.762** | **81.369** | **−0.393** | **−0.48%** ✅ |
| **step 2 的 40 层 MoE ms** | **34.128** | **33.948** | **−0.180** | **−0.53%** ✅ |
| plan 占比 MAE（pp） | 0.2430 | 0.2456 | +0.0026 | +1.05% |
| plan 最大绝对误差（pp） | 0.9033 | 0.9099 | +0.0066 | +0.73% |
| step 2 开 profiler 墙钟 ms | 90.967 | 87.539 | −3.428 | −3.77% |
| 逐层（40 层） | — | — | — | **25 快 / 0 平 / 15 慢** |

**收益幅度很小**（E2E −0.48%、MoE −0.53%），与任务书要求的「明确写出哪项改善、哪项退化」
相比，本轮两项同向但幅度接近计时波动量级（见 §3 分段差）。

## 2. 十个十步的 E2E 分段差（任务书要求）

单位 ms，正数 = stream 更慢：

| 段 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Δ ms | +1.019 | +0.491 | −0.861 | −0.302 | **+1.647** | −1.189 | −1.314 | −1.216 | −0.904 | −1.302 |

- 合计 **−3.93 ms**，均值 **−0.393 ms/段**，与整窗口 −0.393 ms/step 一致。
- **前两段和第五段为正**（stream 更慢），后五段全部为负。
- **段间极差 2.96 ms**（+1.647 到 −1.314），远大于整体均值 0.393 ms，
  说明单次 10 步窗口的噪声已经覆盖了本轮的整体收益。

**这直接支持「收益接近计时波动」的判断。**

## 3. 共同 checkpoint 验证（R5 的核心新增）

| 控制项 | 结果 |
|---|---|
| `checkpoint_match` | ✅ **true** |
| `same_hardware` | ✅ **true**（`hardware.csv` 记录 8 卡 UUID/PCI/驱动/频率，两臂逐项一致）|
| `first_moe_input_match` | ✅ **true**，不同 rank 列表为空 |
| `first_moe_output_match` | ✅ **true**，不同 rank 列表为空 |
| `same_run_controls` | ✅ true |
| `same_physical_routes` | ✅ true |
| `placement_and_initial_state_match` | ✅ true |
| `routes_and_all_slot_counts_match` | ✅ true |
| `matched_outputs` | ✅ true |

**R4 的 `last_warm_input_match = false` 缺陷已被本轮修复路径覆盖**：
R5 在重排后从一份共同 KV checkpoint 分叉，第一个 flush 步的
layer 0 FP8 输入 / FP32 尺度 / BF16 输出与调用 epoch **两臂逐位相同**
（`first_moe_input_different_ranks = []`、`first_moe_output_different_ranks = []`）。

**但 `last_warm_input_match` / `warm_native_match` 仍为 false** ——
这两项衡量的是**重排之前 warmup 历史**的摘要，两臂各自跑完 200 步自由生成预热，
本来就不会逐位相同。checkpoint 是在重排后、切换 stream 之前拍的，
所以它保证了「起点一致」，但不保证「历史一致」。

### 3.1 原生预测差异

| 项 | 值 |
|---|---|
| 采样步 `native_top1_different_of_256` | **41** |
| 最早分歧步 | **step 0** |
| 105 步逐步差异 | 16–41（见 `comparison.json` 的 `native_differences_by_step`）|

任务书要求「105 步原生预测全部一致」才可判 `strict_pair_comparable`，
本轮每步都有 16–41 处不同，未通过。

## 4. 各层快慢

**25 层变快 / 0 层持平 / 15 层变慢**。

| 变化最大 | 层 | baseline µs | stream µs | Δ |
|---|---|---:|---:|---:|
| 改善 | **6** | 940.5 | **838.6** | **−101.9** |
| 改善 | 34 | 898.5 | 879.7 | −18.8 |
| 改善 | 25 | 859.4 | 841.5 | −17.9 |
| 退化 | **35** | 856.1 | **878.1** | **+22.0** |
| 退化 | 10 | 831.4 | 847.3 | +15.8 |
| 退化 | 26 | 848.5 | 863.4 | +14.9 |

### 4.1 layer 6（−101.9 µs）：combine 收尾大幅缩短

| 段 | baseline | stream | Δ |
|---|---:|---:|---:|
| pull token | 294 | 306 | +12 |
| ECO plan | 124 | 122 | −2 |
| GEMM | 601 | 588 | −13 |
| **combine** | **133** | **45** | **−88** |

与 R3（−105）和 R4（−105）的 layer 6 表现高度一致：**逐块发号让 combine 收尾显著变短**。

### 4.2 layer 35（+22.0 µs）：同一段的镜像

| 段 | baseline | stream | Δ |
|---|---:|---:|---:|
| pull token | 266 | 278 | +12 |
| GEMM | 621 | 616 | −5 |
| **combine** | **30** | **58** | **+28** |

同样由 combine 主导，方向相反。幅度（+28 µs）不到 layer 6 收益（−88 µs）的三分之一。

## 5. 版本与环境

| 项 | 值 |
|---|---|
| 实测 SHA（两臂同工作树） | `f301f053105923b812ad169bc7cbd6a6ba29ef74` |
| **实际启动次数** | **2**（baseline 1 + stream 1，符合预算）|
| baseline capture ID | `e56436c9e8e74113886fbc66a15ac1d3` |
| stream capture ID | `00fc182573554531b935b0a6fe902e71` |
| baseline 原始目录 | `/tmp/gcmoe-aligned-8BZ9sZmE` |
| stream 原始目录 | `/tmp/gcmoe-aligned-UFmmLhFi` |
| 共享路由 | `/tmp/gcmoe-route-replay-Q6W0KywL` |
| KV checkpoint（88 GiB 二进制，仅留执行机） | `/tmp/gcmoe-branch-1ocv7xeQ` |
| 工作目录 | `/tmp/gcmoe-r5-1i4orPQi` |
| GPU | 8×H20，driver 580.105.08，500 W / 1980 MHz / 2619 MHz（`hardware.csv`）|
| 软件 | cutlass 4.8.1 / torch 2.13.0+cu130 / CUDA 13.0 / vllm 0.31.0 |
| 代码修改 | **无**（预检全过，未做兼容性修复）|

**预检**（`preflight.log`）：全部 CPU 测试 PASS、`STREAM_CHECKPOINT_CUDA PASS`
（小张量 CUDA 存取）、`compiled roles [1, 4, 2, 5, 3] in 18.2s`。

## 6. 首个失败命令 / 分歧

- **无运行失败**：两臂均 `exit_code=0`，`SHA_OK`。
- **首个原生分歧**：**step 0**，具体请求见 `comparison.json` 的 `native_differences_by_step`。

## 7. 交付物（规格全部达标）

| 项 | 要求 | 实际 |
|---|---:|---:|
| 层总览 | 80 | **80** |
| counts | 80 | **80** |
| absolute Perfetto | 22 | **22** |
| 逐 rank 数据（两臂各 320 行） | 640 | **642**（含表头）|
| checkpoint 清单 | 8 | **8**（`delivery/checkpoint/rank0..7.json`）|
| 硬件记录 | — | `hardware.csv` + `gpu-before/after.csv` ✅ |
| `SHA256SUMS` | 通过 | **通过**（228 条）|

附 `paired.csv`、`layer_pairs.csv`、`comparison.json`、`workload.json`（两臂）、
`placement.json`、`workload.sha256`、`cold-replay-check.log`、`preflight.log`。

**88 GiB KV 二进制按任务书要求只在执行机保留**（`/tmp/gcmoe-branch-1ocv7xeQ`），
GitHub 只回传八份清单与摘要。

## 8. 限制

1. **每臂一次，无重复性误差界**。−0.48% / −0.53% 的幅度与 §2 的段间极差
   （2.96 ms，是均值的 7.5 倍）相比很小，**不宜据此宣称稳定收益**。
2. **「不可比」**：`last_warm_input_match` 与 `warm_native_match` 为 false，
   因为两臂的 200 步自由生成预热历史不同。R5 用 checkpoint 保证了**重排后起点一致**，
   但无法让预热历史一致 —— 这是该设计的固有边界，不是缺陷。
3. **层与 rank 不是独立重复**：40 层来自同一次 capture。
4. **`plan` 误差本轮反而略升**（+1.05% MAE），与主指标方向不同，按任务书分别报告。
5. 本结论只覆盖本次回放负载、固定先后顺序、每臂一次。
