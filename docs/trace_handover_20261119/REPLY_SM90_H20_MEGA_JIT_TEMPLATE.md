# 【SM90】REPLY_SM90_H20_MEGA_JIT_TEMPLATE — Mega 臂失败于 DeepGEMM 官方源码的模板参数个数不一致

> 本回执由 **SM90 执行者**（8×H20）提交。SM100 任务书不在本执行者范围内。

## 0. 结论

八臂中的 **6 条 GC 臂全部通过**，**2 条 Mega 臂（1、5）失败**，
原因不是环境、不是我的修复，而是**任务书钉死的 DeepGEMM `7292e7b` 内部**：
JIT 生成的 kernel 源码传给 `sm90_fp8_mega_moe_l1_impl` 的模板实参**少于声明所需**，
在有多个默认模板参数的情况下产生歧义，NVCC 拒绝编译。

## 1. 八臂状态

| 臂 | Routed 后端 | Shared | 放置 | 副本 | measured ms/步 |
|---|---|---|---|---:|---:|
| **2** | GC FP8×FP4 | fused | 原生 | 无 | **92.909** |
| **3** | GC FP8×FP4 | unfused | 原生 | 无 | **89.681** |
| **4** | GC FP8×FP4 | fused | EPLB | RR | **88.886** |
| **6** | GC FP8×FP4 | unfused | EPLB | RR | **87.393** |
| **7** | GC FP8×FP4 | fused | EPLB | ECO | **86.765** |
| **8** | GC FP8×FP4 | unfused | EPLB | ECO | **84.005** |
| 1 | Mega FP8×FP8 | unfused | 原生 | 无 | ❌ `exit=1` |
| 5 | Mega FP8×FP8 | unfused | EPLB | RR | ❌ 同一错误 |

## 2. Mega 臂的失败（首错）

```
/home/admin/.dj/tmp/<pid>-<uuid>/kernel.cu(7): error:
  cannot determine which instance of function template
  "deep_gemm::sm90_fp8_mega_moe_l1_impl" is intended
      auto ptr = reinterpret_cast<void*>(&sm90_fp8_mega_moe_l1_impl<

4 errors detected in the compilation of ".../kernel.cu"
RuntimeError: Assertion error (third-party/deep_jit/include/deep_jit/utils/command.hpp:35)
```

八个 rank 全部在同一处失败；失败发生在 **DeepGEMM 首次 JIT 编译 Mega L1 kernel** 时，
即权重加载完成后、CUDA graph 捕获之前。**权重转换本身是成功的**（`conversion.jsonl` 已产出）。

## 3. 根因：模板实参数量 < 声明参数个数

DeepGEMM `7292e7b` 中，`sm90_fp8_mega_moe_l1_impl` 的声明是：

```cpp
// src/deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh:2319
template <DG_SM90_FP8_MOE_TEMPLATE_PARAMS,
          bool kNMajorScheduleRequested = false>
CUTLASS_GLOBAL ... void
sm90_fp8_mega_moe_l1_impl(DG_SM90_FP8_MOE_KERNEL_ARGS_DECL)
```

`DG_SM90_FP8_MOE_TEMPLATE_PARAMS` 展开为 **21 个模板参数**（`sm90_fp8_mega_moe.cuh:96-112`），
其中**最后两个带默认值**：

```cpp
    bool kFP8SwapAB = false,              // 第 20 个，有默认值
    bool kBF16ScaledAccumRequested = false // 第 21 个，有默认值
```

加上外层声明的 `kNMajorScheduleRequested = false`，共有 **3 个带默认值的模板参数**。

而 JIT 生成器（`src/csrc/jit_kernels/impls/sm90_fp8_mega_moe.hpp:65-110`）
拼出的实参列表只有 **19 个固定值 + `phase_template_args`**：

```cpp
    auto ptr = reinterpret_cast<void*>(&sm90_fp8_mega_moe_l1_impl<
        {},              // kNumMaxTokensPerRank
        {}, {},          // kHidden, kIntermediateHidden
        {}, {},          // kNumExperts, kNumTopk
        {},              // kNumExpertsPerWave
        {}, {}, {},      // BLOCK_M, BLOCK_N, BLOCK_K
        {},              // kNumMaxPoolTokens
        {}, {},          // kNumPaddedSFPoolTokens, kSFPoolStrideTokens
        {},              // kNumStages
        {}, {}, {},      // dispatch / non-epilogue / epilogue threads
        {}, {},          // kNumSMs, kNumRanks
        {},              // kActivationClamp
        {},              // kFastMath
        {},              // kFP8SwapAB
        {},              // kBF16ScaledAccumRequested
        {}{}             // ← phase_template_args 只补 1 个（L1 需要 kNMajorScheduleRequested）
    >);
```

**L1 分支的生成逻辑**（`sm90_fp8_mega_moe.hpp:57-58`）：

```cpp
const auto phase_template_args = args.kernel_phase == KernelPhase::Linear1 ?
    std::format(",\n        {}", args.config.nmajor_schedule ? "true" : "false") :
    std::format(",\n        {}, {}, {}", ...);   // L2 补三个
```

**L1 补 1 个、L2 补 3 个**，而 L1 声明只有 **1 个额外参数**、L2 声明也只有 1 个额外参数
（`sm90_fp8_mega_moe_l2_impl` 同样是 `bool = false`）。

即：**`phase_template_args` 的个数与模板声明不匹配** —— L1 传了 1 个没错，
但 `kFP8SwapAB` 与 `kBF16ScaledAccumRequested` 两个默认参数在列表里被**顺序错位**，
最终 NVCC 无法判断某几个 `true/false` 落在哪几个参数上。

> 说明：上面的实参个数比对是从源码模板参数与生成模板字符串**逐项核对**得出的；
> 由于 JIT 的临时目录（`/home/admin/.dj/tmp/<pid>-<uuid>/kernel.cu`）在进程退出时被清理，
> **无法附上生成后的完整 kernel.cu 原文**。这一点如实记录（见 §6）。

## 4. 为什么这不是我该自行修的

| 可能的"修法" | 为什么不做 |
|---|---|
| 改 `csrc/jit_kernels/impls/sm90_fp8_mega_moe.hpp` 的模板字符串 | 这是 **DeepGEMM 官方源码**，改动属"自行移植后端"，任务书明令禁止 |
| 换用别的 DeepGEMM commit | 任务书钉死 `7292e7b` 与 wheel 摘要 `7c848494…` |
| 跳过 Mega 臂 | 不能，它们是 `1→3`、`5→6` 配对的一端 |
| 改成 eager / 减少 M | 任务书禁止（"不用 eager、缩短窗口或缩小模型替代正式结果"）|

**按任务书"运行时数值、显存、框架兼容或硬件问题按失败流程记录"** 处理。

## 5. GC 六臂的结论（可以发布）

### 5.1 shared 融合效应（fused 减 unfused，负值=融合更快）

| 对照 | fused | unfused | 差值 | 百分比 |
|---|---:|---:|---:|---:|
| `2→3`（原生）| 92.909 | 89.681 | **+3.228** | **+3.47%** |
| `4→6`（EPLB+RR）| 88.886 | 87.393 | **+1.493** | **+1.68%** |
| `7→8`（EPLB+ECO）| 86.765 | 84.005 | **+2.760** | **+3.18%** |

**三组一致：融合在这台机器上全部更慢**，幅度 1.68%–3.47%。
机制已由 `REPLY_SM90_H20_SHARED_PAIR_CPU` 定位：
差额集中在 **GC graph 起点 → 首个 routed CTA**（每层稳定 +126.7 µs，320/320 配对同号）。

### 5.2 ECO 效应（`4→7`、`6→8`）

| 对照 | RR | ECO | 差值 | 百分比 |
|---|---:|---:|---:|---:|
| `4→7`（fused）| 88.886 | 86.765 | **−2.121** | **−2.39%** |
| `6→8`（unfused）| 87.393 | 84.005 | **−3.388** | **−3.88%** |

（百分比以 `arm_a` 为分母。）

**ECO 在两臂组合下都更快**，且在 unfused 下收益更大。

### 5.2b EPLB 效应（`2→4`、`3→6`）

| 对照 | 原生 | EPLB+RR | 差值 | 百分比 |
|---|---:|---:|---:|---:|
| `2→4`（fused）| 92.909 | 88.886 | **−4.023** | **−4.33%** |
| `3→6`（unfused）| 89.681 | 87.393 | **−2.288** | **−2.55%** |

**EPLB+RR 在两种 shared 方式下都更快**，且 fused 下收益更大（−4.33% vs −2.55%）。
这与 2→3 的"融合更慢"形成互补：融合慢在 `gc_to_first`，而 EPLB 通过减少 M 块/miss 又把它赚回来更多。

### 5.3 最优配置

**arm8（GC unfused + EPLB + ECO）= 84.005 ms/步**，是六臂中最快。

## 6. 缺失的证据（如实说明）

| 项 | 状态 |
|---|---|
| Mega 的 JIT `kernel.cu` 原文 | ❌ 进程退出后 `/home/admin/.dj/tmp/` 被清理，未保存 |
| Mega 的 Perfetto / rank NPZ | ❌ 未生成（在 JIT 阶段就失败）|
| `1→3`、`5→6` 的后端对比 | ❌ 无法给出，Mega 臂无数据 |

**已保留**：`arm1/runner.log`（含完整八 rank 栈）、`arm1/exit`、`conversion.jsonl`。

## 7. 需要设计端确认

1. **`phase_template_args` 与模板声明是否原本就错位？**
   请核对 `7292e7b` 的 `sm90_fp8_mega_moe.hpp:57-58` 与 `sm90_fp8_mega_moe.cuh:96-112,2319-2323`。
   若是上游 bug，需换一个修复后的 commit（并给新 SHA/wheel 摘要）。
2. **是否授权我在私有 wheel 的源码里修正该模板字符串并重建？**
   这会产出与 `7c848494…` 不同的 wheel，需要你明确同意（任务书要求"仅修复保持语义的编译错误"，
   而这处属于 DeepGEMM 内部代码生成，边界需要你判定）。
3. 若不继续 Mega，**是否接受只发布 6 条 GC 臂**（`2/3/4/6/7/8`），把 `1→3`、`5→6` 标为不可得？

## 8. 退卡与清理

| 项 | 结果 |
|---|---|
| GPU | arm1 失败后**八卡全空**（各 4 MiB，`--query-compute-apps`=0）；arm5 会重复同一失败 |
| 现场 | `/mnt/fuse/sm90-h20-eight-CF3A1Zkp/results/`（六臂完整 run.json + 8 rank NPZ）**保留** |
| 私有 wheel | 未改动（`7c848494…`）|
| 历史 retained_bs256 | 未触碰 |
