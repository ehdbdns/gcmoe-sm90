# 【SM90】REPLY_MAIN24_PREFIX_CAPACITY — 在 random 臂的 prefill 原生校验处停止

> 本回执由 **SM90 执行者**（8×H20）提交。SM100 任务书不在本执行者范围内。

## 0. 首个失败

**阶段**：`step=3: capture`，第 2 个臂 `BS=256 / random` 的 **prefill 阶段原生校验**。
**exit_code = 1**（`exit-code.txt` 记录 `1`）。

```
ValueError: prefill: 190 native top-1 mismatches;
  first relative offset 0, req109, position 0, native 671, reference 2581
```

八卡全部以同一信息失败（`Worker_TP0..TP7`），日志中共 17 处该断言记录。
调用栈定位：`integration/long_cost.py:103 begin()`
→ `ref.verify_native_prefix("prefill", ...)`
→ `integration/fixed_tokens.py:57`。

按任务书 §执行「首次差异或资源错误即停止，保留日志和现场」，
**我已停止**，第 3–6 个臂（rr / eco_linear / eco / eco_smr）**均未启动**。

## 1. 已完成的臂

| 臂 | 状态 | E2E ms/步 |
|---|---|---:|
| **vanilla** | ✅ 完成（1000 统计 + 1000 正式） | **91.173** |
| **random** | ❌ prefill 原生校验失败 | — |
| rr / eco_linear / eco / eco_smr | 未启动 | — |

**实际模型启动数：2**（预算 6）。

### 1.1 失败请求

| 项 | 值 |
|---|---|
| 请求号 | **req109** |
| 位置 | position 0（relative offset 0）|
| native | **671** |
| reference | **2581** |
| 该阶段失配数 | **190 / 256** |

失配在 prefill 的第一个 token 位置即出现，且 256 个请求中有 190 个不同 ——
这不是个别请求的抖动，而是**整批 prefill 的计算与参考轨迹系统性不一致**。

## 2. 容量数据（任务书 §验收 1 要求，取自已完成的 vanilla 臂）

### 2.1 显存（八 rank 完全一致）

| 项 | 数值 |
|---|---|
| 每卡总显存 | 102,085,492,736 B = **95.07 GiB** |
| **最低剩余显存** | 24,600,969,216 B = **22.91 GiB** |
| **最高 PyTorch reserved** | 66,337,112,064 B = **61.78 GiB** |
| `GC_MOE_KV_BYTES` | 21,474,836,480 B = **20.00 GiB** |

> 说明：`free` 为任务书要求的「设备剩余显存」读数，**包含 CUDA allocator 之外的占用**
> （NCCL/NVSHMEM 对称堆、CUDA context、驱动预留等），因此它与
> `95.07 − 61.78 = 33.29 GiB` 的差额约 10.4 GiB 来自这些非 allocator 部分。

### 2.2 vLLM 公布的容量

| 项 | 数值 |
|---|---|
| **GPU KV cache size** | **1,530,922 tokens** |
| Maximum concurrency（按 4,643 token 序列）| 329 |

与任务书 §512/1024/2048 容量依据中引用的
「每卡 20 GiB KV、1,530,922 个 token」**完全一致**，说明该基线数据在本机成立。

### 2.3 每步 token 数校验

vanilla 臂的 `stats` 与 `measured` 两个窗口均通过「每步 256 个生成 token」的检查
（否则计时段会被判为「shrinking batch」而报错）。该臂 1000 步正常完成。

## 3. 为什么这构成必须停止的「首次差异」

任务书 §验收 2 规定：

> 只有输入、路由、快照读回、**预热及正式原生输出全部通过**才能发布严格对比。

random 臂的失败发生在 **prefill 阶段**（尚未进入 1000 步统计窗口），
属于任务书列举的「首次差异」，且影响的是**整个臂的可比性**：
若 prefill 轨迹已经分叉，后续的反馈 token 与路由回放都不再是同一份工作量。

因此：
- 不继续运行其余四个臂（它们同样依赖 prefill 原生校验通过）；
- 不把 vanilla 的 91.173 ms 与任何其他臂做对比（只有 1 个臂有数据）；
- 严格对比**无法发布**。

## 4. 版本与环境

| 项 | 值 |
|---|---|
| 测量 SHA | `9093d90029d61293a1a07291b86c6bf1e594a568` |
| 任务书 | `docs/trace_executor_20261119/MAIN24_PREFIX_CAPACITY_TASK.md`（【SM90】）|
| 工作目录 | `/tmp/gcmoe-prefix-LjI6nKxm` |
| vanilla Capture workspace | 见 `delivery/bs256/vanilla/runner.log` 的 `Capture workspace:` 行 |
| **random Capture workspace** | **`/tmp/gcmoe-aligned-EVc9Wfoc`** |
| GPU | 8×H20，driver 580.105.08 |
| 软件 | cutlass 4.8.1 / torch 2.13.0+cu130 / CUDA 13.0 / vllm 0.31.0 |
| 代码修改 | **无**（预检一次通过，未做编译修复）|
| 环境变量 | `GC_MAIN24_SINGLE_BS=256`、`GC_MOE_KV_BYTES=21474836480` |

**预检**：CPU 测试全部 PASS、**六种角色组合编译全部通过**
（含 `[1,4,2,5,6,3]`）、八卡 SM-reuse 门禁两侧 SHA 逐 rank 相同。
注意本版角色 grid 为 **72**（此前为 71），是调度改动，非异常。

## 5. 现场保留

| 内容 | 路径 |
|---|---|
| 失败臂完整日志 | `delivery/bs256/random/runner.log`（含 17 处断言与完整调用栈）|
| 失败臂 status | `delivery/bs256/random/status.txt`（`step=3: capture`，`exit_code=1`）|
| random Capture workspace | `/tmp/gcmoe-aligned-EVc9Wfoc`（status / failed-command / run.log / warm.log）|
| vanilla 完整产物 | `delivery/bs256/vanilla/`（run.json + 8 rank NPZ + checkpoint/prefill）|
| 退出码 | `exit-code.txt` = **1** |
| 预检与进度日志 | `preflight.log`、`progress.log` |

## 6. 需要设计端确认的事项

1. **random 臂为何在 prefill 第一个位置就分叉**：
   该臂与 vanilla 共享同一份 prompt、同一份参考 token，但走随机放置 + RR 而非原生槽。
   若参考轨迹是在 vanilla 的放置下生成的，那么 random 放置下 prefill 的原生 top-1
   本就**不应**与参考一致 —— 需要确认 `verify_native_prefix` 对 random 臂
   是否应当以「自身放置下的首轮输出」为基准，而不是与 vanilla 的参考比较。
2. 若第 1 点成立，则该断言对 random/rr 这类**改变放置**的臂可能需要不同的基准；
   这属于判据口径，超出执行者权限，故停下回传。
3. vanilla 臂的容量数据（§2）完整可用，可直接用于 §512/1024/2048 的规划核算。

## 7. 交付物

本次为失败交付，目录为
`docs/trace_executor_20261119/main24_prefix_capacity_results_blocked/`，
包含 `delivery/`（vanilla 完整产物 + random 失败日志）、`preflight/`、
`exit-code.txt`、`preflight.log`、`progress.log` 及自有 `SHA256SUMS`。
二进制 KV 与路由回放文件按任务书留在执行机现场。
