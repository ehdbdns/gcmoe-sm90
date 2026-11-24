# 【SM90】REPLY_SM90_H20_RETRY_MEGA_1_5 — Mega JIT 编译已修复，两臂在 KV 分配处 **OOM**

> 本回执由 **SM90 执行者**（8×H20）提交。SM100 任务书不在本执行者范围内。

## 0. 结论

| 项 | 结果 |
|---|---|
| 任务 SHA | `a5c75567218c11d9026a34c76d80e4b45599f911` |
| 测量 SHA（钉死）| `14d0e097ea8bfd6438f8fd3a6e47b45cd220d552` |
| **Mega JIT 编译问题** | ✅ **已修复**（L1/L2 两个真实 cubin 均生成成功）|
| **arm1（Mega 原生）** | ❌ `exit=1`，**OOM**（wall 393 s）|
| **arm5（Mega + EPLB + RR）** | ❌ `exit=1`，**OOM**（wall 361 s）|
| GPU | 已释放，八卡全空 |

**上一轮的模板歧义问题被彻底解决了**（见 §1），
但两臂在 **KV cache 分配**处因显存不足失败。这是任务书 §"失败、回传和清理" 预警过的容量问题。

## 1. 编译侧：模板歧义已消除（本轮最大进展）

### 1.1 两阶段预检全部通过

```
nvcc: NVIDIA (R) Cuda compiler driver, release 13.2, V13.2.78
selected-cxx: .../toolchain/bin/x86_64-conda-linux-gnu-g++
              x86_64-conda-linux-gnu-g++ (conda-forge gcc 15.3.0-7) 15.3.0

l1.cubin  75,608 B  ✅
l2.cubin  82,776 B  ✅
```

`DG_JIT_NVCC_COMPILER` 包装器（`scripts/sm90/mega_nvcc.sh`）只对含 SM90 Mega 头文件的编译
追加 `-ccbin <GCC 15.3>`，让 NVCC 用与 wheel 一致的 host 编译器解析模板，
**模板参数个数与声明不再错位**。日志确认：

```
Mega JIT preflight passed: /usr/local/cuda/bin/nvcc with .../x86_64-conda-linux-gnu-g++
```

`l1.log` 中可见 `sm90_fp8_mega_moe_l1_impl<...>` 正常实例化（不再是 "cannot determine which instance"）。

### 1.2 空跑阶段已验证

进入模型后 GPU 一度占用 **95,349 MiB/卡**，说明 Mega routed kernel 已在设备上执行，
**没有再出现上一轮的 `kernel.cu(7): error`**。

## 2. 运行时失败：KV cache 分配 OOM

### 2.1 首错（两臂一致）

```
File ".../vllm/v1/worker/gpu_worker.py", in initialize_from_config
    self.model_runner.initialize_kv_cache(kv_cache_config)
File ".../gpu_model_runner.py", in initialize_kv_cache
    kv_caches = allocate_kv_cache(...)
File ".../kv_cache.py", in allocate_kv_cache
    buf = torch.zeros(buf_size, dtype=torch.int8, device=device)
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 GiB.
  GPU <N> has a total capacity of 95.07 GiB of which 3.00 GiB is free.
  Including non-PyTorch memory, this process has 92.00 GiB memory in use.
  Of the allocated memory 79.38 GiB is allocated by PyTorch,
  and 2.54 GiB is reserved by PyTorch but unallocated.
```

| 项 | 值 |
|---|---|
| 失败阶段 | **`initialize_kv_cache`**（权重加载完成之后、CUDA graph 捕获之前）|
| rank | 8/8 全部 |
| 每卡总容量 | 95.07 GiB |
| 请求分配 | **20.00 GiB**（`GC_MOE_KV_BYTES`，与六条 GC 臂相同）|
| 已用（含非 PyTorch）| 92.00 GiB |
| PyTorch 已分配 | **79.38 GiB** |
| 剩余 | **3.00 GiB** |

### 2.2 与任务书的容量预警一致

任务书 §"失败、回传和清理" 原文：

> 每卡 20 GiB KV 外，Mega 的 40 层 routed FP8 payload 在每 rank 48/56 槽下约占 **63.3/73.8 GiB**，
> 另需 dense 权重、对称缓冲区与 CUDA Graph。显存不足的臂标记 OOM，不填入性能值。

**实测 PyTorch 分配 79.38 GiB**，与"73.8 GiB routed payload + dense/对称缓冲"的量级吻合。
95.07 − 79.38 = 15.69 GiB 可用，而 KV 要 20 GiB → **缺 4.3 GiB**。

### 2.3 为什么六条 GC 臂能跑而 Mega 不能

| 项 | GC 臂（FP8×FP4）| Mega 臂（FP8×FP8）|
|---|---|---|
| routed 权重精度 | FP4（每元素 4 bit）| **FP8（每元素 8 bit）** |
| 40 层 routed payload | 小 | **约 2 倍**（63.3/73.8 GiB）|
| 实测 PyTorch 已分配 | — | **79.38 GiB** |

FP8 权重是 FP4 的两倍大，这是 Geometry 决定的，无法通过调度规避。

## 3. 交付物

`docs/trace_executor_20261119/sm90_h20_bs256_mega_gc_eight_results/mega_retry/`：

| 文件 | 内容 |
|---|---|
| `retry.log` | 完整启动日志（wheel 校验、JIT 预检、两臂调度）|
| `arm1-runner.log` / `arm5-runner.log` | 两臂完整栈（含 8 rank 的 OOM）|
| `arm1-exit.txt` / `arm5-exit.txt` | `1` / `1` |
| `wall-seconds.txt` | arm1 **393 s**、arm5 **361 s** |
| `nvcc.txt` / `selected-cxx` / `cxx.txt` | NVCC 13.2.78 + GCC 15.3.0 路径与版本 |
| `l1.log` / `l2.log` | 两阶段 cubin 编译日志 |
| `l1.cubin` / `l2.cubin` | **两个真实 cubin**（75,608 / 82,776 B）—— 编译侧成功的直接证据 |
| `mega-conversion.jsonl` | 真实第 0 层三投影转换（w1/w3/w2 的 fp4/scale/fp8/block-scale 摘要与误差）|

### 3.1 逐臂明细（任务书 §回执 1）

| 项 | arm1 | arm5 |
|---|---|---|
| exit | 1 | 1 |
| execution.sha | `14d0e097…` | `14d0e097…` |
| wall-seconds | **393** | **361** |
| 8 rank NPZ | ❌ 未产出（OOM 早于测量）| ❌ |
| run.json | ❌ | ❌ |
| installed-layout | ❌ | ❌ |
| runner.log | ✅ | ✅ |
| 模型加载秒数 | 在 wall 内，未单列（失败早于计时）| 同 |
| 正式测量秒数 | 0（未进入）| 0 |

## 4. `1→3`、`5→6` 配对（任务书 §回执 2）

**无法给出**：Mega 臂无 `run.json`，`report_eight.py` 对两臂均报
`No such file or directory: .../arm{1,5}/run.json`。

参照的 GC unfused 对照（本轮已完成）：

| 臂 | 配置 | measured ms/步 |
|---|---|---:|
| 3 | GC unfused · 原生 | **89.681** |
| 6 | GC unfused · EPLB+RR | **87.393** |

## 5. 需要设计端决定（唯一阻塞）

OOM 有四种可能出路，**都需要你们授权，我不擅自改**：

| 方案 | 做法 | 影响 |
|---|---|---|
| **A. 降低 Mega 臂的 KV** | 只把 Mega 两臂的 `GC_MOE_KV_BYTES` 从 20 GiB 降到 ~14 GiB | 破坏"八臂共同的 KV 预算"（任务书要求保持相同），且 1000 步 × BS256 需要约 15 GiB，可能不够 |
| **B. 减少 Mega 的槽数** | 原生 48 → 更少 | 改变实验语义（放置规模）|
| **C. 提高 GPU 可用显存** | 降低 `GPU_UTIL` 之外的非 PyTorch 占用（NCCL/NVSHMEM 对称堆）| 需要重新标定，属环境变更 |
| **D. 放弃 Mega 两臂** | 只发布六条 GC 臂 | 失去 `1→3`、`5→6` 的后端对比 |

**我的判断**：任务书要求"保持八臂共同的 KV 预算、模型、精度与 batch"，因此 A/B 都违反约束；
D 最省事但损失信息；C 需要重新标定显存。**建议 A 的变体**：若设计端允许 Mega 臂单独用较小 KV
并显式标注"该臂 KV 与其他臂不同"，则可以在同一台机器跑通。

## 6. 退卡与清理

| 项 | 结果 |
|---|---|
| GPU | 八卡全空（各 4 MiB，`--query-compute-apps`=0）|
| 现场 | `/mnt/fuse/sm90-h20-eight-CF3A1Zkp/results/`（六臂完整 + 两臂日志）保留 |
| JIT 缓存 | `results/mega-jit-1O900sW1/` 保留（含 cubin 与日志，作为编译成功证据）|
| 私有 wheel | 未改动（`7c848494…`）|
| 六臂产物 | 未触碰 |
