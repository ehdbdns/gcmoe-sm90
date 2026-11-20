# REPLY_MULTILAYER_AB — **失败：coarse 采集窗口覆盖了 2 个 decode step**

## 0. 首个失败点

**步骤：第 3 组进程（`step02 / baseline / coarse`）的步骤 3「capture」**
（即任务书 `run_multilayer_study.sh` 的第 3 个 matrix cell）。

```
exit_code=1
ValueError: rank 0: expected 40 of each role, got
            {'dispatch': 80, 'shared': 80, 'routed': 80, 'plan': 80, 'combine': 80}
```

发生在 `aligned_step.py:49` 的 `read_calls()`：
它要求一次 profile 窗口内每个 role 恰好 `GC_MOE_LAYERS=40` 个 kernel，
实际拿到 **80 个（= 2 个 decode step）**。按任务书 §「执行」末段
「运行、性能、解析、计数、缺数据等问题在首次失败时停下回执」，
**我已停止，未继续 step02/stream/coarse、fine 及 step 5/8/11 的任何进程。**

## 1. 已完成的部分（3/24）

| 组 | 状态 | 说明 |
|---|---|---|
| step02 baseline e2e | ✅ `exit_code=0` | 100 步，无 profiler |
| step02 stream e2e | ✅ `exit_code=0` | 100 步，无 profiler |
| **step02 baseline coarse** | ❌ **首个失败** | 采集本身跑完，解析阶段拒绝 |
| step02 stream coarse | 未运行 | |
| step02 两臂 fine | 未运行 | |
| step05 / step08 / step11 全部 | 未运行 | |

## 2. 版本与设备

| 项 | 值 |
|---|---|
| 测量代码（`MULTILAYER_AB_TASK.md` 首次加入的提交） | `bd507ef13989ef039def354cc8456e342edf42fc` |
| baseline 内核 | `98b6194a56e7468653bb6230bf0b3887c9466dce`（STREAM=0）|
| stream 内核 | `6617ee976fa86eb77eb3a34b8b6df3e60059a586`（STREAM=1） |
| 主仓库 HEAD | `bd507ef`（工作区干净，fast-forward 后） |
| GPU | 8×H20，driver 580.105.08 |
| 软件 | cutlass 4.8.1 / torch 2.13.0+cu130 / CUDA 13.0 / vllm 0.31.0 / ptxas 13.2.78 |
| 模型 | `/home/admin/local_models/DeepSeek-V4.1-Flash` |
| 数据集 | `/mnt/nas2/nanjun.cp/MoE-Bench-Experiments/datasets/dsv4-flash_online_input_ids_26-10-08_1.jsonl` |
| 工作目录 | `/tmp/gcmoe-multilayer-QOsqSAJJ` |
| 交付中间目录 | `/tmp/gcmoe-multilayer-QOsqSAJJ/delivery` |

**CPU 门禁**（`tools` 工作树，全部通过）：

```
ALIGNED_STEP_CPU PASS
PULL_GEMM_DETAIL_CPU PASS
MULTILAYER_STUDY_CPU PASS
```

**未做任何代码修改**：三个工作树都保持原 SHA，`git status` 干净。

## 3. 根因：measure 窗口跨了 2 个 `engine.step()`

### 3.1 直接证据（`/tmp/gcmoe-aligned-C909pjyw/raw/`）

```
rank0.json    : 400 个 GCMoERoleKernel = 80 per role   (= 2 × 40)
capture.invalid.json :
  layers      = 40          (解析器期望)
  first_epoch = 14681
  rank0.calls = 160         (= 4 × 40，设备快照保留的调用数)
  error       = "rank 0: expected 40 of each role, got {...80...}"
```

Kineto trace 里的 `cudaGraphLaunch` **只有 2 次**，且每次对应一整个 40 层 MoE 图：

| 事件 | ts (µs) | dur (µs) |
|---|---:|---:|
| `cudaGraphLaunch` #0 | 4304953171986.8 | 6022.2 |
| `cudaGraphLaunch` #1 | 4304953255542.6 | 4648.9 |

两个 launch 的起点相差 **83,555.8 µs（83.6 ms）**，其间 dispatch #39 → #40 的间隙是
**22.36 ms**；而同一 step 内相邻层的 dispatch 间隙中位数只有 **1.56 ms**。
81.6 µs 的两次 launch 起点的差（83.6 ms）与该 step 的 E2E 63.1 ms 同量级 ——
**这是一次「上一步还没退场、下一步已经提交」的真实跨步泄漏**，不是打点错位。

### 3.2 采集侧的 `_gc_moe_prof_stop` 顺序

`integration/deepep_compare.py` 里 `gc_moe_prof_stop` 的第一步是
`torch.cuda.synchronize()`，然后才 `__exit__`。

但**同步只覆盖"本进程此刻已入队的工作"**。`real_decode_100_100.py` 在
`measure_steps=1` 的循环里只 `engine.step()` 一次，而 vLLM 默认
`async_scheduling=True` 会**预提交一步**（这在本项目早有注释：
「`LLMEngine.step()` only *submits* work … the scheduler runs a batch ahead」）。
`_gc_moe_prof_stop` 的 `synchronize()` 等在 **GPU 侧**，
于是 profiler 的覆盖范围变成了「已入队的这一步 + 被预提交的下一步」。

这条路径上一轮（`98b6194` + `--fine-trace`，`GC_MEASURE_FLUSH=2`，
即任务书上一版的 step2）**没有**触发，同样参数下 trace 是 40 个 dispatch：

| 运行 | `GC_MEASURE_FLUSH` | `dispatch` 数 | `cudaGraphLaunch` |
|---|---:|---:|---:|
| 上轮 `LJa5C9QE`（after/baseline，step2） | 2 | 40 | 1 |
| 上轮 `7QsNdv5A`（after/stream，step2） | 2 | 40 | 1 |
| **本轮 `C909pjyw`（study coarse，step**2**）** | **2** | **80** | **2** |

两者唯一的脚本差异是 `run_aligned_ep8.sh` 第 57 行把
`GC_MEASURE_FLUSH=2` 改成了 `GC_MEASURE_FLUSH="${GC_MOE_STUDY_STEP:-2}"`。
本轮 `GC_MOE_STUDY_STEP=2`，取值相同 —— **所以不是 FLUSH 数值的问题**，
而是 coarse 变体额外设置了 `GC_MOE_LAYERS=40` + `layers=({0..39})`，
使 `aligned_step.py` 的「每 role 恰好 40 个」断言第一次真正生效：
上轮只解析 layer 8 / 20 两层，80 个 dispatch 的 trace 也能通过
（它只挑两层），本轮要全部 40 层，才暴露出窗口里其实是 2 步。

**换句话说：跨步泄漏很可能在上轮就已经存在，只是上轮的解析器按层过滤，掩盖了它。**
这是本轮最有价值的发现，也是必须停下报告而非自行绕过的原因。

### 3.3 为什么不能在执行者权限内修

任务书 §「执行」明确：执行者权限只有「运行和语义等价的最小编译修复」，
且「**禁止调整协议、门限、采样点、调度和解析口径**」。
要修这个问题必须动其中之一：

- 把 profile 窗口收紧到严格单步（需要改 `real_decode_100_100.py` 的
  `_sync()` / `prof_stop` 时序，或抑制 async scheduling 的一步预取）—— **改协议**；
- 或者放宽 `aligned_step.py` 的「每 role 40 个」断言，让它从 80 个里挑一步 —— **改解析口径**。

两者都超出授权，故停下回执。

## 4. 已产出的可用数据（step02 E2E，两臂）

这两组与失败无关，已完整落盘，可供参考：

| 臂 | 执行 SHA | 100 步 E2E | 原始目录 | status |
|---|---|---:|---|---|
| baseline | `98b6194` | **63.135 ms/step** | `/tmp/gcmoe-aligned-iu7XLXO5` | `exit_code=0` |
| stream | `6617ee9` | **60.822 ms/step** | `/tmp/gcmoe-aligned-xFrtb2In` | `exit_code=0` |

stream 比 baseline 快 **2.313 ms/step（−3.7%）**。

### 4.1 必须同时报告的两点保留意见

1. **两臂 prompt 相同，输出 hash 不同**：
   `prompt_hash` 都是 `254af9d8a2752eab68f6419f…`（说明喂的是同一批提示词），
   但 `output_hash` 分别是 `1f04f7dc7e6d04a59a217430…`（baseline）与
   `c4ccc9a87f0ee62cab7a9b79…`（stream）。**即两臂生成的 token 序列不一致。**
   任务书要求「保留每轮原始负载和生成结果摘要」，这里如实报告：
   **本轮 E2E 的 −3.7% 是在两臂输出不同的前提下取得的，不能直接当作净收益。**
2. **只听 1 个 step 的 E2E，且是第 1 个矩阵 cell**：任务书要求 4 个独立进程才构成
   主指标，现在只有 1 个（step 2），且 step 5/8/11 完全未跑。

## 5. 现场保留

| 内容 | 路径 |
|---|---|
| 失败轮完整目录 | `/tmp/gcmoe-aligned-C909pjyw` |
| ├ 8 份 Kineto trace | `raw/rank0.json … rank7.json` |
| ├ 设备快照（含失败原因） | `raw/capture.invalid.json` |
| └ 完整运行日志 | `run.log`（含 USDT profiler_start/stop 与 8 worker 导出） |
| 交付中间目录 | `/tmp/gcmoe-multilayer-QOsqSAJJ/delivery` |
| ├ 已完成 e2e 两组 | `step02/{baseline,stream}/e2e/` |
| └ 失败 coarse 的证据副本 | `step02/baseline/coarse/evidence/` |
| 三个工作树 | `/tmp/gcmoe-multilayer-QOsqSAJJ/{tools,baseline,stream}` |

GPU 已释放（8×4 MiB），无残留进程。

## 6. 建议的下一步（需任务书明确授权）

按影响面从大到小：

1. **先确认跨步泄漏是否真实存在**：用本轮保留的 `raw/rank0.json`，
   在 `aligned_step.py` 上做一次「只取前 80 个 dispatch」的离线试算，
   看 40 层的时间线是否自洽（这只需读数据，不改协议）。
2. 若确认真实存在，则需要改 **测量代码**：在 `_gc_moe_prof_start` 之前
   多一次 `_sync()` 并**确认预取队列排空**（例如 `engine.step()` 之前
   先 `_gc_moe_sync()`，或用 `VLLM_USE_ASYNC_SCHEDULING=0`），
   让 profile 窗口严格等于 1 个 decode step。
3. 若判定 2 行不通，则需要改 **解析器**：允许一个 capture 内存在
   多个完整 decode step，并按 step 边界切分（本轮 `capture.invalid.json`
   里 4 个 step 的边界已经可以从 `stamps[10]` 的 >10 ms 间隙识别：
   `[1929060, 22362, 143657] µs`）。

两项都需要新的任务书授权。当前我停在首次失败，未做任何上述改动。
