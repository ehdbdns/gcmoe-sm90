# [SM100] Reply - V41 L20 host/device cause

执行者：SM100。任务书：`docs/sm100_megamoe/V41_L20_HOST_DEVICE_CAUSE.md`。
代码 SHA：`f2230f1fe39cb6b652e865bd9eddc40052d25e85`。结果目录：
`docs/sm100_megamoe/v41_l20_host_device_results/`。

## 三问（微秒）

1. **L19 combine 结束后，晚 rank 的第一个额外等待**
   - **Mega rank 5**：`all_gather -> cublas_tss`，晚 rank 设备 gap **124.6084 us**。
     其中 host 尚未调用 API **112.8906 us**，`cuLaunchKernelEx` 自身 **7.7412 us**，
     API 返回后设备再等 **3.9766 us**。早 rank 同一边界设备 gap 约 3.0 us，新增约
     **121.6 us**。调用链：`aten::empty -> aten::mm -> cuLaunchKernelEx(corr=11910) ->
     cublas_tss`，后继 CPU op `aten::linear`。
   - **GC rank 2**：`mhc -> all_gather`，晚 rank gap **19.3594 us**，早 rank 0.1279 us，
     新增 **19.2314 us**。host 未进入 API **9.5195 us**，`cudaLaunchKernelExC`
     **6.0850 us**，设备尾等 **3.7549 us**。调用链：
     `aten::empty -> _C_custom_ar::mnnvl_lamport_all_gather -> cudaLaunchKernelExC(corr=11851)
     -> all_gather`。GC 随后 `all_gather -> cublas_tss` 再增加 165.4102 us。

2. **最后一次 reduce-scatter 重新对齐后，重新造成 L20 入口差的调用**
   - 最后 reduce-scatter 结束时，入口差已回到 **5.265 us (Mega) / 4.655 us (GC)**。
   - Mega：`reduce_scatter -> mhc` +73.1201，`mhc -> mega_gate` +103.1680，
     `mega_gate -> prepare` +106.3359，`prepare -> memcpy` +22.6250，
     `memcpy -> fill` +15.0713，`fill -> mega_moe` +18.3369 us。
   - GC：`reduce_scatter -> mhc` +73.8574，`mhc -> mega_gate` +81.2490，
     `mega_gate -> prepare` +122.4307，`prepare -> memcpy` +26.3047，
     `memcpy -> fill` +25.8555，`fill -> gc_moe` +25.8555 us。
   - 六段 post-RS gap 合计 **338.6572 / 355.5527 us**；其中 API 前
     **292.4131 / 291.6172 us**（86% / 82%）。API 自身和返回后的设备排队仅数微秒到二十微秒。

3. **早 rank 的 count exchange 发生在晚 rank 进入 L20 前的部分**
   - **Mega**：早 rank 7 交换区间 **366.0498 us**，其中 **346.0918 us（94.5%）**
     在晚 rank 5 进入前；19.9580 us 重叠或在其后。
   - **GC**：早 rank 4 交换区间 **368.6094 us**，其中 **357.8574 us（97.1%）**
     在晚 rank 2 进入前；10.7520 us 重叠或在其后。

两臂最终 L20 入口差：Mega **347.1162 us**（L19 出口差 0.9912），GC
**357.8896 us**（L19 出口差 1.3232）。全窗口最大单个新增 gap 是
`q_kv_norm -> dense_gemm`：Mega **276.2900 us**、GC **290.1790 us**；其 host 未进入
API 分别为 266.2119 / 277.4804 us。该 gap 位于层内 attention 路径，不属于 post-RS 六段表。

## 闭合与归因

- Mega：window delta 347.0459 = busy delta -1895.9424 + gap delta 2242.9883 us，
  closure error **0.0000**，41 个 gap 全匹配。
- GC：window delta 358.9863 = busy delta -2610.6729 + gap delta 2969.6592 us，
  closure error **0.0000**，42 个 gap 全匹配。
- 正 gap 不能直接相加等于入口差；同流 collective/重叠会让 late rank 的设备 busy
  窗口同步缩短，正负项和残差见 `closure.csv`。
- 两臂**机制同因、首个调用不同**：都是最后 reduce-scatter 对齐后，在
  `mhc/gate/prepare/memcpy/fill/graph` 多处重新累积，新增时间主要在 host 尚未进入
  后继 CUDA API 前。Mega 首个新增等待更大（124.6 us），GC 首个只有 19.2 us，但 GC
  后续 `all_gather -> cublas_tss` 达 165.4 us，最终入口差略大。

## 无 Kineto step 4/6 复核

| 臂 | step | 入口差 us | 早/晚 rank | 早交换 us | 晚交换 us |
|---|---:|---:|---|---:|---:|
| Mega | 1 | 347.1162 | 7/5 | 366.0498 | 18.5918 |
| Mega | 4 | 263.9951 | 2/7 | 281.6318 | 17.7598 |
| Mega | 6 | 289.5518 | 4/5 | 306.9121 | 17.3115 |
| GC | 1 | 357.8896 | 4/2 | 368.6094 | 8.5439 |
| GC | 4 | 266.5117 | 4/5 | 277.6963 | 9.9199 |
| GC | 6 | 273.0605 | 7/1 | 283.1367 | 8.9922 |

step 4/6 关闭 Kineto forward 后仍在 264--290 us（Mega）和 267--281 us（GC），且早晚
rank 轮换，排除固定 rank 绑定和仅 step 1 现象。step 1 加 CPU activity 后相对旧无 CPU
trace 基线，Mega 291.8848 -> 347.1162 us，GC 492.6123 -> 357.8896 us；方向相反，无统一
膨胀量。逐项见 `step146_compare.csv`。

## 性能工具、时钟与缺口

- `perf_event_paranoid=2`；`cpu-clock:u` 199 Hz，记录 **2,219,443** 样本、
  726.078 MB；16 个 worker 主 PID/TID 全部在记录中，每个主 TID 26,957--42,598 个样本，
  见 `perf_pid_tid_coverage.csv`。
- 关键 gap 只有 17--125 us，短于 5.03 ms 采样周期，因此 gap 内通常 0 个栈样本；
  不以空栈推断原因。
- 关键 gap 内 **0 个 SWITCH 记录**，前后分别为 SWITCH IN 与 SWITCH OUT；Mega 的 OUT
  为 preempt（之后可运行），GC 的 OUT 为阻塞，但都发生在 gap 之后。故关键 gap 内线程
  处于 running，无 runnable/blocked 区间；直接记录见 `perf_switch_evidence.csv`。
- 时钟：step 1 两个 live anchor，边界半宽 <=0.8155 us (Mega) / 0.7200 us (GC)；
  host 提交拟合最大残差 37.634 us (Mega rank 5) / 8.380 us (GC)。step 4/6 用单个
  after anchor fallback，半宽 0.864 / 0.9765 us，仅作同量级复核。
- 无 kernel 栈、无 off-CPU 栈；不能把 API 前的 100 us 级 CPU 段进一步归因到缺页、锁
  竞争或 OS 抢占。现有直接证据只支持“host 线程在运行、尚未调用 API”。
- 时间线净差无未解释项（closure error 0）；未解释的是 pre-API 段内部的具体 Python/C++
  语句耗时，不是入口差本身。

## 交付物

`v41_l20_host_device_results/` 包含：

- `README.md`：完整方法、表格、限制和复现命令。
- `mega_step001_l19_l20.perfetto.json`、`gc_step001_l19_l20.perfetto.json`：两臂 8 rank
  公共时轴（CPU op/API、设备 stream、collective、MoE、count exchange、SWITCH 调度轨、样本点）。
- `arm_summary.csv`、`l20_arrival_and_exchange.csv`、`exchange_intersection.csv`、
  `boundary_alignment.csv`、`call_chain.csv`、`gap_attribution.csv`、
  `key_call_answers.csv`、`closure.csv`、`perf_coverage.csv`、
  `perf_pid_tid_coverage.csv`、`perf_switch_evidence.csv`、
  `step146.csv`、`step146_compare.csv`、`analyze_l20_host_device.py`。

## 原始证据、环境、退出码、GPU 释放

- 原始路径：`/data0/home/admin/workspace/megamoe/v41-l20-host-device-T2qg4jm8/run`；
  清单 `raw_evidence_paths.txt`，SHA256 `raw_evidence.sha256`（114 条）。
- 关键 SHA：Mega `profile-rank0.json` =
  `b3ad60347ff4934055f00c71c4defd5e1bb30be2c7cb1982a696a4f8dc5c94c2`；
  GC `profile-rank0.json` =
  `9f0a658290e8d6dbb2c6a486348f7f65847aa8bd4e75eca48ceed671b3add259`；
  `worker-sched.data` =
  `86fb8dde4a8f24f741531a60c9b8a6cf4d7912306a4a45615ca3b81a685f2635`；
  `worker-sched.txt` =
  `ccfd35ea3a3ef7e66ff36087df927ae46a476c653378d1491e5ee319de76e1f2`。
- 环境：torch 2.13.0，vLLM 0.31.0 wheel，nvidia-cutlass-dsl 4.8.1，CUDA 13.5 ptxas；
  完整版本、GPU UUID、perf 权限见 `environment.txt`。
- 退出码：Mega start 14:16:01 / end 14:28:49 / exit **0**；GC start 14:28:51 /
  end 14:36:43 / exit **0**。组锁 acquired 14:15:50，released 14:36:46；
  释放后 `nvidia-smi --query-compute-apps` 为空，见 `gpu_release.txt`。

## 清理

分析完成后清理本次私有代码 worktree、DSL 环境、overlay、DG cache、临时脚本和锁；
保留 `run/bs512` 原始 trace/result/stats、perf 数据与命令日志供复查。共享仓库和已有
环境不改动。
