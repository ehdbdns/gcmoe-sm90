# SM100 回执：V41 host-drift origin

执行者：**SM100**（不是 SM90）。本轮只执行
`docs/sm100_megamoe/V41_HOST_DRIFT_ORIGIN.md`，没有启动模型，也没有占用
GPU。源提交为 `b74b2eca60bedffe41e232b1659ad78d8831d464`，原始归档为
`/data0/home/admin/workspace/megamoe/v41-l20-host-device-T2qg4jm8/run`。

## 结果先说

GC 从 **L1** 开始落后：L1 新增 **609.194 us**，到 L19 累计 **9120.500 us**，
L20 再新增 **535.876 us** 后为 **9656.500 us**。Mega 的 L1 是
**-34.716 us**，到 **L2** 才首次明显落后 **186.919 us**，最大新增出现在
**L10（301.093 us）**；L19 为 **1986.000 us**，L20 新增为
**-32.834 us**，因此 L20 的 **1953.250 us** 几乎全是继承的提交 skew。

两臂的共同点是 L20 设备入口差由“继承的提交差 + 早 rank 已排队时间”闭合，
不是晚 rank 在 L20 新长出同样的整段时间。不同点是 GC 从 L1 起持续累积，
Mega 则先负、L2 正分叉、L7 回吐、L10 最大、L20 不再新增；不能把两臂写成
同一个逐层放大机制。

## 时钟与 L0 anchor

原始 Kineto `ts` 是各 rank 本地时钟。GC 的跨 rank 原始 `ts` 差
**-80343.743 us**、Mega 为 **-31213.742 us**，这是时钟偏移，不是 L1 的
层间工作。改用同一进程的 monotonic/device global timer 后：

| arm | anchor→首次提交 us | L0 launch skew us | L0 device skew us |
|---|---:|---:|---:|
| GC | 82375.470 | 2032.000 | 576.500 |
| Mega | 32077.724 | 864.250 | 649.500 |

这 82/32 ms 是 anchor 等待，不能计入 L1–19 或 L20。

## 层间闭合与 L20 inherited/new

`layer_closure.csv` 的 host closure error 和 exclusive closure error 均小于
0.3 us。GC 的 L1–19 新增 **7088.585 us**、L20 **535.876 us**、L1–20
**7624.461 us**；Mega 分别为 **1121.944 us**、**-32.834 us**、
**1089.110 us**。早 rank 的 L19 launch→entry 队列时间分别为
**8690.857 us**（GC）和 **1697.049 us**（Mega）。

| arm | L20 inherited us | L20 new us | submit skew us | queue delta us | 计算入口差 us | 实测 device skew us |
|---|---:|---:|---:|---:|---:|---:|
| GC | 9120.500 | 535.876 | 9656.500 | -9298.474 | 358.026 | 357.750 |
| Mega | 1986.000 | -32.834 | 1953.250 | -1606.183 | 347.067 | 347.000 |

入口公式为 `device skew = submit skew + (late queue - early queue)`；差值分别
只有 **-0.276 us** 和 **-0.067 us**。GC L19 的入口差为 450.373 us（实测
450.500 us），Mega L19 为 303.362 us（实测 303.500 us）。

## exclusive 归因和原始 Kineto

完整累计和逐事件见 `v41_host_drift_origin_results/exclusive_semantic_summary.csv`、
`top_exclusive.csv`、`exclusive_evidence.csv` 和 `exclusive_occurrences.csv`。

GC L1–19：`unobserved` **+4032.209 us**，`mm_mxfp8` **+647.402 us**，
`cudaMemcpyAsync` **+297.801 us**，`cudaGraphLaunch` **+193.694 us**，
`cuLaunchKernelEx` **+184.798 us**。首层最大正事件是 occurrence 9 的
`unobserved`：早/晚 **108.734/194.386 us**，差 **85.651 us**；L1 的
`mm_mxfp8` occurrence 1 还差 **35.838 us**。L20 的 **535.876 us** 中，
`unobserved` **+318.567 us**、`mm_mxfp8` **+78.752 us**、
`aten::copy_` 的 launch **+24.647 us**（correlation 12310）。

Mega L1–19：`unobserved` **+1197.236 us**，`cudaMemcpyAsync`
**+101.353 us**，`aten::empty` **+48.731 us**，`cuLaunchKernelEx`
**+48.366 us**；反过来 `aten::slice` **-319.282 us**、`mm_mxfp8`
**-66.206 us**。首次明显分叉在 L2，代表事件是 occurrence 48 的
`unobserved` **+34.184 us** 和 occurrence 0 的 `mm_mxfp8`
**+28.051 us**；最大新增在 L10，代表事件是 occurrence 77 的
`unobserved` **+32.848 us**。L20 的净新增是 **-32.834 us**，最大正向为
`sparse_attn_indexer` occurrence 14 **+36.146 us** 和 `persistent_topk`
launch occurrence 0 **+23.347 us**（correlation 12294），但被
`mm_mxfp8` **-83.324 us** 等负项抵消。

所有正负值、重复调用序号、PID/TID、前后事件和 correlation 都保留在 CSV。
`unobserved` 只表示原始 Kineto 中没有包围的具名 CPU op，不能改称 Python
调度、锁竞争或缺页；本轮没有做最近时间替代，也没有为图内节点虚构逐节点
host 提交。

## perf 和 step 1/4/6

perf 为 **199 Hz**，采样周期 **5025.125 us**。L1–19 的两臂 16 个 rank
共 304 个 layer-window、命中主线程样本 **107** 个；单窗口通常 0–1 个样本。
只有 GC rank7 在窗口内出现 **2 次 SWITCH、35.577 us off-CPU**，其余 rank
窗口均为 running。热点分散在 Python interpreter、allocator、loader/TLS、
libc、sync/scheduler、CUDA collective 和 unknown；没有一条栈能解释 GC
的 4.0 ms 或 Mega 的 1.2 ms 剩余量。时钟最大拟合残差为 GC **8.380 us**、
Mega **37.634 us**。逐 rank 覆盖见 `perf_stack_summary.csv`，不能以空栈
作为归因。

step 4/6 使用各自原始 monotonic 曲线和各自早晚 rank：

| arm | step | early/late | L19 skew us | L20 skew us |
|---|---:|---|---:|---:|
| GC | 1 | 4/2 | 9120.466 | 9657.327 |
| GC | 4 | 4/5 | 2430.880 | 2548.190 |
| GC | 6 | 7/1 | 1289.602 | 1382.033 |
| Mega | 1 | 7/5 | 1986.350 | 1953.378 |
| Mega | 4 | 2/7 | 1792.477 | 2120.465 |
| Mega | 6 | 4/5 | 722.612 | 787.997 |

早晚 rank 在 step 间轮换，排除固定机器位置；不能跨不同 step 直接做绝对放大
比较。详见 `step146_mono.csv`、`step146_mono_summary.csv`。

## L19→L20 图覆盖与 SM90 对照

两臂八 rank 的 L19 MoE graph launch 到 L20 MoE graph launch 之间均为：
**3 次 `cudaGraphLaunch`**（1 次 MoE + 2 次 dense local）、**30 次直接
kernel launch**（合计 33 次）、**8 次 `cudaMemcpyAsync`**、**9 次
`aten::copy_`**、**1 个逻辑 `aten::zero_`**（内层 2 条 `aten::fill_`）。
GC MoE 图有 3 个 kernel 节点，Mega MoE 图有 1 个；三图合计 graph-node
kernel 事件为 **7 / 5**。replay 后审计 copy 3 个，下一次 replay 前 scratch
copy 1 个、zero/fill 1 个；模型侧约 5 copy、4 memcpy API、29 直接 kernel。
逐图偏移和节点别名见 `graph_launch_details.csv` 与
`graph_coverage_compare.csv`。

SM90 对照文件为
`docs/trace_executor_20261119/sm90_main24_stable_decode_bs512_results/delivery/bs512/vanilla/run.json`
（SHA256 `2dcc9aed821594ae4e5927d5deb5cbc32d7a2cd941538391d6c97fc42b50f1f2`），
其记录为 batch 512、40 层、`ENFORCE_EAGER=0`、
`CUDAGRAPH_MODE=FULL_DECODE_ONLY`。当前 SM100 `five_matrix.py:83` 固定
`enforce_eager=True`，`five_worker.py` 只捕获单层 MoE，dense 走局部图，
不是同一个 full-decode 覆盖层级。

安装环境为 vLLM `0.31.0`（commit `gdb9527a46`）、torch `2.13.0+cu130`。
full decode capture 入口在 `vllm/v1/worker/gpu_worker.py:880-883`、
`vllm/v1/worker/gpu_model_runner.py:6755-6825,6861-6958`；图输入地址和
replay 约束在 `vllm/compilation/cuda_graph.py:281-321,345-363`。切整图前
必须静态化按层的 `current/call`、`host_submit`、scratch/copyback、plan/quota
view、ECO table、dense graph lookups 和审计地址，清单见
`full_decode_adjustments.csv`；源码路径、逐文件 SHA 和最短摘录见
`source_locations.md`。本轮没有翻转 `enforce_eager` 或修改模型。

## 产物与证据

结果目录：`docs/sm100_megamoe/v41_host_drift_origin_results/`

- `README.md`：完整方法、逐 rank 表、闭合、perf 和源码对照。
- `windows.csv`、`ranks.csv`、`layer_summary.csv`、`layer_closure.csv`：层曲线和闭合。
- `exclusive_cumulative.csv`、`exclusive_semantic_summary.csv`、`exclusive_evidence.csv`、`exclusive_occurrences.csv`、`top_exclusive.csv`：独占时间和原始事件证据。
- `perf_stack_summary.csv`、`perf_stack_hotspots.csv`、`perf_window_windows.csv`、`perf_clock_fit.csv`：perf 覆盖和时钟拟合。
- `step146_mono.csv`、`step146_mono_summary.csv`：step 1/4/6 原始曲线。
- `graph_coverage_compare.csv`、`graph_launch_details.csv`：L19→L20 图覆盖。
- `source_locations.md`、`full_decode_adjustments.csv`：源码/版本和整图调整项。
- `evidence_sha256.txt`：原始 trace、perf、脚本和 SM90 对照 SHA。
- `SHA256SUMS`：本目录自校验。

剩余微秒：GC L1–19 `unobserved` **4032.209 us**、L20 **318.567 us**；
Mega L1–19 **1197.236 us**、L20 **8.874 us**。它们只能定位到 host 提交
窗口内没有包围具名 op 的区间，现有 5 ms perf 采样不能进一步细分。
