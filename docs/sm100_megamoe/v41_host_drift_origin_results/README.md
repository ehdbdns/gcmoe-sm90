# SM100 V41 host-drift origin analysis

本目录是 SM100 对 kept 原始 trace 的只读复算，源提交为
`b74b2eca60bedffe41e232b1659ad78d8831d464`。原始数据在
`/data0/home/admin/workspace/megamoe/v41-l20-host-device-T2qg4jm8/run`，
本轮没有启动模型或占用 GPU。

## 一句话结论

GC 从 **L1** 就开始落后，L1 新增 **609.194 us**，到 L19 为 **9120.500 us**，
L20 再新增 **535.876 us** 后为 **9656.500 us**。Mega 的 L1 反而为
**-34.716 us**，到 **L2** 才首次明显落后 **186.919 us**，最大新增在
**L10（301.093 us）**；L19 为 **1986.000 us**，L20 的新增为
**-32.834 us**，所以 L20 的 **1953.250 us** 基本全是继承 skew，而不是
L20 新长出来的。两臂都不支持“固定机器位置”解释；step 4/6 的早晚 rank
发生了轮换。

## 时钟边界与 L0

Kineto 的 `ts` 是各 rank 本地时钟，不能把两个 rank 的原始 `ts` 直接相减当作
跨 rank 绝对时间。GC 的原始 `ts` 差为 **-80343.743 us**，Mega 为
**-31213.742 us**，这是时钟偏移；真正的 anchor 到首次提交时间分别为
**82375.470 us** 和 **32077.724 us**。这两段 82/32 ms 的 anchor 等待属于
L0 提交前的等待，不能摊进 L1–19 或 L20 的层间新增。

| arm | 原始 Kineto `ts` 差 (us) | anchor→首次提交 (us) | L0 launch skew (us) | L0 device skew (us) |
|---|---:|---:|---:|---:|
| GC | -80343.743 | 82375.470 | 2032.000 | 576.500 |
| Mega | -31213.742 | 32077.724 | 864.250 | 649.500 |

跨 rank 的设备时间使用 device global timer；提交时间的层间曲线使用同一
进程的 monotonic clock。原始关联只接受实际 MoE 设备 kernel 和提交线程的
CUDA graph launch 的唯一 correlation。本轮没有使用最近时间匹配。

## 八个 rank 的关键层

下表是每个 rank 自身的 host submission window（us），不是跨 rank 差值。
GC 的固定早晚对是 `4/2`，Mega 是 `7/5`；`first` 和 `max` 指该臂首次明显
分叉层和最大新增层的窗口值。完整逐层数据见 `rank_key_layers.csv`、
`windows.csv` 和 `ranks.csv`。

| arm | rank | role | L0 | first layer/value | max layer/value | L19 | L20 |
|---|---:|---|---:|---|---|---:|---:|
| GC | 0 | other | 25437.709 | L1 2399.690 | L1 2399.690 | 1532.174 | 2994.857 |
| GC | 1 | other | 50861.632 | L1 2465.262 | L1 2465.262 | 1544.902 | 3009.276 |
| GC | 2 | late | 133826.907 | L1 2942.427 | L1 2942.427 | 1874.802 | 3565.005 |
| GC | 3 | other | 31764.246 | L1 2452.146 | L1 2452.146 | 1574.676 | 3013.426 |
| GC | 4 | early | 51451.438 | L1 2333.232 | L1 2333.232 | 1538.277 | 3029.129 |
| GC | 5 | other | 35612.131 | L1 2458.053 | L1 2458.053 | 1604.345 | 3086.879 |
| GC | 6 | other | 50845.692 | L1 2419.096 | L1 2419.096 | 1555.913 | 2963.736 |
| GC | 7 | other | 51307.532 | L1 2617.900 | L1 2617.900 | 1591.884 | 3185.309 |
| Mega | 0 | other | 24518.371 | L2 2430.649 | L10 1493.684 | 1519.215 | 2960.354 |
| Mega | 1 | other | 25324.791 | L2 3126.219 | L10 1500.180 | 1522.010 | 3064.095 |
| Mega | 2 | other | 21853.492 | L2 2424.080 | L10 1592.283 | 1600.273 | 3089.506 |
| Mega | 3 | other | 27155.165 | L2 2422.439 | L10 1509.796 | 1515.774 | 2987.286 |
| Mega | 4 | other | 82204.256 | L2 2446.676 | L10 1458.919 | 1562.052 | 3032.058 |
| Mega | 5 | late | 55181.766 | L2 2593.946 | L10 1780.688 | 1566.661 | 3099.203 |
| Mega | 6 | other | 182762.083 | L2 2473.859 | L10 1455.559 | 1543.646 | 2929.839 |
| Mega | 7 | early | 23104.042 | L2 2407.027 | L10 1479.596 | 1517.323 | 3132.037 |

## 层间闭合与 L20 入口

`layer_closure.csv` 对 L1–20 的相邻层有符号增量做闭合。GC 的 L1–19
新增为 **7088.585 us**，L20 新增为 **535.876 us**，合计 **7624.461 us**；
Mega 的 L1–19 为 **1121.944 us**，L20 为 **-32.834 us**，合计
**1089.110 us**。两臂的 host closure error 都在 0.2 us 以内。

L20 的 GPU 入口差按
`device skew = total launch skew + (late queue - early queue)` 闭合：

| arm | L20 inherited (us) | L20 new (us) | L20 submit skew (us) | queue delta (us) | 计算入口差 (us) | 实测 device skew (us) |
|---|---:|---:|---:|---:|---:|---:|
| GC | 9120.500 | 535.876 | 9656.500 | -9298.474 | 358.026 | 357.750 |
| Mega | 1986.000 | -32.834 | 1953.250 | -1606.183 | 347.067 | 347.000 |

GC L19 的入口差为 `9120.500 + (20.730 - 8690.857) = 450.373 us`，实测
450.500 us；Mega L19 为 `1986.000 + (14.411 - 1697.049) = 303.362 us`，
实测 303.500 us。误差小于 0.3 us，说明 L20 设备空档主要是早 rank 已经
排队的时间，而不是晚 rank 在 L20 才额外多花了同样长的时间。

## exclusive 归因

`exclusive_semantic_summary.csv` 保留负值和 `unobserved`。`unobserved`
只表示原始 Kineto 中没有包围的具名 CPU op，不能改写成 Python 调度、
锁竞争或缺页。

GC L1–19 的独占净差为 **7088.585 us**，其中 `unobserved` 为
**+4032.209 us**，`mm_mxfp8` 为 **+647.402 us**，`cudaMemcpyAsync` 为
**+297.801 us**，`cudaGraphLaunch` 为 **+193.694 us**，
`cuLaunchKernelEx` 为 **+184.798 us**。GC 从 L1 起持续新增，首层最大的
正事件是无包围 occurrence 9：早晚持续 **108.734/194.386 us**，差
**85.651 us**；L1 的 `mm_mxfp8` occurrence 1 还贡献 **35.838 us**。
GC L20 的 **535.876 us** 中，`unobserved` **+318.567 us**，
`mm_mxfp8` **+78.752 us**，`cudaLaunchKernel@aten::copy_` occurrence 0
**+24.647 us**（correlation 12310）。

Mega L1–19 的独占净差为 **1121.944 us**，其中 `unobserved`
**+1197.236 us**，`cudaMemcpyAsync` **+101.353 us**，
`aten::empty` **+48.731 us**，`cuLaunchKernelEx` **+48.366 us**；
相反 `aten::slice` 为 **-319.282 us**，`mm_mxfp8` 为 **-66.206 us**，
因此这不是单调累加。首次明显正分叉在 L2：`unobserved` occurrence 48
代表层 **+34.184 us**，`mm_mxfp8` occurrence 0 **+28.051 us**；
最大新增在 L10，代表事件是 `unobserved` occurrence 77 **+32.848 us**。
Mega L20 的净新增只有 **-32.834 us**；正向最大是
`sparse_attn_indexer` occurrence 14 **+36.146 us** 和
`persistent_topk` 的 launch occurrence 0 **+23.347 us**
（correlation 12294），但被 `mm_mxfp8` **-83.324 us** 和
`aten::empty` **-28.187 us** 抵消。

逐事件、PID/TID、correlation 和邻居见 `top_exclusive.csv`、
`exclusive_evidence.csv`、`exclusive_occurrences.csv`。L20 的两个主要正
事件都能回到原始 Kineto 的前后事件；没有把图内节点伪造成逐节点 host 提交。

## perf 覆盖

归档 perf 的 CPU clock 为 **199 Hz**，名义采样周期 **5025.125 us**。
L1–19 的 16 个 rank 共覆盖 304 个 layer-window，命中主线程样本 **107** 个；
每个单独窗口通常只有 0–1 个样本，不能把一次空栈当成归因。

| arm | rank | role | wall/实例 us | 样本/实例 | window 样本 | off-CPU us | switch | 最大首帧类别/占比 |
|---|---:|---|---:|---:|---:|---:|---:|---|
| GC | 0 | other | 1791.049 | 0.316 | 6 | 0.000 | 0 | unknown 50.0% |
| GC | 1 | other | 1790.255 | 0.368 | 7 | 0.000 | 0 | unknown 28.6% |
| GC | 2 | late | 2133.727 | 0.421 | 8 | 0.000 | 0 | unknown 75.0% |
| GC | 3 | other | 1811.021 | 0.368 | 7 | 0.000 | 0 | unknown 14.3% |
| GC | 4 | early | 1760.594 | 0.263 | 5 | 0.000 | 0 | python_interpreter 40.0% |
| GC | 5 | other | 1835.803 | 0.316 | 6 | 0.000 | 0 | python_interpreter 33.3% |
| GC | 6 | other | 1787.811 | 0.368 | 7 | 0.000 | 0 | unknown 57.1% |
| GC | 7 | other | 1983.387 | 0.316 | 6 | 35.577 | 2 | python_interpreter 66.7% |
| Mega | 0 | other | 1723.900 | 0.368 | 7 | 0.000 | 0 | unknown 42.9% |
| Mega | 1 | other | 1785.626 | 0.368 | 7 | 0.000 | 0 | unknown 28.6% |
| Mega | 2 | other | 1781.362 | 0.316 | 6 | 0.000 | 0 | python_interpreter 50.0% |
| Mega | 3 | other | 1737.030 | 0.368 | 7 | 0.000 | 0 | sync_scheduler 42.9% |
| Mega | 4 | other | 1764.524 | 0.368 | 7 | 0.000 | 0 | unknown 42.9% |
| Mega | 5 | late | 1869.289 | 0.368 | 7 | 0.000 | 0 | libtorch_python 28.6% |
| Mega | 6 | other | 1766.478 | 0.368 | 7 | 0.000 | 0 | libtorch_cpu 42.9% |
| Mega | 7 | early | 1810.230 | 0.368 | 7 | 0.000 | 0 | python_interpreter 42.9% |

只有 GC rank7 在窗口内出现 2 次 SWITCH / 35.577 us off-CPU；其余 rank 的
主线程窗口都为 running。热点在 Python interpreter、allocator、loader/TLS、
libc、sync/scheduler、CUDA collective 和 unknown 之间分散，没有一条栈能
解释 GC 的 4.0 ms 或 Mega 的 1.2 ms 剩余。时钟最大拟合残差为 GC 8.380 us、
Mega 37.634 us（Mega rank5），不足以改变层间结论。

## step 4/6 的早晚 rank 轮换

`step146_mono_summary.csv` 使用每一步自己的早晚 rank，而不是固定复用 step1：

| arm | step | early/late | L19 skew us | L20 skew us |
|---|---:|---|---:|---:|
| GC | 1 | 4/2 | 9120.466 | 9657.327 |
| GC | 4 | 4/5 | 2430.880 | 2548.190 |
| GC | 6 | 7/1 | 1289.602 | 1382.033 |
| Mega | 1 | 7/5 | 1986.350 | 1953.378 |
| Mega | 4 | 2/7 | 1792.477 | 2120.465 |
| Mega | 6 | 4/5 | 722.612 | 787.997 |

早晚 rank 在 step 间换了位置，排除固定 rank 绑定；同一 rank 在不同 step 的
skew 也变化，不能用同一步的绝对放大直接比较。

## L19→L20 图覆盖与 SM90 对照

两臂八个 rank 的 L19 MoE graph launch 到 L20 MoE graph launch 之间都是：
3 次 `cudaGraphLaunch`（1 次 MoE + 2 次 dense local）、30 次直接 kernel
launch（总共 33 次 kernel launch）、8 次 `cudaMemcpyAsync`、9 次
`aten::copy_`、1 个逻辑 `aten::zero_`（内层 2 条 `aten::fill_`）。
GC 的 MoE 图含 3 个 kernel 节点，Mega 含 1 个；三图合计 graph-node kernel
事件分别为 7 和 5。计数/审计部分是 replay 后 3 个输出/计数 copy，加上
下一次 replay 前 1 个 scratch copy 和 1 个 zero/fill；模型部分约 5 个 copy、
4 个 memcpy API、29 个直接 kernel launch。详见
`graph_coverage_compare.csv` 和 `graph_launch_details.csv`。

早 rank 的 L19 launch→entry 队列时间为 GC **8690.857 us**、Mega
**1697.049 us**；L20 为 GC **9319.313 us**、Mega **1621.188 us**。这与
上面 L20 inherited 的量级一致，说明晚 rank 的 L20 空档不能整段归因给
L20 新工作。

SM90 已交付基线 `docs/trace_executor_20261119/sm90_main24_stable_decode_bs512_results/delivery/bs512/vanilla/run.json`
（SHA256 `2dcc9aed821594ae4e5927d5deb5cbc32d7a2cd941538391d6c97fc42b50f1f2`）
记录 `ENFORCE_EAGER=0`、`CUDAGRAPH_MODE=FULL_DECODE_ONLY`，batch 512、
40 层，`ms_per_step=166.466`。当前 SM100 的 `five_matrix.py:83` 固定
`enforce_eager=True`，`five_worker.py` 只捕获单层 MoE，dense 走
`native_dense_replay.py` 的局部图，不是同一个 full-decode 覆盖层级。

安装环境为 vLLM `0.31.0`（commit `gdb9527a46`）、torch `2.13.0+cu130`。
full decode capture 入口在 `vllm/v1/worker/gpu_worker.py:880-883`、
`vllm/v1/worker/gpu_model_runner.py:6755-6825` 和 `6861-6958`；
replay 地址约束在 `vllm/compilation/cuda_graph.py:281-321,345-363`。
当前按层变化的 Python `current/call`、`host_submit`、scratch/copyback、
plan/quota view、ECO table、dense graph key 和审计地址在整图回放下都必须
改成静态的 40 层副本或图外一次交换，具体清单见
`full_decode_adjustments.csv`；路径、逐文件 SHA 和最短摘录见
`source_locations.md`。

## 证据边界与剩余微秒

- 所有 8 rank 的实际 MoE kernel 都能唯一关联到提交线程的 graph launch；host closure 和 exclusive closure 误差均小于 0.3 us，没有最近时间替代。
- 两臂共同点是 L20 的 submit skew 已被 L19 继承，入口差由 queue 差闭合；但 GC 从 L1 起持续新增，Mega 在 L1 为负、L2 首次正分叉、L7 回吐、L10 最大、L20 无新增，因此不能称为同一个逐层放大机制。
- 仍未细分的剩余量：GC L1–19 `unobserved` **4032.209 us**、L20 **318.567 us**；Mega L1–19 **1197.236 us**、L20 **8.874 us**。这些量只能定位到提交窗口内、无包围具名 op 的区间；5 ms perf 采样不能进一步把它们拆成 Python、锁、缺页或同步。
- 原始证据 SHA 见 `evidence_sha256.txt`，目录内容校验见 `SHA256SUMS`。

## 产物索引

| 文件 | 内容 |
|---|---|
| `windows.csv` | 固定早晚 rank 的 L0–L20 launch/new/device/queue 曲线 |
| `ranks.csv` | 每个 rank 每层的 host API、device entry、window、correlation |
| `layer_summary.csv` | L0 anchor、首次分叉、最大新增、L19/L20 汇总 |
| `layer_closure.csv` | L1–19、L20、L1–20 的 host/device/exclusive 闭合 |
| `exclusive*.csv` | exclusive 累计、occurrence、top、语义桶和 Kineto 证据 |
| `perf_*.csv` | perf 样本覆盖、栈热点、逐窗口 running/off-CPU、时钟拟合 |
| `step146_mono*.csv` | step 1/4/6 的原始 monotonic 提交曲线 |
| `graph_coverage_compare.csv` | L19→L20 图/直接 kernel/copy/fill 覆盖 |
| `graph_launch_details.csv` | 每张图的多次 launch、offset、节点别名 |
| `source_locations.md` | 当前 SM100 与安装 vLLM 的源码位置、SHA、摘录 |
| `full_decode_adjustments.csv` | 切 full decode 前必须静态化的按层状态 |
