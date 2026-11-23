# SM100 V41 L20 主机/设备错峰归因（f2230f1）

本目录是 **SM100 执行者** 对 `docs/sm100_megamoe/V41_L20_HOST_DEVICE_CAUSE.md` 的结果。
两臂在同一组 8 张 L20D（`GPU-3aad6312...` 至 `GPU-31a509fd...`）上串行运行；Mega 为 step 1
带 Kineto CPU activity，GC 随后运行；step 4/6 关闭 Kineto forward 用于无采样复核。
Mega/GC 退出码均为 0。原始 trace、perf 数据和日志保留在
`/data0/home/admin/workspace/megamoe/v41-l20-host-device-T2qg4jm8/run`。

## 三个问题的微秒答案

| 臂 | 早/晚 rank | L19 出口差 | L20 入口差 | 首个额外等待 | 早 rank 交换在晚 rank 进入前 |
|---|---:|---:|---:|---|---:|
| Mega | 7 / 5 | 0.9912 | **347.1162** | `all_gather -> cublas_tss`，晚 rank gap 124.6084 | **346.0918** |
| GC | 4 / 2 | 1.3232 | **357.8896** | `mhc -> all_gather`，晚 rank gap 19.3594 | **357.8574** |

### Q1：第 19 层 combine 结束后，晚 rank 的第一个额外等待

- **Mega rank 5**：`all_gather -> cublas_tss`。晚 rank 设备 gap **124.6084 us**；早 rank
  同一 边界设备 gap 约 3.0 us（早 rank 的 `cuLaunchKernelEx` 已提前提交），额外约
  **121.6 us**。晚 rank 的 124.6084 us 中，**112.8906 us 发生在 host 进入 API 之前**，
  `cuLaunchKernelEx` 自身 7.7412 us，API 返回后设备再等 3.9766 us。调用链为
  `aten::empty -> aten::mm -> cuLaunchKernelEx(corr=11910) -> cublas_tss`；
  `aten::linear` 是后继 CPU op。
- **GC rank 2**：`mhc -> all_gather`。晚 rank gap **19.3594 us**，早 rank 0.1279 us，
  额外 **19.2314 us**。其中 host 未进入 API 9.5195 us，`cudaLaunchKernelExC` 6.0850 us，
  设备尾等 3.7549 us。调用链为
  `aten::empty -> _C_custom_ar::mnnvl_lamport_all_gather -> cudaLaunchKernelExC(corr=11851) -> all_gather`。
  GC 随后在 `all_gather -> cublas_tss` 还有更大的 165.4102 us gap，但首个新增等待是上面的
  `mhc -> all_gather`。

### Q2：最后一次 reduce-scatter 重新对齐后，谁重新造成 L20 入口差

最后一次 reduce-scatter 结束时，入口差已被压回：Mega **5.265 us**，GC **4.655 us**。
之后下列六段调用边界再次把差拉开；数字是晚 rank gap 增量，括号内是其中 host 尚未进入
后继 CUDA API 的时间。

| 臂 | 调用边界 | gap 增量 us | host 未进入 API us | API 执行 us | API 后设备等待 us |
|---|---|---:|---:|---:|---:|
| Mega | `reduce_scatter -> mhc` | 73.1201 | 66.3203 | 3.883 | 2.917 |
| Mega | `mhc -> mega_gate` | 103.1680 | 96.6641 | 3.622 | 3.266 |
| Mega | `mega_gate -> prepare` | 106.3359 | 100.8213 | 3.448 | 3.347 |
| Mega | `prepare -> memcpy` | 22.6250 | 12.7695 | 8.095 | 3.169 |
| Mega | `memcpy -> fill` | 15.0713 | 9.9834 | 4.645 | 3.163 |
| Mega | `fill -> mega_moe` | 18.3369 | 5.8545 | 11.110 | 3.100 |
| GC | `reduce_scatter -> mhc` | 73.8574 | 65.7070 | 4.859 | 3.291 |
| GC | `mhc -> mega_gate` | 81.2490 | 73.6094 | 4.415 | 3.225 |
| GC | `mega_gate -> prepare` | 122.4307 | 116.4512 | 4.783 | 3.662 |
| GC | `prepare -> memcpy` | 26.3047 | 14.7480 | 9.953 | 3.107 |
| GC | `memcpy -> fill` | 25.8555 | 11.0420 | 14.391 | 2.695 |
| GC | `fill -> gc_moe` | 25.8555 | 10.0596 | 20.341 | 0.000 |

六段 post-RS gap 合计 **338.6572 us (Mega) / 355.5527 us (GC)**；其中 API 前
**292.4131 / 291.6172 us**，占 86% / 82%。API 自身和 API 返回后的设备排队都只有
数微秒到二十微秒。`prepare -> memcpy`、`memcpy -> fill` 与图提交段已不是纯 Python
op，但仍以 host 边界等待为主；图内节点不虚构逐节点 host 提交，统一按同一次
`cudaGraphLaunch`/graph-submit 记。

### Q3：早 rank 的 count exchange 有多少发生在晚 rank 进入 L20 前

- Mega：早 rank 7 的交换区间长 **366.0498 us**，起点在自身 L20 入口后 1.0244 us；
  其中 **346.0918 us (94.5%)** 在晚 rank 5 进入 L20 前，19.9580 us 与晚 rank 入口重叠或在其后。
- GC：早 rank 4 的交换区间长 **368.6094 us**，起点在自身 L20 入口后 0.0322 us；
  其中 **357.8574 us (97.1%)** 在晚 rank 2 进入 L20 前，10.7520 us 重叠或在其后。

## 时间线闭合与最大单 gap

两臂都用同一个时间窗口闭合，误差为 0：

| 臂 | window delta us | union busy delta us | gap delta us | closure error us |
|---|---:|---:|---:|---:|
| Mega | 347.0459 | -1895.9424 | 2242.9883 | 0.0000 |
| GC | 358.9863 | -2610.6729 | 2969.6592 | 0.0000 |

因此不能把正 gap 简单相加当成最终入口差；collective/同流重叠使 late rank 的设备 busy
窗口同时缩短。正负 gap 与 busy 变化已全部写入 `closure.csv`，`matched_gaps=41/42`
（Mega/GC），`unmatched_gaps=0`。全窗口最大单个新增 gap 是
`q_kv_norm -> dense_gemm`：**276.2900 us (Mega)**、**290.1790 us (GC)**；
其 host 未进入 API 分别为 266.2119 / 277.4804 us。这个 gap 位于层内 attention 路径，
不计入上面的 post-RS 六段表。

## 可复查调用链

Mega 晚 rank 5，第一个额外等待：

```text
device all_gather end
  -> host CPU gap 112.8906 us inside aten::mm (previous aten::empty, following aten::linear)
  -> cuLaunchKernelEx 7.7412 us (pid=tid=3729610, corr=11910)
  -> device tail 3.9766 us
  -> cublas_tss start
```

GC 晚 rank 2，第一个额外等待：

```text
device mhc end
  -> host CPU gap 9.5195 us inside _C_custom_ar::mnnvl_lamport_all_gather
  -> cudaLaunchKernelExC 6.0850 us (pid=tid=695966, corr=11851)
  -> device tail 3.7549 us
  -> all_gather start
```

两臂的完整边界、correlation、前后 CPU op、stream、API enter/exit 和 gap 归因见
`boundary_alignment.csv`、`call_chain.csv`、`gap_attribution.csv`。

## 8 rank 公共时轴 Perfetto

- `mega_step001_l19_l20.perfetto.json`
- `gc_step001_l19_l20.perfetto.json`

每个文件有 8 个 rank 进程；`tid=2000` 是 host submitter/CPU op/API，设备 stream 用
`tid=1000+stream`，`tid=3000` 是 L19 出口、最后 collective、mhc、gate、prepare、
L20 graph launch、设备入口、count exchange 和 MoE 出口；`tid=5000` 是 perf SWITCH
重建的 running/runnable/blocked 轨，`tid=5001` 是 perf 用户态样本点。所有事件均以
最早 L19 出口为零点，单位 us。

## step 4/6 无 Kineto 复核

| 臂 | step | 入口差 us | 早/晚 rank | 早 rank 交换 us | 晚 rank 交换 us |
|---|---:|---:|---:|---:|---:|
| Mega | 1 | 347.1162 | 7/5 | 366.0498 | 18.5918 |
| Mega | 4 | 263.9951 | 2/7 | 281.6318 | 17.7598 |
| Mega | 6 | 289.5518 | 4/5 | 306.9121 | 17.3115 |
| GC | 1 | 357.8896 | 4/2 | 368.6094 | 8.5439 |
| GC | 4 | 266.5117 | 4/5 | 277.6963 | 9.9199 |
| GC | 6 | 273.0605 | 7/1 | 283.1367 | 8.9922 |

step 4/6 关闭 Kineto forward，入口差仍在 264--290 us（Mega）和 267--281 us（GC），
且早晚 rank 轮换（Mega 2/7、4/5；GC 4/5、7/1），说明不是固定 rank 绑定，也不是只在
step 1 出现的现象。step 1 的 Kineto CPU activity 使绝对值与旧无 CPU trace 基线不同：
Mega 291.8848 -> 347.1162 us，GC 492.6123 -> 357.8896 us；方向相反，没有统一的
膨胀量。`step146_compare.csv` 给出逐项 delta。

## 两臂是否同因

**机制同因、首个调用不同。** 两臂都是最后一次 reduce-scatter 结束时已对齐（5.265 /
4.655 us），随后在 `mhc/gate/prepare/memcpy/fill/graph` 多个边界逐次累积，且新增部分
主要落在后继 CUDA API 尚未被调用前的 host CPU 路径；API 自身与 API 后的设备排队都小。
Mega 的第一个额外等待更大（124.6 us），GC 首个只有 19.2 us，但 GC 的
`all_gather -> cublas_tss` 随后增加到 165.4 us，最终入口差反而略大（357.9 vs 347.1 us）。

## 未解释的净差与采样限制

- 时间线净差已闭合：两臂 closure error 都是 **0.0000 us**，所有 41/42 个 gap 都匹配，
  没有 unmatched gap。未解释的是“这些 100 us 级 pre-API CPU 段具体在哪个 Python/C++
  语句上消耗”，不是入口差本身。
- 外部 perf：`perf_event_paranoid=2`，`cpu-clock:u` 用户态采样可用，199 Hz（采样周期
  5.03 ms），记录 2,219,443 个样本、726.078 MB；16 个 worker 主 PID/TID 都在记录中
  （每个主 TID 26,957--42,598 个样本，`perf_pid_tid_coverage.csv`）。关键 gap 只有
  17--125 us，短于采样周期，因此这些 gap 内通常 **0 个栈样本**；不能用空栈推断原因。
- SWITCH 证据：关键 gap 内 0 个切换记录，前后分别是 SWITCH IN 与 SWITCH OUT；Mega 的
  OUT 为 preempt（之后可运行），GC 的 OUT 为阻塞，但切换都发生在 gap 之后。即关键
  gap 内线程处于 running，没有被抢占/阻塞区间；直接证据见 `perf_switch_evidence.csv`
  和 `gap_attribution.csv` 的 `perf_running/runnable/blocked_*` 列。
- 时钟：step 1 两个 live anchor，设备边界半宽 <=0.8155 us（Mega）/0.7200 us（GC）；
  host 提交到 trace 的线性拟合最大残差 37.634 us（Mega rank 5）/8.380 us（GC）。
  API 前后与设备 gap 均在 Kineto 同一时间轴内，host fit 残差不用于改写 gap；但它限制
  host 提交时轴与设备时轴的严格同步解释。step 4/6 只有单个 after anchor，使用
  _single_after_anchor_fallback_，半宽 0.864/0.9765 us，只用于同量级复核。
- 没有 kernel 栈和 off-CPU 栈（paranoid=2）；因此不能把 pre-API 段进一步归因到缺页、
  锁竞争或 OS 抢占。现有证据只支持“host 线程在运行且尚未调用 API”。

## 文件清单

| 文件 | 内容 |
|---|---|
| `arm_summary.csv` | 两臂早晚 rank、L19/L20 差、正负 gap、host fit 残差 |
| `l20_arrival_and_exchange.csv` | 8 rank 的 L19 出口、L20 入口、窗口、busy/gap、交换区间 |
| `exchange_intersection.csv` | 早 rank 交换与晚 rank 入口的交集 |
| `boundary_alignment.csv` | 逐边界共同序号、前后 kernel/stream/correlation、累计 delta |
| `call_chain.csv` | 同上，供调用链复查 |
| `gap_attribution.csv` | 全部 >=10 us gap、API 三段拆分、CPU op、perf 调度 |
| `key_call_answers.csv` | 三问的逐臂答案 |
| `closure.csv` | window = busy + gap 闭合 |
| `perf_coverage.csv` | 全部 worker PID/TID 样本和 switch 计数 |
| `perf_pid_tid_coverage.csv` | 16 个主 TID 的精简覆盖表 |
| `perf_switch_evidence.csv` | 关键 gap 前后 SWITCH 记录 |
| `step146.csv` / `step146_compare.csv` | step 1/4/6 与 c909c64 基线对照 |
| `*_step001_l19_l20.perfetto.json` | 两臂 8 rank 公共时轴 |
| `analysis_validation.txt` | 闭合、时间戳映射、perf 是否加载的自动校验 |

## 复现命令

```bash
cd /data0/home/admin/workspace/megamoe/v41-l20-host-device-T2qg4jm8/code
python -m py_compile docs/sm100_megamoe/v41_l20_host_device_results/analyze_l20_host_device.py

python docs/sm100_megamoe/v41_l20_host_device_results/analyze_l20_host_device.py \
  ../run/bs512 docs/sm100_megamoe/v41_l20_host_device_results \
  --perf ../run/worker-sched.txt \
  --perf-pids 3729594,3729597,3729602,3729606,3729608,3729610,3729612,3729613,695964,695965,695966,695967,695968,695969,695970,695971 \
  --perf-tids ../run/worker-sched-fields.txt
```

`worker-sched-fields.txt` 是 `perf script -F comm,pid,tid,time,event --show-switch-events --ns`
按 16 个 worker PID 过滤得到的字段文件；`worker-sched.txt` 是默认字段加用户态栈。

## 原始证据、环境与释放

- 原始路径与大小：`raw_evidence_paths.txt`；SHA256：`raw_evidence.sha256`
  （114 个原始 trace/result/stats/log/perf 条目）。
- 关键 SHA：Mega `profile-rank0.json` =
  `b3ad60347ff4934055f00c71c4defd5e1bb30be2c7cb1982a696a4f8dc5c94c2`；
  GC `profile-rank0.json` =
  `9f0a658290e8d6dbb2c6a486348f7f65847aa8bd4e75eca48ceed671b3add259`；
  `worker-sched.data` =
  `86fb8dde4a8f24f741531a60c9b8a6cf4d7912306a4a45615ca3b81a685f2635`；
  `worker-sched.txt` =
  `ccfd35ea3a3ef7e66ff36087df927ae46a476c653378d1491e5ee319de76e1f2`。
- 环境：torch 2.13.0，vLLM 0.31.0 wheel，nvidia-cutlass-dsl 4.8.1，CUDA 13.5
  ptxas；完整版本、GPU UUID、perf 权限在 `environment.txt`。
- 退出码与释放：Mega start 14:16:01 / end 14:28:49 / exit 0；GC start 14:28:51 /
  end 14:36:43 / exit 0。组锁 acquired 14:15:50，released 14:36:46；
  `gpu_release.txt` 记录释放后 compute-apps 为空。
- 本结果目录只提交分析产物；原始 profile、stats、measure、日志和 perf 文件保留在
  `run/` 下，未写入仓库。
