# 【SM100】BS512 原生 GCMoE / MegaMoE 五层阶段 Perfetto 回执

执行者：**SM100**。本轮只执行 `V41_NATIVE_PHASE_PERFETTO.md`，未执行任何 SM90
任务书。`mega`、`gc` 各启动一次模型，均以退出码 **0** 完成。

## 结论

- 两臂都完成 1000 个统计步和 3 个测量步，测量窗口的 8 个 rank 均报告
  `validated_layer_graphs=40`、`block_m=16`。
- 每臂生成五层图，共 **10 张 Perfetto**；每张图含 8 个 rank，每张
  `counts.json` 的八 rank token 合计均为 **3072**（`512 × 6`）。
- 五层八 rank 的 M 块合计为：Mega `413, 440, 433, 427, 415`；GC
  `411, 433, 431, 437, 420`。
- 关键 rank 按该层八卡完整 MoE 设备跨度最大者选择，同一 rank 用于所有阶段。
  Mega 五层关键 rank 为 `0, 5, 6, 0, 0`，GC 为 `0, 7, 0, 4, 5`。
- 全 rank 最大绝对时钟对齐漂移为 Mega **1.083805 us**、GC **1.621055 us**；
  所有阶段边界都在该 rank 的完整 MoE 区间内，锚点也落在被追踪步内。
- 本轮 `result.json` 中的墙钟包含 profiler 开销，只用于证明运行完成，**不作为
  速度胜负结论**；速度结论仍沿用直跑表。

## 任务、源码与环境

| 项目 | 值 |
|---|---|
| 任务书 | `docs/sm100_megamoe/V41_NATIVE_PHASE_PERFETTO.md` |
| 任务书 SHA256 | `86b8b2a1eea3a1c566af27bfbe48c7637437b048e9d95f3987ce90f46d4fcf41` |
| 执行源码提交 | `cbfb7ecdf3163e943acbd1261cd02f48eb1245df` |
| 回执打包基线 | `47e7839a4a3fc0424a5a4e1a80e302f3a9c4fa3a` |
| GCMoE 上游 | `fb8b65fa829c97962eb2dbc9a4ad8b759b9cd736` |
| GPU | 8 × NVIDIA L20D，UUID 见 `gpus.txt` |
| Python / Torch | 3.12 / 2.13.0 |
| vLLM | wheel SHA256 `39132b433967f0962d03d8a5f633b9bee6b32708f566c7d5637b79f8d0ef3433` |
| CuTe DSL | 4.8.1 |
| ptxas | CUDA 13.5.37 |
| 模型权重索引 SHA256 | `74b0686a3d2891980d5e303251b075a3bccae2c2ff650747db2620a649b98fa8` |
| 提示 SHA256 | `b4fee0121ad43c17fdd5fccf5c4809602468725776257facc16f867e4c25cd86` |

运行配置为 BS512、TP8/EP8、40 个专家层、top-6、融合 shared expert、
32 GiB KV/rank、原生 greedy decode。只运行 `mega` 与 `gc`，没有 RR、ECO
或额外臂。时间线：

| Arm | 启动 | 结束 | 退出码 |
|---|---|---|---:|
| `mega` | 2026-11-22T20:10:29+08:00 | 2026-11-22T20:16:43+08:00 | 0 |
| `gc` | 2026-11-22T20:16:44+08:00 | 2026-11-22T20:24:14+08:00 | 0 |

为避开公网 wheel 下载停顿，CuTe DSL 4.8.1 使用本地缓存中的五个同版本官方
wheel；五个文件的 SHA256 在 `dsl-wheelhouse.sha256`，安装日志在
`dsl-install.log`。这只改变安装途径，不改变模型、kernel、输入或计时语义。
本轮没有修改上游实现，也不需要编译修复。

## 五层关键 rank 阶段

单位为 **us**。所有列都来自同一个关键 rank；区间可重叠，不能横向相加。
`routed L2+push` 已包含 `push/epi`，`entry→routed` 也可能与 shared 计算重叠。

### Mega

| 层 | rank | token | M块 | MoE | entry→routed | idx/barrier | pull | shared L1 | shared L2 | routed L1 | routed L2+push | push/epi | combine |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 00 | 0 | 400 | 54 | 826.112 | 672.096 | 662.784 | 14.560 | 662.912 | 5.472 | 135.648 | 128.288 | 124.672 | 11.968 |
| 08 | 5 | 234 | 45 | 552.896 | 378.080 | 371.264 | 13.216 | 371.616 | 5.472 | 119.392 | 109.568 | 106.208 | 48.832 |
| 20 | 6 | 293 | 50 | 607.488 | 410.528 | 401.920 | 13.056 | 402.048 | 6.784 | 127.744 | 115.424 | 111.168 | 66.368 |
| 32 | 0 | 486 | 62 | 546.144 | 374.176 | 367.904 | 14.048 | 368.096 | 5.984 | 144.896 | 134.816 | 131.040 | 21.088 |
| 39 | 0 | 425 | 53 | 563.360 | 415.200 | 405.984 | 13.856 | 405.888 | 4.768 | 120.256 | 113.024 | 109.088 | 20.480 |

### GC

实际绿上下文为 `dispatch=12, compute=136`。

| 层 | rank | token | M块 | MoE | entry→routed | idx/barrier | pull | shared L1 | shared L2 | routed L1 | routed L2+push | push/epi | combine |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 00 | 0 | 393 | 51 | 761.344 | 593.248 | 587.936 | 25.536 | 15.104 | 12.224 | 133.600 | 126.080 | 122.240 | 24.832 |
| 08 | 7 | 658 | 68 | 559.264 | 373.344 | 367.872 | 44.768 | 15.232 | 11.264 | 166.656 | 163.200 | 160.384 | 6.272 |
| 20 | 0 | 608 | 65 | 589.952 | 364.384 | 358.976 | 39.328 | 16.416 | 9.952 | 162.080 | 155.712 | 152.608 | 53.632 |
| 32 | 4 | 246 | 45 | 560.992 | 379.264 | 373.376 | 17.440 | 14.656 | 11.904 | 115.264 | 107.520 | 104.576 | 55.616 |
| 39 | 5 | 380 | 53 | 520.192 | 357.152 | 349.024 | 22.656 | 15.648 | 11.840 | 136.288 | 127.136 | 123.968 | 18.304 |

原始数值在 `phase_table.csv`；每层八个 rank 的计数、19 个设备时钟和漂移在相邻的
`step002_layerXX.counts.json`，完整 Perfetto 事件在对应的
`step002_layerXX.perfetto.json`。

## 十张 Perfetto 链接

| Arm | 层 | 文件 |
|---|---:|---|
| Mega | 0 | https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/sm100_megamoe/v41_native_phase_results/mega/perfetto/step002_layer00.perfetto.json |
| Mega | 8 | https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/sm100_megamoe/v41_native_phase_results/mega/perfetto/step002_layer08.perfetto.json |
| Mega | 20 | https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/sm100_megamoe/v41_native_phase_results/mega/perfetto/step002_layer20.perfetto.json |
| Mega | 32 | https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/sm100_megamoe/v41_native_phase_results/mega/perfetto/step002_layer32.perfetto.json |
| Mega | 39 | https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/sm100_megamoe/v41_native_phase_results/mega/perfetto/step002_layer39.perfetto.json |
| GC | 0 | https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/sm100_megamoe/v41_native_phase_results/gc/perfetto/step002_layer00.perfetto.json |
| GC | 8 | https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/sm100_megamoe/v41_native_phase_results/gc/perfetto/step002_layer08.perfetto.json |
| GC | 20 | https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/sm100_megamoe/v41_native_phase_results/gc/perfetto/step002_layer20.perfetto.json |
| GC | 32 | https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/sm100_megamoe/v41_native_phase_results/gc/perfetto/step002_layer32.perfetto.json |
| GC | 39 | https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/sm100_megamoe/v41_native_phase_results/gc/perfetto/step002_layer39.perfetto.json |

## 校验与测量产物

`validation.txt` 记录了两臂 10 个 counts 的逐层检查：每层 8 个 rank、token 总数
3072、slot 行之和等于 token、M 块重算一致、19 个时钟存在、所有阶段边界满足
`start <= a <= b <= end`、Perfetto 可解析。Mega 每张图有 96 个事件，GC 因 GC
kernel 更多每张有 112 个事件；差异来自 Kineto kernel 数量，不影响设备阶段轨道。

- [phase_table.csv](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/sm100_megamoe/v41_native_phase_results/phase_table.csv)
- [validation.txt](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/sm100_megamoe/v41_native_phase_results/validation.txt)
- [Mega result.json](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/sm100_megamoe/v41_native_phase_results/mega/result.json)
- [GC result.json](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/sm100_megamoe/v41_native_phase_results/gc/result.json)
- [SHA256SUMS](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/sm100_megamoe/v41_native_phase_results/SHA256SUMS)

每臂 8 个 `measure-rank0..7.npz` 均已提交；每个文件形状为
`clocks=(3,40,19)`、`counts=(3,40,48)`。带 profiler 的测量墙钟分别为 Mega
`798.516311 ms/步`、GC `534.380541 ms/步`；这些数字受采样影响，不能据此判断
两臂快慢。

## 原始证据、释放与清理

原始 Torch profile 与 anchor 保留在执行机
`/data0/home/admin/workspace/megamoe/v41-phase-bs512-ku9Hv473/bs512/{mega,gc}/`，
共 32 个文件；相对路径和 SHA256 在 `raw-evidence.sha256`。清理前、清理后均以
该清单校验为 **32/32 OK**。大文件不提交 Git。

- 释放检查：`nvidia-smi --query-compute-apps` 为空，八卡利用率均为 0%，
  `/dev/nvidia0..7` 的 `fuser` 均为空；见 `gpu_release.txt`。
- 运行锁与组锁均无持有者，`released.log` 为 `2026-11-22T20:24:15+08:00`；
  见 `lock-release.txt`。没有存活的 run/worker 进程，见
  `processes-stopped.txt`。
- 私有 upstream、DSL 环境、overlay、DG 缓存和 wheelhouse 已用 Python
  `shutil.rmtree` 清理；运行目录由 853,273,620 字节降至 88,711,914 字节，
  共回收 764,561,706 字节。原始 profile/anchor、measure NPZ 和日志保留；见
  `cleanup.txt`。

结果入口：`docs/sm100_megamoe/v41_native_phase_results/`。
