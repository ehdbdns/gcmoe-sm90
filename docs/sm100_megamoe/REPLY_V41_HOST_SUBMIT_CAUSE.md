# 【SM100】BS512 主机提交延迟 CPU 归因回执

执行者：**SM100**。本轮只执行 `V41_HOST_SUBMIT_CAUSE.md`，未执行任何 SM90
任务书。全程离线只读已有 Kineto/profile 与时钟锚点，**未申请 GPU、未启动模型**，
没有修改模型、kernel、输入 profile、锚点或计时语义。

## 结论

- 2 臂 × 8 rank × 40 个 CPU 窗口全部齐全，共 **640** 个窗口、**274,995** 个
  最内层独占时间片段。层 1–39 的段长总和与相邻 `launch_host_ns` 差一致，
  最大误差 **0.001453 us**（GC rank5 L39），远小于 0.1 us 容差。
- 第 20 层 Mega 晚 entry rank4 相对早 entry rank6：第 0 层已继承 **1349.492 us**；
  第 1 层首先新增 **+279.460 us**，第 20 层再新增 **+478.917 us**，最终累计
  **2521.898 us**。第 1–20 层新增量合计 **1172.406 us**，与
  `2521.898 - 1349.492` 完全一致。
- 第 20 层 GC 晚 entry rank2 相对早 entry rank0：第 0 层继承 **865.180 us**；
  第 1 层先新增 **+173.479 us**，第 20 层新增 **+159.324 us**，最终累计
  **2313.156 us**。第 1–20 层新增量合计 **1447.976 us**。
- 第 20 层最大扩大窗口不是由单个同步 API 或单个函数造成。Mega 的真实调用实例中，
  `vllm::sparse_attn_indexer` 一次调用贡献 **+169.300 us**，随后是
  `vllm::mm_mxfp8` **+46.354 us** 与 **+26.367 us**、`vllm::mxfp8_quantize`
  **+18.979 us** 与 **+15.477 us**；另有 Kineto 未覆盖区间合计 **+242.669 us**。
  GC 对应最大项为 quantize **+12.874 us**、mm_mxfp8 **+12.556/+10.758 us**、
  quantize **+7.514 us**、sparse indexer **+4.944 us**；未覆盖区间合计
  **+117.285 us**。
- 第 20 层 `MXFP8 quantize → dense blockscaled GEMM` 的最大空档里，晚 rank 的
  空档几乎全部是 **CPU 尚未提交**，不是 GPU 已排队但未执行：Mega rank4 空档
  **269.537 us**，CPU `cudaLaunchKernelExC` 在空档开始后 **259.588 us** 才发生，
  提交后到 GPU 开始仅 **9.949 us**，dense kernel 本身 **11.008 us**（corr
  **12072**）。GC rank2 分别为 **217.505 / 207.529 / 9.976 / 11.264 us**。
- `cudaStreamSynchronize` 与 `cudaEventSynchronize` 在提交线程上均为 **0** 次；
  `cudaDeviceSynchronize` 只在层 0 出现，每 rank 一次，共 16 次。第 20 层选中
  rank 的 `cudaStreamWaitEvent` 最大仅 **1.937 us**，没有发现慢同步等待。
- 现有 trace 中没有 `record_function`、five_worker 拷贝/审计标记或运行时包围事件。
  因而不能把这些延迟唯一指定给 five_worker 的拷贝、审计函数或某个源码函数；
  能确认的是具体模型 forward 调用的独占时间变长，以及若干两端调用明确的
  Kineto 未覆盖区间。**不能仅凭 unobserved 空白推断操作系统抢占**。

## 任务与输入

| 项目 | 值 |
|---|---|
| 任务书 | `docs/sm100_megamoe/V41_HOST_SUBMIT_CAUSE.md` |
| 任务书 SHA256 | `81dc17db0b3e02753b74979c866d9a227e10a6f7801b598204672c1e69c222ef` |
| 分段分析脚本 | `scripts/sm100/native_host_submit.py` |
| 分段脚本 SHA256 | `42c7541d49787d555c24ba838a06c55dba29dcf5dbb7a697e3683f949318036f` |
| 报告脚本 | `scripts/sm100/native_host_submit_report.py` |
| 报告脚本 SHA256 | `6161838a2f04276fca3fbf1812f1806c103d104f4016e0d06328422137d879cf` |
| 执行源码提交 | `b0c1b848234913550469235b5bae7f5ae951c335` |
| Python | 3.12.7 |
| CPU | 256 × Intel Xeon 6776P |
| 原始证据目录 | `/data0/home/admin/workspace/megamoe/v41-phase-bs512-ku9Hv473/bs512/` |
| 原始清单 SHA256 | `2c0ff8919a5a4ef94adb5f7d7a022d00e917d17fe0713034bf702be5ec471ff0` |
| `host_segments.csv.gz` SHA256 | `5acfd05129d69942b347788e01e8989b89d1ce13834db72b284ec565a5d17883` |

原始目录在分析前后均执行 `sha256sum --quiet -c raw-evidence.sha256`，
**32/32 SHA_OK**。原始 profile、anchor 与上轮结果均未改写。

## 全部窗口与审计

- 640 个窗口分别对应 2 臂 × 8 rank × 40 层；层 0 是启动锚点到首次图提交，
  层 1–39 是前一次图提交到本次图提交。
- 624 个层间窗口全部识别到 **40 次有序 `cudaGraphLaunch`**，提交 pid/tid 唯一；
  没有无法归属提交者的窗口，不需要回退交付原始 trace 片段。
- 层 1–39 的分段总和恒等式共 624 组，最大绝对误差 **0.001453 us**。
- 选中窗口的 category 总差与新增提交差共 16 组，最大误差 **0.001952 us**。
- dense GEMM 空档拆分 `尚未提交 + 已提交待执行 = 空档` 的误差为 **0**。
- 调用序列按 `(category,name,enclosing_cpu_op)` 对齐。8 组选中 rank 对中，
  7 组在层 0 到目标层全部一致；只有 GC L8 的层 0 启动窗口相差 1 个片段
  （1238 vs 1237），该处已明确保留而不是强行对齐。所有 16 个“首次显著扩大 /
  后续最大扩大”窗口的真实调用序列均一致，因此重复同名调用按同一线程的
  per-key occurrence 匹配。一个调用若有多个嵌套独占片段，先汇总为该真实调用
  实例的独占时间，再比较两端，避免把同一调用拆成多项重复计数。

## 第 20 层逐层提交差

`累计` 是晚 entry rank 的 `launch_host_ns` 减早 entry rank 的
`launch_host_ns`；`新增` 是本层相对上一层的变化。第 0 层没有“上一层新增”。
两臂相加关系均为 `累计[L-1] + 新增[L] = 累计[L]`，21/21 层成立。

| 层 | Mega 累计 (us) | Mega 新增 (us) | GC 累计 (us) | GC 新增 (us) |
|---:|---:|---:|---:|---:|
| 0 | 1349.492 | — | 865.180 | — |
| 1 | 1628.952 | +279.460 | 1038.659 | +173.479 |
| 2 | 1701.520 | +72.568 | 1187.586 | +148.927 |
| 3 | 1888.347 | +186.827 | 1282.325 | +94.739 |
| 4 | 2007.017 | +118.670 | 1328.581 | +46.256 |
| 5 | 2067.223 | +60.206 | 1395.212 | +66.631 |
| 6 | 2112.917 | +45.694 | 1464.191 | +68.979 |
| 7 | 1935.684 | -177.233 | 1496.188 | +31.997 |
| 8 | 1918.446 | -17.238 | 1555.920 | +59.732 |
| 9 | 1825.963 | -92.483 | 1611.854 | +55.934 |
| 10 | 1766.670 | -59.293 | 1673.216 | +61.362 |
| 11 | 1717.221 | -49.449 | 1725.006 | +51.790 |
| 12 | 1710.456 | -6.765 | 1786.149 | +61.143 |
| 13 | 1725.076 | +14.620 | 1825.258 | +39.109 |
| 14 | 1624.150 | -100.926 | 1975.372 | +150.114 |
| 15 | 1623.805 | -0.345 | 2017.518 | +42.146 |
| 16 | 1580.042 | -43.763 | 2059.852 | +42.334 |
| 17 | 1874.490 | +294.448 | 2095.304 | +35.452 |
| 18 | 2019.284 | +144.794 | 2125.951 | +30.647 |
| 19 | 2042.981 | +23.697 | 2153.832 | +27.881 |
| 20 | **2521.898** | **+478.917** | **2313.156** | **+159.324** |

表中负新增表示该窗口晚 rank 比早 rank 缩回了一些已有差；不能把最终累计差归给
最后一次调用。Mega L20 的最终累计中，**1349.492 us 已在第 0 层继承**；
GC L20 中 **865.180 us 已在第 0 层继承**。

## 层 0 锚点与启动偏差

首锚点差是两 rank 进入采集窗口的启动先后，不是 CPU 时钟校准偏移；第二锚点是
近同步端点，八组仅相差 **-7.218 至 +23.971 us**。因此逐层表使用同一
`baseTimeNanoseconds` 下的主机 launch 时间，并只用第二锚点核对两端处在同一
同步后窗口。

| Arm | 目标层 | 早/晚 rank | 首锚点差 (us) | 第二锚点差 (us) | L0 提交差 (us) | 最终累计 (us) |
|---|---:|---:|---:|---:|---:|---:|
| Mega | 8 | 5 / 4 | +799.783 | +20.768 | 1821.044 | 1878.737 |
| Mega | 20 | 6 / 4 | +1007.106 | +23.971 | 1349.492 | 2521.898 |
| Mega | 32 | 0 / 4 | -40.456 | +5.852 | 2403.227 | 9001.467 |
| Mega | 39 | 0 / 4 | -40.456 | +5.852 | 2403.227 | 10405.394 |
| GC | 8 | 7 / 2 | +76760.711 | -4.856 | 816.210 | 1490.313 |
| GC | 20 | 0 / 2 | +17354.169 | -7.218 | 865.180 | 2313.156 |
| GC | 32 | 4 / 2 | +73401.561 | +3.872 | 1046.840 | 3956.058 |
| GC | 39 | 5 / 2 | +10043.827 | +9.790 | 999.128 | 2182.810 |

GC 首锚点差很大，但 L0 提交差只有数百至约一千微秒，说明不能用首锚点差直接
平移提交时间。第二锚点差和最终同步端点均在几十微秒内，支持两臂使用同一
主机时间基准比较。

## 层 8 / 20 / 32 / 39 扩大位置

“首次”是层 1 起第一个新增差 ≥100 us 的窗口；“后续最大”是除首次以外新增差
最大的窗口。`目标新增` 是目标层自身相对上一层的新增量，不是最终累计。

| Arm | 目标层 | 早/晚 rank | 首次：层 / 新增 (us) | 后续最大：层 / 新增 (us) | 目标层新增 (us) | 目标层累计 (us) |
|---|---:|---:|---:|---:|---:|---:|
| Mega | 8 | 5 / 4 | L3 / +156.156 | L6 / +128.567 | -154.463 | 1878.737 |
| Mega | 20 | 6 / 4 | L1 / +279.460 | L20 / +478.917 | +478.917 | 2521.898 |
| Mega | 32 | 0 / 4 | L1 / +443.281 | L20 / +544.840 | +166.271 | 9001.467 |
| Mega | 39 | 0 / 4 | L1 / +443.281 | L20 / +544.840 | +239.600 | 10405.394 |
| GC | 8 | 7 / 2 | L1 / +201.985 | L2 / +94.628 | +80.196 | 1490.313 |
| GC | 20 | 0 / 2 | L1 / +173.479 | L20 / +159.324 | +159.324 | 2313.156 |
| GC | 32 | 4 / 2 | L1 / +232.349 | L14 / +196.437 | +101.121 | 3956.058 |
| GC | 39 | 5 / 2 | L1 / +137.753 | L14 / +100.820 | +54.549 | 2182.810 |

第 8/32/39 层可以确认“从第 1 层起出现显著扩大”，但目标层自身的增量可能缩回；
例如 Mega L8 目标层新增为 **-154.463 us**。因此不能简单把最终差全部归给目标层
最后一个调用。第 20 层是两臂都持续扩大、且存在最大 dense 前空档的层，下面重点
展开。

## 第 20 层类别分解

表中的 `未覆盖` 是两端都匹配的 Kineto unobserved 片段差；其他类别是真实调用
独占时间按 category 汇总后的差。`合计` 与新增提交差的误差最多 **0.001952 us**。

| Arm | 窗口 | 未覆盖 (us) | cpu_op (us) | cuda_driver (us) | cuda_runtime (us) | 合计 / 新增 (us) |
|---|---|---:|---:|---:|---:|---:|
| Mega | L1 首次 | +134.612 | +140.591 | +0.563 | +3.694 | +279.460 / +279.460 |
| Mega | L20 最大 | +242.669 | +266.233 | -5.972 | -24.012 | +478.919 / +478.917 |
| GC | L1 首次 | +164.134 | +15.599 | +4.414 | -10.669 | +173.478 / +173.479 |
| GC | L20 最大 | +117.285 | +46.645 | +2.688 | -7.295 | +159.322 / +159.324 |

`cuda_driver` / `cuda_runtime` 的负值说明晚 rank 的部分 API 独占时间反而更短；
真正扩大的是模型 `cpu_op` 和未覆盖时间，而不是 CUDA API 本身。

## 第 20 层首次显著扩大：真实调用 top 5

`call#` 是窗口内真实调用的顺序；`occ` 是同一 `(category,name,enclosing_cpu_op)`
的第几次调用。`父时长` 是该真实调用的完整 span，`独占` 是内部所有最内层片段
之和。表中 `早 → 晚`，差值均为晚减早。

| Arm | call# / occ | 调用 | 父时长 (us) | 独占 (us) | 差 (us) |
|---|---:|---|---:|---:|---:|
| Mega | 101 / 2 | `vllm::mm_mxfp8` | 181.779 → 236.044 | 166.686 → 213.504 | +46.818 |
| Mega | 27 / 1 | `vllm::all_gather` | 158.300 → 200.608 | 123.333 → 166.879 | +43.546 |
| Mega | 166 / 3 | `vllm::mxfp8_quantize` | 63.175 → 115.790 | 48.429 → 87.521 | +39.093 |
| Mega | 171 / 4 | `vllm::mm_mxfp8` | 190.383 → 203.097 | 168.253 → 189.406 | +21.153 |
| Mega | 115 / 3 | `vllm::mm_mxfp8` | 156.332 → 178.531 | 142.075 → 162.837 | +20.762 |
| GC | 56 / 1 | `vllm::mm_mxfp8` | 214.327 → 237.529 | 198.226 → 221.103 | +22.877 |
| GC | 51 / 1 | `vllm::mxfp8_quantize` | 73.715 → 84.959 | 56.840 → 66.128 | +9.288 |
| GC | 131 / 1 | `aten::copy_` | 19.818 → 24.880 | 13.418 → 18.371 | +4.953 |
| GC | 115 / 3 | `vllm::mm_mxfp8` | 152.721 → 156.682 | 138.684 → 142.995 | +4.312 |
| GC | 124 / 1 | `_C::fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert` | 35.561 → 36.404 | 18.476 → 21.906 | +3.431 |

首次窗口的未覆盖 top 3：Mega **+25.923 / +22.879 / +20.812 us**；
GC **+54.784 / +27.251 / +24.103 us**。完整 top 5 在
`contributions.csv`，两端调用与 correlation 在 `call_evidence.csv`。

## 第 20 层后续最大扩大：真实调用 top 5

| Arm | call# / occ | 调用 | 父时长 (us) | 独占 (us) | 差 (us) | 关联 |
|---|---:|---|---:|---:|---:|---|
| Mega | 147 / 1 | `vllm::sparse_attn_indexer` | 514.697 → 655.117 | 307.639 → 476.938 | **+169.300** | 嵌套 corr 12296…12380；indexer/topk kernels |
| Mega | 248 / 4 | `vllm::mm_mxfp8` | 220.305 → 263.746 | 202.101 → 248.454 | +46.354 | corr 12511；dense blockscaled GEMM |
| Mega | 74 / 1 | `vllm::mm_mxfp8` | 243.431 → 268.626 | 224.171 → 250.538 | +26.367 | corr 12072；dense blockscaled GEMM |
| Mega | 69 / 1 | `vllm::mxfp8_quantize` | 81.768 → 105.787 | 64.526 → 83.506 | +18.979 | corr 12061；MXFP8 quantize kernel |
| Mega | 243 / 2 | `vllm::mxfp8_quantize` | 74.283 → 92.838 | 60.073 → 75.550 | +15.477 | corr 12500；MXFP8 quantize kernel |
| GC | 69 / 1 | `vllm::mxfp8_quantize` | 73.528 → 87.211 | 57.953 → 70.827 | +12.874 | corr 12061；MXFP8 quantize kernel |
| GC | 248 / 4 | `vllm::mm_mxfp8` | 196.602 → 209.957 | 181.980 → 194.536 | +12.556 | corr 12511；dense blockscaled GEMM |
| GC | 74 / 1 | `vllm::mm_mxfp8` | 204.258 → 215.649 | 189.003 → 199.761 | +10.758 | corr 12072；dense blockscaled GEMM |
| GC | 243 / 2 | `vllm::mxfp8_quantize` | 75.043 → 75.925 | 54.311 → 61.824 | +7.514 | corr 12500；MXFP8 quantize kernel |
| GC | 147 / 1 | `vllm::sparse_attn_indexer` | 415.619 → 427.762 | 280.589 → 285.533 | +4.944 | 嵌套 corr 12296…12380；indexer/topk kernels |

未覆盖片段 top 5：Mega **+26.831 / +23.580 / +20.772 / +17.344 / +13.309 us**；
GC **+20.558 / +19.702 / +17.517 / +12.516 / +7.151 us**。这些片段的相邻 CPU op
和 CUDA API 已写入 `call_evidence.csv`；其两端之外的空白没有 span，不能证明
是操作系统抢占，也不能指定某个源码函数。

## dense GEMM 前长空档：未提交 vs 已排队

第 20 层最大空档都从 `MXFP8 quantize` kernel 结束开始，到 dense blockscaled GEMM
开始结束。下表分别给出 CPU 在空档开始后多久发出 `cudaLaunchKernelExC`，以及
发出后多久 GPU 真正开始执行。

| Arm | rank 角色 | 空档 (us) | CPU 延迟 (us) | 提交后待执行 (us) | dense kernel (us) | correlation |
|---|---:|---:|---:|---:|---:|---:|
| Mega | 早 rank6 | 223.842 | 213.265 | 10.577 | 7.200 | 12511 |
| Mega | 晚 rank4 | **269.537** | **259.588** | **9.949** | 11.008 | **12072** |
| GC | 早 rank0 | 204.162 | 194.607 | 9.555 | 6.911 | 12511 |
| GC | 晚 rank2 | **217.505** | **207.529** | **9.976** | 11.264 | **12072** |

三列关系为 `空档 = CPU 尚未提交 + 提交后待执行`，四行误差均为 **0**。
晚 rank 与早 rank 的差异几乎全部发生在 CPU 提交之前：Mega 为
`259.588 - 213.265 = 46.323 us`，GC 为 `207.529 - 194.607 = 12.922 us`；
提交后的待执行尾差只有约 0.6 us 和 0.4 us。因此这里不是 dense GEMM 自身变慢，
也不是 GPU 队列里已有大量工作等待，而是对应 CPU 启动路径在提交 dense kernel
之前多花了时间。

## 同步与 five_worker 证据边界

- `cudaStreamSynchronize=0`、`cudaEventSynchronize=0`。
- `cudaDeviceSynchronize=16`，恰好每臂每 rank 一次，且全部在层 0。
- `cudaStreamWaitEvent=768`，全局最大 **7.103 us**，超过 10 us 的记录为 0；
  第 20 层四个选中 rank 最大 **1.937 us**。没有“慢同步 API 等设备”的证据。
- 202,692 个保留事件中没有 `record_function`、five_worker 或 audit 标记。
  选中调用序列是正常模型 forward 的 `vllm::mm_mxfp8`、`mxfp8_quantize`、
  `sparse_attn_indexer`、`all_gather`、fused norm/rope 等；没有可据以指定
  five_worker 拷贝/审计函数的包围事件。

## 覆盖范围与限制

- 全部 640 个窗口都完成了分段和层 1–39 恒等式核验；但详细逐调用归因只覆盖
  `L8/20/32/39` 的 8 组 entry 极值 rank 对，不能把某一组的 top 调用外推到所有
  rank 或所有层。
- 第 20 层两臂都显示：首次显著扩大发生在 L1，后续最大扩大仍在 L20；L0 已继承
  的差必须单列，不能归给 L20 最后一次提交。
- `unobserved` 只能说明 Kineto 在该时段没有 CPU/CUDA span 覆盖。回执给出了其
  两端的具体调用及微秒数，但没有足够观测量证明 OS 抢占、缺页、锁竞争或某个
  具体函数；若要闭环，需要更细的 userspace 采样或显式包围事件。
- 本轮没有重跑 GPU、没有新增模型实验，也没有改动原始证据。

## 交付文件

- [host_segments.csv.gz](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/sm100_megamoe/v41_host_submit_results/host_segments.csv.gz)：274,995 行分段。
- [cumulative_submission_diffs.csv](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/sm100_megamoe/v41_host_submit_results/cumulative_submission_diffs.csv)：206 行逐层累计/新增提交差。
- [expansion_windows.csv](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/sm100_megamoe/v41_host_submit_results/expansion_windows.csv)：8 组首次/最大/目标窗口。
- [contributions.csv](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/sm100_megamoe/v41_host_submit_results/contributions.csv)：类别总差、真实调用 top 5、未覆盖片段 top 5。
- [call_evidence.csv](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/sm100_megamoe/v41_host_submit_results/call_evidence.csv)：160 行调用/空档证据。
- [key_intervals.csv](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/sm100_megamoe/v41_host_submit_results/key_intervals.csv)：第 20 层关键窗口类别恒等式。
- [launch_gap_evidence.csv](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/sm100_megamoe/v41_host_submit_results/launch_gap_evidence.csv)：dense 前“未提交/待执行”拆分。
- [sequence_alignment.txt](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/sm100_megamoe/v41_host_submit_results/sequence_alignment.txt)：片段与真实调用序列对齐记录。
- [validation.txt](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/sm100_megamoe/v41_host_submit_results/validation.txt)、[versions.txt](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/sm100_megamoe/v41_host_submit_results/versions.txt)、[SHA256SUMS](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/sm100_megamoe/v41_host_submit_results/SHA256SUMS)：核验与版本。

核验结论：无错误、无 GPU 使用、无模型启动；原始输入 32/32 校验通过。
