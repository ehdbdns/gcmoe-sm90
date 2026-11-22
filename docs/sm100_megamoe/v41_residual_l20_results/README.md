# V41 residual L20 analysis (SM100)

本目录是 `docs/sm100_megamoe/V41_RESIDUAL_L20.md` 的 CPU-only 离线分析产物。
未启动模型、未申请 GPU、未改动模型/ kernel / dense 回放语义。

## 输入与对齐

- 原始证据：`/data0/home/admin/workspace/megamoe/v41-dense-submit-5u4m9Asy/run/bs512`
- 结果归档：`docs/sm100_megamoe/v41_dense_submit_results`
- 原始 SHA 校验：`sha256sum --quiet -c raw-evidence.sha256`，结果为 `RAW_SHA_OK`。
- step 1 使用每 rank 的两个 `sm100_phase_clock_anchor` 将 device globaltimer
  映射到公共 Kineto 时间轴：
  `scale=(mark2-mark1)/(dev2-dev1)`，
  `trace_us=mark1+(device_ns-dev1)*scale`。
- 16 个 step 1 窗口的并集、活动和空档均以所有 kernel/gpu_memcpy/gpu_memset
  区间取并集后计算；`window = union_busy + gap_total` 的最大闭合误差为 `0 us`。
- Perfetto 对所有 rank 只减去同一个公共 Kineto 基准，不按各 rank 的 L19 出口置零。

原始 `scripts/sm100/residual_l20.py` 会分别输出每个重叠事件区间，不能把同 rank
事件时长直接相加。这里的 `window.csv` 是修正后的并集 busy/gap 表；
`window_events.csv` 保留逐个设备事件，供核对重叠。

## step 1 窗口

| Arm | Rank | 角色 | window (us) | busy union (us) | gap total (us) | >=10us gaps | max gap (us) |
|---|---:|---|---:|---:|---:|---:|---:|
| GC | 0 | late | 4255.029 | 360.125 | 3894.904 | 36 | 357.762 |
| GC | 1 |  | 4250.733 | 354.308 | 3896.426 | 36 | 357.313 |
| GC | 2 | early | 3762.173 | 3709.502 | 52.671 | 0 | 2.976 |
| GC | 3 |  | 3763.355 | 3706.877 | 56.479 | 0 | 4.287 |
| GC | 4 |  | 3762.706 | 3706.867 | 55.839 | 0 | 3.424 |
| GC | 5 |  | 3766.010 | 3706.233 | 59.776 | 0 | 4.640 |
| GC | 6 |  | 3765.269 | 3712.084 | 53.185 | 0 | 4.289 |
| GC | 7 |  | 3762.240 | 3704.638 | 57.603 | 0 | 3.584 |
| Mega | 0 |  | 2749.488 | 2691.053 | 58.436 | 0 | 3.488 |
| Mega | 1 |  | 2750.440 | 2692.268 | 58.173 | 0 | 3.200 |
| Mega | 2 |  | 2752.214 | 2691.923 | 60.291 | 0 | 6.240 |
| Mega | 3 | early | 2747.637 | 2692.660 | 54.977 | 0 | 3.359 |
| Mega | 4 | late | 3039.759 | 357.120 | 2682.639 | 32 | 251.745 |
| Mega | 5 |  | 2747.922 | 2692.111 | 55.811 | 0 | 3.392 |
| Mega | 6 |  | 2752.655 | 2698.672 | 53.983 | 0 | 3.487 |
| Mega | 7 |  | 2748.187 | 2692.923 | 55.264 | 0 | 3.840 |

GC 的 L19 出口跨 rank 差 `0.996 us`，L20 入口差 `492.612 us`；Mega 分别为
`0.784 us` 和 `291.885 us`。正常 rank 的窗口只有 `53.98–60.29 us` 空档，晚到
rank 的大空档来自 32–36 个 `>=10 us` 的离散设备空档。这里的 busy union 包含
collective kernel 内部的等待，不等于纯计算时间：GC 早 rank 的 reduce-scatter/
mhc#2 位于同 stream，却各记录约 `3.26 ms`，且互相重叠约 `3251 us`；Mega 早 rank
对应约 `2.31 ms`，重叠约 `2295 us`。这是原始 Kineto 重叠，不能把两者相加或据此
推断因果。

最大空档两侧的调用在 `largest_gaps.csv` 中逐条给出。GC rank 0 最大空档为
`357.762 us`，从 q_kv_norm（stream 23，correlation 12034）到 dense_gemm
（stream 23，correlation 12053）；Mega rank 4 的同位置空档为 `251.745 us`。

## collective 前后

| Arm | 边界 | early (us) | late (us) | start delta (us) | end delta (us) |
|---|---|---:|---:|---:|---:|
| GC | all_gather | 27.962–158.587 | 145.883–158.012 | +117.921 | -0.575 |
| GC | post-AG cublas_tss | 164.379–176.923 | 376.925–386.205 | +212.546 | +209.282 |
| GC | last collective reduce-scatter | 467.038–3728.215 | 3711.279–3729.071 | +3244.241 | +0.856 |
| GC | post-collective mhc#2 | 477.213–3740.151 | 3835.600–3847.729 | +3358.387 | +107.577 |
| GC | mega_gate | 3740.086–3748.151 | 3963.824–3972.112 | +223.738 | +223.961 |
| GC | prepare | 3749.688–3752.023 | 4142.865–4145.201 | +393.178 | +393.178 |
| GC | L20 entry | 3762.173 | 4255.029 | +492.856 | +492.856 |
| Mega | all_gather | 30.544–97.553 | 87.744–98.560 | +57.200 | +1.007 |
| Mega | post-AG cublas_tss | 101.201–110.226 | 269.568–278.912 | +168.367 | +168.687 |
| Mega | last collective reduce-scatter | 405.299–2710.725 | 2698.508–2716.300 | +2293.209 | +5.575 |
| Mega | post-collective mhc#2 | 415.636–2721.701 | 2781.806–2794.125 | +2366.170 | +72.424 |
| Mega | mega_gate | 2721.861–2730.501 | 2857.518–2865.998 | +135.656 | +135.497 |
| Mega | prepare | 2733.093–2735.973 | 2967.854–2970.222 | +234.761 | +234.249 |
| Mega | L20 entry | 2747.637 | 3039.759 | +292.122 | +292.122 |

两个 all-gather 的结束边界已经对齐（GC `-0.575 us`、Mega `+1.007 us`）；
第一个有证据的再次扩大发生在它之后：GC 下游 cublas_tss 起点差 `212.546 us`，
Mega 为 `168.367 us`。最后一个 collective 的结束边界也重新对齐到 `0.856/5.575 us`，
但随后 mhc#2、mega_gate、prepare 和 L20 入口的空档逐段扩大。kernel 自身时长基本
相同（差异通常小于 1 us），扩大来自 kernel 之间的设备空档，而不是这些 kernel
变慢。早 rank 的 reduce-scatter 持续时间很长、晚 rank 只有 `17.792 us`，与早
rank 在 collective 内等待其余 rank 一致，但本 trace 没有 host API/依赖证据，
因此不把 CPU 调度或线程抢占写成原因。

`boundary_alignment.csv` 含两臂各 44 个共同边界（共 88 行），按 stream、correlation、
kernel 名和调用顺序匹配，simultaneous 重叠流单列；两臂均为 `88/88 matched`。

## step 4/6 外部锚点核对

step 4/6 没有 Kineto kernel；`clock-before` 的 `ts=0,dur=0`，因此使用
`clock-after` 单锚点把 device globaltimer 映射到公共时间轴。该回退完全来自归档中的
外部锚点，不借用 step 1 kernel。step 1 仍使用双锚点。

| Arm | Step | dense | early rank | late rank | L19 spread (us) | L20 spread (us) | span early/late (us) |
|---|---:|---|---:|---:|---:|---:|---:|
| GC | 1 | on | 2 | 0 | 0.996 | 492.612 | 3762.173 / 4255.029 |
| GC | 4 | on | 2 | 3 | 1.828 | 270.719 | 2145.664 / 2415.328 |
| GC | 6 | on | 5 | 0 | 1.153 | 281.537 | 2302.880 / 2584.448 |
| Mega | 1 | on | 3 | 4 | 0.784 | 291.885 | 2747.637 / 3039.759 |
| Mega | 4 | on | 0 | 7 | 1.253 | 269.888 | 2146.624 / 2416.256 |
| Mega | 6 | on | 3 | 0 | 1.054 | 260.311 | 2032.896 / 2293.248 |

step 4/6 仍复现 `260–282 us` 的入口差与几乎相同的 span 差，但早晚 rank 会轮换，
不是固定某一张卡。由于单锚点没有跨锚点漂移修正，跨越几十微秒的 late rank 判断稳健，
而聚集在数微秒内的多个 early rank 先后次序只具有该回退方法的精度。

## 文件

- `window.csv`：16 个 step 1 窗口的并集 busy/gap 段。
- `window_events.csv`：逐个设备事件及同 stream 重叠标记。
- `window_summary.csv`：逐 rank 窗口、并集、空档闭合、最大 gap 与相邻调用。
- `largest_gaps.csv`：逐 rank 全部空档按大小排序，附两侧 kernel/stream/correlation。
- `boundary_alignment.csv`：两臂 44 个共同边界，含累计/新增 start、end 差和重叠标记。
- `boundary_delta_contributors.csv`：新增差超过 1 us 的边界贡献。
- `collective_boundary_summary.csv`：all-gather、最后一个 collective 与 L20 入口汇总。
- `step146_window.csv`：step 1/4/6 的八 rank 外部锚点结果。
- `gc_step001_l19_l20.perfetto.json`、`mega_step001_l19_l20.perfetto.json`：八 rank 图。
- `analysis_manifest.json`、`analysis_validation.txt`：方法、覆盖和校验。
- `raw-evidence.sha256`、`raw-evidence-paths.txt`、`raw_sha_validation.txt`：原始证据。

## 未归因残差

设备侧已闭合到公共时间轴的窗口，但现有证据没有 host CUDA API、CPU 线程运行状态或
跨 rank 依赖边。因此只能确认晚 rank 的 32–36 个离散设备空档与 collective 内的等待，
不能判定这些空档来自 CPU 提交、线程抢占、图调度还是未显式记录的依赖。
