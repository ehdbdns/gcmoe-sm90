# 【SM100】V41 dense 回放后 L19→L20 剩余错峰回执

执行者：**SM100**。本轮只执行 `docs/sm100_megamoe/V41_RESIDUAL_L20.md`，
未执行任何 SM90 任务书。

**本轮是纯 CPU 离线分析：没有申请 GPU、没有启动模型、没有占用 GPU，也没有修改
模型、kernel、dense 回放、输入或逐层计时语义。** step 4/6 只读取已归档 NPZ 和外部
时钟锚点，没有把 step 1 的 kernel 移植到其他步。

## 输入与校验

| 项目 | 值 |
|---|---|
| 任务书 SHA256 | `74fc965719dee079f1e7efba900c015aa966856c07ee74b7bb4833c155618edd` |
| 执行 checkout | `ab27d06e785f04f0ba22f86a2eea8f4ef81482b9` |
| 原始证据 | `/data0/home/admin/workspace/megamoe/v41-dense-submit-5u4m9Asy/run/bs512` |
| 历史归档 | `docs/sm100_megamoe/v41_dense_submit_results` |
| 原始 SHA256 清单 | `8e4e0b8d5665aedcb94adffedaeb38a2f158004d9ce3dc4e773475d4c7a77c2e` |
| 原始 SHA 复验 | `sha256sum --quiet -c raw-evidence.sha256` → `RAW_SHA_OK` |
| GPU | **未使用** |

step 1 的两锚点映射为
`scale=(mark2-mark1)/(dev2-dev1)`、
`trace_us=mark1+(device_ns-dev1)*scale`。窗口活动按所有
kernel/gpu_memcpy/gpu_memset 区间取并集。16 个窗口均满足
`window = union_busy + gap_total`，最大绝对闭合误差为 **0 us**。

原始 `scripts/sm100/residual_l20.py` 会分别列出每个重叠区间；同 rank 直接求和会
重复计算，因此本回执的 gap 总量使用并集补集。逐个原始设备事件保留在
`v41_residual_l20_results/window_events.csv`，并集表为 `window.csv`。

## 结论

- GC step 1 的 L19 出口跨 rank 差为 `0.996 us`，但 L20 入口差为 `492.612 us`。
  Mega 分别为 `0.784 us` 和 `291.885 us`。**剩余错峰不是 L19 combine 出口造成的。**
- 正常 rank 的层间设备空档只有 `52.671–60.291 us`；晚到 rank 的空档为：
  GC rank0 `3894.904 us`、rank1 `3896.426 us`，Mega rank4 `2682.639 us`。
- 晚到 rank 的残差由离散空档组成：GC rank0/1 各有 **36** 个 `>=10 us` 空档，
  合计 `3878.424/3880.971 us`；Mega rank4 有 **32** 个，合计 `2664.078 us`。
  其余 13 个 rank 的 `>=10 us` 空档数全部为 **0**。
- all-gather 的结束边界基本对齐后，第一条有证据的再次扩大出现在其后的设备空档：
  GC 下游 cublas_tss 起点晚 `212.546 us`，Mega 晚 `168.367 us`。这是设备侧
  空档，不是 kernel 时长增长；本 trace 没有 host API/线程状态证据，不能归因为
  CPU 抢占。
- 最后一个 collective reduce-scatter 的结束边界重新收敛：GC 晚 `0.856 us`、
  Mega 晚 `5.575 us`。随后空档再次逐段扩大，最终到 L20 入口的
  `492.856/292.122 us`。因此残差跨越“早段空档积压 → collective 内等待 →
  collective 后再次空档扩大”的完整链条，而不是单个 kernel 自身变慢。
- step 4/6 在无 profiler 的条件下复现同量级：入口差 `260.311–281.537 us`，
  span 差 `260.352–281.568 us`；但早晚 rank 会轮换，不是固定某一张卡。

## step 1 逐 rank 并集与空档

| Arm | Rank | 角色 | window us | busy union us | gap total us | >=10us gaps | max gap us |
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

busy union 包含 collective kernel 内部的 peer wait，不代表纯计算时间。GC 早 rank
的 reduce-scatter 与 mhc#2 都记录约 `3.26 ms`，但同 stream 上互相重叠约
`3251 us`；Mega 早 rank 对应约 `2.31 ms`、重叠约 `2295 us`。这是原始 Kineto
重叠，不能相加，也不能据此推断因果。晚 rank 上 reduce-scatter 只有 `17.792 us`，
说明早 rank 的长时长与等待其余 rank 相符。

最大的空档及两侧真实调用：

| Arm/rank | gap us | 位置 us | 前 kernel, stream, correlation | 后 kernel, stream, correlation |
|---|---:|---:|---|---|
| GC 0 | 357.762 | 889.344–1247.105 | q_kv_norm, 23, 12034 | dense_gemm, 23, 12053 |
| GC 1 | 357.313 | 888.658–1245.972 | q_kv_norm, 23, 12034 | dense_gemm, 23, 12053 |
| Mega 4 | 251.745 | 592.643–844.388 | q_kv_norm, 23, 12034 | dense_gemm, 23, 12053 |
| GC 0 | 218.913 | 158.012–376.925 | all_gather, 23, 11851 | cublas_tss, 35, 11910 |
| Mega 4 | 171.009 | 98.560–269.568 | all_gather, 23, 11851 | cublas_tss, 35, 11910 |

`largest_gaps.csv` 给出全部空档及其两侧 kernel/stream/correlation。

## collective 前后边界

| Arm | 边界 | early us | late us | start delta us | end delta us |
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

collective 结束后，mhc#2、gate、prepare 的 kernel 时长差异均远小于 1 us，空档差却
逐段增大；这说明残差来自 kernel 间的未活跃区间，而不是这些 kernel 变慢。没有
host CUDA API、CPU 线程运行状态或显式依赖边，因此只能报告设备空档。

`boundary_alignment.csv` 按同 stream、correlation、kernel 名和调用顺序对齐，包含
两臂各 44 个共同边界；共 **88/88 matched，0 unmatched**。重叠事件有显式标记，
其中早 rank 的 reduce-scatter/mhc#2 重叠为主证据中的 trace 重叠，不用于因果结论。

## step 4/6 无 profiler 核对

`clock-before` 的 Kineto 事件是 `ts=0,dur=0`，step 4/6 使用同一归档的
`clock-after` 单锚点映射 device globaltimer。该方法没有跨锚点漂移修正；late rank
与其余 rank 相差 260 us 以上，判断稳健，聚集在数微秒内的 early rank 次序只具有
单锚点精度。step 1 仍使用双锚点。

| Arm | Step | dense | early rank | late rank | L19 spread us | L20 spread us | span early/late us |
|---|---:|---|---:|---:|---:|---:|---:|
| GC | 1 | on | 2 | 0 | 0.996 | 492.612 | 3762.173 / 4255.029 |
| GC | 4 | on | 2 | 3 | 1.828 | 270.719 | 2145.664 / 2415.328 |
| GC | 6 | on | 5 | 0 | 1.153 | 281.537 | 2302.880 / 2584.448 |
| Mega | 1 | on | 3 | 4 | 0.784 | 291.885 | 2747.637 / 3039.759 |
| Mega | 4 | on | 0 | 7 | 1.253 | 269.888 | 2146.624 / 2416.256 |
| Mega | 6 | on | 3 | 0 | 1.054 | 260.311 | 2032.896 / 2293.248 |

step 4/6 的量级与 step 1 的 `260–282 us` 残差一致，且 rank 角色轮换。未把 step 1
的 kernel 或持续时间移植到其他步。

## Perfetto

- GC：`docs/sm100_megamoe/v41_residual_l20_results/gc_step001_l19_l20.perfetto.json`
- Mega：`docs/sm100_megamoe/v41_residual_l20_results/mega_step001_l19_l20.perfetto.json`

两张图均为八 rank、公共 Kineto 基准、真实跨 rank 偏移；没有按各 rank 的 L19 出口
置零。公共基准分别为 GC `4503005873172.310 us`、Mega
`4502618852975.567 us`，图内绝对偏移只减这个公共值。图中同时标出 L19 出口、
L20 入口和全流并集设备空档。

## 首个有证据的差值与本轮残差

**首个有证据的再次扩大**发生在 all-gather 结束对齐之后：GC 下游 cublas_tss 起点差
`212.546 us`，Mega 为 `168.367 us`；其后又出现 32–36 个离散设备空档。最后
collective 的结束边界虽收敛到 `0.856/5.575 us`，但 post-collective mhc#2 结束差
已扩大到 `107.577/72.424 us`，之后继续扩大到 L20 入口。

本轮能确认的是：**晚到 rank 的 kernel 时长没有整体增长，残差由 repeated device
gaps 与 collective 内部等待共同构成。** 仍缺的观测是 host CUDA API 时间线、
CPU 线程调度/抢占记录、以及跨 rank collective 的显式依赖边；没有这些证据，不把
空档归因于线程抢占、CPU 提交或某个未观测依赖。

## 产物

全部结果在 `docs/sm100_megamoe/v41_residual_l20_results/`：

- `window.csv`、`window_events.csv`、`window_summary.csv`、`largest_gaps.csv`
- `boundary_alignment.csv`、`boundary_delta_contributors.csv`、`collective_boundary_summary.csv`
- `step146_window.csv`
- `gc_step001_l19_l20.perfetto.json`、`mega_step001_l19_l20.perfetto.json`
- `analyze_residual_l20.py`、`README.md`、`analysis_manifest.json`、`analysis_validation.txt`
- `raw-evidence.sha256`、`raw-evidence-paths.txt`、`raw_sha_validation.txt`

复验命令：

```bash
/data0/home/admin/workspace/megamoe/venv/bin/python \
  docs/sm100_megamoe/v41_residual_l20_results/analyze_residual_l20.py \
  /data0/home/admin/workspace/megamoe/v41-dense-submit-5u4m9Asy/run/bs512 \
  docs/sm100_megamoe/v41_dense_submit_results \
  docs/sm100_megamoe/v41_residual_l20_results --task-commit ab27d06
```

回执结束。**SM100 本轮未占 GPU，未执行 SM90。**
