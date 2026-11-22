# 【SM100】BS512 层间到达偏差 CPU 复核回执

执行者：**SM100**。本轮只执行 `V41_ENTRY_SKEW_CPU.md`，未执行任何 SM90
任务书。分析全程只读上轮 Kineto 与时钟锚点，**未申请 GPU、未启动模型**。

## 结论

- 两臂各 8 rank、每 rank 39 个层间区间，共 **624 行**；每行都有 40 次有序
  `cudaGraphLaunch`，`prev_end_to_launch_us` 与 `launch_to_entry_us` 均非空，
  因此不需要交付 trace 回退片段。
- 对层 8/20/32/39，entry 最早与最晚 rank 的区间长度差为
  **340.677–389.599 us**；同一对 rank 的 `entry_host_ns` 差为
  **341.974–389.190 us**，两者一致到亚微秒。
- 较晚 entry 的 rank 在更长区间内有更少的 GPU 活动并集、更多的空档；八组
  `Δgap = Δdevice_active_union + Δuncovered` 的误差全部为 **0**。
- entry 最晚的 rank，图提交到 GPU 入口仅需 **19.552–22.584 us**；entry 最早的
  rank，图提交到入口为 **1.151–10.039 ms**，其中三组提交发生在上一层设备结束前
  （负值），显示 CPU 图提交队列存在明显提前量差异。
- 最晚 entry rank 的最长空档通常为 **200.321–346.819 us**，边界多为
  `MXFP8 quantize` 或 `Q/KV norm+quant` 后等待 dense GEMM；但在层 32/39 最早
  entry rank 上，最长空档只有 **3.072–4.063 us**。这组关系与 CPU 提交偏差一致，
  但空档名称本身不足以单独证明某个 kernel 或调度器是根因。
- 两臂的单 rank kernel 分组总和范围是 **236.382–5550.111 us**。较早 entry 的
  rank 主要由 `MNNVL reduce_scatter` 与 `Mega MHC GEMM` 构成，较晚 entry 的 rank
  只剩约 100 us 的 fused norm/rope/attn 和 24–29 us 的 dense/MHC 分组。
  kernel 可重叠，这些总和不是 GPU 活动并集，也不能直接作为因果归因。

## 任务与输入

| 项目 | 值 |
|---|---|
| 任务书 | `docs/sm100_megamoe/V41_ENTRY_SKEW_CPU.md` |
| 任务书 SHA256 | `367c2cbf28c2b55eeb87f941936adabe935c86c5353599561b5c6e77219d84f0` |
| 分析源码 | `scripts/sm100/native_entry_skew.py` |
| 源码 SHA256 | `976955d178b3466d56955b84cabc9002f34b51e07eea2d70643afe28bec07fa3` |
| 执行源码提交 | `1636e5e3b9f1b29f42618bf9b88ac32cd5114d0a` |
| Python / numpy | 3.12.7 / 2.3.5 |
| CPU | 256 × Intel Xeon 6776P |
| 原始证据目录 | `/data0/home/admin/workspace/megamoe/v41-phase-bs512-ku9Hv473/bs512/` |
| 原始证据清单 SHA256 | `2c0ff8919a5a4ef94adb5f7d7a022d00e917d17fe0713034bf702be5ec471ff0` |

运行前从原始目录执行 `sha256sum --quiet -c source-raw-evidence.sha256`，32/32
通过；分析后再次校验仍为 **32/32 SHA_OK**。原始 profile/anchor 未复制或改写。
本轮直接复用既有执行者 checkout，没有新建失败残留的临时代码 worktree。

## 极值 rank 的区间分解

按每个 `arm/layer` 的 `entry_host_ns` 选最早和最晚 rank。单位为 **us**；每格是
`最早 rank / 最晚 rank / 最晚减最早`。`active` 为区间内 kernel/memcpy/memset
的设备活动并集，`uncovered` 为该并集之外的空档。

| Arm | 层 | rank 早/晚 | gap | active union | uncovered | 恒等式误差 |
|---|---:|---:|---:|---:|---:|---:|
| Mega | 8 | 5 / 4 | 2460.144 / 2811.690 / **+351.547** | 2173.549 / 266.239 / **-1907.310** | 286.595 / 2545.451 / **+2258.856** | 0 |
| Mega | 20 | 6 / 4 | 3779.767 / 4163.248 / **+383.481** | 3046.000 / 346.176 / **-2699.824** | 733.767 / 3817.072 / **+3083.306** | 0 |
| Mega | 32 | 0 / 4 | 2349.196 / 2698.507 / **+349.311** | 2315.729 / 294.017 / **-2021.712** | 33.468 / 2404.490 / **+2371.022** | 0 |
| Mega | 39 | 0 / 4 | 1607.593 / 1997.191 / **+389.599** | 1587.019 / 236.382 / **-1350.637** | 20.574 / 1760.810 / **+1740.235** | 0 |
| GC | 8 | 7 / 2 | 2256.436 / 2614.484 / **+358.049** | 1748.301 / 259.684 / **-1488.617** | 508.135 / 2354.801 / **+1846.666** | 0 |
| GC | 20 | 0 / 2 | 2989.841 / 3337.979 / **+348.138** | 2632.240 / 351.586 / **-2280.654** | 357.601 / 2986.393 / **+2628.792** | 0 |
| GC | 32 | 4 / 2 | 2358.698 / 2721.365 / **+362.667** | 2323.143 / 303.039 / **-2020.104** | 35.556 / 2418.326 / **+2382.771** | 0 |
| GC | 39 | 5 / 2 | 1413.705 / 1754.382 / **+340.677** | 1390.477 / 237.374 / **-1153.103** | 23.229 / 1517.008 / **+1493.779** | 0 |

## 图提交与最长空档

`prev→launch` 是 `cudaGraphLaunch` 相对上一层设备结束的时间，`launch→entry`
是提交到当前 MoE 入口的时间。负的 `prev→launch` 表示 CPU 已提前排队。
`gap` 列是最长 `uncovered` 时长及前后边界；缩写 `q` 为 MXFP8 quantize，
`qk` 为 Q/KV norm+quant，`dense` 为 FlashInfer dense blockscaled GEMM。

| Arm | 层 | 最早 rank：prev→launch / launch→entry | 最晚 rank：prev→launch / launch→entry | 最长空档 早 / 晚 |
|---|---:|---:|---:|---|
| Mega | 8 | 912.997 / 1547.146 | 2791.808 / 19.883 | q→dense 180.034 / qk→dense 236.545 |
| Mega | 20 | 1619.669 / 2160.098 | 4141.615 / 21.633 | q→dense 223.842 / q→dense 269.537 |
| Mega | 32 | **-6325.003** / 8674.199 | 2676.602 / 21.905 | MoE→Memcpy 3.072 / qk→dense 213.345 |
| Mega | 39 | **-8431.195** / 10038.788 | 1974.607 / 22.584 | Memcpy→elementwise 2.624 / q→dense 213.473 |
| GC | 8 | 1104.999 / 1151.437 | 2594.240 / 20.244 | q→dense 203.426 / q→dense 228.642 |
| GC | 20 | 1006.743 / 1983.098 | 3318.427 / 19.552 | q→dense 204.162 / q→dense 217.505 |
| GC | 32 | **-1255.669** / 3614.367 | 2699.034 / 22.331 | dense→qk 3.808 / qk→dense 346.819 |
| GC | 39 | **-448.300** / 1862.005 | 1733.211 / 21.171 | GC MoE→Memcpy 4.063 / q→dense 200.321 |

## Kernel 分组对比

表中只列各 rank 最大的两个分组，数值是所有同名的 kernel 在区间内的累计时间；
完整分组在 `kernels.csv`，top 5 在 `kernel_top.csv`。kernel 可重叠，累计值不等于
活动并集；同步或 collective kernel 的运行时间也可能包含等待。

| Arm | 层 | 最早 entry rank：kernel 总和；最高两组 | 最晚 entry rank：kernel 总和；最高两组 |
|---|---:|---:|---|
| Mega | 8 | 4012.539；reduce_scatter 1910.030，MHC 1827.629 | 266.239；fused norm 102.240，MHC 24.704 |
| Mega | 20 | 5550.111；reduce_scatter 2570.350，MHC 2498.189 | 346.176；fused norm 105.728，dense 29.216 |
| Mega | 32 | 4361.244；MHC 2053.452，reduce_scatter 2040.716 | 294.017；fused norm 105.825，dense 28.417 |
| Mega | 39 | 2896.085；MHC 1325.192，reduce_scatter 1312.200 | 236.382；fused norm 103.296，MHC 25.024 |
| GC | 8 | 3202.746；reduce_scatter 1509.517，MHC 1431.916 | 259.684；fused norm 100.320，MHC 24.512 |
| GC | 20 | 4890.899；reduce_scatter 2307.692，MHC 2242.829 | 351.586；fused norm 105.249，dense 29.312 |
| GC | 32 | 4379.538；MHC 2057.515，reduce_scatter 2045.450 | 303.039；fused norm 105.024，dense 29.185 |
| GC | 39 | 2558.354；MHC 1180.295，reduce_scatter 1166.887 | 237.374；fused norm 103.136，MHC 24.512 |

这里能确认的是：较晚 entry 的 rank 在更长的 CPU 区间内没有等到对应的 GPU
工作，且提交到入口的提前量远小于较早 entry 的 rank。仅凭 kernel 名称或空档
边界，仍不能把原因唯一归到 `reduce_scatter`、dense GEMM 或某个 CPU 算子。

## 文件与核验

- [intervals.csv](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/sm100_megamoe/v41_entry_skew_results/intervals.csv)：624 行区间、设备活动并集、空档边界、图提交时间。
- [kernels.csv](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/sm100_megamoe/v41_entry_skew_results/kernels.csv)：12278 行按 kernel 名称分组的累计设备时间。
- [skew_summary.csv](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/sm100_megamoe/v41_entry_skew_results/skew_summary.csv)：16 个极值 rank 的汇总。
- [kernel_top.csv](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/sm100_megamoe/v41_entry_skew_results/kernel_top.csv)：对应 top 5 分组。
- [validation.txt](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/sm100_megamoe/v41_entry_skew_results/validation.txt)：行数、几何、恒等式、图提交完整性和版本记录。
- [versions.txt](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/sm100_megamoe/v41_entry_skew_results/versions.txt)：执行环境与输入 SHA。
- [SHA256SUMS](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/sm100_megamoe/v41_entry_skew_results/SHA256SUMS)：本目录提交文件清单。

核验结果：两臂 16 个 arm/rank 组合各有 39 个区间；层号恰为 1..39；
`device_active_union_us + uncovered_us == gap_us` 最大误差为 `0`；624/624 个区间
都有 `launches_seen=40`；两份 CSV 的 SHA256 已写入 `validation.txt`。没有错误，
没有修改模型、输入、计时语义或上游实现。
