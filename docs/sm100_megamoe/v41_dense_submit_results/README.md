# V41 BS512 dense-submit results (SM100)

本目录只包含 **SM100** 执行者按 `docs/sm100_megamoe/V41_BS512_DENSE_SUBMIT_FIX.md`
产生的 BS512 原生 greedy decode 结果；未执行任何 SM90 任务书。

## 结果摘要

- Mega 和 GC 各启动一次，退出码均为 `0`。两臂均为 1000 个统计步加 7 个测量步，
  测量步 dense 回放开关为 `off,on,on,off,on,off,on`。
- 每臂 8 个 rank 均记录 **82/82** 张经过 eager 逐元素校验的 dense 图；每 rank
  `validated_layer_graphs=40`，`block_m=16`，RR 调用为 0。
- step 1 的 L20 量化→dense 同流间隔为 `-0.863..+0.321 us`；旧归档对应记录为
  `204.162..269.537 us`。负值是相邻 Kineto 事件边界的亚微秒重叠，按物理串行含义
  应视为无可观测正间隔，不能把负值当额外收益。
- L20 的两次 `关→开` 转换中，Mega 均下降；GC 在 3→4 下降，5→6 只有 L20/L39
  下降，L0/L8 上升。不能用单一方向概括跨 rank 错峰结论。

## 文件

- `step001_dense_gaps.csv`：两臂 8 rank × 2 个 L20 quantize→dense 同流配对。
- `step001_dense_gap_match.csv`：与旧 `launch_gap_evidence.csv` 的 4 条记录逐条匹配。
- `within_run_skew.csv`：五层 × step 2–6 的 dense 开关、host replay 调用、到达时差、
  最长交换、完整八卡 MoE 跨度、token 与 M 块。
- `step002_skew.csv`：同臂旧 step 2 与新 step 2 的对照；host 两列不是同一种 API 计时。
- `clock_alignment_check.csv`：`clock-before` Kineto 时间戳缺失时的单锚点回退核验。
- `mega/`、`gc/`：结果 JSON、退出码、日志、8 个 measure NPZ、60 张 Perfetto 与
  60 张 overview。
- `raw-evidence.sha256`、`raw-evidence-paths.txt`：未提交的原始 profile、校准会话、
  锚点 NPY 和 stats NPZ 的 SHA256 与执行机路径。
- `analyzer-fallback.diff`、`postprocess.log`：仅分析器的单锚点回退，不改变设备时钟、
  模型、kernel 或测量语义。
- `gpu_release.txt`、`lock_release.txt`、`processes-stopped.txt`：释放证据。

## 口径

`host_replay_spread_us` 是跨 rank 的 Python `graph.replay` 调用时差，旧归档的
host spread 是 CUDA API 入口时差，两者不可当作同一 API 的精确时长。`wall_ms`
包含准备、RPC 与导出，不能替代长窗吞吐。L20 的 GitHub 链接和完整表格见
`../REPLY_V41_BS512_DENSE_SUBMIT_FIX.md`。
