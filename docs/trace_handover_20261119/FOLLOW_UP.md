# 【SM90】BS256 prefill 自对照诊断

本轮只跑两次 vanilla prefill 并检查层 19/20 的模块输入输出，回执写入 REPLY_SM90_PREFIX_BLOCK_AA.md。[【SM90】执行任务书](../trace_executor_20261119/SM90_PREFIX_BLOCK_AA.md)

# 历史任务 · 修正 EPLB 后的 24 臂主表

回执写入本目录的 `REPLY_MAIN24_COST.md`。每档 BS 固定工作量，新增关闭 SM-reuse 的线性 ECO 价格对照，测量完整 MoE、端到端、各 rank 预测误差与 ECO 拍卖成交。[八卡任务书](../trace_executor_20261119/MAIN24_COST_TASK.md)

# 历史任务 · 1000 + 1000 步的两臂 fine cost 对照

在同一台八卡 H20 上固定 stream，先测线性价格，再测逐 valid 价格。主指标为重排后 1000 步完整 MoE 耗时。请回执至本目录的 `REPLY_LONG_COST_1000.md`。[八卡执行任务书](../trace_executor_20261119/LONG_COST_1000_TASK.md)

# 历史任务 · 固定 stream，验证 fine cost

执行线性价格和逐 valid 价格两臂，各启动一次模型。回执写入本目录的 `REPLY_VALID_COST.md`。[执行任务书](../trace_executor_20261119/VALID_COST_TASK.md)

# 历史任务 · R6 严格验证 stream

请执行 [R6 八卡任务书](../trace_executor_20261119/FIXED_STREAM_R6_TASK.md)，每臂一次，回执写入本目录的 `REPLY_FIXED_STREAM_R6.md`。

R5 已复核。首个 MoE 调用八卡输入输出一致，105 步仍有 2,782/26,880 个原生预测不同。100 步 E2E 和 step 2 整层 MoE 分别下降 0.48% 和 0.53%，严格对照未通过。[R5 数据](../trace_executor_20261119/fixed_stream_results_r5/comparison.json)

# R4 八卡回执复核

R4 已完成，共启动两次模型。两臂均使用 `f73ceaa`、修复后的 ECO 和旧线性价格 `26 × M 块数 + routed rows`。100 个计时步的 E2E 从 83.362 降至 82.200 ms/步（−1.39%）。step 2 的 40 层 MoE 从 34.142 升至 34.523 ms（+1.12%），其中 9 层变快，31 层变慢。当前证据不足以确认 stream 的等价加速。每臂只有一次运行。[R4 回执](REPLY_FIXED_STREAM_R4.md) · [配对指标](../trace_executor_20261119/fixed_stream_results_r4/paired.csv)

复核了文件校验和、两臂的审计记录及 40 层逐槽计数。回放的 105 步 × 40 层都使用相同的路由 ID、权重、源内行号和逐槽物理行数。两臂都用 baseline 预热，但八个 rank 最后一次预热的量化输入和尺度摘要均有差异，预热原生预测摘要也不同。回放第 0 步已有 44/256 个原生 top‑1 不同，step 2 为 51/256，105 步共 3,640/26,880 个不同。强制反馈的 token 序列相同。现有采样无法定位预热内首次发生数值分歧的调用。[两臂审计判定](../trace_executor_20261119/fixed_stream_results_r4/comparison.json)

逐块调度已在设备上生效。11 个详细采样层共有 512 个多 M 块槽实例。以同一 rank、同一槽的首个 L1 tile 与该槽 pull 结束比较，baseline 有 0 个提前开算，stream 有 261 个。以同一 M 块的最后一个 L1 tile 和首个 L2 tile 比较，两臂各 3,066 块均满足依赖。第 20 层 rank 0 的槽 51 在两臂都接收 320 行，baseline 在整槽 pull 完成后 217 µs 开始 L1，stream 则提前 72 µs 开始。[第 20 层 baseline absolute](../trace_executor_20261119/fixed_stream_results_r4/step02/baseline/paired/step002_layer20_epoch14581.absolute.perfetto.json) · [第 20 层 stream absolute](../trace_executor_20261119/fixed_stream_results_r4/step02/stream/paired/step002_layer20_epoch14581.absolute.perfetto.json)

退化最大的第 3 层，八卡最大 MoE 从 861.0 升至 1,071.7 µs。rank 6 的有效领票至 routed 结束从 623.7 延至 846.7 µs。它的 combine 从 31.7 降至 17.1 µs，其余七个 rank 在 stream 的 combine 内停留 235.8–275.7 µs。combine 等待同伴发布 compute_ready，观察与 rank 6 晚到相符。这一层只有总览 trace，无法将其 routed 尾部归因到某个 CTA、槽或队列。[第 3 层 baseline counts](../trace_executor_20261119/fixed_stream_results_r4/step02/baseline/paired/step002_layer03_epoch14564.counts.json) · [第 3 层 stream counts](../trace_executor_20261119/fixed_stream_results_r4/step02/stream/paired/step002_layer03_epoch14564.counts.json)

R4 已结案，原始数据和日志由执行者保留。
