# SM100 执行者 · BS1024 五臂图重放复测

旧矩阵在 eager 模式反复调用后端。尤其是 ECO 成交计数传入每次层调用独有的地址，它参与 GC 子图缓存键，导致每 rank 正式窗口出现 40,000 个不同的参数组合。此次把 MoE 后端调用逐层预热、捕获，在计时前验证重放，正式窗口五臂均使用图重放。模型外层仍使用 eager，保留原表采集口径。

在已更新的干净 main checkout，按原五臂任务书取得同一组空闲且无人预约的八卡，确保 sparse checkout 包含 integration 及输入归档。在持久终端执行一次：

```bash
set -o pipefail
work=/data0/home/admin/workspace/megamoe/v41-five-graph-bs1024-01
mkdir -p "$work"
SM100_MATRIX_BS=1024 bash scripts/sm100/run_five.sh "$work" 2>&1 | tee "$work/launcher.log"
```

只跑 BS1024，五臂各一次，1000+1000。每个 rank 必须完成 40 个层图的 eager/replay 输出、实际接收数和拍卖成交数校验，失败时交付完整堆栈及首错。仅允许语义等价的编译修复并附 diff。原生 gate 或 token 严格校验失败时仍交付全表。报告各 rank 两窗口首个观测到的原生 gate 与回放差异，以及各臂端到端、完整 MoE、block M、峰值 M 和 ECO 成交数。把同臂数据与旧 BS1024 并排，分别说明固定工作量计时和原生一致性结果。两处启动改动同时生效，差额代表合计影响。

结果与日志放入 docs/sm100_megamoe/v41_graph_bs1024_results/，回执写入 docs/sm100_megamoe/REPLY_V41_GRAPH_BS1024.md。按原五臂任务书收尾清理、复查八卡释放后推送 main。
