# SM90 · 定位 BS1024 fine cost 的端到端差额

上轮 fine 比 linear 多 7.113 ms/步，同轮完整 MoE 相差约 0.001 ms。999 个跨步区间平均多 7.106 ms，20 对细分打点指向采样返回至下次执行入口。本轮检验这个差额能否在相反顺序下复现，并在同一模型步的 driver 调用路径上定位额外 elapsed 时间。

## 执行

仅在同一组八张 H20 空闲且获分配后执行。在 [BS1024 任务书](SM90_MAIN24_STABLE_BS1024.md) 的完整 shell 命令块中，紧接 `export GC_STEP_BOUNDARY=1` 加一行 `export GC_COST_HOST_PROBE=1`，执行一次整段。沿用两份数据文件、相同 prompt 哈希、20 GiB/卡 KV、stream、稳定 top-k、同步计时及八 rank 公平性检查。脚本自动依次启动 vanilla、eco_linear、eco。vanilla 只用于重新生成先前按约释放的 KV 与回放基准；正式成本函数对照共两臂，每臂 1000 步统计、1000 步测量。两臂从同一新基准恢复，并核对上轮的逐层放置。保持旧价/slack 90、新价/slack 11469。只有等价编译修复可以由执行者改动，附完整 diff 和首错。

## 交付与判读

1. 成功时提交本轮 delivery 中的 pairs.csv、orders.csv、exclusive_host.csv、host_balance.csv、cost_boundary.csv、三个 run.json、八 rank 的原始 NPZ、日志、SHA 清单；保留版本、执行顺序和 GPU 记录。失败时给出首次失败及已有证据。结果目录为 `docs/trace_executor_20261119/sm90_bs1024_cost_host_results/`，回执为 `docs/trace_handover_20261119/REPLY_SM90_BS1024_COST_HOST.md`。沿用 BS1024 任务书的清理要求，推送 main。
2. 首先列出本轮两臂 E2E、完整 MoE、跨步间隔及差额，确认预热、正式原生输出、实际路由、KV、权重和布局校验通过。对照上轮 fine 先跑和本轮 linear 先跑各自的组内差值。若慢的方向反转或差额消失，记录观察值和波动；两轮打点密度不同，每种顺序各一次，保留次序与运行时环境的解释空间。
3. 主机表按互斥区间列出 fine 减 linear 的均值、配对中位数及 fine 更慢的步数。区分 scheduler、执行提交、worker 结果等待、采样 API、输出处理和剩余 engine.step 时间，并列出调用线程的 CPU 耗时。wall 减线程 CPU 包括被抢占、阻塞、等待其他线程或设备；worker_result 表示 driver 等结果的经过时间。列出可解释的毫秒数、占 E2E 差额的比例及余量，仅在差额明显且同向时计算比例。用已有 collect 和十步同步壁钟检查闭合，报告插桩外层残差。GPU 区间单独核对，主机和 GPU 两条时间轴不能相加。
4. 指出重复出现的最大区间，说明证据支持到哪一层。如差额仍落在未细分等待或换序后未复现，在回执中保留该结论。
