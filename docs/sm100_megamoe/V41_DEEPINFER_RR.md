# SM100 · 两后端整模 DeepInfer RR 路由验证

沿用上轮已核验的初始放置，单独切换 RR 副本选择。每个 rank 使用本地 token 行号和 EP rank 选号，同时按实际物理槽记录有效 token 的 load。此次检查 RR 接入和输出；逐层 DeepInfer 重排和 ECO 另行验证。

1. 拉取 main 并记录 SHA。按 [上轮两臂任务书](V41_EPLB_DECODE.md) 的步骤 1–2，使用各自的新工作目录，分别运行 MegaMoE 和 GCMoE。仅在指定八卡均空闲且无人预约时持锁执行，有占用或预约就等待。
2. 第一次持锁、设置环境后，启动模型前执行 `python scripts/sm100/check_v41_rr.py 2>&1 | tee "$work/rr-preflight.log"`。保留 pipefail，预检失败即回执。
3. 两臂均在上轮模型命令上追加 `--deepinfer-rr`。每臂一次启动、同一 prompt 连续生成四步。记录八个 rank 在生成期间的 rr_calls、逐层布局哈希、输出 ID、退出码、释放时间。对比两臂以及上轮的 40 个布局哈希与输出 [75,200,18,200]。有差异时标出首处，保留原始数据。
4. 回执写入 docs/sm100_megamoe/REPLY_V41_DEEPINFER_RR.md，日志及校验和写入同目录 v41_deepinfer_rr_results/，推送 main。仅允许等价编译修复并附 diff，其他失败附首个完整调用栈。此轮不采性能。
