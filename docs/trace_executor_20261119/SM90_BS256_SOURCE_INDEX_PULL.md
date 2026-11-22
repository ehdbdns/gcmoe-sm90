# SM90 · BS256 源端副本索引与当前 ECO 对比

仅在原八张空闲 H20 的预约内串行运行两臂。每臂启动一次模型，1000 步统计 + 重排后 1000 步测量。baseline 使用当前默认 ECO；source_index 只打开 GC_MOE_SM90_SOURCE_INDEX_PULL=1。两臂均使用逐层 EPLB、ECO fine cost mode 2、slack 11469、M64、stream、6 SM dispatch，dispatch/shared/routed CTA 为 24/71/71。以本轮 baseline 为分母。

新版在源 rank 按逻辑专家和源内序号写入每条多活跃副本路由，沿用第一阶段的 system fence 和 epoch 发布。plan 发布原有配额信号。目标 CTA 按 quota 从源端读取连续的小组索引，每 warp 最多 4 行，尾组及小配额立即处理。单活跃副本沿用目标侧索引。保留上轮串行 split 与 plan 分区的代码供追溯，两臂运行时均关闭它们。

## 命令

在 SM90 主仓和原 venv 环境执行。沿用已通过校验的 BS256 KV 与 replay 现场；旧现场若原样搬迁，设置 SM90_RETAINED 为其新目录。使用八卡预约，其他用户占用时等待。

```bash
set -euo pipefail
git fetch origin main
book=docs/trace_executor_20261119/SM90_BS256_SOURCE_INDEX_PULL.md
sha=$(git log -1 --format=%H origin/main -- "$book")
checkout=$(mktemp -d /tmp/sm90-source-index-code-XXXXXXXX)
git worktree add --detach "$checkout" "$sha"
(cd "$checkout" && bash scripts/trace_layer/run_bs256_eco_exchange.sh source-index)
```

记录 Trial workspace 与每臂 Capture workspace。脚本检查历史工作量、两窗口审计和设备配置，并逐 rank 比较 token、epoch、物理槽计数、预测值及拍卖数组。若编译失败，执行者只能作保持语义的编译修复，附 diff 和实跑 SHA，并以同一版重测两臂。其余失败提交首栈和已完成臂证据。

## 指标与交付

1. 提交 moe.csv。主指标为剔除 profiler step 2 的 999 步完整 MoE ms/步，附含采样步的 1000 步结果。逐层选 graph 至 combine 跨度最大的 rank，在同一 rank 拆首个有效 routed CTA 前、首末 CTA 间及末 CTA 后三段。报告相对同轮 baseline 的有符号变化（负值更快）。
2. step 2 的层 0、8、20、32、39 每臂各交一份粗粒度与 absolute/detail Perfetto JSON、counts.json 和 detail.json，列出八 rank 的 token、M 块、活跃槽及慢 rank。每个有实际接收行的多副本槽，detail.json 的 replica_timing 和 Perfetto 包含本 rank 的 quota 发布前标记、首个 warp 发起激活复制前的标记、最后一个 L1 M 块发布。三者都在本 rank 的同一设备时间轴上。分别报告 quota 到首笔激活 issue、首笔 issue 到最后一块就绪，并对照原有 replica wait、首个有效 CTA、pull tail 和完整 MoE。首笔标记测的是发起复制，含实际发出指令前的少量插桩。保留零行槽计数。
3. 归档到 docs/trace_executor_20261119/sm90_bs256_source_index_results/，含计时、每臂 run.json、rank0–7.npz、两种 trace、counts、日志、验证输出、版本、SHA256SUMS。回执写入 docs/trace_handover_20261119/REPLY_SM90_BS256_SOURCE_INDEX_PULL.md 并推送 main。注明本轮判定、失败点或不确定性。
4. 完成或失败后确认作业退出、八卡释放。归档核验后只清理本任务新建的 checkout 和临时输出；保留历史 KV 和 replay。
