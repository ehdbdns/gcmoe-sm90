# SM90 · BS256 单活跃副本 token 直推

在原 8 张空闲 H20 的预约内串行跑两臂。两臂各一次，1000 步统计、重排后 1000 步测量。唯一实验开关是 GC_MOE_SM90_SINGLE_PUSH。baseline 关闭，single_push 打开。使用相同固定回放、KV、逐层 EPLB、ECO fine cost mode 2、slack 11469、M64、stream、6 SM dispatch 和原有 CUDA Graph。硬件须支持 GPU 对之间的原生原子操作。

单活跃副本按源 rank 前缀和源内路由序号定位最终行。dispatch 的前 8 个 CTA 各用 warp 推送 activation、scale、gate 和 combine 回址，每行完成后向目标 M 块计数一次。目标 dispatch CTA 观察整块或尾块收齐后发布现有 ntile 工作。多活跃副本使用 baseline 的索引广播、plan 配额和目标端 pull。包含 push 的是 stream 分支。固定行序仍需要第一阶段的源计数就绪。接收 CTA 按原有槽序访问，后续槽可能有排队时间。

## 执行

在 SM90 主仓运行。预约期间有人占用任一卡就等待。沿用上次通过校验的 KV 和 replay，原样迁移时设置 SM90_RETAINED 为新目录。先保存 nvidia-smi topo -m 和 nvidia-smi topo -p2p a 输出并核实所用的 8 张卡之间双向支持原子操作。基线与新臂使用同一 checkout。

```bash
set -euo pipefail
git fetch origin main
book=docs/trace_executor_20261119/SM90_BS256_SINGLE_PUSH.md
sha=$(git log -1 --format=%H origin/main -- "$book")
checkout=$(mktemp -d /tmp/sm90-single-push-code-XXXXXXXX)
git worktree add --detach "$checkout" "$sha"
(cd "$checkout" && bash scripts/trace_layer/run_bs256_eco_exchange.sh single-push)
```

记录 Trial workspace 和两臂 Capture workspace。脚本逐臂检查历史输入输出、统计及测量窗口的八 rank 审计、物理槽计数、拍卖数组和预测值。每个采样层还检查新臂各源 CTA 合计发送行数等于全体单活跃副本槽的接收行数，以及每个有效 M 块的发布、valid 和 ntile 领取记录。编译失败时只允许做保持语义的编译修复，附 diff 与实跑 SHA，并用该版本跑两臂。超时、挂起、回放或校验失败时停止，提交首个错误和已有产物。保持 kernel 算法、实验参数、回放与校验条件原样。

## 回执

1. 交付 moe.csv。主判据为剔除采样步的 999 步完整 MoE ms/步，相对变化为 100 × (single_push / baseline − 1)，负值表示更快。同时报告含采样步的 1000 步结果，以及同一关键 rank 的 graph 至首个有效 CTA、首末有效 CTA 之间和最后 CTA 至 combine 三段。
2. step 2 的层 0、8、20、32、39 每臂交付 torch profiler 对齐的粗粒度 Perfetto JSON、absolute/detail Perfetto JSON、counts.json、detail.json。列出各 rank 的 token、M 块、活跃槽及慢 rank。新臂的 Singleton push CTA 是源端扫描并推送的区间，rows 是实际推送行数。目标单副本槽的区间包含等待和发号。结合 L1 publish、首个计算 CTA、末次 L2 push 和完整 MoE，分析收益或损失。不同 GPU 的原始 globaltimer 不能直接相减。
3. 归档到 docs/trace_executor_20261119/sm90_bs256_single_push_results/，包含两臂 run.json、rank0–7.npz、moe.csv、上述五层 trace 与 counts、验证日志、拓扑、版本和 SHA256SUMS。回执推送到 main 的 docs/trace_handover_20261119/REPLY_SM90_BS256_SINGLE_PUSH.md，给出两个臂的可点击 GitHub 链接。
4. 完成或失败时停止本任务的进程并确认八卡释放。核验归档后清理本次创建的临时 checkout 和输出，保留之前的 KV、replay 和他人进程。
