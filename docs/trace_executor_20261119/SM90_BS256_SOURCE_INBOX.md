# SM90 执行任务 · BS256 source inbox 与 pull baseline

收到即执行。预约同一组八张空闲 H20，串行跑两臂，各一次。baseline 使用当前主表的 EPLB + ECO fine cost、pull + stream；source_inbox 仅打开 `GC_MOE_SM90_SOURCE_INBOX=1`。两臂 plan 均在通信分区，dispatch/shared/routed/plan 的 CTA 数均为 20/72/72/1，均关闭 SM 复用。每臂前 1000 步统计，重排后 1000 步测量。用本轮新跑的 baseline 作分母。

## 执行命令

在执行机原仓库运行下列命令。脚本校验同组 GPU、历史参考和两窗口 KV 清单。保留现场默认在 /tmp/sm90-bs256-trace-pbpMbH4u；如果该现场已搬迁，先将 SM90_RETAINED 指向含同一份 kv/ 和 replay/ 的目录。记录 Trial workspace 和两臂的 Capture workspace。

```bash
set -euo pipefail
git fetch origin main
book=docs/trace_executor_20261119/SM90_BS256_SOURCE_INBOX.md
sha=$(git log -1 --format=%H origin/main -- "$book")
checkout=$(mktemp -d /tmp/sm90-source-inbox-code-XXXXXXXX)
git worktree add --detach "$checkout" "$sha"
(cd "$checkout" && bash scripts/trace_layer/run_bs256_eco_exchange.sh source-inbox)
```

## 判读和回执

1. 保留两臂验证日志，核对固定输入、原生输出、逐层逐槽 token 数、放置、拍卖与实际 grid。按同一步、同层、同 rank 配对。任一正确性检查失败时提交首错、日志、退出码和已生成的产物。
2. 交付 first_m_block.csv。采样测量 step 2 的 L0、8、20、32、39，分别给出每个 rank 从本地 MoE graph 起点到首次发布可领取 L1 任务的耗时，以及首次发布满 M 块的耗时。每项列两臂原值、差值（inbox 减 baseline）、有效配对数、提前/持平/变晚数量与配对差值中位数。保留没有满块的空值。附两臂五层 absolute Perfetto、对应 counts/detail 文件以及每 rank 的 token 和 M 块数。
3. 交付 moe.csv。主指标取剔除 profiler 步的 999 步完整 MoE 均值（ms/步），同时列差值和有符号的相对变化，附全 1000 步值。首块发布提前且完整 MoE 持平或降低，才支持本轮目标。若结果接近计时波动，逐项列出观测并注明不确定性。
4. 提交两臂 run.json、八 rank 的测量 NPZ、验证日志、代码 SHA、退出码和产物 SHA256。执行者仅可做保持语义的编译修复并附 diff；若修复，两臂使用同一最终版本。调度、参数、计时和验收条件按本任务书执行。
5. 回执写入 docs/trace_handover_20261119/REPLY_SM90_BS256_SOURCE_INBOX.md，产物写入 docs/trace_executor_20261119/sm90_bs256_source_inbox_results/，推送 main 并核对远端提交。登记原始证据路径，释放 GPU，清理本轮私有 checkout 和临时缓存，保留共享环境及原始回放现场。
