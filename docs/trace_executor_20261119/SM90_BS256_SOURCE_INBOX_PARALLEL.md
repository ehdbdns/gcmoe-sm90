# SM90 执行任务 · BS256 并行 source inbox 与 pull baseline

收到即执行。预约同一组八张空闲 H20，串行跑两臂，各一次。baseline 使用当前主表的 EPLB + ECO fine cost、pull + stream；source_inbox 仅打开 `GC_MOE_SM90_SOURCE_INBOX=1`。两臂 plan 均在通信分区，dispatch/shared/routed/plan 的 CTA 数均为 20/72/72/1，均关闭 SM 复用。每臂前 1000 步统计，重排后 1000 步测量。用本轮新跑的 baseline 作分母。

## 执行命令

沿用上轮八卡预约、HCA 和可见 GPU 设置，确认八卡空闲后开始。在执行机原仓库运行下列命令。脚本校验同组 GPU、历史参考和两窗口 KV 清单。保留现场默认在 /tmp/sm90-bs256-trace-pbpMbH4u；如果该现场已搬迁，先将 SM90_RETAINED 指向含同一份 kv/ 和 replay/ 的目录。记录 Trial workspace 和两臂的 Capture workspace。

```bash
set -euo pipefail
git fetch origin main
book=docs/trace_executor_20261119/SM90_BS256_SOURCE_INBOX_PARALLEL.md
sha=$(git log -1 --format=%H origin/main -- "$book")
checkout=$(mktemp -d /tmp/sm90-source-inbox-code-XXXXXXXX)
git worktree add --detach "$checkout" "$sha"
cd "$checkout"
(
  source scripts/env.sh
  export PYTHONPATH="$PWD/source:$PWD/patches:$PWD/integration" GC_MOE_SRC="$PWD/source"
  unset GC_MOE_SM90_SPLIT_EXCHANGE GC_MOE_SM90_SOURCE_INDEX_PULL GC_MOE_SM90_SINGLE_PUSH
  export GC_MOE_SM90_SOURCE_INBOX=1 GC_MOE_SM90_STREAM_M_BLOCKS=1 GC_MOE_SM90_PULL_TMA=1
  export GC_MOE_ECO=1 GC_MOE_ECO_MAX_COPIES=8 GC_MOE_MB_CLAIM=1 GC_MOE_BLOCK_M_FORCE=64 GC_MOE_SM90_DISPATCH_SMS=6
  export GC_MOE_FIXED_DIR="$checkout/synthetic-fixed" TL_PUSH_PAYLOAD=1
  export TL_CAP=128 TL_TOKENS=17 TL_HOT_BLOCKS=1 TL_CHECK_PUSH_RETURN=1
  timeout 600 "$GC_ENV/bin/torchrun" --standalone --nproc-per-node=8 scripts/verify_eco_capture.py
) 2>&1 | tee "$checkout/smoke.log"
for n in 135 0 64 65; do grep -q "PUSH_PAYLOAD_OK hot_rows=$n replays=3" "$checkout/smoke.log"; done
grep -q 'ECO_CAPTURE OK' "$checkout/smoke.log"
bash scripts/trace_layer/run_bs256_eco_exchange.sh source-inbox
```

## 判读和回执

1. 保留两臂验证日志，核对固定输入、原生输出、逐层逐槽 token 数、放置、拍卖与实际 grid。按同一步、同层、同 rank 配对。任一正确性检查失败时提交首错、日志、退出码和已生成的产物。
2. 交付 first_m_block.csv。采样测量 step 2 的 L0、8、20、32、39，分别给出每个 rank 从本地 MoE graph 起点到首次发布可领取 L1 任务的耗时，以及首次发布满 M 块的耗时。每项列两臂原值、差值（inbox 减 baseline）、有效配对数、提前/持平/变晚数量与配对差值中位数。保留没有满块的空值。附两臂五层 absolute Perfetto、对应 counts/detail 文件以及每 rank 的 token 和 M 块数。
3. 交付 moe.csv。主指标取剔除 profiler 步的 999 步完整 MoE 均值（ms/步），同时列差值和有符号的相对变化，附全 1000 步值。首块发布提前且完整 MoE 持平或降低，才支持本轮目标。若结果接近计时波动，逐项列出观测并注明不确定性。
4. 提交两臂 run.json、八 rank 的测量 NPZ、验证日志、代码 SHA、退出码和产物 SHA256。执行者仅可做保持语义的编译修复并附 diff；若修复，两臂使用同一最终版本。调度、参数、计时和验收条件按本任务书执行。
5. 回执写入 docs/trace_handover_20261119/REPLY_SM90_BS256_SOURCE_INBOX_PARALLEL.md，产物写入 docs/trace_executor_20261119/sm90_bs256_source_inbox_parallel_results/，推送 main 并核对远端提交。登记原始证据路径，释放 GPU，清理本轮私有 checkout 和临时缓存，保留共享环境及原始回放现场。

另附旧 inbox（69d288e）与本轮 inbox 在 step 2 L20 的逐 rank 诊断表。以各自本地 dispatch 起点为零，列末次单副本发送完成、末次副本发送完成、首个满块及末个块发布时间，缺失记为 NA。发送事件描述本 rank 出站，发布描述本 rank 入站。报告完整 MoE 的 graph→first、first→last、last→combine 三段及各段差值，保存 smoke.log 后清理。旧数据直接读回执，不增加第三臂。
