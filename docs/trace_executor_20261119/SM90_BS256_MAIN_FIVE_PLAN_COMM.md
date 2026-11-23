# SM90 · BS256 主表五臂重测

在原八卡 H20 上串行运行 vanilla、random（随机放置 + RR）、rr（逐层 EPLB + RR）、eco（逐层 EPLB + fine ECO）、eco_smr（同一 fine ECO + SM-reuse）。后两臂均为 mode 2、slack 11469，plan 占通信分区一个 SM，dispatch 20 CTA，shared / routed 各 72 CTA；eco_smr 在 dispatch 完成后额外启用 role 6。五臂均用原 pull、stream 和 FULL_DECODE_ONLY，稳定 prefill/decode top-k，M64，TP8/EP8，20 GiB KV/rank。每臂启动一次，1000 步统计、重排后 1000 步正式测量；主表口径使用完整正式窗口。

## 执行

预约原八张空闲卡，有人使用时等待。沿用原模型、数据和运行环境。选择能容纳两份 KV 和 route bank 的本地 checkpoint 目录。

```bash
set -euo pipefail
git fetch origin main
book=docs/trace_executor_20261119/SM90_BS256_MAIN_FIVE_PLAN_COMM.md
rev=$(git log -1 --format=%H origin/main -- "$book")
work=$(mktemp -d /tmp/sm90-main-five-XXXXXXXX)
git worktree add --detach "$work/code" "$rev"
cd "$work/code"
unset GC_STEP_BOUNDARY GC_COST_HOST_PROBE GC_BS256_TRACE_DIR GC_MOE_SM90_SINGLE_PUSH GC_MOE_SM90_SPLIT_EXCHANGE GC_MOE_SM90_SOURCE_INDEX_PULL
unset GC_MOE_DECODE_STATS_AA GC_MOE_VANILLA_STATS_REPLAY GC_MOE_PREFIX_DIAG GC_MOE_PREFIX_SPARSE_AA GC_MOE_PREFIX_BLOCK_AA GC_MOE_PREFIX_LAYER_AA GC_MOE_PREFIX_HIDDEN_AA
export GC_MOE_ORDER_PREFILL=1 GC_MOE_STABLE_PREFILL_TOPK=1 GC_MOE_STABLE_DECODE_TOPK=1
export GC_MOE_MAIN_FINE_ONLY=1 GC_MAIN24_SINGLE_BS=256 GC_MOE_KV_BYTES=21474836480 GC_MOE_SM90_PLAN_DISPATCH=1
export GC_MOE_CHECKPOINT_ROOT="${GC_MOE_CHECKPOINT_ROOT:-/tmp}"
rc=0
bash scripts/trace_layer/run_main24.sh "$work" || rc=$?
echo "$rc" > "$work/exit-code.txt"
exit "$rc"
```

## 验收与交付

- vanilla 发布本轮两个边界的 KV、原生 token 和路由参考，其余四臂严格复用。验收 delivery/bs256/pairs.csv 共 10 对，逐对 warm_native_equal、strict_pair_comparable 为真，native_diff 为零。检查输入输出哈希、各层放置及 route bank 校验；两个 ECO 臂逐槽 counts 及拍卖计数完全一致。保留两臂实际 role-5 价格、通信分区日志以及主机预检 role grid 的输出。报告 8×H20、测量 SHA、显存和每步实际 BS。
- 回执打印 README 主表全部列。完整 MoE 按同一步同层八 rank 的最大 graph→combine 跨度求和，再对 1000 步取平均。相对变化 = (本臂 MoE / 本轮 vanilla MoE − 1) × 100%，保留正负号；同时列 token IR、总/峰值 M 块、活跃物理槽、M 块 IR、routed/graph 预测误差（百分点）、ECO 成交数。列出 eco 对 rr、eco_smr 对 eco 的同轮 MoE 差异。正式数据入表前由设计端复核。
- 将 delivery、preflight、preflight.log、progress.log、exit-code.txt 和回执分别提交 docs/trace_executor_20261119/sm90_bs256_main_five_plan_comm_results/ 与 docs/trace_handover_20261119/REPLY_SM90_BS256_MAIN_FIVE_PLAN_COMM.md。仅允许正式测量前保持语义的编译修复，附 diff 并固定同一执行 SHA；其他失败保留首错和已完成产物。推 main、核验远端后释放八卡，清理本轮已归档的私有 worktree/临时缓存，保留复查所需证据及历史现场。
