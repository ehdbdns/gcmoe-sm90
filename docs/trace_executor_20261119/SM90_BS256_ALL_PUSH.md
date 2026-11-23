# SM90 · BS256 单副本与多副本源端直推、空闲 CTA 领任务

原八张 H20 空闲且在预约内时串行执行。保留固定 KV、两窗口路由、EPLB、ECO 价格、block M、stream 和 launch 配置。此次两臂为同期 baseline 和全直推（沿用 runner 中的 single_push 标识）。只运行两次正式模型，均为 1000 步统计 + 重排后 1000 步测量。原生正确性失败即交首错，执行者只可做保持语义的编译修复，两臂须同 SHA。沿用已验证 HCA 和 SM90_RETAINED 的历史现场。

```bash
set -euo pipefail
git fetch origin main
book=docs/trace_executor_20261119/SM90_BS256_ALL_PUSH.md
rev=$(git log -1 --format=%H origin/main -- "$book")
area=$(mktemp -d /tmp/sm90-all-push-XXXXXXXX)
git worktree add --detach "$area/code" "$rev"
cd "$area/code"
archive=docs/trace_executor_20261119/sm90_bs256_async_return_fix_results
fixture=docs/trace_executor_20261119/sm90_main24_stable_decode_bs256_results/delivery/bs256
if [[ $(git config --bool core.sparseCheckout || true) == true ]]; then
  git sparse-checkout add "$archive" "$fixture" scripts source integration patches
fi
until busy=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader) && [[ -z "$busy" ]]; do sleep 30; done
(
  source scripts/env.sh
  unset GC_MOE_SM90_SPLIT_EXCHANGE GC_MOE_SM90_SOURCE_INDEX_PULL GC_MOE_SM90_PLAN_DISPATCH
  export PYTHONPATH="$PWD/source:$PWD/patches:$PWD/integration" GC_MOE_SRC="$PWD/source"
  export GC_MOE_ECO=1 GC_MOE_ECO_MAX_COPIES=8 GC_MOE_MB_CLAIM=1 GC_MOE_BLOCK_M_FORCE=64
  export GC_MOE_SM90_SINGLE_PUSH=1 GC_MOE_SM90_STREAM_M_BLOCKS=1 GC_MOE_SM90_DISPATCH_SMS=6 GC_MOE_SM90_PULL_TMA=1
  export GC_MOE_FIXED_DIR="$area/synthetic-fixed" TL_PUSH_PAYLOAD=1
  export TL_CAP=128 TL_TOKENS=17 TL_HOT_BLOCKS=1 TL_CHECK_PUSH_RETURN=1
  timeout 600 "$GC_ENV/bin/torchrun" --standalone --nproc-per-node=8 scripts/verify_eco_capture.py
) 2>&1 | tee "$area/return-smoke.log"
for rows in 135 0 64 65; do grep -q "PUSH_PAYLOAD_OK hot_rows=$rows replays=3" "$area/return-smoke.log"; done
grep -q 'ECO_CAPTURE OK' "$area/return-smoke.log"
bash scripts/trace_layer/run_bs256_eco_exchange.sh single-push 2>&1 | tee "$area/pair.log"
work=$(sed -n 's/^Trial workspace: //p' "$area/pair.log")
test -f "$work/moe.csv"
```

## 核对

1. 冒烟四种单副本边界与每步 136 行的多副本热专家同场运行。每组依同图 legacy 路径取参考，再连跑三次直推，检查所有逻辑路由唯一、源激活/scale/gate、各副本收到的精确源路由集合与 plan 的 canonical quota 切片一致、L2 写回与最终 BF16。正式两窗口检查 native token、逐槽 counts、auction、布局和硬件。
2. perfetto 的 replica_sends 以源 rank 为进程，记录领取任务的 CTA、目的物理槽、行数和发送起止。每个源对每个非空交集至多一条；汇总八源后严格等于每个副本拥有的行数。核对每个 L1 块都附 declared_sources、finished_sources，逐层列满块提前发布数和尾块数。基线继续检查原 pull；候选图的 local route preparation 为本地计数和路由准备，source push and destination completion 含发送与目的收尾，ECO plan 独立计时。
3. 以 999 个非采样步、每层关键 rank 的完整 MoE 求和比较同期 baseline。给出三段均值、ms/步、变化率 100 × (candidate / baseline − 1)，并列出上一轮 35.7605 / 36.7759 ms。按样本逐源计算本源 quota_publications 到 replica_sends 开始的时差、分 CTA 任务数与行数、最后发送/最后发布与 combine 的关系；区分源计时与目的计时。第 20 层列 rank 0 槽 45 在八个源上的实际发送 CTA 和时间。其他源 CTA 的时间要用各自 Kineto 锚点对齐。
4. 交付两臂 step 2、层 0/8/20/32/39 的十份 absolute detail Perfetto JSON 的 GitHub blob 链接，以及 coarse、counts、逐 rank token/M 块、detail JSON、rank NPZ、moe.csv、smoke 和验证日志、原始证据路径及 SHA。回执写 docs/trace_handover_20261119/REPLY_SM90_BS256_ALL_PUSH.md，产物归档 docs/trace_executor_20261119/sm90_bs256_all_push_results/。推 main 后清理本轮临时 checkout、释放八卡，保留历史 KV/replay 和原始证据。
