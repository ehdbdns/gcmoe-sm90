# SM90 · BS256 异步直推回址修复复测

仅使用原 8 张空闲 H20，在原预约内串行执行。旧回执 7d225d3 的 single_push 首个 decode token 已失配。本版修复 route_meta 三元组的存储地址。先跑无需模型的一次八卡回址检查，再用同一提交启动正式 baseline、single_push 各一次。若 GPU 被他人使用则等待。沿用上轮 HCA、KV、replay、P2P 原子与硬件校验。

```bash
set -euo pipefail
git fetch origin main
book=docs/trace_executor_20261119/SM90_BS256_ASYNC_RETURN_FIX.md
sha=$(git log -1 --format=%H origin/main -- "$book")
wt=$(mktemp -d /tmp/sm90-return-fix-XXXXXXXX)
git worktree add --detach "$wt" "$sha"
cd "$wt"
while [[ -n $(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader) ]]; do sleep 30; done
(
  source scripts/env.sh
  unset GC_MOE_SM90_SPLIT_EXCHANGE GC_MOE_SM90_SOURCE_INDEX_PULL
  export PYTHONPATH="$PWD/source:$PWD/patches:$PWD/integration" GC_MOE_SRC="$PWD/source"
  export GC_MOE_ECO=1 GC_MOE_ECO_MAX_COPIES=8 GC_MOE_MB_CLAIM=1 GC_MOE_BLOCK_M_FORCE=64
  export GC_MOE_SM90_SINGLE_PUSH=1 GC_MOE_SM90_STREAM_M_BLOCKS=1 GC_MOE_SM90_DISPATCH_SMS=6 GC_MOE_SM90_PULL_TMA=1
  export GC_MOE_FIXED_DIR="$wt/synthetic-fixed" TL_PUSH_PAYLOAD=1
  export TL_CAP=128 TL_TOKENS=17 TL_HOT_BLOCKS=1 TL_CHECK_PUSH_RETURN=1
  timeout 600 "$GC_ENV/bin/torchrun" --standalone --nproc-per-node=8 scripts/verify_eco_capture.py
) 2>&1 | tee return-smoke.log
for rows in 135 0 64 65; do
  grep -q "PUSH_PAYLOAD_OK hot_rows=$rows replays=3" return-smoke.log
done
grep -q 'ECO_CAPTURE OK' return-smoke.log
bash scripts/trace_layer/run_bs256_eco_exchange.sh single-push
```

热专家依次收 135、0、64、65 行，使用合成固定 ordinal，源 rank 和轮次改变输入、gate 和 top-k 槽位。每组在同一图上先执行旧 pull 取参考，再连续执行三次异步 push，覆盖两代状态复用。每次检查八卡全部物理槽的逐行激活、scale、gate、回址和逻辑专家；每个源 token 的每个 top-k 选择必须恰好出现一次，逐路 L2 写回与最终 BF16 输出分别和参考精确比较。GC_MOE_FIXED_DIR 在冒烟子 shell 仅用于启用固定回放 ABI；正式实验仍由 runner 设置真实目录。门禁通过后执行上轮正式任务书的两臂、1000+1000、原生输出与审计要求。以本轮 999 个非采样步的完整 MoE 为主指标。两臂分别交第 2 步五层的 Perfetto、detail、各 rank token/M 块数，汇报新臂提前发号比例。若任一校验失败，保存首错及已完成数据，不能报告失败臂速度。只允许保持语义的编译修复，附 diff、SHA 并使用一致版本。回执写 docs/trace_handover_20261119/REPLY_SM90_BS256_ASYNC_RETURN_FIX.md，结果归档到 docs/trace_executor_20261119/sm90_bs256_async_return_fix_results/，含 return-smoke.log。推送 main 后清理本轮临时文件并核实八卡释放，保留历史回放与 KV。
