# 【SM90】BS256 两次同配置 vanilla 回放：定位 sampler 输入边界

只做两次相同的 vanilla prefill，两次都是参考轨迹消费者，使用相同代码、模型、GPU、请求顺序和诊断开关。每个接受 token 记录传入 sample 的最后一行 hidden 字节摘要。原始发布者随原生结果生成参考，而之前的一次 vanilla 回放使用强制反馈；本次固定这项差异。保留严格输出判据。此次不测性能。

在原八卡 H20 空闲时按以下步骤执行。只允许语义等价的编译修复；任何运行错误保留现场回执。

```bash
set -eo pipefail
git pull --ff-only origin main
repo=$(git rev-parse --show-toplevel)
cd "$repo"
test -z "$(git status --porcelain)"
sha=$(git rev-parse HEAD)
work=$(mktemp -d /tmp/gcmoe-prefix-aa-XXXXXXXX)
old=$(cat docs/trace_executor_20261119/main24_prefix_capacity_results_blocked/bs256/state-path.txt)
mkdir "$work/replay" "$work/kv"
cp "$old/replay/tokens.json" "$work/replay/"
sha256sum "$work/replay/tokens.json" > "$work/reference.sha"
export GC_MOE_EXEC_ROOT="$repo" GC_MOE_FIXED_DIR="$work/replay" GC_MOE_STRICT_STATE_DIR="$work/kv"
export GC_MOE_MATRIX_SIZE=24 GC_MOE_MATRIX_BS=256 GC_MOE_PREFIX_DIAG=1 GC_MOE_LONG_COST=1
export GC_MOE_KV_BYTES=21474836480 GC_MOE_STUDY_STEP=0 GC_MOE_ORDER_EXTRA=1 GC_MOE_SM90_STREAM_M_BLOCKS=1
export VLLM_USE_V2_MODEL_RUNNER=1 VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1 GC_CAPTURE_TOKENS=1
unset GC_MOE_FIXED_REUSE GC_MOE_FIRST_STEP_AUDIT GC_MOE_AA_CONTROL GC_MOE_COST_STUDY
export GC_MOE_PREFIX_HIDDEN_AA=1
for trial in first second; do
  export GC_MOE_MATRIX_ARM=vanilla
  bash scripts/trace_layer/run_aligned_ep8.sh "$sha" --study-main20 > "$work/$trial.log" 2>&1
  raw=$(sed -n 's/^Capture workspace: //p' "$work/$trial.log")
  cp "$raw/run.json" "$work/$trial.json"
done
```

将 first.json、second.json、reference.sha、两份日志提交到 docs/trace_executor_20261119/sm90_prefix_hidden_aa_results/，回执写入 docs/trace_handover_20261119/REPLY_SM90_PREFIX_HIDDEN_AA.md 并 push main。确认两个报告的 prompt_hash、positions、batch_order、prefill_steps、samples 完全相同。按 rank、引擎步、请求对应每个摘要，并核对各请求的接受次数等于 positions。由 mismatches 与共同参考还原两次输出差异，分别统计“两次原生 token 不同且至少一个 rank 的 hidden 不同”和“两次原生 token 不同且所有 rank 的 hidden 相同”。列出最早发生差异的引擎步、请求、请求内位置。第一种情况只定位到 sampler 输入已经不同，第二种才继续查 sampler 及其下游。若两次完全一致也按实交付。
