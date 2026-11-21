# 【SM90】BS256 vanilla 两次 prefill：定位首处 MoE 边界差异

原始记录复算后的最早 sampler hidden 差异在引擎步 13，最早原生 token 差异在步 19。按原协议跑两次同配置 vanilla prefill，前 20 步在每个实际执行的 MoE 层记录输入 hidden、传入的路由 ID 和权重、输出摘要。摘要涉及同步，两次均打开，只用于定位，不计性能。正式实验判据保持不变。

仅运行下列命令；仅允许有记录的语义等价编译修复，其他失败保留现场。

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
export GC_MOE_PREFIX_HIDDEN_AA=1 GC_MOE_PREFIX_LAYER_AA=1
for trial in first second; do
  export GC_MOE_MATRIX_ARM=vanilla
  bash scripts/trace_layer/run_aligned_ep8.sh "$sha" --study-main20 > "$work/$trial.log" 2>&1
  raw=$(sed -n 's/^Capture workspace: //p' "$work/$trial.log")
  cp "$raw/run.json" "$work/$trial.json"
done
```

将两个 JSON、日志、reference.sha 归档到 docs/trace_executor_20261119/sm90_prefix_layer_aa_results/；回执 docs/trace_handover_20261119/REPLY_SM90_PREFIX_LAYER_AA.md，提交并 push main。按 (step,layer,rank) 比较两次 moe_by_rank，先检查 20 步 × 40 层 × 8 rank 的覆盖、顺序、tokens 均一致；缺失则注明，不能判为相等。给出最早不同边界的步和层以及各字段不同的 rank。若上一步层输出相同而本层输入已不同，只能定位到中间路径；相同输入与路由而输出不同则记录相关 rank。还要复核本次 sampler hidden 与原生 token 是否分叉。探针改变同步时序，若未复现也按实回执。
