# 【SM90】定位固定索引后首个 sampler hidden 差异（第 38 步，req134）

上一轮两次 vanilla 的 0–19 步 40 层 MoE 边界全部相同；采样 hidden 的首个差异在第 38 步，八个 rank 都是 req134。只跑两次相同 vanilla，保留 prefill 和 decode 的索引排序，仅在第 20–38 步记录每层 MoE 的输入、路由和输出。

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
unset GC_MOE_PREFIX_SPARSE_AA GC_MOE_PREFIX_BLOCK_AA
export GC_MOE_ORDER_PREFILL=1 GC_MOE_PREFIX_HIDDEN_AA=1 GC_MOE_PREFIX_LAYER_AA=1
export GC_MOE_PREFIX_LAYER_FIRST=20 GC_MOE_PREFIX_LAYER_STOP=39
for trial in first second; do
  export GC_MOE_MATRIX_ARM=vanilla
  bash scripts/trace_layer/run_aligned_ep8.sh "$sha" --study-main20 > "$work/$trial.log" 2>&1
  raw=$(sed -n 's/^Capture workspace: //p' "$work/$trial.log")
  cp "$raw/run.json" "$work/$trial.json"
done
```

提交 first/second 的完整 JSON、日志、reference.sha 与校验和到 docs/trace_executor_20261119/sm90_order_step38_results/，回执 docs/trace_handover_20261119/REPLY_SM90_ORDER_STEP38.md。逐 rank 检查每步 40 层覆盖、最早不同的步、层、字段与该层之前的最后相同边界；同时给出最早不同的 sampler hidden（步和请求 ID）、两次原生采样差异。探针有同步开销，只作为公平性诊断。仅允许等价编译修复。
