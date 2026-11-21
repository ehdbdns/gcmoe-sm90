# 【SM90】固定 prefill 有效索引顺序，复验 vanilla A/A

两次相同 vanilla 都打开 GC_MOE_ORDER_PREFILL=1。在实际 combine_topk_swa_indices 返回后，仅将每行有效前缀内的非负索引升序排列，保留重复次数、掩码位置、长度、尾部。decode 与原任务一致。仅作公平性诊断，含同步探针；不计性能。只允许等价编译修复。

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
export GC_MOE_ORDER_PREFILL=1 GC_MOE_PREFIX_SPARSE_AA=1 GC_MOE_PREFIX_HIDDEN_AA=1 GC_MOE_PREFIX_LAYER_AA=1 GC_MOE_PREFIX_BLOCK_AA=1
for trial in first second; do
  export GC_MOE_MATRIX_ARM=vanilla
  bash scripts/trace_layer/run_aligned_ep8.sh "$sha" --study-main20 > "$work/$trial.log" 2>&1
  raw=$(sed -n 's/^Capture workspace: //p' "$work/$trial.log")
  cp "$raw/run.json" "$work/$trial.json"
done
```

逐个 rank、步和 chunk 检查 L20.flash_mla_sparse_fwd 的进入及返回覆盖，先报最早不同的字段，再对照 wo_b 输入。注意 mixed batch 中本探针只覆盖 prefill token；零记录回报为未覆盖。提交两个完整 JSON、日志与 SHA256SUMS 到 docs/trace_executor_20261119/sm90_order_prefill_aa_results/，回执 docs/trace_handover_20261119/REPLY_SM90_ORDER_PREFILL_AA.md。

还须比较两次运行自身的全部 positions、batch_order、逐 request 已采样 token（由相对同一参考的 mismatches 重建）、hidden_by_rank、moe_by_rank。相对于旧参考的失配与两次彼此是否一致分别报告，并给出剩余首处分歧。
