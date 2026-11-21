# 【SM90】检查 prefill 稀疏 attention 的实际输入和原位输出

仅重复相同 vanilla 两次，在步 12、13 的第 20 层按调用顺序对比 FlashMLA 的真实 heads、有效索引顺序、每行多重集、长度、选中 KV、sink、scale 和写出的结果。保留原有 block 与采样摘要。只允许等价编译修复。

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
export GC_MOE_PREFIX_SPARSE_AA=1 GC_MOE_PREFIX_HIDDEN_AA=1 GC_MOE_PREFIX_LAYER_AA=1 GC_MOE_PREFIX_BLOCK_AA=1
for trial in first second; do
  export GC_MOE_MATRIX_ARM=vanilla
  bash scripts/trace_layer/run_aligned_ep8.sh "$sha" --study-main20 > "$work/$trial.log" 2>&1
  raw=$(sed -n 's/^Capture workspace: //p' "$work/$trial.log")
  cp "$raw/run.json" "$work/$trial.json"
done
```

逐个 rank、步和 chunk 检查 L20.flash_mla_sparse_fwd 的进入及返回覆盖，先报最早不同的字段，再对照 wo_b 输入。注意 mixed batch 中本探针只覆盖 prefill token；零记录回报为未覆盖。提交两个完整 JSON、日志与 SHA256SUMS 到 docs/trace_executor_20261119/sm90_sparse_prefill_aa_results/，回执 docs/trace_handover_20261119/REPLY_SM90_SPARSE_PREFILL_AA.md。
