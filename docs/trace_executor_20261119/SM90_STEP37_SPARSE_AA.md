# 【SM90】固定索引后，核对第 37 步第 20 层真实 attention 输入和输出

刚返回的 6080 条原始记录显示，所有 rank 在第 37 步第 19 层 MoE 的输出仍一致，第 20 层 MoE 输入 hidden 已不同。两次相同 vanilla 均继续开启 prefill/decode 排序。只对步 36、37 探测稀疏 attention 和模块边界，逐项核对已排序索引、每行多重集、有效长度、Q、选中的 KV 及实际输出。若输入多重集不同，结合模型源码检查选块或选 token；若相同，先找实际不同字段。仅允许等价编译修复。

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
export GC_MOE_PREFIX_BLOCK_FIRST=36 GC_MOE_PREFIX_BLOCK_STOP=38
export GC_MOE_PREFIX_LAYER_FIRST=36 GC_MOE_PREFIX_LAYER_STOP=38
export GC_MOE_ORDER_PREFILL=1 GC_MOE_PREFIX_SPARSE_AA=1 GC_MOE_PREFIX_HIDDEN_AA=1 GC_MOE_PREFIX_LAYER_AA=1 GC_MOE_PREFIX_BLOCK_AA=1
for trial in first second; do
  export GC_MOE_MATRIX_ARM=vanilla
  bash scripts/trace_layer/run_aligned_ep8.sh "$sha" --study-main20 > "$work/$trial.log" 2>&1
  raw=$(sed -n 's/^Capture workspace: //p' "$work/$trial.log")
  cp "$raw/run.json" "$work/$trial.json"
done
```

逐个 rank、步和 chunk 检查 L20.flash_mla_sparse_fwd 的进入及返回覆盖，先报最早不同的字段，再对照 wo_b 输入。注意 mixed batch 中本探针只覆盖 prefill token；零记录回报为未覆盖。提交两个完整 JSON、日志与 SHA256SUMS 到 docs/trace_executor_20261119/sm90_step37_sparse_aa_results/，回执 docs/trace_handover_20261119/REPLY_SM90_STEP37_SPARSE_AA.md。

还须比较两次运行自身的全部 positions、batch_order、逐 request 已采样 token（由相对同一参考的 mismatches 重建）、hidden_by_rank、moe_by_rank。相对于旧参考的失配与两次彼此是否一致分别报告，并给出剩余首处分歧。
