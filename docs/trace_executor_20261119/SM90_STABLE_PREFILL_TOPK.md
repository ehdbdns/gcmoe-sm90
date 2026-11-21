# 【SM90】稳定选择 prefill top-k 后复验逐请求公平性

第 37 步 layer 20 的 rank 2/4/6/7 在 FlashMLA 入口已经选中了不同的每行索引。vLLM 0.31.0 的 sampler.cu 在同分时用原子写入阈值桶的位置打破平局。此诊断改用原分数降序、同分按原逻辑位置升序，保留短行语义、有效范围、-1 填充。两次 vanilla 使用相同配置；只做数值验收，不计时间。

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
source config/experiment.env
source scripts/env.sh
"$GC_ENV/bin/python" tests/test_stable_prefill_topk_cpu.py
export GC_MOE_STABLE_PREFILL_TOPK=1
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

逐个 rank、步和 chunk 检查 L20.flash_mla_sparse_fwd 的进入及返回覆盖，先报最早不同的字段，再对照 wo_b 输入。注意 mixed batch 中本探针只覆盖 prefill token；零记录回报为未覆盖。提交两个完整 JSON、日志与 SHA256SUMS 到 docs/trace_executor_20261119/sm90_stable_topk_results/，回执 docs/trace_handover_20261119/REPLY_SM90_STABLE_TOPK.md。

还须比较两次运行自身的全部 positions、batch_order、逐 request 已采样 token（由相对同一参考的 mismatches 重建）、hidden_by_rank、moe_by_rank。相对于旧参考的失配与两次彼此是否一致分别报告，并给出剩余首处分歧。
