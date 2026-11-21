# 【SM90】定位第 19、20 个 decoder block 内部的首处差异（层号从 0 开始）

复用两次相同 vanilla prefill；保留此前 MoE 与 sampler 摘要。仅在步 12、13 对层 19/20 的实际模块及两级子模块，在前向进入及返回时记录张量摘要。同步可能影响复现；此任务没有性能验收。

按以下步骤执行，仅允许有记录的等价编译修复，其他问题交回设计端。

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
export GC_MOE_PREFIX_HIDDEN_AA=1 GC_MOE_PREFIX_LAYER_AA=1 GC_MOE_PREFIX_BLOCK_AA=1
for trial in first second; do
  export GC_MOE_MATRIX_ARM=vanilla
  bash scripts/trace_layer/run_aligned_ep8.sh "$sha" --study-main20 > "$work/$trial.log" 2>&1
  raw=$(sed -n 's/^Capture workspace: //p' "$work/$trial.log")
  cp "$raw/run.json" "$work/$trial.json"
done
```

核对 block_by_rank 两次执行的覆盖、形状、模块和进入/返回顺序。定位八卡上按时间顺序最早的张量差异；给出所属模块、调用边界、字段、该模块输入是否相同，以及此前确认一致的边界。无记录、残差原地修改或编译路径跳过的模块不能当作相等。给出实际覆盖的子模块名称，并报告原 step 13/layer 20 MoE 差异是否复现。提交两个 JSON、日志、reference.sha 到 docs/trace_executor_20261119/sm90_prefix_block_aa_results/，回执 docs/trace_handover_20261119/REPLY_SM90_PREFIX_BLOCK_AA.md，push main。
