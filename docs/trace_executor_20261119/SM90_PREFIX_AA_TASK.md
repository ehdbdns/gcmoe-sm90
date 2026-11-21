# 【SM90】BS256 prefill 原生 top-1 自对照与随机放置诊断

只在原 8×H20 运行。复用已完成 vanilla 的 tokens.json，只跑 vanilla、random 各一次 prefill。两者均回放同一份反馈，保留替换前的原生 token。两次都在所有请求完成 prefill 后退出；无需重做 1000+1000。先确认同一 vanilla 后端重复执行能否复现原始参考，再检查随机放置引入了多少额外差异。性能主表和严格判据沿用原版。

在干净 checkout 中执行以下命令，使用上轮相同的模型、数据、软件和八张空闲卡。执行者只执行命令；只允许语义等价的编译修复，需记录独立 SHA。缺失参考文件或遇到错误即保留日志、提交失败回执。

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
for arm in vanilla random; do
  export GC_MOE_MATRIX_ARM="$arm"
  bash scripts/trace_layer/run_aligned_ep8.sh "$sha" --study-main20 > "$work/$arm.log" 2>&1
  raw=$(sed -n 's/^Capture workspace: //p' "$work/$arm.log")
  cp "$raw/run.json" "$work/$arm.json"
done
```

将两份 JSON、reference.sha、两臂日志（失败则含原始异常）提交到 docs/trace_executor_20261119/sm90_prefix_aa_results/。回执写入 docs/trace_handover_20261119/REPLY_SM90_PREFIX_AA.md，提交并推送 main。检查两份报告的 prompt_hash、prefill_steps、positions 和 batch_order 是否逐项相同。分别给出 mismatches 条数 / samples、涉及的不同请求数、位置 0 的失配数。每条明细按 [请求编号, 请求内位置, 原生 token, 参考 token] 记录。两臂之间也按同一请求和位置比较原生值；未列入明细的位置等于该位置的参考值。报告 req109 的位置 0 在两次启动中的结果。所有前缀都要保留，发现数值差异时按诊断结果交付。
