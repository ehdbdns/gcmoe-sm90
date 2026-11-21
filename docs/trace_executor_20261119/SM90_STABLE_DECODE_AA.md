# 【SM90】稳定 decode top-k 的 1000 步 vanilla 发布与消费复验

纯 decode 的 token top-k 与 prefill 分别调用不同的 vLLM 入口。此诊断在 SparseAttnIndexer 的 decode 调用上采用原分数降序、同分逻辑位置升序，在全新目录生成 vanilla 参考并由相同 vanilla 回放。不改主表，不把诊断选择器计入性能结论。

```bash
set -eo pipefail
git pull --ff-only origin main
repo=$(git rev-parse --show-toplevel)
sha=$(git rev-parse HEAD)
work=$(mktemp -d /tmp/gcmoe-vanilla-consume-XXXXXXXX)
git worktree add --detach "$work/code" "$sha"
cd "$work/code"
state=$(mktemp -d "${GC_MOE_CHECKPOINT_ROOT:-/tmp}/gcmoe-decode-aa-XXXXXXXX")
export GC_MOE_FIXED_DIR="$state/replay" GC_MOE_STRICT_STATE_DIR="$state/kv"
mkdir "$GC_MOE_FIXED_DIR" "$GC_MOE_STRICT_STATE_DIR"
df -h "$state" # 单份八卡 KV 和 stats 路由需约 170 GiB 可用空间
export GC_MOE_EXEC_ROOT="$PWD" GC_MOE_MATRIX_SIZE=24 GC_MOE_MATRIX_ARM=vanilla GC_MOE_MATRIX_BS=256
export GC_MOE_LONG_COST=1 GC_MOE_MATRIX_PREFIX_CHECK=1 GC_MOE_ORDER_EXTRA=1 GC_MOE_ORDER_PREFILL=1
export GC_MOE_STABLE_PREFILL_TOPK=1 GC_MOE_DECODE_STATS_AA=1 GC_MOE_STABLE_DECODE_TOPK=1 GC_MOE_KV_BYTES=21474836480
export GC_MOE_SM90_STREAM_M_BLOCKS=1 GC_MOE_STUDY_STEP=0 VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1 GC_CAPTURE_TOKENS=1
unset GC_MOE_FIXED_REUSE GC_MOE_FIRST_STEP_AUDIT GC_MOE_AA_CONTROL GC_MOE_COST_STUDY GC_MOE_PREFIX_DIAG GC_MOE_PREFIX_LAYER_AA GC_MOE_PREFIX_BLOCK_AA GC_MOE_PREFIX_HIDDEN_AA
source config/experiment.env
source scripts/env.sh
"$GC_ENV/bin/python" tests/test_stable_prefill_topk_cpu.py
for phase in publisher consumer; do
  unset GC_MOE_VANILLA_STATS_REPLAY
  if [[ $phase == consumer ]]; then export GC_MOE_VANILLA_STATS_REPLAY=1; fi
  if bash scripts/trace_layer/run_aligned_ep8.sh "$sha" --study-main20 > "$work/$phase.log" 2>&1; then code=0; else code=$?; fi
  echo "$code" > "$work/$phase.exit"
  if ((code)); then break; fi
done
```

分别保存 run.json、日志、请求数、状态、参考 SHA256、state 路径。若失败报告首次差异、窗口原生失配总数。原生 token、prefill 及物理工作量验收保持原有条件。回执 docs/trace_handover_20261119/REPLY_SM90_STABLE_DECODE_AA.md，产物 docs/trace_executor_20261119/sm90_stable_decode_aa_results/。仅允许等价编译修复。
