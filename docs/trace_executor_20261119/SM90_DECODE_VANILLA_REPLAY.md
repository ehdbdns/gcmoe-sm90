# 【SM90】用 vanilla 消费原 vanilla 快照，定位统计窗口 4 处原生分歧

目的：在与失败 random 相同的消费者路径上只换回 vanilla 放置。读取前一轮同一份 vanilla 的 tokens、统计路由和 prefill KV，保留全部严格验证。只启动一次模型，完成 1000 步统计即退出。此次用来定位，不进入主表。若同样分歧，应优先查共享 decode 或回放；若这一次完全一致，则转查 random 放置。无需重跑发布者或其他臂。

```bash
set -eo pipefail
git pull --ff-only origin main
repo=$(git rev-parse --show-toplevel)
sha=$(git rev-parse HEAD)
work=$(mktemp -d /tmp/gcmoe-vanilla-consume-XXXXXXXX)
git worktree add --detach "$work/code" "$sha"
cd "$work/code"
state=$(cat docs/trace_executor_20261119/sm90_main24_stable_bs256_results_blocked/delivery/bs256/state-path.txt)
export GC_MOE_FIXED_DIR="$state/replay" GC_MOE_STRICT_STATE_DIR="$state/kv"
test "$(sha256sum "$GC_MOE_FIXED_DIR/tokens.json" | cut -d' ' -f1)" = 880da497abce4830742fb3d060cf52bc7db575b8b71c475de246731c16d536b8
export GC_MOE_EXEC_ROOT="$PWD" GC_MOE_MATRIX_SIZE=24 GC_MOE_MATRIX_ARM=vanilla GC_MOE_MATRIX_BS=256
export GC_MOE_LONG_COST=1 GC_MOE_MATRIX_PREFIX_CHECK=1 GC_MOE_ORDER_EXTRA=1 GC_MOE_ORDER_PREFILL=1
export GC_MOE_STABLE_PREFILL_TOPK=1 GC_MOE_VANILLA_STATS_REPLAY=1 GC_MOE_KV_BYTES=21474836480
export GC_MOE_SM90_STREAM_M_BLOCKS=1 GC_MOE_STUDY_STEP=0 VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1 GC_CAPTURE_TOKENS=1
unset GC_MOE_FIXED_REUSE GC_MOE_FIRST_STEP_AUDIT GC_MOE_AA_CONTROL GC_MOE_COST_STUDY GC_MOE_PREFIX_DIAG GC_MOE_PREFIX_LAYER_AA GC_MOE_PREFIX_BLOCK_AA GC_MOE_PREFIX_HIDDEN_AA
if bash scripts/trace_layer/run_aligned_ep8.sh "$sha" --study-main20 > "$work/runner.log" 2>&1; then code=0; else code=$?; fi
echo "$code" > "$work/exit-code.txt"
```

保留原有参考文件。提交 exit-code、runner.log、Capture workspace 的 run.json（若成功）、runtime.log、失败栈或成功的八卡核验结果、参考 SHA。交到 docs/trace_executor_20261119/sm90_decode_vanilla_replay_results/ 与 docs/trace_handover_20261119/REPLY_SM90_DECODE_VANILLA_REPLAY.md。仅允许等价编译修复，其余错误原样回执。
