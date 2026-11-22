#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
sha=$(git rev-parse HEAD)
source scripts/env.sh
py="$GC_ENV/bin/python"
previous=${SM90_RETAINED:-/tmp/sm90-bs256-trace-pbpMbH4u}
ref="$PWD/docs/trace_executor_20261119/sm90_main24_stable_decode_bs256_results/delivery/bs256"
work=$(mktemp -d /tmp/sm90-bs256-eco-XXXXXXXX)
echo "Trial workspace: $work"
export GC_MOE_EXEC_ROOT="$PWD" GC_MOE_MATRIX_SIZE=24 GC_MOE_MATRIX_BS=256 GC_MOE_MATRIX_ARM=eco
export GC_MOE_MATRIX_PREFIX_CHECK=1 GC_MOE_LONG_COST=1 GC_MOE_ORDER_EXTRA=1 GC_MOE_ORDER_PREFILL=1
export GC_MOE_STABLE_PREFILL_TOPK=1 GC_MOE_STABLE_DECODE_TOPK=1 GC_MOE_STUDY_STEP=0
export GC_MOE_SM90_STREAM_M_BLOCKS=1 GC_MOE_KV_BYTES=21474836480 GC_CAPTURE_TOKENS=1
export VLLM_USE_V2_MODEL_RUNNER=1 VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1
export GC_MOE_STRICT_STATE_DIR="$previous/kv"
export DATASET=/mnt/nas2/nanjun.cp/MoE-Bench-Experiments/datasets/dsv4-flash_online_input_ids_26-10-08_1.jsonl
(cd "$ref/.." && sha256sum --quiet -c SHA256SUMS)
for rank in {0..7}; do
  for phase in '' 'prefill/'; do
    cmp "$ref/checkpoint/${phase}rank$rank.json" "$previous/kv/${phase}rank$rank.json"
  done
done
for trial in baseline split dispatch; do
  export SM90_ECO_TRIAL="$trial"
  unset GC_MOE_SM90_SPLIT_EXCHANGE GC_MOE_SM90_PLAN_DISPATCH
  grid=24
  case "$trial" in
    split) export GC_MOE_SM90_SPLIT_EXCHANGE=1 ;;
    dispatch) export GC_MOE_SM90_PLAN_DISPATCH=1; grid=20 ;;
  esac
  dir="$work/$trial"
  export GC_MOE_FIXED_DIR="$dir/replay" GC_BS256_TRACE_DIR="$dir/raw"
  mkdir -p "$dir/replay"
  for file in stats-rank{0..7}.pt measure-rank{0..7}.pt tokens.json placement.json stats-demand.npz; do
    test -r "$previous/replay/$file"
    ln -s "$previous/replay/$file" "$dir/replay/$file"
  done
  nvidia-smi --query-gpu=index,uuid,name,pci.bus_id,driver_version,power.limit,clocks.max.sm,clocks.max.memory --format=csv,noheader > "$dir/hardware.csv"
  diff -u "$ref/eco/hardware.csv" "$dir/hardware.csv"
  while [[ -n $(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader) ]]; do sleep 30; done
  timeout 120m bash scripts/trace_layer/run_aligned_ep8.sh "$sha" --study-main20 2>&1 | tee "$dir/runner.log"
  raw=$(sed -n 's/^Capture workspace: //p' "$dir/runner.log")
  grep -Fxq exit_code=0 "$raw/status.txt"
  printf '%s\n' "$raw" > "$dir/measurement.path"
  cmp "$ref/eco/installed-layout.json" "$dir/replay/eco-layout.json"
  "$py" scripts/trace_layer/aligned_step.py "$dir/raw" "$dir/perfetto" --layers 0 8 20 32 39 --dispatch-grid "$grid"
  "$py" scripts/trace_layer/pull_gemm_detail.py "$dir/raw" "$dir/perfetto" --require-streamed-blocks --require-batch-claims
  "$py" scripts/trace_layer/check_bs256_trace.py "$ref/eco" "$raw" eco "$dir/perfetto" | tee "$dir/verification.txt"
done
"$py" scripts/trace_layer/bs256_eco_times.py "$work" | tee "$work/moe.csv"
