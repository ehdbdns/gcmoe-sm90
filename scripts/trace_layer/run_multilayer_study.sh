#!/usr/bin/env bash
# Run both arms in the same pinned measurement checkout.
set -eo pipefail
if [[ $# != 2 && ! ( $# == 3 && $3 == --valid-cost ) && ! ( $# == 4 && ( $3 == --valid-cost-from || $3 == --replay-from || $3 == --same-state-from || $3 == --trace-first-step-from || $3 == --negative-control-from || $3 == --ordered-decode-from ) ) ]]; then
  echo "Usage: $0 common-worktree new-output-dir [--valid-cost | --valid-cost-from directory | --replay-from directory | --same-state-from directory | --trace-first-step-from directory | --negative-control-from directory | --ordered-decode-from directory]" >&2; exit 2
fi
here=$(cd "$(dirname "$0")" && pwd)
root=$(realpath "$1")
bundle=$(realpath -m "$2")
sha=$(git -C "$here" rev-parse HEAD)
test "$(git -C "$root" rev-parse HEAD)" = "$sha"
analysis_python=$(cd "$root" && source scripts/env.sh && printf '%s/bin/python' "$GC_ENV")
test -x "$analysis_python"
unset GC_MOE_COST_STUDY GC_MOE_ORDER_EXTRA GC_MOE_AA_CONTROL GC_MOE_FIRST_STEP_AUDIT GC_MOE_FIXED_DIR GC_MOE_FIXED_REUSE GC_MOE_STRICT_STATE_DIR GC_MOE_KV_BYTES
protocol=${3:-}
replay=${4:-}
if [[ $protocol == --ordered-decode-from ]]; then
  export GC_MOE_ORDER_EXTRA=1
  protocol=--negative-control-from
fi
results="$here/../../docs/trace_executor_20261119"
if [[ $protocol == --valid-cost || $protocol == --valid-cost-from ]]; then
  if [[ $protocol == --valid-cost ]]; then
    replay=$(cat "$results/fixed_stream_results_r8/workload-path.txt")
  fi
  export GC_MOE_COST_STUDY=1 GC_MOE_ORDER_EXTRA=1
  protocol=--trace-first-step-from
fi
mkdir "$bundle"
save_partial_checkpoint() {
  local status=$?
  if (( status != 0 )) && [[ -n ${GC_MOE_STRICT_STATE_DIR:-} ]]; then
    mkdir -p "$bundle/checkpoint"
    for manifest in "$GC_MOE_STRICT_STATE_DIR"/rank*.json; do
      if [[ -f "$manifest" ]]; then cp "$manifest" "$bundle/checkpoint/"; fi
    done
    if [[ -d "$GC_MOE_STRICT_STATE_DIR/treatment" ]]; then
      mkdir -p "$bundle/checkpoint/treatment"
      for control in "$GC_MOE_STRICT_STATE_DIR"/control-step-mode*.json; do
        if [[ -f "$control" ]]; then cp "$control" "$bundle/checkpoint/"; fi
      done
      for manifest in "$GC_MOE_STRICT_STATE_DIR"/treatment/rank*.json; do
        if [[ -f "$manifest" ]]; then cp "$manifest" "$bundle/checkpoint/treatment/"; fi
      done
    fi
  fi
  return "$status"
}
trap save_partial_checkpoint EXIT
echo "$sha" > "$bundle/measurement.sha"
if [[ $protocol == --same-state-from || $protocol == --trace-first-step-from || $protocol == --negative-control-from ]]; then
  GC_MOE_STRICT_STATE_DIR=$(mktemp -d "${GC_MOE_CHECKPOINT_ROOT:-/tmp}/gcmoe-branch-XXXXXXXX")
  export GC_MOE_STRICT_STATE_DIR GC_MOE_KV_BYTES=11811160064
  if [[ $protocol == --negative-control-from ]]; then
    export GC_MOE_AA_CONTROL=1
    mkdir "$GC_MOE_STRICT_STATE_DIR/treatment"
  fi
  "$analysis_python" - "$GC_MOE_STRICT_STATE_DIR" "$GC_MOE_KV_BYTES" <<'CHECK_SPACE'
from pathlib import Path
import shutil
import sys
available = shutil.disk_usage(sys.argv[1]).free
needed = 8 * int(sys.argv[2]) * (2 if Path(sys.argv[1], 'treatment').is_dir() else 1)
if available < needed:
    sys.exit(f'KV checkpoint needs up to {needed} bytes, mount has {available}; set GC_MOE_CHECKPOINT_ROOT')
CHECK_SPACE
  echo "$GC_MOE_STRICT_STATE_DIR" > "$bundle/checkpoint-path.txt"
  if [[ $protocol == --trace-first-step-from || $protocol == --negative-control-from ]]; then
    export GC_MOE_FIRST_STEP_AUDIT=1
    if [[ ${GC_MOE_COST_STUDY:-0} == 1 ]]; then
      echo fixed_valid_cost > "$bundle/comparison.txt"
    elif [[ ${GC_MOE_ORDER_EXTRA:-0} == 1 ]]; then
      echo fixed_stream_r8 > "$bundle/comparison.txt"
    elif [[ ${GC_MOE_AA_CONTROL:-0} == 1 ]]; then
      echo fixed_stream_r7 > "$bundle/comparison.txt"
    else
      echo fixed_stream_r6 > "$bundle/comparison.txt"
    fi
  else
    echo fixed_stream_r5 > "$bundle/comparison.txt"
  fi
elif [[ $protocol == --replay-from ]]; then
  echo fixed_stream_r4 > "$bundle/comparison.txt"
else
  echo fixed_stream > "$bundle/comparison.txt"
fi
GC_MOE_FIXED_DIR=$(mktemp -d /tmp/gcmoe-route-replay-XXXXXXXX)
export GC_MOE_FIXED_DIR VLLM_USE_V2_MODEL_RUNNER=1 VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1
echo "$GC_MOE_FIXED_DIR" > "$bundle/workload-path.txt"
"$analysis_python" "$here/multilayer_study.py" reference \
  "$results/multilayer_ab_results/step02/baseline/paired" "$GC_MOE_FIXED_DIR" \
  --prepared "$results/fixed_stream_results_failed/tokens.json"
if [[ $protocol == --replay-from || $protocol == --same-state-from || $protocol == --trace-first-step-from || $protocol == --negative-control-from ]]; then
  replay=$(realpath "$replay")
  cmp "$GC_MOE_FIXED_DIR/tokens.json" "$replay/tokens.json"
  cp "$replay"/{placement.json,routes-rank{0..7}.pt} "$GC_MOE_FIXED_DIR/"
  export GC_MOE_FIXED_REUSE=1
  PYTHONPATH="$root/source:$root/integration" "$analysis_python" "$here/check_cold_replay.py" \
    "$GC_MOE_FIXED_DIR" "$results/fixed_stream_results_r2_blocked/step02/baseline/paired/runner.log" \
    | tee "$bundle/cold-replay-check.log"
fi
export GC_CAPTURE_TOKENS=1
for arm in baseline stream; do
  flag=0
  cost=0
  if [[ $arm == stream ]]; then flag=1; fi
  if [[ ${GC_MOE_COST_STUDY:-0} == 1 ]]; then
    flag=1
    if [[ $arm == stream ]]; then cost=2; fi
  fi
  out="$bundle/step02/$arm/paired"
  mkdir -p "$out"
  if [[ -n ${GC_MOE_STRICT_STATE_DIR:-} ]]; then
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name --format=csv,noheader > "$out/preexisting-processes.csv"
    if [[ -s "$out/preexisting-processes.csv" ]]; then
      echo "GPUs already in use; see $out/preexisting-processes.csv" >&2
      exit 1
    fi
    nvidia-smi --query-gpu=index,pstate,temperature.gpu,clocks.current.sm,clocks.current.memory,power.draw \
      --format=csv,noheader > "$out/gpu-before.csv"
    nvidia-smi --query-gpu=index,uuid,name,pci.bus_id,driver_version,power.limit,clocks.max.sm,clocks.max.memory \
      --format=csv,noheader > "$out/hardware.csv"
  fi
  echo "step02 $arm paired"
  GC_MOE_STUDY_COSTFN="$cost" GC_MOE_EXEC_ROOT="$root" GC_MOE_STUDY_STEP=2 GC_MOE_SM90_STREAM_M_BLOCKS="$flag" \
    bash "$here/run_aligned_ep8.sh" "$sha" --study-paired > "$out/runner.log" 2>&1
  raw=$(sed -n 's/^Capture workspace: //p' "$out/runner.log")
  grep -Fxq exit_code=0 "$raw/status.txt"
  test "$(cat "$raw/execution.sha")" = "$sha"
  PYTHONPATH="$root/source" "$analysis_python" "$here/multilayer_study.py" collect "$raw" "$out" paired 2
  mv "$out/runner.log" "$raw/study-runner.log"
  cp "$raw"/{execution.sha,status.txt,gpu.log,runtime.log} "$out/"
  if [[ -n ${GC_MOE_STRICT_STATE_DIR:-} ]]; then cp "$raw/gpu-after.csv" "$out/"; fi
  sha256sum "$raw/result.json" > "$out/result.sha256"
  cp "$raw/sha256.txt" "$out/raw-sha256.txt"
done
if [[ -n ${GC_MOE_STRICT_STATE_DIR:-} ]]; then
  mkdir "$bundle/checkpoint"
  cp "$GC_MOE_STRICT_STATE_DIR"/rank{0..7}.json "$bundle/checkpoint/"
  if [[ ${GC_MOE_AA_CONTROL:-0} == 1 ]]; then
    mkdir "$bundle/checkpoint/treatment"
    cp "$GC_MOE_STRICT_STATE_DIR"/treatment/rank{0..7}.json "$bundle/checkpoint/treatment/"
  fi
fi
if [[ -n ${GC_MOE_FIXED_DIR:-} ]]; then
  cp "$GC_MOE_FIXED_DIR/placement.json" "$bundle/"
  sha256sum "$GC_MOE_FIXED_DIR"/{placement.json,tokens.json,routes-rank{0..7}.pt} > "$bundle/workload.sha256"
fi
PYTHONPATH="$root/source" "$analysis_python" "$here/multilayer_study.py" compare "$bundle"
(cd "$bundle" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
