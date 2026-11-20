#!/usr/bin/env bash
# Run exactly one linear and one valid-cost process, in the current clean checkout.
set -eo pipefail
[[ $# == 1 ]] || { echo "Expected a new delivery directory" >&2; exit 2; }
cd "$(git rev-parse --show-toplevel)"
out=$(realpath -m "$1")
test -z "$(git status --porcelain)"
mkdir "$out"
sha=$(git rev-parse HEAD)
echo "$sha" > "$out/measurement.sha"
source scripts/env.sh
analysis_python="$GC_ENV/bin/python"
test -x "$analysis_python"
state=$(mktemp -d "${GC_MOE_CHECKPOINT_ROOT:-/tmp}/gcmoe-long-cost-XXXXXXXX")
echo "$state" > "$out/state-path.txt"
export GC_MOE_FIXED_DIR="$state/replay" GC_MOE_STRICT_STATE_DIR="$state/kv"
mkdir "$GC_MOE_FIXED_DIR" "$GC_MOE_STRICT_STATE_DIR"
export GC_MOE_KV_BYTES=21474836480 GC_MOE_LONG_COST=1 GC_MOE_COST_STUDY=1 GC_MOE_ORDER_EXTRA=1
export GC_MOE_EXEC_ROOT="$PWD" GC_MOE_SM90_STREAM_M_BLOCKS=1 GC_MOE_STUDY_STEP=0
export VLLM_USE_V2_MODEL_RUNNER=1 VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1 GC_CAPTURE_TOKENS=1
unset GC_MOE_FIXED_REUSE GC_MOE_FIRST_STEP_AUDIT GC_MOE_AA_CONTROL
"$analysis_python" - "$state" <<'PY'
import shutil,sys
# Eight 20-GiB KV images and two independent 40,000-call route banks per rank.
if shutil.disk_usage(sys.argv[1]).free < 8*21474836480 + 2*8*40000*256*6*(8+4):
    raise SystemExit('Insufficient local space for the KV checkpoint and the two route banks')
PY
for arm in linear fine; do
  mkdir "$out/$arm"
  log="$out/$arm/runner.log"
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name --format=csv,noheader > "$out/$arm/preexisting-processes.csv"
  test ! -s "$out/$arm/preexisting-processes.csv"
  nvidia-smi --query-gpu=index,uuid,name,pci.bus_id,driver_version,power.limit,clocks.max.sm,clocks.max.memory --format=csv,noheader > "$out/$arm/hardware.csv"
  awk -F, '$3 !~ /NVIDIA H20/ { bad=1 } END { exit (NR != 8 || bad) }' "$out/$arm/hardware.csv"
  export GC_MOE_STUDY_COSTFN=0
  if [[ $arm == fine ]]; then export GC_MOE_STUDY_COSTFN=2; fi
  bash scripts/trace_layer/run_aligned_ep8.sh "$sha" --study-long-cost > "$log" 2>&1
  raw=$(sed -n 's/^Capture workspace: //p' "$log")
  grep -Fxq exit_code=0 "$raw/status.txt"
  test "$(cat "$raw/execution.sha")" = "$sha"
  cp "$raw"/{run.json,rank{0..7}.npz,runtime.log,status.txt,gpu.log,execution.sha} "$out/$arm/"
done
mkdir "$out/checkpoint"
cp "$GC_MOE_STRICT_STATE_DIR"/rank{0..7}.json "$out/checkpoint/"
cp "$GC_MOE_FIXED_DIR/placement.json" "$out/"
(cd "$GC_MOE_FIXED_DIR" && sha256sum stats-rank*.pt measure-rank*.pt stats-demand.pt tokens.json placement.json) > "$out/workload.sha256"
"$analysis_python" scripts/trace_layer/long_cost_report.py "$out"
(cd "$out" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
