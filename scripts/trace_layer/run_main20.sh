#!/usr/bin/env bash
# Each BS has one publisher and all other arms consume its replay bank.
set -eo pipefail
[[ $# == 1 ]] || { echo "Expected a new delivery directory" >&2; exit 2; }
out=$(realpath -m "$1")
cd "$(git rev-parse --show-toplevel)"
test -z "$(git status --porcelain)"
mkdir "$out"
sha=$(git rev-parse HEAD)
echo "$sha" > "$out/measurement.sha"
source scripts/env.sh
py="$GC_ENV/bin/python"
export GC_MOE_KV_BYTES="${GC_MOE_KV_BYTES:-21474836480}" GC_MOE_LONG_COST=1 GC_MOE_ORDER_EXTRA=1
export GC_MOE_EXEC_ROOT="$PWD" GC_MOE_SM90_STREAM_M_BLOCKS=1 GC_MOE_STUDY_STEP=0
export VLLM_USE_V2_MODEL_RUNNER=1 VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1 GC_CAPTURE_TOKENS=1
unset GC_MOE_FIXED_REUSE GC_MOE_FIRST_STEP_AUDIT GC_MOE_AA_CONTROL GC_MOE_COST_STUDY
arms=(vanilla random rr)
report_args=()
case ${GC_MOE_MATRIX_SIZE:-20} in
  20) ;;
  24) arms+=(eco_linear); report_args+=(--with-linear) ;;
  *) exit 2 ;;
esac
arms+=(eco eco_smr)
if [[ ${GC_STEP_BOUNDARY:-0} == 1 ]]; then
  test "${GC_MOE_MATRIX_SIZE:-}" = 24
  # Counterbalance the cost-arm order from BS256/512. Vanilla still publishes.
  arms=(vanilla random rr eco eco_linear eco_smr)
fi
batches=(32 64 128 256)
if [[ -n ${GC_MAIN24_SINGLE_BS:-} ]]; then
  test "${GC_MOE_MATRIX_SIZE:-20}" = 24
  case "$GC_MAIN24_SINGLE_BS" in 256|512|1024|2048) ;; *) exit 2 ;; esac
  batches=("$GC_MAIN24_SINGLE_BS")
fi
export GC_MOE_MATRIX_PREFIX_CHECK=1
for bs in "${batches[@]}"; do
  export GC_MOE_MATRIX_BS="$bs"
  group="$out/bs$bs"
  mkdir "$group"
  state=$(mktemp -d "${GC_MOE_CHECKPOINT_ROOT:-/tmp}/gcmoe-main${GC_MOE_MATRIX_SIZE:-20}-bs$bs-XXXXXXXX")
  echo "$state" > "$group/state-path.txt"
  export GC_MOE_FIXED_DIR="$state/replay" GC_MOE_STRICT_STATE_DIR="$state/kv"
  mkdir "$GC_MOE_FIXED_DIR" "$GC_MOE_STRICT_STATE_DIR"
  "$py" - "$state" "$bs" <<'PY'
import os, shutil, sys
required = 2*8*int(os.environ["GC_MOE_KV_BYTES"]) + 2*8*40000*int(sys.argv[2])*6*12 + 4*1024**3
if shutil.disk_usage(sys.argv[1]).free < required:
    raise SystemExit(f"Need {required / 1024**3:.1f} free GiB on the checkpoint filesystem")
PY
  for arm in "${arms[@]}"; do
    export GC_MOE_MATRIX_ARM="$arm"
    dst="$group/$arm"
    mkdir "$dst"
    log="$dst/runner.log"
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name --format=csv,noheader > "$dst/preexisting-processes.csv"
    test ! -s "$dst/preexisting-processes.csv"
    nvidia-smi --query-gpu=index,uuid,name,pci.bus_id,driver_version,power.limit,clocks.max.sm,clocks.max.memory --format=csv,noheader > "$dst/hardware.csv"
    awk -F, '$3 !~ /NVIDIA H20/ { bad=1 } END { exit (NR != 8 || bad) }' "$dst/hardware.csv"
    echo "Starting BS=$bs arm=$arm"
    if bash scripts/trace_layer/run_aligned_ep8.sh "$sha" --study-main20 > "$log" 2>&1; then code=0; else code=$?; fi
    raw=$(sed -n 's/^Capture workspace: //p' "$log")
    if [[ -d "$raw" ]]; then
      for name in execution.sha runtime.log status.txt gpu.log failed-command.txt run.json rank{0..7}.npz; do
        if [[ -f "$raw/$name" ]]; then cp "$raw/$name" "$dst/"; fi
      done
    fi
    test "$code" = 0 || exit "$code"
    grep -Fxq exit_code=0 "$dst/status.txt"
    test "$(cat "$dst/execution.sha")" = "$sha"
    cp "$GC_MOE_FIXED_DIR/$arm-layout.json" "$dst/installed-layout.json"
    if [[ $arm == vanilla ]]; then
      mkdir "$group/checkpoint"
      mkdir "$group/checkpoint/prefill"
      cp "$GC_MOE_STRICT_STATE_DIR"/prefill/rank{0..7}.json "$group/checkpoint/prefill/"
      cp "$GC_MOE_STRICT_STATE_DIR"/rank{0..7}.json "$group/checkpoint/"
      cp "$GC_MOE_FIXED_DIR/stats-demand.npz" "$group/"
    fi
    if [[ $arm == rr ]]; then cp "$GC_MOE_FIXED_DIR/placement.json" "$group/"; fi
  done
  (cd "$GC_MOE_FIXED_DIR" && sha256sum stats-rank*.pt measure-rank*.pt stats-demand.npz tokens.json placement.json *-layout.json) > "$group/workload.sha256"
  "$py" scripts/trace_layer/main20_report.py "$out" "${report_args[@]}" --bs "$bs"
  if [[ ${GC_STEP_BOUNDARY:-0} == 1 ]]; then
    "$py" scripts/trace_layer/cost_boundary_report.py "$group" > "$group/cost_boundary.csv"
  fi
  # The group report has verified every arm's readback against its publisher.
  # Only release this job's private disposable payloads; preserve evidence on failure.
  if [[ ${GC_MAIN20_KEEP_KV:-0} != 1 ]]; then
    rm -- "$GC_MOE_STRICT_STATE_DIR"/{prefill/,}rank{0..7}.bin
  fi
done
if [[ -z ${GC_MAIN24_SINGLE_BS:-} ]]; then
  "$py" scripts/trace_layer/main20_report.py "$out" "${report_args[@]}"
fi
(cd "$out" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
