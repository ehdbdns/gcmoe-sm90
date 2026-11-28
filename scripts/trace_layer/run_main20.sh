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
four=${SM90_FULL_MODEL_FOUR:-0}
wa_pair=${SM90_LINEAR_WA_PAIR:-0}
report_args=()
case ${GC_MOE_MATRIX_SIZE:-20} in
  20) ;;
  24)
    if [[ $wa_pair == 1 ]]; then
      [[ $four == 0 && ${GC_MAIN24_SINGLE_BS:-} == 256 && ${GC_STEP_BOUNDARY:-0} == 0 && ${GC_COST_HOST_PROBE:-0} == 0 ]] || exit 2
      export GC_MOE_SM90_PLAN_DISPATCH=1
    elif [[ $four == 1 ]]; then
      if [[ ${GC_MOE_EP8_LINEAR:-0} == 1 ]]; then
        if [[ -n ${GC_MOE_SOURCE_BATCH:-} ]]; then
          [[ $GC_MOE_SOURCE_BATCH =~ ^(8|16|32|64|128|256)$ && $((8*GC_MOE_SOURCE_BATCH)) == "$GC_MAIN24_SINGLE_BS" ]] || exit 2
        else
          [[ ${GC_MAIN24_SINGLE_BS:-} =~ ^(32|64|128|256|512|1024)$ ]] || exit 2
        fi
        report_args+=(--linear-ep8)
      else
        [[ ${GC_MAIN24_SINGLE_BS:-} =~ ^(512|1024)$ ]] || exit 2
      fi
      [[ ${GC_STEP_BOUNDARY:-0} == 0 && ${GC_COST_HOST_PROBE:-0} == 0 ]] || exit 2
      export GC_MOE_SM90_PLAN_DISPATCH=1
      report_args+=(--four-arms)
    elif [[ ${GC_MOE_MAIN_FINE_ONLY:-0} == 1 ]]; then
      [[ ${GC_MAIN24_SINGLE_BS:-} == 256 && ${GC_STEP_BOUNDARY:-0} == 0 && ${GC_COST_HOST_PROBE:-0} == 0 ]] || exit 2
      export GC_MOE_SM90_PLAN_DISPATCH=1
      report_args+=(--fine-only)
    else
      arms+=(eco_linear); report_args+=(--with-linear)
    fi ;;
  *) exit 2 ;;
esac
arms+=(eco)
if [[ $wa_pair == 1 ]]; then arms=(vanilla eco_linear eco_linear_wa1); fi
if [[ $four == 1 || $wa_pair == 1 ]]; then
  [[ ${GC_MOE_MATRIX_SIZE:-20} == 24 ]] || exit 2
  exec 9>/home/admin/workspace/aop_lab/app_source/sm90_persist_20261123/h20-eight.lock
else
  arms+=(eco_smr)
fi
if [[ ${GC_STEP_BOUNDARY:-0} == 1 ]]; then
  test "${GC_MOE_MATRIX_SIZE:-}" = 24
  # Counterbalance the cost-arm order from BS256/512. Vanilla still publishes.
  arms=(vanilla random rr eco eco_linear eco_smr)
fi
batches=(32 64 128 256)
if [[ -n ${GC_MAIN24_SINGLE_BS:-} ]]; then
  test "${GC_MOE_MATRIX_SIZE:-20}" = 24
  case "$GC_MAIN24_SINGLE_BS" in 32|64|128) test "${GC_MOE_EP8_LINEAR:-0}" = 1 ;; 256|512|1024|2048) ;; *) exit 2 ;; esac
  batches=("$GC_MAIN24_SINGLE_BS")
fi
if [[ ${GC_COST_HOST_PROBE:-0} == 1 ]]; then
  test "${GC_STEP_BOUNDARY:-}" = 1 && test "${GC_MAIN24_SINGLE_BS:-}" = 1024 || exit 2
  arms=(vanilla eco_linear eco)
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
required = 2*8*int(os.environ["GC_MOE_KV_BYTES"]) + 2*8*40000*int(os.getenv("GC_MOE_SOURCE_BATCH") or sys.argv[2])*6*12 + 4*1024**3
if shutil.disk_usage(sys.argv[1]).free < required:
    raise SystemExit(f"Need {required / 1024**3:.1f} free GiB on the checkpoint filesystem")
PY
  if [[ ${GC_COST_HOST_PROBE:-0} == 1 ]]; then
    ref=docs/trace_executor_20261119/sm90_main24_stable_decode_bs1024_results/delivery/bs1024
    cp "$ref/placement.json" "$GC_MOE_FIXED_DIR/placement.json"
  fi
  if [[ ${SM90_IR_REFERENCE:-0} == 1 ]]; then [[ $four == 1 && ${GC_MOE_EP8_LINEAR:-0} == 1 ]] || exit 2; arms=(vanilla rr); fi
  if [[ ${SM90_AUCTION_PAIR:-0} == 1 ]]; then
    [[ $four == 1 && ${GC_MOE_EP8_LINEAR:-0} == 1 && $bs =~ ^(256|512|1024)$ ]] || exit 2
    arms=(eco eco_small)
    "$py" scripts/sm90/auction_pair.py stage "$GC_MOE_FIXED_DIR" "$group" --bs "$bs"
    export SM90_AUCTION_REPLAY
    SM90_AUCTION_REPLAY=$(cat "$group/reference-mode.txt")
    export GC_MOE_SM90_SPLIT_SHARED=1 GC_MOE_SM90_SPLIT_SHARED_M=32
  fi
  for arm in "${arms[@]}"; do
    export GC_MOE_MATRIX_ARM="$arm"
    if [[ ${SM90_AUCTION_PAIR:-0} == 1 ]]; then
      export GC_MOE_MATRIX_ARM=eco GC_MOE_ECO_FINE_AUCTION=0
      if [[ $arm == eco_small ]]; then export GC_MOE_ECO_FINE_AUCTION=1; fi
    fi
    dst="$group/$arm"
    mkdir "$dst"
    log="$dst/runner.log"
    if [[ $four == 1 || $wa_pair == 1 ]]; then
      while true; do
        flock 9
        nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name --format=csv,noheader > "$dst/preexisting-processes.csv"
        if [[ ! -s "$dst/preexisting-processes.csv" ]]; then break; fi
        flock -u 9
        sleep 60
      done
    else
      nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name --format=csv,noheader > "$dst/preexisting-processes.csv"
    fi
    test ! -s "$dst/preexisting-processes.csv"
    nvidia-smi --query-gpu=index,uuid,name,pci.bus_id,driver_version,power.limit,clocks.max.sm,clocks.max.memory --format=csv,noheader > "$dst/hardware.csv"
    awk -F, '$3 !~ /NVIDIA H20/ { bad=1 } END { exit (NR != 8 || bad) }' "$dst/hardware.csv"
    echo "Starting BS=$bs arm=$arm"
    if bash scripts/trace_layer/run_aligned_ep8.sh "$sha" --study-main20 > "$log" 2>&1; then code=0; else code=$?; fi
    if [[ $four == 1 || $wa_pair == 1 ]]; then flock -u 9; fi
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
    if [[ $arm == vanilla || ( ${SM90_AUCTION_PAIR:-0} == 1 && $arm == eco ) ]]; then
      if [[ $arm == eco ]]; then cp "$GC_MOE_FIXED_DIR/placement.json" "$group/"; fi
      mkdir "$group/checkpoint"
      mkdir "$group/checkpoint/prefill"
      cp "$GC_MOE_STRICT_STATE_DIR"/prefill/rank{0..7}.json "$group/checkpoint/prefill/"
      cp "$GC_MOE_STRICT_STATE_DIR"/rank{0..7}.json "$group/checkpoint/"
      cp "$GC_MOE_FIXED_DIR/stats-demand.npz" "$group/"
    fi
    if [[ $arm == rr || ( $wa_pair == 1 && $arm == eco_linear ) ]]; then cp "$GC_MOE_FIXED_DIR/placement.json" "$group/"; fi
  done
  (cd "$GC_MOE_FIXED_DIR" && sha256sum stats-rank*.pt measure-rank*.pt stats-demand.npz tokens.json placement.json *-layout.json) > "$group/workload.sha256"
  if [[ ${SM90_IR_REFERENCE:-0} == 1 ]]; then
    test -s "$group/rr/installed-layout.json"
  elif [[ ${SM90_AUCTION_PAIR:-0} == 1 ]]; then
    "$py" scripts/sm90/auction_pair.py report "$out" --bs "$bs"
  elif [[ $wa_pair == 1 ]]; then
    "$py" scripts/trace_layer/linear_wa_report.py "$out"
  elif [[ ${GC_COST_HOST_PROBE:-0} == 1 ]]; then
    "$py" scripts/trace_layer/cost_host_report.py "$group" "$ref"
  else
    "$py" scripts/trace_layer/main20_report.py "$out" "${report_args[@]}" --bs "$bs"
  fi
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
