#!/usr/bin/env bash
# Run both windows only if the preferred one has a documented capacity failure.
set -euo pipefail
[[ $# == 1 ]] || { echo "usage: run_six_memory_retry.sh OUTPUT_ROOT" >&2; exit 2; }
job=$(realpath -m "$1")
mkdir -p "$job"
exec 7>"$job/six.lock"
flock -n 7 || { echo "six-arm run already active in $job" >&2; exit 1; }
cd "$(git rev-parse --show-toplevel)"
sha=$(git rev-parse HEAD)
for steps in 1000 500; do
  out="$job/window-$steps"
  args=()
  if [[ -d $out ]]; then
    test "$(cat "$out/measurement.sha")" = "$sha"
    test "$(cat "$out/six-steps")" = "$steps"
    args+=(--resume)
  fi
  if [[ ! -f "$out/capacity-failed-arm" ]]; then
    rc=0
    SM90_SIX_STEPS=$steps bash scripts/sm90/run_eight.sh --six-bs256 "${args[@]}" "$out" \
      >> "$job/window-$steps.log" 2>&1 || rc=$?
    echo "$rc" > "$job/window-$steps.exit"
    if [[ $rc != 75 || ! -f "$out/capacity-failed-arm" ]]; then exit "$rc"; fi
  fi
  # Also covers a restart between a recorded OOM and the next window.
  arm=$(cat "$out/capacity-failed-arm")
  [[ $arm =~ ^[0-6]$ ]]
  test "$(cat "$out/arm$arm/exit")" -ne 0
  echo "Capacity failure at $steps+$steps, arm $arm; evidence in $out/arm$arm/runner.log"
done
exit 75
