#!/usr/bin/env bash
# Consume the BS256 ordered-RR scan and run all eight formal arms at one common KV tier.
set -euo pipefail
[[ $# == 2 ]] || { echo "usage: run_tuned_eight.sh BS256_SCAN.csv JOB" >&2; exit 2; }
scan=$(realpath -e "$1")
job=$(realpath -m "$2")
mkdir -p "$job"
exec 7>"$job/eight.lock"
flock -n 7 || { echo "job already running" >&2; exit 1; }
cd "$(git rev-parse --show-toplevel)"
source scripts/env.sh
SM90_TUNED_EXTRA=$("$GC_ENV/bin/python" scripts/sm90/tuned_eight.py select "$scan")
export SM90_TUNED_EXTRA
if [[ -f "$job/selection.csv" ]]; then cmp "$scan" "$job/selection.csv"; else cp "$scan" "$job/selection.csv"; fi
for kv in 10 7 5; do
  out="$job/kv-$kv"
  args=()
  export SM90_TUNED_KV=$kv SM90_SIX_STEPS=1000
  if [[ $kv == 5 ]]; then export SM90_SIX_STEPS=500; fi
  if [[ -d $out ]]; then
    test "$(cat "$out/measurement.sha")" = "$(git rev-parse HEAD)"
    test "$(cat "$out/tuned-extra")" = "$SM90_TUNED_EXTRA"
    args+=(--resume)
  fi
  if [[ ! -f "$out/capacity-failed-arm" ]]; then
    code=0
    bash scripts/sm90/run_eight.sh --tuned-eight "${args[@]}" "$out" >> "$job/kv-$kv.log" 2>&1 || code=$?
    echo "$code" > "$job/kv-$kv.exit"
    if [[ $code != 75 || ! -f "$out/capacity-failed-arm" ]]; then exit "$code"; fi
  fi
  arm=$(cat "$out/capacity-failed-arm")
  [[ $arm =~ ^[0-8]$ ]] && test "$(cat "$out/arm$arm/exit")" -ne 0
done
exit 75
