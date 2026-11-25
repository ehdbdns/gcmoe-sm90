#!/usr/bin/env bash
# Prefer four redundant slots. Reduce replication before lowering the common KV tier.
set -euo pipefail
case $# in
  1) job=$(realpath -m "$1") ;;
  2) job=$(realpath -m "$2") ;; # Former SCAN.csv JOB invocation.
  *) echo "usage: run_tuned_eight.sh JOB" >&2; exit 2 ;;
esac
mkdir -p "$job"
exec 7>"$job/eight.lock"
flock -n 7 || { echo "job already running" >&2; exit 1; }
cd "$(git rev-parse --show-toplevel)"
source scripts/env.sh
sha=$(git rev-parse HEAD)
export SM90_TUNED_EXTRA=4
for kv in 10 7 5; do
  out="$job/kv-$kv"
  export SM90_TUNED_KV=$kv SM90_SIX_STEPS=1000
  if [[ $kv == 5 ]]; then export SM90_SIX_STEPS=500; fi
  while true; do
    args=()
    if [[ -d $out ]]; then
      test "$(cat "$out/measurement.sha")" = "$sha"
      test "$(cat "$out/tuned-kv")" = "$kv"
      if [[ -f "$out/failed-k4/capacity-failed-arm" ]]; then
        # Complete a possibly interrupted transition before resuming any arm.
        if [[ $(cat "$out/tuned-extra") == 4 ]]; then
          "$GC_ENV/bin/python" scripts/sm90/tuned_eight.py reduce "$out"
        fi
        export SM90_TUNED_EXTRA=1
      fi
      test "$(cat "$out/tuned-extra")" = "$SM90_TUNED_EXTRA"
      args+=(--resume)
    fi
    if [[ ! -f "$out/capacity-failed-arm" ]]; then
      code=0
      bash scripts/sm90/run_eight.sh --tuned-eight "${args[@]}" "$out" >> "$job/kv-$kv-k$SM90_TUNED_EXTRA.log" 2>&1 || code=$?
      echo "$code" > "$job/kv-$kv-k$SM90_TUNED_EXTRA.exit"
      if [[ $code != 75 || ! -f "$out/capacity-failed-arm" ]]; then exit "$code"; fi
    fi
    arm=$(cat "$out/capacity-failed-arm")
    [[ $arm =~ ^[0-8]$ ]]
    test "$(cat "$out/arm$arm/exit")" -ne 0
    if [[ $SM90_TUNED_EXTRA == 4 && $arm -ge 4 ]]; then
      "$GC_ENV/bin/python" scripts/sm90/tuned_eight.py reduce "$out"
      export SM90_TUNED_EXTRA=1
      continue
    fi
    # A publisher/native failure is independent of replication. A one-slot
    # failure already used the smaller budget. Retry at the next common tier.
    break
  done
done
exit 75
