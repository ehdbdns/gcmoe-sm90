#!/usr/bin/env bash
# Run from the measurement checkout with two clean, pinned kernel worktrees.
set -eo pipefail
if [[ $# != 3 ]]; then echo "Usage: $0 baseline-worktree stream-worktree new-output-dir" >&2; exit 2; fi
here=$(cd "$(dirname "$0")" && pwd)
baseline=$(realpath "$1")
stream=$(realpath "$2")
bundle=$(realpath -m "$3")
mkdir "$bundle"
git -C "$here" rev-parse HEAD > "$bundle/measurement.sha"
base_sha=98b6194a56e7468653bb6230bf0b3887c9466dce
stream_sha=${GC_MOE_STUDY_STREAM_SHA:-6617ee976fa86eb77eb3a34b8b6df3e60059a586}
analysis_python=$(cd "$baseline" && source scripts/env.sh && printf '%s/bin/python' "$GC_ENV")
test -x "$analysis_python"
export GC_CAPTURE_TOKENS=1
for step in 2 5 8 11; do
  arms=(baseline stream)
  if [[ $step == 5 || $step == 8 ]]; then arms=(stream baseline); fi
  for mode in e2e coarse fine; do
    for arm in "${arms[@]}"; do
      root=$baseline; sha=$base_sha; flag=0
      if [[ $arm == stream ]]; then root=$stream; sha=$stream_sha; flag=1; fi
      printf -v tag 'step%02d' "$step"
      out="$bundle/$tag/$arm/$mode"
      mkdir -p "$out"
      # The runner keeps full logs (including generated tokens) with the raw trace.
      echo "$tag $arm $mode"
      GC_MOE_EXEC_ROOT="$root" GC_MOE_STUDY_STEP="$step" GC_MOE_SM90_STREAM_M_BLOCKS="$flag" \
        bash "$here/run_aligned_ep8.sh" "$sha" "--study-$mode" > "$out/runner.log" 2>&1
      raw=$(sed -n 's/^Capture workspace: //p' "$out/runner.log")
      grep -Fxq exit_code=0 "$raw/status.txt"
      test "$(cat "$raw/execution.sha")" = "$sha"
      "$analysis_python" "$here/multilayer_study.py" collect "$raw" "$out" "$mode" "$step"
      # Keep large per-run logs on the machine, with a stable path in the bundle.
      mv "$out/runner.log" "$raw/study-runner.log"
      cp "$raw"/{execution.sha,status.txt,gpu.log,runtime.log} "$out/"
      sha256sum "$raw/result.json" > "$out/result.sha256"
      if [[ -f "$raw/sha256.txt" ]]; then cp "$raw/sha256.txt" "$out/raw-sha256.txt"; fi
    done
  done
done
"$analysis_python" "$here/multilayer_study.py" compare "$bundle"
(cd "$bundle" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
