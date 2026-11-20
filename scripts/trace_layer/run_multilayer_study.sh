#!/usr/bin/env bash
# Run from the measurement checkout with two clean, pinned kernel worktrees.
set -eo pipefail
if [[ $# != 3 && ( $# != 4 || $4 != --valid-cost ) ]]; then
  echo "Usage: $0 baseline-worktree stream-worktree new-output-dir [--valid-cost]" >&2; exit 2
fi
here=$(cd "$(dirname "$0")" && pwd)
baseline=$(realpath "$1")
stream=$(realpath "$2")
bundle=$(realpath -m "$3")
base_sha=98b6194a56e7468653bb6230bf0b3887c9466dce
stream_sha=6617ee976fa86eb77eb3a34b8b6df3e60059a586
analysis_python=$(cd "$baseline" && source scripts/env.sh && printf '%s/bin/python' "$GC_ENV")
test -x "$analysis_python"
if [[ ${4:-} == --valid-cost ]]; then
  phase_one="$here/../../docs/trace_executor_20261119/multilayer_ab_results/comparison.json"
  "$analysis_python" - "$phase_one" <<'CHECK'
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
if not path.is_file() or json.loads(path.read_text()).get('scheduling_gate_passed') is not True:
    sys.exit('Valid-cost study is on hold pending a successful scheduling comparison.')
CHECK
fi
mkdir "$bundle"
git -C "$here" rev-parse HEAD > "$bundle/measurement.sha"
if [[ ${4:-} == --valid-cost ]]; then
  base_sha=$(git -C "$here" rev-parse HEAD)
  stream_sha=$base_sha
  echo valid_cost > "$bundle/comparison.txt"
fi
export GC_CAPTURE_TOKENS=1
for arm in baseline stream; do
  root=$baseline
  sha=$base_sha
  flag=0
  cost=0
  if [[ $arm == stream ]]; then
    root=$stream
    sha=$stream_sha
    flag=1
  fi
  if [[ ${4:-} == --valid-cost ]]; then
    flag=1
    if [[ $arm == stream ]]; then cost=2; fi
  fi
  out="$bundle/step02/$arm/paired"
  mkdir -p "$out"
  echo "step02 $arm paired"
  GC_MOE_STUDY_COSTFN="$cost" GC_MOE_EXEC_ROOT="$root" GC_MOE_STUDY_STEP=2 GC_MOE_SM90_STREAM_M_BLOCKS="$flag" \
    bash "$here/run_aligned_ep8.sh" "$sha" --study-paired > "$out/runner.log" 2>&1
  raw=$(sed -n 's/^Capture workspace: //p' "$out/runner.log")
  grep -Fxq exit_code=0 "$raw/status.txt"
  test "$(cat "$raw/execution.sha")" = "$sha"
  PYTHONPATH="$root/source" "$analysis_python" "$here/multilayer_study.py" collect "$raw" "$out" paired 2
  mv "$out/runner.log" "$raw/study-runner.log"
  cp "$raw"/{execution.sha,status.txt,gpu.log,runtime.log} "$out/"
  sha256sum "$raw/result.json" > "$out/result.sha256"
  cp "$raw/sha256.txt" "$out/raw-sha256.txt"
done
"$analysis_python" "$here/multilayer_study.py" compare "$bundle"
(cd "$bundle" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
