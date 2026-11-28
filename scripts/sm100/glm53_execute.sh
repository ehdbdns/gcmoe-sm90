#!/usr/bin/env bash
# One private directory per launch. Packaging and publishing can be retried separately.
set -euo pipefail
umask 077
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
job=$(realpath "${1:?pass the formal.XXXXXXXX path created during bootstrap}")
unset GLM_ECO_ONLY GLM_REFIT4 GLM_SLOT_BUDGETS GLM_ARM_LABEL
if [[ ${2:-} == --refit4 || ${2:-} == --slot-budgets ]]; then
  if [[ $2 == --slot-budgets ]]; then export GLM_SLOT_BUDGETS=1; fi
  [[ $# == 3 ]] || { echo "expected --refit4 previous-three-arm-results" >&2; exit 2; }
  export GLM_REFIT4=1
  GLM_PRICE_REFERENCE=$(realpath -e "$3")
  export GLM_PRICE_REFERENCE
  unset GLM_RESUME_FROM GLM_LAUNCH_REFERENCE
  set -- "$1"
fi
if [[ $# == 3 && ( $2 == --resume || $2 == --eco-only ) ]]; then
  GLM_RESUME_FROM=$(realpath -e "$3")
  export GLM_RESUME_FROM
  test -s "$GLM_RESUME_FROM/vanilla/result.json"
  if [[ $2 == --eco-only ]]; then
    test -s "$GLM_RESUME_FROM/rr/result.json"
    export GLM_ECO_ONLY=1
  fi
elif [[ $# == 1 ]]; then
  unset GLM_RESUME_FROM
else
  echo "usage: glm53_execute.sh NEW_JOB [--resume BASELINE_RESULTS | --eco-only COMPLETED_RR_RESULTS]" >&2; exit 2
fi
results="$job/results"
dest="$job/receipt/$(basename "$job")"
test -r "$repo/SOURCE_REVISION"
mkdir "$results"
if [[ ${GLM_SLOT_BUDGETS:-0} == 1 ]]; then echo '["vanilla","rr","eco8","eco12"]' > "$results/scheduled-arms.json"; fi
# Serialize this task even while preparing on CPU, including across login accounts.
lock=/tmp/gcmoe-glm53-bs256-execution.lock
if [[ ! -e $lock ]]; then (umask 022; set -C; : > "$lock") 2>/dev/null || test -e "$lock"; fi
exec 8<"$lock"
rc=0
(
  set -euo pipefail
  flock -n 8 || { echo "An existing GLM BS256 executor is active on this node."; exit 75; }
  active=$(pgrep -af '^([^ ]*/)?bash[[:space:]]+([^ ]*/)?run_glm53_three[.]sh([[:space:]]|$)' || true)
  if [[ -n $active ]]; then
    echo "An earlier formal launcher is active: $active"
    exit 75
  fi
  # Keep credentials out of the CUDA worker processes.
  unset GH_TOKEN GITHUB_TOKEN
  bash "$repo/scripts/sm100/run_glm53_three.sh" "$results"
) 2>&1 | tee "$results/run.log" || rc=$?
if [[ ! -f $results/revision ]]; then cp "$repo/SOURCE_REVISION" "$results/revision"; fi
python3 "$repo/scripts/sm100/glm53_receipt.py" "$results" "$dest" "$rc"
echo "Private run exit=$rc, receipt=$dest"
python3 "$repo/scripts/sm100/glm53_publish.py" "$dest"
exit "$rc"
