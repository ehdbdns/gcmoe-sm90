#!/usr/bin/env bash
# Preflight, then six arms per BS on the same clean measurement revision.
set -eo pipefail
[[ $# == 1 ]] || { echo "Expected the task workspace directory" >&2; exit 2; }
work=$(realpath "$1")
cd "$(git rev-parse --show-toplevel)"
test -z "$(git status --porcelain)"
test ! -e "$work/preflight" && test ! -e "$work/delivery" || exit 2
export GC_MOE_MATRIX_SIZE=24
bash scripts/trace_layer/preflight_main20.sh "$work/preflight" 2>&1 | tee "$work/preflight.log"
bash scripts/trace_layer/run_main20.sh "$work/delivery" 2>&1 | tee "$work/progress.log"
(cd "$work/delivery" && sha256sum --quiet -c SHA256SUMS)
