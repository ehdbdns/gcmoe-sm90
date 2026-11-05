#!/bin/bash
# ============================================================================
# Run one GC MoE arm (vanilla or EPLB) with this experiment's pinned sources.
#
#   ./scripts/run_arm.sh <epbl:0|1> <arm-label> [--seqs-per-rank N ...]
#
# Everything the arm needs comes from THIS directory: the gc_moe source under
# source/, the vLLM backend under integration/, the patches under patches/.
# ============================================================================
set -u
ARM_EPLB=$1; ARM_LABEL=$2; shift 2

HERE=$(cd "$(dirname "$0")/.." && pwd)
source "$HERE/config/experiment.env"
source "$HERE/scripts/env.sh"

export GC_MOE_EPLB=$ARM_EPLB
# The EPLB arm re-plans once between the statistics window and the measured
# window unless explicitly disabled: warm = statistics, measure = timed.
if [ "${GC_MOE_EPLB_REARRANGE:-1}" = "1" ]; then
    export GC_MOE_EPLB_NO_REARRANGE=0
else
    export GC_MOE_EPLB_NO_REARRANGE=1
fi
export GC_MOE_SRC=$HERE/source
export PYTHONPATH=$HERE/source:$HERE/patches:$HERE/integration:${PYTHONPATH:-}

LOG=$HERE/logs/real_${ARM_LABEL}.log
OUT=$HERE/runs/${ARM_LABEL}.json
mkdir -p "$HERE/logs" "$HERE/runs"
echo "[run] $ARM_LABEL start $(date +%H:%M:%S) eplb=$GC_MOE_EPLB" > "$LOG"

"$GC_ENV/bin/python" -u "$HERE/scripts/real_decode_100_100.py" \
    --label "$ARM_LABEL" --output "$OUT" \
    --prefill-lo "$PREFILL_LO" --prefill-hi "$PREFILL_HI" \
    --seqs-per-rank "${SEQS_PER_RANK:-4}" \
    --warm-steps "${WARM_STEPS:-5}" --measure-steps "${MEASURE_STEPS:-10}" \
    "$@" >> "$LOG" 2>&1
echo "[run] exit=$? $(date +%H:%M:%S)" >> "$LOG"
