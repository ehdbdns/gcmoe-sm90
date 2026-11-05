#!/bin/bash
# ============================================================================
# [ECO-SM90] Run one arm with the SM90 ECO plan.
#
#   ./scripts/run_arm_eco.sh <epbl:0|1> <eco:0|1> <arm-label> [extra args]
#
#   (0 0) vanilla            : no replication, no plan
#   (1 0) eplb-rr            : EPLB placement + host round-robin slot choice
#   (1 1) eplb-eco           : EPLB placement + in-kernel plan (role 5)
#
# Everything comes from THIS directory: source/, integration/, patches/.
# ============================================================================
set -u
ARM_EPLB=$1; ARM_ECO=$2; ARM_LABEL=$3; shift 3

HERE=$(cd "$(dirname "$0")/.." && pwd)
source "$HERE/config/experiment.env"
source "$HERE/scripts/env.sh"

export GC_MOE_EPLB=$ARM_EPLB
export GC_MOE_ECO=$ARM_ECO
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
echo "[run] $ARM_LABEL start $(date +%H:%M:%S) eplb=$ARM_EPLB eco=$ARM_ECO" > "$LOG"

"$GC_ENV/bin/python" -u "$HERE/scripts/real_decode_100_100.py" \
    --label "$ARM_LABEL" --output "$OUT" \
    --prefill-lo "$PREFILL_LO" --prefill-hi "$PREFILL_HI" \
    --seqs-per-rank "${SEQS_PER_RANK:-4}" \
    --warm-steps "${WARM_STEPS:-5}" --measure-steps "${MEASURE_STEPS:-10}" \
    "$@" >> "$LOG" 2>&1
echo "[run] exit=$? $(date +%H:%M:%S)" >> "$LOG"
