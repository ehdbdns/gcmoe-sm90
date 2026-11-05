#!/bin/bash
# ============================================================================
# Serial driver: three arms x four batch sizes, one after another.
#
#   ./scripts/run_matrix.sh [label-prefix]
#
# Every arm runs with GC_MOE_ECO_TRACE=1 so each rank's timeline is recorded
# (the dispatch-side rows exist for RR and random arms too -- see the layout's
# `trace` flag). Redundancy is fixed at 4 slots/rank (P=416) for all runs, so
# the only thing that changes between groups is the batch size.
# ============================================================================
set -u
PREFIX=${1:-M}
HERE=$(cd "$(dirname "$0")/.." && pwd)
cd "$HERE" || exit 1

run_one() {   # $1=label $2=eplb $3=eco $4=seqs_per_rank $5=extra_args
    local label=$1 eplb=$2 eco=$3 spr=$4 extra=$5
    echo "=== $label  eplb=$eplb eco=$eco seqs/rank=$spr  $(date +%H:%M:%S) ==="
    # shellcheck disable=SC2086
    env SEQS_PER_RANK=$spr WARM_STEPS=100 MEASURE_STEPS=100 \
        GC_MOE_EPLB_EXTRA_PER_RANK=4 GC_MOE_ECO_TRACE=1 \
        GC_MOE_ECO_WB=260 GC_MOE_ECO_WA=111 $extra \
        ./scripts/run_arm_eco.sh "$eplb" "$eco" "$label" >> "/tmp/${label}.driver" 2>&1
    local rc=$?
    echo "    -> rc=$rc  $(grep -oE '\[measure\] [0-9.]+ ms/step' "/tmp/${label}.driver" | tail -1)"
    return $rc
}

for group in 32:4 64:8 128:16; do
    bs=${group%%:*}; spr=${group##*:}
    run_one "${PREFIX}_${bs}_random" 0 0 "$spr" "GC_MOE_SLOT_PLACEMENT=random"
    run_one "${PREFIX}_${bs}_rr"     1 0 "$spr" ""
    run_one "${PREFIX}_${bs}_eco"    1 1 "$spr" "GC_MOE_ECO=1"
done
echo "ALL DONE $(date +%H:%M:%S)"
