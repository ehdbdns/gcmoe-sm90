#!/bin/bash
# ============================================================================
# [IR-STEP] Four-arm matrix with the three independent switches.
#
#   ./scripts/run_matrix_ir.sh <label-prefix> <seqs-per-rank> [extra env...]
#
# Arms (all share the same placement/redundancy unless overridden):
#   vanilla       : no EPLB, no ECO, no SM-reuse
#   random        : random placement + RR slot choice
#   eplb+rr       : EPLB placement + RR slot choice
#   eplb+eco      : EPLB placement + in-kernel plan
#   eplb+eco+smr  : the above + role 6 (dispatch SM handed to compute)
#
# The switches are independent:
#   GC_MOE_EPLB=1                     placement (with GC_MOE_SLOT_PLACEMENT=random -> random)
#   GC_MOE_ECO=1                      in-kernel plan (needs EPLB placement)
#   GC_MOE_SM90_SM_REUSE=1            role 6 helper compute
# ============================================================================
set -u
PREFIX=$1; SPR=$2
HERE=$(cd "$(dirname "$0")/.." && pwd)
cd "$HERE" || exit 1
L=${LOG:-/tmp/ir_matrix_$PREFIX.log}

COMMON="SEQS_PER_RANK=$SPR WARM_STEPS=${WARM_STEPS:-100} MEASURE_STEPS=${MEASURE_STEPS:-100}
        GC_MOE_EPLB_REDUNDANCY=${GC_MOE_EPLB_REDUNDANCY:-0.10}
        GC_MOE_EPLB_EXTRA_PER_RANK=${GC_MOE_EPLB_EXTRA_PER_RANK:-8}
        GC_MOE_ECO_WB=${GC_MOE_ECO_WB:-260} GC_MOE_ECO_WA=${GC_MOE_ECO_WA:-111}
        GC_MOE_ECO_MAX_COPIES=${GC_MOE_ECO_MAX_COPIES:-8}
        GC_MEASURE_IR_STEP=${GC_MEASURE_IR_STEP:-1}
        GC_IRSTEP_STEPS=${GC_IRSTEP_STEPS:-${MEASURE_STEPS:-100}}"

run () {   # $1=suffix $2=eplb $3=eco, rest = extra env assignments
  local suf=$1 eplb=$2 eco=$3; shift 3
  local lab="${PREFIX}_${suf}"
  for i in $(seq 1 40); do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader|awk '{print $1}'|sort -n|tail -1)
    [ "${u:-99999}" -lt 1000 ] && break; sleep 5
  done
  echo "START $lab max_used=${u:-?} $(date +%H:%M:%S)" >> "$L"
  # `run_arm_eco.sh` sets GC_MOE_EPLB/GC_MOE_ECO from its args, so the arm
  # identity comes from argv; everything else (placement kind, SM-reuse,
  # redundancy, IR sampling) comes from the environment that prefixes it.
  env $COMMON "$@" ./scripts/run_arm_eco.sh "$eplb" "$eco" "$lab" >> "$L" 2>&1
  echo "DONE $lab rc=$? $(tr '\r' '\n' < logs/real_$lab.log 2>/dev/null | grep -oE '\[ir-step\].*' | tail -1) $(date +%H:%M:%S)" >> "$L"
}

run vanilla  0 0 GC_MOE_SM90_SM_REUSE=0
run random   1 0 GC_MOE_SM90_SM_REUSE=0 GC_MOE_SLOT_PLACEMENT=random
run rr       1 0 GC_MOE_SM90_SM_REUSE=0
run eco      1 1 GC_MOE_SM90_SM_REUSE=0
run eco_smr  1 1 GC_MOE_SM90_SM_REUSE=1
echo "ALL DONE $(date +%H:%M:%S)" >> "$L"
