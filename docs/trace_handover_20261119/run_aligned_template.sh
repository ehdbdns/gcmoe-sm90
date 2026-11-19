#!/bin/bash
# Template for a run where the three windows can overlap.
#   trace        : profiled over the measured steps
#   ir_step      : 1024-step capacity, keyed by the KERNEL-WRITTEN epoch
#   pull_tl ring : last 256 layer calls (= 6.4 steps at 40 layers)
#
# IMPORTANT: GC_MOE_MEASURE_FLUSH=0 is what keeps the ring from being filled by
# the post-rearrange flush steps. Without it the ring only holds flush+measure
# steps and the trace (which starts after the flush) does not overlap it.
#
# KNOWN ISSUE (see HANDOVER.md 7.1): the profiler currently captures only 2
# steps regardless of --measure-steps. Verify call counts before trusting a run.
set -u
HERE=/home/admin/workspace/aop_lab/app_source/gcmoe_exp/t19_costfn
cd "$HERE" || exit 1
source "$HERE/config/experiment.env"
source "$HERE/scripts/env.sh"
export PYTHONPATH=$HERE/source:$HERE/patches:$HERE/integration
export MODEL=/home/admin/local_models/DeepSeek-V4.1-Flash
export DATASET=/mnt/nas2/nanjun.cp/MoE-Bench-Experiments/datasets/dsv4-flash_online_input_ids_26-10-08_1.jsonl

export GC_MOE_EPLB=1 GC_MOE_ECO=1
export GC_MOE_EPLB_EXTRA_PER_RANK=8 GC_MOE_ECOPLB_PER_LAYER=1
export GC_MOE_ECO_WB=26 GC_MOE_ECO_WA=0 GC_MOE_ECO_MAX_COPIES=8
export GC_MOE_MEASURE_FLUSH=0
export GC_MOE_MB_CLAIM=1 GC_MOE_SM90_PULL_TMA=1
export GC_MOE_ECO_TRACE=1 GC_MOE_PULL_TRACE=1
export GC_MOE_FINE_TRACE=1 GC_MOE_FINE_EPOCHS=2 GC_MOE_FINE_RECS=262144
export GC_MOE_IR_STEP=1 GC_MEASURE_IR_STEP=1
export GC_IRSTEP_STEPS=1024
export GC_MOE_SHARED_IN_GC=0
export ENFORCE_EAGER=0 GPU_UTIL=0.72 MAX_BATCHED=2048

# 4 measured steps = 160 layer calls < 256 ring slots
"$GC_ENV/bin/python" -u "$HERE/scripts/real_decode_100_100.py" \
  --label ALIGNEDQ --output "$HERE/runs/ALIGNEDQ.json" \
  --prefill-lo 256 --prefill-hi 2048 --seqs-per-rank 32 \
  --warm-steps 200 --measure-steps 4 \
  --profile-out "$HERE/prof/alignedq" \
  > "$HERE/logs/alignedq.log" 2>&1
echo "ALIGNEDQ done $?" >> /tmp/rt/alignedq_progress.log
