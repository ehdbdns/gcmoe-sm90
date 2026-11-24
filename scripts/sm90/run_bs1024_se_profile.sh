#!/usr/bin/env bash
set -eo pipefail
cd "$(git rev-parse --show-toplevel)"
out=$(realpath -m "${1:?expected a new output directory}")
arms=(rr eco)
profile_options=()
export_options=()
if [[ $# == 3 ]]; then
  bs=$2
  group=$(realpath "$3")
  [[ $bs == 512 || $bs == 1024 ]] && [[ -f "$group/stats-demand.npz" ]] || exit 2
  arms=(vanilla random rr eco eco_smr)
  profile_options=(--bs "$bs" --group "$group" --layers 8 20 32)
  export_options=(--bs "$bs" --matrix)
elif [[ $# != 1 ]]; then
  exit 2
fi
mkdir "$out"
git rev-parse HEAD > "$out/execution.sha"
source scripts/env.sh
export PYTHONPATH="$PWD/source:$PWD/source/tests:$PWD/integration:$GC_SHIM"
export OMP_NUM_THREADS=2 GC_MOE_LAYERS=1 GC_MOE_MB_CLAIM=1 GC_MOE_BLOCK_M_FORCE=64
export GC_MOE_SM90_PULL_TMA=1 GC_MOE_SM90_STREAM_M_BLOCKS=1 GC_MOE_SM90_DISPATCH_SMS=6 GC_MOE_SM90_PLAN_DISPATCH=1
export GC_MOE_SM90_SM_REUSE=0 GC_MOE_ECO_PER_LAYER=1 GC_MOE_ECO_MAX_COPIES=8
export GC_MOE_ECO_WB=26 GC_MOE_ECO_WA=0 GC_MOE_ECO_COSTFN=2 GC_MOE_ECO_ROUNDS=1 GC_MOE_ECO_COLD_ROUNDS=6
export GC_MOE_ECO_NOSPLIT=0 GC_MOE_ECO_BLOCKS_ONLY=0 GC_MOE_ECO_TOKENS_ONLY=0 GC_MOE_ECO_NULL_PLAN=0 GC_MOE_ECO_NO_AUCTION=0
export GC_MOE_SM90_SOURCE_INBOX=0 GC_MOE_SM90_SINGLE_PUSH=0 GC_MOE_SM90_SOURCE_INDEX_PULL=0 GC_MOE_SM90_SPLIT_EXCHANGE=0
export GC_MOE_LONG_COST=1 GC_MOE_ECO_TRACE=1 GC_MOE_PULL_TRACE=1 GC_MOE_STEP_PROFILE=1
export GC_MOE_IR_STEP=0 GC_MOE_FINE_TRACE=0 GC_MOE_FIXED_DIR="$out" GC_MOE_EPLB_WINDOW=1000
persist=/home/admin/workspace/aop_lab/app_source/sm90_persist_20261123
exec 9>"$persist/h20-eight.lock"
idle() {
  nvidia-smi --query-gpu=uuid,name --format=csv,noheader > "$out/gpus.csv" || return 1
  awk -F, '$2 !~ /NVIDIA H20/ {bad=1} END {exit (bad || NR!=8)}' "$out/gpus.csv" || return 1
  nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader > "$out/processes.csv" || return 1
  test ! -s "$out/processes.csv"
}
export NCCL_IB_HCA=mlx5_4:1,mlx5_5:1,mlx5_9:1,mlx5_17:1,mlx5_21:1,mlx5_29:1
for arm in "${arms[@]}"; do
  flock 9
  while ! idle; do flock -u 9; sleep 60; flock 9; done
  export GC_MOE_MATRIX_ARM=$arm GC_MOE_SM90_SM_REUSE=0
  if [[ $arm == eco_smr ]]; then export GC_MOE_SM90_SM_REUSE=1; fi
  "$GC_ENV/bin/torchrun" --standalone --nproc-per-node=8 source/benchmarks/benchmark_gc_moe_se.py \
    --sm90-profile --arm "$arm" --output "$out" "${profile_options[@]}" 2>&1 | tee "$out/$arm.log"
  flock -u 9
done
if [[ $# == 3 ]]; then
  for layer in 8 20 32; do
    "$GC_ENV/bin/python" scripts/sm90/se_perfetto.py "$out/layer$layer" "${export_options[@]}"
  done
  "$GC_ENV/bin/python" scripts/sm90/se_replay_report.py "$out"
else
  "$GC_ENV/bin/python" scripts/sm90/se_perfetto.py "$out"
fi
