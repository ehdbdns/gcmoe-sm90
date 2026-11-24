#!/usr/bin/env bash
# Run from the clean execution worktree. Keep all outputs on a persistent job volume.
set -eo pipefail
[[ $# == 2 && ( $2 == 256 || $2 == 512 || $2 == 1024 ) ]] || exit 2
cd "$(git rev-parse --show-toplevel)"
root=$(realpath -m "$1")
bs=$2
mkdir -p "$root"
sha=$(git rev-parse HEAD)
if [[ -e "$root/execution.sha" ]]; then
  [[ $(cat "$root/execution.sha") == "$sha" ]] || exit 2
else
  echo "$sha" > "$root/execution.sha"
fi
# The restarted H20 host uses the six active adapters from the main-table runs.
export NCCL_IB_HCA=mlx5_4:1,mlx5_5:1,mlx5_9:1,mlx5_17:1,mlx5_21:1,mlx5_29:1
export NVSHMEM_HCA_LIST="$NCCL_IB_HCA"
group="$root/collection"
if [[ ! -f "$group/capture.json" ]]; then
  # Do not overwrite a partial collector. Preserve its first error for diagnosis.
  mkdir "$group"
  mkdir "$group/routes" "$group/vanilla" "$group/kv"
  (
    exec 9>/home/admin/workspace/aop_lab/app_source/sm90_persist_20261123/h20-eight.lock
    while true; do
      flock 9
      nvidia-smi --query-gpu=uuid,name --format=csv,noheader > "$group/gpus.csv"
      awk -F, '$2 !~ /NVIDIA H20/ {bad=1} END {exit (bad || NR!=8)}' "$group/gpus.csv"
      nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader > "$group/processes.csv"
      if [[ ! -s "$group/processes.csv" ]]; then break; fi
      flock -u 9
      sleep 60
    done
    export SM90_COLLECT_SE_ROUTES=1 GC_MOE_MATRIX_ARM=vanilla GC_MOE_MATRIX_SIZE=24 GC_MOE_MATRIX_BS="$bs"
    export GC_MOE_ORDER_PREFILL=1 GC_MOE_ORDER_EXTRA=1 GC_MOE_STABLE_PREFILL_TOPK=1 GC_MOE_STABLE_DECODE_TOPK=1
    export GC_MOE_MATRIX_PREFIX_CHECK=1 GC_MOE_STUDY_STEP=0 GC_MOE_LONG_COST=1 GC_CAPTURE_TOKENS=1
    export GC_MOE_FIXED_DIR="$group" GC_MOE_STRICT_STATE_DIR="$group/kv" GC_MOE_EXEC_ROOT="$PWD"
    export GC_MOE_KV_BYTES=21474836480 VLLM_USE_V2_MODEL_RUNNER=1 VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1
    export GC_MOE_SM90_STREAM_M_BLOCKS=1 GC_MOE_SM90_PLAN_DISPATCH=1
    unset GC_MOE_VANILLA_STATS_REPLAY GC_MOE_FIXED_REUSE GC_MOE_PREFIX_DIAG GC_MOE_DECODE_STATS_AA
    unset SM90_EIGHT_ARM SM90_EIGHT_BACKEND SM90_PAIRED_SMALL GC_BS256_TRACE_DIR GC_STEP_BOUNDARY GC_KV_OVERHEAD
    unset GC_MOE_SM90_SOURCE_INBOX GC_MOE_SM90_SINGLE_PUSH GC_MOE_SM90_SPLIT_EXCHANGE GC_MOE_SM90_SOURCE_INDEX_PULL
    export MODEL=/mnt/dsv41/models/DeepSeek-V4.1-Flash
    dataset_dir=/mnt/dsv41/nanjun.cp/MoE-Bench-Experiments/datasets
    first="$dataset_dir/dsv4-flash_online_input_ids_26-10-08_1.jsonl"
    second="$dataset_dir/dsv4-flash_online_input_ids_26-10-08_2.jsonl"
    export DATASET="$first,$second"
    sha256sum -c docs/trace_executor_20261119/sm90_h20_bs256_mega_gc_eight_results/model_and_dataset.sha256 > "$group/input-check.log"

    export GC_KV_BUDGET=2300000
    unset GC_REPEAT_BS512_PROMPT_SHA256
    if [[ $bs == 256 ]]; then
      # Match the single-file corpus and prompt selection of the BS256 full-model table.
      export DATASET="$first"
      unset GC_KV_BUDGET
      sha256sum "$first" > "$group/dataset.sha256"
    else
      sha256sum "$first" "$second" > "$group/dataset.sha256"
    fi
    if [[ $bs == 1024 ]]; then
      export GC_KV_BUDGET=5000000 GC_REPEAT_BS512_PROMPT_SHA256=e7c4aee9db91a34c018bd54464e67355681b74112d515401844f473c053cfcef
    fi
    bash scripts/trace_layer/run_aligned_ep8.sh "$sha" --study-main20
  ) 2>&1 | tee "$group/collector.log"
fi
# No GPU reservation is needed while preparing the compact immutable replay inputs.
(
  source scripts/env.sh
  export PYTHONPATH="$PWD/source:$PWD/integration:$GC_SHIM" OMP_NUM_THREADS=2
  "$GC_ENV/bin/python" scripts/sm90/prepare_se_capture.py "$group" "$bs"
)
bash scripts/sm90/run_bs1024_se_profile.sh "$root/replay" "$bs" "$group" --fresh
