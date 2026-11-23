#!/usr/bin/env bash
# Run inside the site's eight-H20 allocation. Each invocation owns a new output tree.
set -eo pipefail
cd "$(git rev-parse --show-toplevel)"
sha=$(git rev-parse HEAD)
test -z "$(git status --porcelain)"
out=$(realpath -m "${1:?expected new private output directory}")
mkdir "$out"
echo "$sha" > "$out/measurement.sha"
persist=/home/admin/workspace/aop_lab/app_source/sm90_persist_20261123
export SM90_DG_SITE="$persist/dg-sm90-build-ohe59RVP/site"
wheels="$persist/dg-sm90-build-ohe59RVP/wheels"
export MODEL=/mnt/dsv41/models/DeepSeek-V4.1-Flash
export DATASET=/mnt/dsv41/nanjun.cp/MoE-Bench-Experiments/datasets/dsv4-flash_online_input_ids_26-10-08_1.jsonl
source scripts/env.sh
py="$GC_ENV/bin/python"
sha256sum -c docs/trace_executor_20261119/sm90_h20_bs256_mega_gc_eight_results/model_and_dataset.sha256 > "$out/input.sha256.log"
echo "7c848494dbd01daa77ea80765888073cb63b85d7f74054fbcead404598ddf61e  $wheels/deep_gemm-2.8.2+7292e7b-cp310-cp310-linux_x86_64.whl" | sha256sum -c -
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 "$py" scripts/sm90/check_mega_weights.py "$MODEL" > "$out/conversion.jsonl"
# A private-extension import failure only blocks the two routed Mega arms.
CUDA_VISIBLE_DEVICES='' "$py" integration/sm90_deepgemm.py > "$out/attention-import.log" 2>&1
mega_import=0
CUDA_VISIBLE_DEVICES='' "$py" integration/sm90_deepgemm.py mega > "$out/mega-import.log" 2>&1 || mega_import=$?
for nic in mlx5_{4,5,9,17,21,29}; do
  grep -qw ACTIVE "/sys/class/infiniband/$nic/ports/1/state"
done
export NCCL_IB_HCA=mlx5_4:1,mlx5_5:1,mlx5_9:1,mlx5_17:1,mlx5_21:1,mlx5_29:1
export NVSHMEM_HCA_LIST="$NCCL_IB_HCA"
export GC_MOE_KV_BYTES=21474836480 GC_MOE_LONG_COST=1 GC_MOE_ORDER_EXTRA=1 GC_MOE_EXEC_ROOT="$PWD"
export GC_MOE_SM90_STREAM_M_BLOCKS=1 GC_MOE_STUDY_STEP=0 GC_MOE_MATRIX_SIZE=24 GC_MOE_MATRIX_BS=256
export GC_MOE_ORDER_PREFILL=1 GC_MOE_STABLE_PREFILL_TOPK=1 GC_MOE_STABLE_DECODE_TOPK=1
export GC_MOE_MATRIX_PREFIX_CHECK=1 GC_MOE_SM90_PLAN_DISPATCH=1 VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1 GC_CAPTURE_TOKENS=1
unset GC_MOE_FIXED_REUSE GC_MOE_FIRST_STEP_AUDIT GC_MOE_AA_CONTROL GC_MOE_COST_STUDY
export GC_MOE_FIXED_DIR="$out/replay" GC_MOE_STRICT_STATE_DIR="$out/kv"
mkdir "$GC_MOE_FIXED_DIR" "$GC_MOE_STRICT_STATE_DIR"
test "$(df -PB1 "$out" | awk 'END{print $4}')" -gt 405000000000
# Site scheduler/booking takes precedence over the local advisory lock.
exec 9>"$persist/h20-eight.lock"
flock 9
idle() {
  nvidia-smi --query-gpu=uuid,name --format=csv,noheader > "$out/gpus.csv" || return 1
  awk -F, '$2 !~ /NVIDIA H20/ {bad=1} END {exit (bad || NR!=8)}' "$out/gpus.csv" || return 1
  nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader > "$out/processes.csv" || return 1
  test ! -s "$out/processes.csv"
}
while ! idle; do flock -u 9; sleep 60; flock 9; done
selection=$(cut -d, -f1 "$out/gpus.csv" | paste -sd, -)
if [[ -z ${CUDA_VISIBLE_DEVICES:-} ]]; then export CUDA_VISIBLE_DEVICES="$selection"; fi
# Preserve the scheduler-assigned device mask. The runtime gate verifies eight SM90 devices.
for id in 2 3 4 6 7 8 1 5; do
  dst="$out/arm$id"
  mkdir "$dst"
  if [[ $id != 2 && ! -f "$out/arm2/complete" ]]; then echo publisher_failed > "$dst/skipped"; continue; fi
  if [[ $id == 7 || $id == 8 ]]; then
    if [[ ! -s "$GC_MOE_FIXED_DIR/placement.json" ]]; then echo missing_rr_placement > "$dst/skipped"; continue; fi
  fi
  if [[ ($id == 1 || $id == 5) && $mega_import != 0 ]]; then
    echo private_extension_import_failed > "$dst/skipped"
    continue
  fi
  case $id in
    1|2|3) export GC_MOE_MATRIX_ARM=vanilla ;;
    4|5|6) export GC_MOE_MATRIX_ARM=rr ;;
    7|8) export GC_MOE_MATRIX_ARM=eco ;;
  esac
  case $id in 1|5) export SM90_EIGHT_BACKEND=mega ;; *) export SM90_EIGHT_BACKEND=gc ;; esac
  case $id in 2|4|7) export SM90_EIGHT_SHARED=1 ;; *) export SM90_EIGHT_SHARED=0 ;; esac
  export SM90_EIGHT_ARM=$id DG_PRINT_CONFIGS=1
  while ! idle; do flock -u 9; sleep 60; flock 9; done
  code=0
  bash scripts/trace_layer/run_aligned_ep8.sh "$sha" --study-main20 > "$dst/runner.log" 2>&1 || code=$?
  echo "$code" > "$dst/exit"
  raw=$(sed -n 's/^Capture workspace: //p' "$dst/runner.log" | head -1)
  if [[ -n $raw && -d $raw ]]; then
    echo "$raw" > "$dst/raw-location"
    for file in run.json rank{0..7}.npz runtime.log status.txt execution.sha failed-command.txt; do
      if [[ -f "$raw/$file" ]]; then cp "$raw/$file" "$dst/"; fi
    done
  fi
  if [[ $code == 0 ]]; then
    test "$(cat "$dst/execution.sha")" = "$sha"
    cp "$GC_MOE_FIXED_DIR/$id-layout.json" "$dst/installed-layout.json"
    touch "$dst/complete"
  fi
done
flock -u 9
"$py" scripts/sm90/report_eight.py "$out"
