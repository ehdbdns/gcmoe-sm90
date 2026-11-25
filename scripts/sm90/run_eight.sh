#!/usr/bin/env bash
# Run inside the site's eight-H20 allocation. Resume only on the same measurement revision.
set -eo pipefail
launcher_dir=$(cd "$(dirname "$0")" && pwd)
cd "$(git rev-parse --show-toplevel)"
sha=$(git rev-parse HEAD)
test -z "$(git status --porcelain)"
resume=0
mega_only=0
compact=0
six=0
unset SM90_PAIRED_SMALL SM90_SIX_ARM
if [[ ${1:-} == --six-bs256 ]]; then six=1; export SM90_SIX_ARM=1; shift; fi
if [[ ${1:-} == --paired-bs32 ]]; then compact=1; export SM90_PAIRED_SMALL=1; shift; fi
arms=(2 3 4 6 7 8 1 5)
publisher=2
if (( six )); then
  test "$compact" = 0
  publisher=0
  arms=(0 5 1 2 3 4 6)
fi
if (( compact )); then arms=(2 3 1 6 5); fi
if [[ ${1:-} == --resume-mega ]]; then
  [[ $compact == 0 && $six == 0 ]]
  resume=1; mega_only=1; arms=(1 5); shift
  reference=c1a176dbf45f1fd8d433189c1f24533aa44a7212
  test "$(git diff --name-only "$reference" "$sha")" = integration/sm90_mega_weights.py
  cmp integration/sm90_mega_weights.py "$launcher_dir/../../integration/sm90_mega_weights.py"
fi
if [[ ${1:-} == --resume ]]; then resume=1; shift; fi
[[ $# == 1 ]] || { echo "usage: run_eight.sh [--paired-bs32|--six-bs256] [--resume|--resume-mega] OUTPUT" >&2; exit 2; }
out=$(realpath -m "$1")
if (( resume )); then
  test "$six" = "$(cat "$out/six-mode" 2>/dev/null || echo 0)"
  test "$compact" = "$(cat "$out/compact-mode" 2>/dev/null || echo 0)"
  test "$(cat "$out/measurement.sha")" = "${reference:-$sha}" || { echo "resume revision mismatch" >&2; exit 1; }
else
  mkdir "$out"
  echo "$sha" > "$out/measurement.sha"
  echo "$six" > "$out/six-mode"
  echo "$compact" > "$out/compact-mode"
fi
exec 8>"$out/run.lock"
flock -n 8 || { echo "output directory already in use" >&2; exit 1; }
persist=/home/admin/workspace/aop_lab/app_source/sm90_persist_20261123
export SM90_DG_SITE="$persist/dg-sm90-build-ohe59RVP/site"
wheels="$persist/dg-sm90-build-ohe59RVP/wheels"
export MODEL=/mnt/dsv41/models/DeepSeek-V4.1-Flash
export DATASET=/mnt/dsv41/nanjun.cp/MoE-Bench-Experiments/datasets/dsv4-flash_online_input_ids_26-10-08_1.jsonl
if (( six )); then
  export MODEL
  MODEL=$(realpath -e "${SM90_LOCAL_MODEL:-/mnt/fuse/models/DeepSeek-V4.1-Flash}")
  findmnt -T "$MODEL" > "$out/model-mount.txt"
  case $(findmnt -T "$MODEL" -n -o FSTYPE) in nfs*|cifs|ceph|lustre|fuse.sshfs) exit 2 ;; esac
fi
source scripts/env.sh
py="$GC_ENV/bin/python"
input_manifest=docs/trace_executor_20261119/sm90_h20_bs256_mega_gc_eight_results/model_and_dataset.sha256
sed "s|/mnt/dsv41/models/DeepSeek-V4.1-Flash|$MODEL|g" "$input_manifest" > "$out/input.sha256"
sha256sum -c "$out/input.sha256" > "$out/input.sha256.log"
echo "7c848494dbd01daa77ea80765888073cb63b85d7f74054fbcead404598ddf61e  $wheels/deep_gemm-2.8.2+7292e7b-cp310-cp310-linux_x86_64.whl" | sha256sum -c -
if (( mega_only )); then
  conversion=$(mktemp "$out/mega-conversion-XXXXXXXX.jsonl")
  CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 "$py" scripts/sm90/check_mega_weights.py "$MODEL" > "$conversion"
  test "$(wc -l < "$conversion")" -eq 3
  cmp "$out/conversion.jsonl" "$conversion"
else
  CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 "$py" scripts/sm90/check_mega_weights.py "$MODEL" > "$out/conversion.jsonl"
fi
# A private-extension import failure only blocks the two routed Mega arms.
CUDA_VISIBLE_DEVICES='' "$py" integration/sm90_deepgemm.py > "$out/attention-import.log" 2>&1
mega_import=0
CUDA_VISIBLE_DEVICES='' "$py" integration/sm90_deepgemm.py mega > "$out/mega-import.log" 2>&1 || mega_import=$?
if (( mega_only )); then
  # The six controls and publisher state belong to the original measurement checkout.
  for id in 2 3 4 6 7 8; do
    test -f "$out/arm$id/complete"
    (cd "$out/arm$id" && sha256sum --quiet -c artifacts.sha256)
  done
  test -s "$out/replay/placement.json"
fi
if (( compact || six )); then
  cmp "$out/conversion.jsonl" docs/trace_executor_20261119/sm90_h20_bs256_mega_gc_eight_results/mega_retry/mega-conversion.jsonl
fi
if (( compact || mega_only || six )); then
  test "$mega_import" = 0
  source "$launcher_dir/select_mega_compiler.sh"
fi
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
if (( compact )); then
  export GC_MOE_KV_BYTES=4294967296 GC_MOE_MATRIX_BS=32
fi
if (( six )); then
  export GC_MOE_KV_BYTES=10737418240 GC_KV_BUDGET=740000 GC_PREFILL_MARGIN=64
  unset GC_REPEAT_BS512_PROMPT_SHA256 GC_KV_OVERHEAD GC_PREFILL_SLACK
  unset SM90_COLLECT_SE_ROUTES GC_STEP_BOUNDARY GC_COST_HOST_PROBE GC_BS256_TRACE_DIR
fi
if [[ ! -f "$out/arm$publisher/complete" ]]; then
  test "$(df -PB1 "$out" | awk 'END{print $4}')" -gt "$((compact ? 90000000000 : six ? 250000000000 : 405000000000))"
  if (( resume )); then
    # A failed publisher may already have written exclusive-create KV and route files.
    prior=$(mktemp -d "$out/reference-attempt-XXXXXXXX")
    for tree in "$GC_MOE_FIXED_DIR" "$GC_MOE_STRICT_STATE_DIR"; do
      if [[ -d $tree ]]; then mv "$tree" "$prior/"; fi
    done
  fi
  mkdir "$GC_MOE_FIXED_DIR" "$GC_MOE_STRICT_STATE_DIR"
else
  test -s "$GC_MOE_FIXED_DIR/tokens.json"
  for rank in {0..7}; do
    for phase in stats measure; do test -s "$GC_MOE_FIXED_DIR/$phase-rank$rank.pt"; done
    for prefix in "" prefill/; do test -s "$GC_MOE_STRICT_STATE_DIR/${prefix}rank$rank.bin"; done
  done
fi
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
for id in "${arms[@]}"; do
  dst="$out/arm$id"
  if [[ -f "$dst/complete" ]]; then
    (cd "$dst" && sha256sum --quiet -c artifacts.sha256)
    echo "Reusing completed arm $id"
    continue
  fi
  if [[ -d "$dst" ]]; then
    prior=$(mktemp -d "$out/previous-arm$id-XXXXXXXX")
    mv "$dst" "$prior/"
    layout="$GC_MOE_FIXED_DIR/$id-layout.json"
    if [[ -f $layout ]]; then mv "$layout" "$prior/"; fi
  fi
  mkdir "$dst"
  if [[ $id != "$publisher" && ! -f "$out/arm$publisher/complete" ]]; then echo publisher_failed > "$dst/skipped"; continue; fi
  if [[ $id == 7 || $id == 8 ]]; then
    if [[ ! -s "$GC_MOE_FIXED_DIR/placement.json" ]]; then echo missing_rr_placement > "$dst/skipped"; continue; fi
  fi
  if [[ ($id == 1 || $id == 5) && $mega_import != 0 ]]; then
    echo private_extension_import_failed > "$dst/skipped"
    continue
  fi
  case $id in
    0|1|2|3) export GC_MOE_MATRIX_ARM=vanilla ;;
    4|5|6) export GC_MOE_MATRIX_ARM=rr ;;
    7|8) export GC_MOE_MATRIX_ARM=eco ;;
  esac
  case $id in 1|5) export SM90_EIGHT_BACKEND=mega ;; *) export SM90_EIGHT_BACKEND=gc ;; esac
  case $id in 0|2|4|7) export SM90_EIGHT_SHARED=1 ;; *) export SM90_EIGHT_SHARED=0 ;; esac
  if (( compact )); then export SM90_EIGHT_SHARED=0; fi
  export SM90_EIGHT_ARM=$id DG_PRINT_CONFIGS=1
  while ! idle; do flock -u 9; sleep 60; flock 9; done
  code=0
  arm_started=$SECONDS
  bash scripts/trace_layer/run_aligned_ep8.sh "$sha" --study-main20 > "$dst/runner.log" 2>&1 || code=$?
  echo "$((SECONDS - arm_started))" > "$dst/wall-seconds"
  echo "$code" > "$dst/exit"
  raw=$(sed -n 's/^Capture workspace: //p' "$dst/runner.log" | head -1)
  if [[ -n $raw && -d $raw ]]; then
    echo "$raw" > "$dst/raw-location"
    for file in run.json rank{0..7}.npz runtime.log status.txt execution.sha failed-command.txt; do
      if [[ -f "$raw/$file" ]]; then cp "$raw/$file" "$dst/"; fi
    done
  fi
  if (( (compact || six) && id == publisher && code != 0 )); then exit "$code"; fi
  if [[ $code == 0 ]]; then
    test "$(cat "$dst/execution.sha")" = "$sha"
    cp "$GC_MOE_FIXED_DIR/$id-layout.json" "$dst/installed-layout.json"
    (cd "$dst" && sha256sum run.json rank{0..7}.npz installed-layout.json execution.sha > artifacts.sha256)
    touch "$dst/complete"
  fi
done
flock -u 9
"$py" scripts/sm90/report_eight.py "$out"
