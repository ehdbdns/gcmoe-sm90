#!/usr/bin/env bash
# Run the fixed EP8 task from the validated revision; keep evidence on failure.
set -eo pipefail
analysis_dir=$(cd "$(dirname "$0")" && pwd)
variant=${2:-}
study=0
layers=(8 20)
fine_layers=("${layers[@]}")
case "$variant" in
  --study-long-cost|--study-main20) study=1 ;;
  --study-paired)
    study=1
    layers=({0..39})
    fine_layers=(0 4 8 12 16 20 24 28 32 36 39)
    ;;
  --study-fine|--study-coarse|--study-e2e)
    study=1
    layers=(0 4 8 12 16 20 24 28 32 36 39)
    fine_layers=("${layers[@]}")
    if [[ $variant == --study-coarse ]]; then layers=({0..39}); fi
    ;;
  ''|--fine-trace) ;;
  *) exit 2 ;;
esac
if [[ $# -lt 1 || $# -gt 2 ]]; then echo "Expected execution SHA and optional capture mode" >&2; exit 2; fi
cd "${GC_MOE_EXEC_ROOT:-$analysis_dir/../..}"
out=$(mktemp -d /tmp/gcmoe-aligned-XXXXXXXX)
echo "Capture workspace: $out"
step='1: checkout and environment'
trap 'code=$?; printf "step=%s\nexit_code=%s\n" "$step" "$code" > "$out/status.txt"' EXIT
trap 'printf "command=%s\n" "$BASH_COMMAND" >> "$out/failed-command.txt"' ERR
test -z "$(git status --porcelain)"
test "$(git rev-parse HEAD)" = "$(git rev-parse --verify "$1^{commit}")"
git rev-parse HEAD > "$out/start.sha"

nvidia-smi --query-gpu=index,name,driver_version,memory.used,memory.total --format=csv \
  | tee "$out/gpu.log"

export MODEL="${MODEL:-/home/admin/local_models/DeepSeek-V4.1-Flash}"
export DATASET="${DATASET:-/mnt/nas2/nanjun.cp/MoE-Bench-Experiments/datasets/dsv4-flash_online_input_ids_26-10-08_1.jsonl}"
test -d "$MODEL"
IFS=, read -ra dataset_files <<< "$DATASET"
for dataset_file in "${dataset_files[@]}"; do
  dataset_file=${dataset_file#"${dataset_file%%[![:space:]]*}"}
  dataset_file=${dataset_file%"${dataset_file##*[![:space:]]}"}
  [[ -z "$dataset_file" || -r "$dataset_file" ]] || { echo "Unreadable dataset: $dataset_file" >&2; exit 1; }
done
source config/experiment.env
source scripts/env.sh
test -x "$GC_ENV/bin/python"
export GC_MOE_SRC="$PWD/source"
export PYTHONPATH="$PWD/source:$PWD/patches:$PWD/integration"
if [[ $variant == --study-paired || $variant == --study-long-cost || $variant == --study-main20 ]]; then
  # The in-process EngineCore advances only when the driver calls step().
  # The separate TP/EP worker processes still own the eight GPUs.
  export VLLM_ENABLE_V1_MULTIPROCESSING=0
  export PYTHONPATH="$PWD/source:$analysis_dir/../../patches:$analysis_dir/../../integration"
fi
if [[ -n ${GC_MOE_FIXED_DIR:-} ]]; then
  export GC_MOE_ECO_NOSPLIT=0 GC_MOE_ECO_TOKENS_ONLY=0 GC_MOE_ECO_BLOCKS_ONLY=0
  export GC_MOE_ECO_NULL_PLAN=0 GC_MOE_ECO_NO_AUCTION=0 GC_MOE_SM90_BLOCK_READY=0
  export GC_MOE_ECO_ROUNDS=1 GC_MOE_ECO_COLD_ROUNDS=6 PYTHONHASHSEED=0
fi
export WORLD=8 GPU_UTIL=0.72 GC_MOE_NO_PREFETCH=1
export GC_MOE_EPLB=1 GC_MOE_EPLB_NO_REARRANGE=0 GC_MOE_SLOT_PLACEMENT=identity
export GC_MOE_ECOPLB_PER_LAYER=1 GC_MOE_ECO_PER_LAYER=1 GC_MOE_EPLB_EXTRA_PER_RANK=8
export GC_MOE_ECO=1 GC_MOE_ECO_WB=26 GC_MOE_ECO_WA=0 GC_MOE_ECO_COSTFN=0
if (( study )); then export GC_MOE_ECO_COSTFN="${GC_MOE_STUDY_COSTFN:-0}"; fi
export GC_MOE_ECO_MAX_COPIES=8 GC_MOE_MB_CLAIM=1 GC_MOE_SM90_PULL_TMA=1
export GC_MOE_SM90_STREAM_M_BLOCKS="${GC_MOE_SM90_STREAM_M_BLOCKS:-1}"
test "$GC_MOE_SM90_STREAM_M_BLOCKS" = 0 || test "$GC_MOE_SM90_STREAM_M_BLOCKS" = 1
export GC_MOE_SHARED_IN_GC=1 GC_MOE_SM90_SM_REUSE=0 GC_MOE_SM90_DISPATCH_SMS=6
export GC_MOE_CAPACITY=2048 GC_MOE_BOUND=2048 GC_MOE_BLOCK_M_FORCE=64 MAX_BATCHED=2048
export GC_MOE_PULL_TRACE=1 GC_MOE_ECO_TRACE=1 GC_MOE_FINE_TRACE=0
export GC_MOE_IR_STEP=0 GC_MEASURE_IR_STEP=0 GC_MEASURE_IR=0 GC_MEASURE_LOAD=0
if [[ $variant == --study-main20 ]]; then
  export GC_MOE_ECO_COSTFN=2 GC_MOE_EPLB_WINDOW=1000 GC_MOE_STEP_PROFILE=1
  export GC_MOE_EPLB_RANDOM_SEED=0
  case "$GC_MOE_MATRIX_ARM" in
    vanilla) export GC_MOE_EPLB=0 GC_MOE_ECO=0 GC_MOE_EPLB_NO_REARRANGE=1 ;;
    random) export GC_MOE_ECO=0 GC_MOE_SLOT_PLACEMENT=random GC_MOE_EPLB_NO_REARRANGE=1 ;;
    rr) export GC_MOE_ECO=0 ;;
    eco_linear) test "${GC_MOE_MATRIX_SIZE:-20}" = 24; export GC_MOE_ECO_COSTFN=0 ;;
    eco) ;;
    eco_smr) export GC_MOE_SM90_SM_REUSE=1 ;;
    *) exit 2 ;;
  esac
fi
if [[ -n ${GC_BS256_TRACE_DIR:-} ]]; then
  [[ $variant == --study-main20 && $GC_MOE_MATRIX_BS == 256 && $GC_MOE_MATRIX_ARM =~ ^(rr|eco)$ ]] || exit 2
  export GC_MOE_FINE_TRACE=1 GC_MOE_FINE_FLOOR=1 GC_MOE_FINE_EPOCHS=1 GC_MOE_FINE_RECS=4194304 GC_MOE_FINE_MIN_EPOCH=0
fi
fine_args=()
if [[ $variant == --fine-trace || $variant == --study-fine || $variant == --study-paired ]]; then
  export GC_MOE_FINE_TRACE=1 GC_MOE_FINE_FLOOR=1 GC_MOE_FINE_EPOCHS=1
  export GC_MOE_FINE_RECS=4194304 GC_MOE_FINE_MIN_EPOCH=0
  fine_args=(--fine-layers "${fine_layers[@]}")
  if [[ $variant == --study-paired ]]; then fine_args+=(--timing-tail-steps 100); fi
fi
export GC_MOE_LAYERS=40 GC_MEASURE_FLUSH="${GC_MOE_STUDY_STEP:-2}" ENFORCE_EAGER=0 CUDAGRAPH_MODE=FULL_DECODE_ONLY
"$GC_ENV/bin/python" -c 'import torch, vllm; print(torch.__version__, torch.version.cuda, vllm.__version__); assert torch.cuda.device_count() == 8; assert all(torch.cuda.get_device_capability(i)[0] == 9 for i in range(8))' \
  | tee "$out/runtime.log"

if (( study )); then
  git rev-parse HEAD > "$out/execution.sha"
else
  step="2: CPU regression and CUDA compile"
  "$GC_ENV/bin/python" tests/test_aligned_step_cpu.py 2>&1 | tee "$out/cpu.log"
  GC_MOE_STEP_PROFILE=0 GC_MOE_FINE_TRACE=0 GC_MOE_SM90_STREAM_M_BLOCKS=0 \
    "$GC_ENV/bin/python" scripts/smoke_compile_eco.py \
    2>&1 | tee "$out/compile-base.log"
  if [ "${#fine_args[@]}" -gt 0 ]; then
    "$GC_ENV/bin/python" tests/test_pull_gemm_detail_cpu.py 2>&1 | tee "$out/cpu-fine.log"
  fi
  GC_MOE_STEP_PROFILE=1 SMOKE_ECO_WB=26 "$GC_ENV/bin/python" scripts/smoke_compile_eco.py \
    2>&1 | tee "$out/compile-aligned.log"
  if [[ ${GC_MOE_SM90_PLAN_DISPATCH:-1} == 1 ]]; then
    base_roles='[5, 7, 1, 2, 3]'; aligned_roles='[5, 7, 1, 4, 2, 3]'
  else
    base_roles='[1, 2, 5, 3]'; aligned_roles='[1, 4, 2, 5, 3]'
  fi
  grep -Fq "compiled roles $base_roles" "$out/compile-base.log"
  grep -Fq "compiled roles $aligned_roles" "$out/compile-aligned.log"
  git rev-parse HEAD > "$out/execution.sha"

  if [ "$GC_MOE_SM90_STREAM_M_BLOCKS" = 1 ]; then
    step="2b: hot-slot CUDA Graph parity"
    "$GC_ENV/bin/python" tests/test_stream_mblock_cpu.py 2>&1 | tee "$out/cpu-stream.log"
    for mode in 0 1; do
      GC_MOE_STEP_PROFILE=0 GC_MOE_FINE_TRACE=0 GC_MOE_SM90_STREAM_M_BLOCKS="$mode" \
        GC_MOE_MB_CLAIM="$mode" \
        TL_TOKENS=17 TL_CAP=128 TL_HOT_BLOCKS=1 \
        timeout 300 "$GC_ENV/bin/torchrun" --standalone --nproc-per-node=8 \
        scripts/verify_eco_capture.py 2>&1 | tee "$out/hot-parity-$mode.log"
      # torchrun interleaves the eight ranks' stdout without adding newlines, so
      # two STREAM_CHECK records can land on one physical line. Split them out
      # before counting; one record per rank is what the gate actually requires.
      grep -o 'STREAM_CHECK rank=[0-7] sha256=[0-9a-f]*' "$out/hot-parity-$mode.log" \
        | sort > "$out/hot-parity-$mode.sha"
      test "$(wc -l < "$out/hot-parity-$mode.sha")" -eq 8
      for rank in {0..7}; do
        grep -q "^STREAM_CHECK rank=$rank sha256=" "$out/hot-parity-$mode.sha"
      done
    done
    diff -u "$out/hot-parity-0.sha" "$out/hot-parity-1.sha"
  fi
fi

step="3: capture"
bash scripts/warm_page_cache.sh "$MODEL" 2>&1 | tee "$out/warm.log"
driver="$analysis_dir/../real_decode_100_100.py"
if [[ $variant == --study-long-cost || $variant == --study-main20 ]]; then
  export GC_MOE_STEP_PROFILE=1
  "$GC_ENV/bin/python" -u "$driver" --long-cost-study --label LONG_COST --output "$out/run.json" \
    --seqs-per-rank "$(( ${GC_MOE_MATRIX_BS:-256} / 8 ))" --prefill-lo 256 --prefill-hi 2048 --warm-steps 1000 --measure-steps 1000 \
    2>&1 | tee "$out/run.log"
  exit 0
fi
if [[ $variant == --study-e2e ]]; then
  export GC_MOE_PULL_TRACE=0 GC_MOE_ECO_TRACE=0 GC_MOE_FINE_TRACE=0 GC_MOE_STEP_PROFILE=0
  "$GC_ENV/bin/python" -u "$driver" --label STUDY_E2E --output "$out/result.json" \
    --seqs-per-rank 32 --prefill-lo 256 --prefill-hi 2048 --warm-steps 200 --measure-steps 100 \
    2>&1 | tee "$out/run.log"
  exit 0
fi
"$GC_ENV/bin/python" -u "$driver" \
  --label ALIGNED_EP8 --output "$out/result.json" \
  --seqs-per-rank 32 --prefill-lo 256 --prefill-hi 2048 \
  --warm-steps 200 --measure-steps 1 --profile-aligned-step \
  --profile-out "$out/raw" "${fine_args[@]}" 2>&1 | tee "$out/run.log"
if [[ -n ${GC_MOE_STRICT_STATE_DIR:-} ]]; then
  nvidia-smi --query-gpu=index,pstate,temperature.gpu,clocks.current.sm,clocks.current.memory,power.draw \
    --format=csv,noheader > "$out/gpu-after.csv"
fi
test -s "$out/raw/capture.json"
for rank in {0..7}; do
  test -s "$out/raw/rank$rank.json"
done

step="4: parse and verify"
dispatch_grid=24
if [[ $GC_MOE_ECO == 1 && ${GC_MOE_SM90_PLAN_DISPATCH:-1} == 1 ]]; then dispatch_grid=20; fi
overlap_args=(--require-pull-overlap)
if (( study )); then overlap_args=(); fi
"$GC_ENV/bin/python" "$analysis_dir/aligned_step.py" "$out/raw" "$out/deliver" \
  --layers "${layers[@]}" --dispatch-grid "$dispatch_grid" "${overlap_args[@]}" \
  2>&1 | tee "$out/parse.log"
test "$(find "$out/deliver" -maxdepth 1 -name '*.perfetto.json' | wc -l)" -eq "${#layers[@]}"
test "$(find "$out/deliver" -maxdepth 1 -name '*.counts.json' | wc -l)" -eq "${#layers[@]}"
raw_capture=("$out/raw/capture.json")
if [ "${#fine_args[@]}" -gt 0 ]; then
  step="5: validate and join fine trace"
  test -s "$out/raw/fine.json"
  detail_args=()
  if [ "$GC_MOE_SM90_STREAM_M_BLOCKS" = 1 ]; then
    detail_args+=(--require-streamed-blocks --require-batch-claims)
  fi
  "$GC_ENV/bin/python" "$analysis_dir/pull_gemm_detail.py" "$out/raw" "$out/deliver" \
    "${detail_args[@]}" 2>&1 | tee "$out/detail.log"
  test "$(find "$out/deliver" -maxdepth 1 -name '*.detail.perfetto.json' | wc -l)" -eq "${#fine_layers[@]}"
  test "$(find "$out/deliver" -maxdepth 1 -name '*.detail.json' | wc -l)" -eq "${#fine_layers[@]}"
  raw_capture+=("$out/raw/fine.json")
  if [[ $variant == --study-fine || $variant == --study-paired ]]; then raw_capture+=("$out"/raw/fine-epoch-*.json); fi
fi
sha256sum "${raw_capture[@]}" "$out/raw"/rank*.json "$out/deliver"/*.json \
  | tee "$out/sha256.txt"
