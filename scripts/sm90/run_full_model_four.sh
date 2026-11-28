#!/usr/bin/env bash
# Full-model four-arm groups on the shared H20 node.
set -eo pipefail
unset SM90_AUCTION_PAIR SM90_AUCTION_REPLAY GC_MOE_ECO_FINE_AUCTION
unset SM90_SHARED_PLAN_OVERLAP SM90_SHARED_FIRST_WA1 SM90_WA1_PROFILE_DIR SM90_FINE_WA1_REFERENCE GC_MOE_EP8_LINEAR SM90_LINEAR_WA_PAIR SM90_SIX_ARM SM90_IR_REFERENCE
if [[ ${1:-} == --fine-bs512-1024 ]]; then
  shift
  batches=(512 1024)
else
  if [[ ${1:-} == --linear-ep8 ]]; then shift; fi
  export GC_MOE_EP8_LINEAR=1 GC_PREFILL_MARGIN=256
  batches=(32 64 128 256 512 1024)
  if [[ ${1:-} == --auction-bs256 ]]; then
    shift; batches=(256); export SM90_AUCTION_PAIR=1
  elif [[ ${1:-} == --ir-reference ]]; then
    [[ ${2:-} =~ ^(32|64|128|256|512|1024)$ ]] || exit 2
    batches=("$2"); export SM90_IR_REFERENCE=1; shift 2
  fi
fi
[[ $# == 1 && $1 != --* ]] || { echo "usage: run_full_model_four.sh [--linear-ep8|--fine-bs512-1024|--ir-reference BS|--auction-bs256] JOB_DIRECTORY" >&2; exit 2; }
cd "$(git rev-parse --show-toplevel)"
job=$(realpath -m "$1")
mkdir -p "$job"
exec 8>"$job/launcher.lock"
flock -n 8 || { echo "job already active" >&2; exit 2; }
[[ ! -e "$job/shared-first-wa1" && ! -e "$job/shared-plan-overlap" && ! -e "$job/fine-reference" ]] || exit 2
if [[ ${SM90_IR_REFERENCE:-0} == 1 ]]; then
  if [[ -e $job/measurement.sha ]]; then test -e "$job/ir-reference"; fi
  touch "$job/ir-reference"
else
  test ! -e "$job/ir-reference"
fi
if [[ ${SM90_AUCTION_PAIR:-0} == 1 ]]; then
  if [[ -e $job/measurement.sha ]]; then test -e "$job/auction-pair"; fi
  touch "$job/auction-pair"
else
  test ! -e "$job/auction-pair"
fi
sha=$(git rev-parse HEAD)
if [[ -f "$job/measurement.sha" ]]; then
  [[ $(cat "$job/measurement.sha") == "$sha" ]] || exit 2
  stored=0
  if [[ -f "$job/linear-ep8" ]]; then stored=$(cat "$job/linear-ep8"); fi
  [[ $stored == ${GC_MOE_EP8_LINEAR:-0} ]] || exit 2
else
  echo "${GC_MOE_EP8_LINEAR:-0}" > "$job/linear-ep8"
  echo "$sha" > "$job/measurement.sha"
fi
MODEL=$(realpath -e "${SM90_LOCAL_MODEL:-/mnt/fuse/models/DeepSeek-V4.1-Flash}")
export MODEL
test -d "$MODEL"
fstype=$(findmnt -T "$MODEL" -n -o FSTYPE)
case "$fstype" in nfs*|cifs|ceph|lustre|fuse.sshfs) echo "local model required: $MODEL ($fstype)" >&2; exit 2 ;; esac
findmnt -T "$MODEL" > "$job/model-mount.txt"
data=/mnt/dsv41/nanjun.cp/MoE-Bench-Experiments/datasets
sed "s|/mnt/dsv41/models/DeepSeek-V4.1-Flash|$MODEL|g" docs/trace_executor_20261119/sm90_h20_bs256_mega_gc_eight_results/model_and_dataset.sha256 > "$job/inputs.sha256"
sha256sum -c "$job/inputs.sha256" > "$job/input-check.log"
python3 - "$MODEL" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
shards = set(json.loads((root/'model.safetensors.index.json').read_text())['weight_map'].values())
if len(shards) != 48 or any(not (root/name).is_file() or (root/name).stat().st_size == 0 for name in shards):
    raise SystemExit('local model is missing indexed weight shards')
PY
export NCCL_IB_HCA=mlx5_4:1,mlx5_5:1,mlx5_9:1,mlx5_17:1,mlx5_21:1,mlx5_29:1
export NVSHMEM_HCA_LIST="$NCCL_IB_HCA"
export SM90_FULL_MODEL_FOUR=1 GC_MOE_MATRIX_SIZE=24 GC_MOE_ORDER_PREFILL=1
export GC_MOE_STABLE_PREFILL_TOPK=1 GC_MOE_STABLE_DECODE_TOPK=1 GC_MOE_KV_BYTES=21474836480
export GC_MOE_NO_PREFETCH=1 GC_MOE_SM90_PLAN_DISPATCH=1
export GC_MOE_CHECKPOINT_ROOT="${GC_MOE_CHECKPOINT_ROOT:-$job/checkpoints}"
mkdir -p "$GC_MOE_CHECKPOINT_ROOT"
unset GC_PREFILL_SLACK GC_MOE_FIXED_REUSE
unset SM90_COLLECT_SE_ROUTES SM90_EIGHT_ARM SM90_EIGHT_BACKEND SM90_PAIRED_SMALL GC_BS256_TRACE_DIR
unset GC_STEP_BOUNDARY GC_COST_HOST_PROBE GC_MOE_VANILLA_STATS_REPLAY GC_MOE_DECODE_STATS_AA GC_MOE_PREFIX_DIAG
unset GC_MOE_SM90_SOURCE_INBOX GC_MOE_SM90_SINGLE_PUSH GC_MOE_SM90_SPLIT_EXCHANGE GC_MOE_SM90_SOURCE_INDEX_PULL
unset GC_MOE_MAIN_FINE_ONLY GC_MOE_PREFIX_SPARSE_AA GC_MOE_PREFIX_BLOCK_AA GC_MOE_PREFIX_LAYER_AA GC_MOE_PREFIX_HIDDEN_AA
failed=0
for bs in "${batches[@]}"; do
  delivery="$job/delivery-$bs"
  if [[ -f "$delivery/SHA256SUMS" && $(cat "$delivery/measurement.sha") == "$sha" ]]; then
    (cd "$delivery" && sha256sum --quiet -c SHA256SUMS)
    continue
  fi
  if [[ -e "$delivery" ]]; then
    echo "Preserving partial BS$bs run; inspect $job/bs$bs.log" >&2
    if [[ ${GC_MOE_EP8_LINEAR:-0} == 1 ]]; then failed=1; continue; fi
    exit 1
  fi
  export GC_MAIN24_SINGLE_BS=$bs
  unset GC_REPEAT_BS512_PROMPT_SHA256 GC_KV_OVERHEAD
  export DATASET="$data/dsv4-flash_online_input_ids_26-10-08_1.jsonl"
  export GC_KV_BUDGET=1150000
  if (( bs >= 512 )); then
    export DATASET="$DATASET,$data/dsv4-flash_online_input_ids_26-10-08_2.jsonl"
    export GC_KV_BUDGET=2300000
  fi
  if [[ $bs == 1024 ]]; then
    export GC_KV_BUDGET=5000000 GC_REPEAT_BS512_PROMPT_SHA256=e7c4aee9db91a34c018bd54464e67355681b74112d515401844f473c053cfcef
  fi
  code=0
  bash scripts/trace_layer/run_main20.sh "$delivery" > "$job/bs$bs.log" 2>&1 || code=$?
  echo "$code" > "$job/bs$bs.exit"
  if (( code )); then
    if [[ ${GC_MOE_EP8_LINEAR:-0} != 1 ]]; then exit "$code"; fi
    failed=$code
  fi
done
if (( failed )); then exit "$failed"; fi
if [[ ${SM90_IR_REFERENCE:-0} == 1 || ${SM90_AUCTION_PAIR:-0} == 1 ]]; then exit 0; fi
python3 - "$job" "${batches[@]}" <<'PY'
import csv, pathlib, sys
job = pathlib.Path(sys.argv[1])
for name, count in [('main_table', 4), ('pairs', 6), ('auction', 1), ('layers', 160)]:
    merged = []
    for bs in map(int, sys.argv[2:]):
        with (job/f'delivery-{bs}'/f'bs{bs}'/f'{name}.csv').open() as f:
            rows = list(csv.DictReader(f))
        if len(rows) != count or any(int(r['bs']) != bs for r in rows):
            raise ValueError(f'incomplete {name} in BS{bs}')
        merged.extend(rows)
    with (job/f'{name}.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=merged[0].keys())
        writer.writeheader()
        writer.writerows(merged)
PY
if [[ ${GC_MOE_EP8_LINEAR:-0} == 1 ]]; then
  python3 scripts/sm90/compare_ep8_linear.py "$job"
fi
