#!/usr/bin/env bash
# One BS256 reference publication and two measured consumers.
set -eo pipefail
[[ $# == 1 ]] || { echo "usage: run_linear_wa_pair.sh JOB_DIRECTORY" >&2; exit 2; }
cd "$(git rev-parse --show-toplevel)"
job=$(realpath -e "$1")
export MODEL="${SM90_LOCAL_MODEL:-/mnt/fuse/models/DeepSeek-V4.1-Flash}"
export DATASET=/mnt/dsv41/nanjun.cp/MoE-Bench-Experiments/datasets/dsv4-flash_online_input_ids_26-10-08_1.jsonl
manifest=docs/trace_executor_20261119/sm90_h20_bs256_mega_gc_eight_results/model_and_dataset.sha256
python3 - "$MODEL" "$DATASET" "$manifest" <<'PY'
import hashlib, json, sys
from pathlib import Path
model, dataset, manifest = map(Path, sys.argv[1:])
expected = {Path(path).name: h for h, path in (row.split() for row in manifest.read_text().splitlines())}
for path in (model / 'config.json', model / 'model.safetensors.index.json', dataset):
    with path.open('rb') as f:
        h = hashlib.sha256()
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    if h.hexdigest() != expected[path.name]:
        raise SystemExit(f'input hash mismatch: {path}')
shards = set(json.loads((model / 'model.safetensors.index.json').read_text())['weight_map'].values())
if len(shards) != 48 or any(not (model / s).is_file() or not (model / s).stat().st_size for s in shards):
    raise SystemExit('incomplete local model shards')
PY
unset SM90_FULL_MODEL_FOUR SM90_SIX_ARM SM90_EIGHT_ARM SM90_EIGHT_BACKEND SM90_PAIRED_SMALL SM90_COLLECT_SE_ROUTES
unset GC_BS256_TRACE_DIR GC_STEP_BOUNDARY GC_COST_HOST_PROBE GC_MOE_VANILLA_STATS_REPLAY GC_MOE_PREFIX_DIAG
unset GC_MOE_DECODE_STATS_AA GC_MOE_MAIN_FINE_ONLY GC_REPEAT_BS512_PROMPT_SHA256 GC_KV_OVERHEAD
unset GC_MOE_SM90_SINGLE_PUSH GC_MOE_SM90_SOURCE_INBOX GC_MOE_SM90_SPLIT_EXCHANGE GC_MOE_SM90_SOURCE_INDEX_PULL
export SM90_LINEAR_WA_PAIR=1 GC_MOE_MATRIX_SIZE=24 GC_MAIN24_SINGLE_BS=256
export GC_MOE_SM90_PLAN_DISPATCH=1 GC_MOE_STABLE_DECODE_TOPK=1 GC_MOE_STABLE_PREFILL_TOPK=1 GC_MOE_ORDER_PREFILL=1
export GC_PREFILL_MARGIN=256 GC_KV_BUDGET=1150000 GC_MOE_KV_BYTES=21474836480 GC_MOE_NO_PREFETCH=1
export NCCL_IB_HCA=mlx5_4:1,mlx5_5:1,mlx5_9:1,mlx5_17:1,mlx5_21:1,mlx5_29:1
export NVSHMEM_HCA_LIST="$NCCL_IB_HCA"
export GC_MOE_CHECKPOINT_ROOT="${GC_MOE_CHECKPOINT_ROOT:-$job/checkpoints}"
mkdir -p "$GC_MOE_CHECKPOINT_ROOT"
exec bash scripts/trace_layer/run_main20.sh "$job/delivery"
