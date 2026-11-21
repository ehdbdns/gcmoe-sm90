#!/usr/bin/env bash
# Run in the executor's confirmed eight-GPU reservation. Each model is a new process.
set -euo pipefail
root=$(realpath -m "${1:?supply a persistent result directory}")
repo=$(git rev-parse --show-toplevel)
cd "$repo"
mkdir -p "$root"
exec 8>"$root/run.lock"
flock -n 8
sha=$(git rev-parse HEAD)
if [[ -e "$root/revision" ]]; then test "$(cat "$root/revision")" = "$sha"; else echo "$sha" > "$root/revision"; fi
test -z "$(git status --porcelain --untracked-files=no)"
scheduler_visible=${CUDA_VISIBLE_DEVICES-}
set +u
source /home/admin/workspace/megamoe/env.sh
source /data0/home/admin/workspace/megamoe/native-v41-stsFS4w7/venv/bin/activate
set -u
export SM100_V41_MODEL=/data0/home/admin/workspace/megamoe/native-v41-KO1hTbTj/model
export GC_MOE_PTXAS="$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cu13/bin/ptxas"
export GC_MOE_NVCC="${GC_MOE_NVCC:-$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)}"
test -x "$GC_MOE_PTXAS" && test -x "$GC_MOE_NVCC"
export VLLM_USE_V2_MODEL_RUNNER=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1 HF_HUB_OFFLINE=1
export GC_SM100_MATRIX_CLOCKS=1 PYTHONHASHSEED=0
unset GC_MOE_ECO_UNIT_CFG GC_MOE_ECO_OBJECTIVE GC_MOE_PREFIX_DIAG GC_MOE_VANILLA_STATS_REPLAY
export PYTHONPATH="$repo/source:$repo/integration:$repo/scripts/sm100"
expected=$(python -c 'import json,re; d=json.load(open("docs/sm100_megamoe/ENV.json"));print(",".join(re.findall(r"GPU-[0-9A-Fa-f-]{36}",d["commands"]["gpu_list"]["output"])))')
test "$(tr , '\n' <<< "$expected" | sort -u | wc -l)" = 8
# Preserve a scheduler/container restriction. Check the actual UUIDs before any model.
export CUDA_VISIBLE_DEVICES="${scheduler_visible:-$expected}"
command -v fuser >/dev/null
wait_idle() {
  local busy
  until busy=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader) && [[ -z "$busy" ]] && ! fuser /dev/nvidia{0..7} >/dev/null 2>&1; do
    sleep 60
  done
}
exec 9>>/tmp/gcmoe-sm100-gpu-group.lock
while true; do
  flock 9
  if busy=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader) && test -z "$busy" && ! fuser /dev/nvidia{0..7} >/dev/null 2>&1; then break; fi
  flock -u 9
  sleep 60
done
python - "$expected" <<'PY'
import sys,torch
want=set(sys.argv[1].lower().replace('gpu-','').split(','))
got={str(torch.cuda.get_device_properties(i).uuid).lower().removeprefix('gpu-') for i in range(torch.cuda.device_count())}
assert len(want)==8 and got==want,(want,got)
PY
date -Is >> "$root/acquired.log"
trap 'date -Is >> "$root/released.log"' EXIT
export SM100_MATRIX_OVERLAY="$root/overlay"
if [[ ! -e "$root/overlay.ready" ]]; then
  tmp=$(mktemp -d "$root/package-XXXXXXXX")
  python scripts/sm100/five_mega_header.py "$tmp" > "$root/header.log"
  mv "$tmp" "$SM100_MATRIX_OVERLAY"
  touch "$root/overlay.ready"
fi
export PYTHONPATH="$SM100_MATRIX_OVERLAY:$PYTHONPATH"
export DG_JIT_CACHE_DIR="$root/dg-cache" GC_MOE_ECO_CACHE_DIR="$root/eco-cache"
python -m pip freeze > "$root/versions.txt"
nvidia-smi -L > "$root/gpus.txt"
"$GC_MOE_PTXAS" --version > "$root/ptxas.txt"
sha256sum "$SM100_V41_MODEL/model.safetensors.index.json" > "$root/weights.sha256"
python scripts/sm100/check_v41_rr.py | tee "$root/rr-preflight.log"
rc=0
for bs in 256 512 1024 2048; do
  group="$root/bs$bs"
  mkdir -p "$group"
  for arm in mega gc mega_rr gc_rr gc_eco; do
    dest="$group/$arm"
    mkdir -p "$dest"
    if [[ -e "$dest/result.json" && -f "$dest/exit-code" && $(cat "$dest/exit-code") == 0 ]]; then continue; fi
    if [[ "$arm" != mega && ! -e "$group/mega/result.json" ]]; then break; fi
    if [[ -e "$dest/exit-code" ]]; then rc=1; continue; fi
    wait_idle
    code=0
    date -Is > "$dest/start"
    python -u scripts/sm100/five_matrix.py "$group" "$arm" "$bs" > "$dest/model.log" 2>&1 || code=$?
    date -Is > "$dest/end"
    echo "$code" > "$dest/exit-code"
    if [[ "$code" != 0 ]]; then rc=1; tail -30 "$dest/model.log"; fi
  done
  python scripts/sm100/five_report.py "$group" "$bs" || rc=1
done
python scripts/sm100/five_collect.py "$root"
exit "$rc"
