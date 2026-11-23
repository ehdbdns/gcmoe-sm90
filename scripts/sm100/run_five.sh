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
unset SM100_NATIVE_GREEDY SM100_FULL_DECODE SM100_DIRECT_512 SM100_PHASE_TRACE SM100_QUIET_TRACE
if [[ ${2:-} == quiet-native-bs512 ]]; then export SM100_QUIET_TRACE=1 SM100_PHASE_TRACE=1; fi
if [[ ${2:-} == trace-native-bs512 ]]; then export SM100_PHASE_TRACE=1; fi
if [[ ${2:-} == direct-bs512 || -n ${SM100_PHASE_TRACE:-} ]]; then
  export SM100_DIRECT_512=1
fi
if [[ ${2:-} == upstream-greedy-bs512 ]]; then
  export SM100_NATIVE_GREEDY=1 SM100_FULL_DECODE=1 SM100_PHASE_TRACE=1
fi
if [[ ${SM100_DIRECT_512:-} == 1 || ${SM100_NATIVE_GREEDY:-} == 1 || ${2:-} == full-native-bs512 ]]; then
  fixture=docs/trace_executor_20261119/sm90_main24_stable_decode_bs256_results/delivery/bs256/vanilla/rank0.npz
  echo "ceaa52dc3f3b09fa95e08c59b83b56b97d90b8335b3437a506cb5dbd42fe09f2  $fixture" | sha256sum -c -
fi
if [[ ${2:-} == full-native-bs512 ]]; then export SM100_FULL_DECODE=1 SM100_PHASE_TRACE=1; fi
if [[ ${SM100_FULL_DECODE:-} == 1 || ${2:-} == formal-bs512 || ${SM100_DIRECT_512:-} == 1 ]]; then
  export SM100_FORMAL_512=1 SM100_MATRIX_BS=512 GC_MOE_DISPATCH_SMS=8
  if [[ ${SM100_NATIVE_GREEDY:-} == 1 ]]; then
    unset GC_MOE_DISPATCH_SMS
  fi
  export PATH="$root/dsl-env/bin:$PATH"
  test "$(command -v python)" = "$root/dsl-env/bin/python"
  export SM100_CLOCKED_UPSTREAM="$root/upstream"
  test "$(git -C "$root/upstream" rev-parse HEAD)" = fb8b65fa829c97962eb2dbc9a4ad8b759b9cd736
  git -C "$root/upstream" diff --exit-code HEAD
  git -C "$root/upstream" apply "$repo/scripts/sm100/natureal_matrix_audit.patch"
  if [[ -n ${SM100_PHASE_TRACE:-} ]]; then python scripts/sm100/native_phase_instrument.py "$root/upstream"; fi
  if [[ ${SM100_NATIVE_GREEDY:-} == 1 ]]; then
    git -C "$root/upstream" apply "$repo/docs/sm100_megamoe/v41_full_decode_pair_results/compat/gc_capture_fix.diff"
  fi
  cp -R "$repo/source/gc_moe/eplb" "$root/upstream/gc_moe/"
  cp "$repo/source/gc_moe/eco/"meta_sm{90,100}.py "$root/upstream/gc_moe/eco/"
  unset GC_MOE_ECO_BLOCK_M GC_MOE_ECO_PLAN_DISPATCH GC_MOE_ECO_HELPER GC_MOE_ECO_TRACE
  python -c 'from importlib.metadata import version; assert version("nvidia-cutlass-dsl") == "4.8.1"'
  unset GC_MOE_PRIMARY_CTX GC_MOE_BALANCED_WORKQUEUES
fi
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
if [[ ${2:-} == native-pair || ${2:-} == upstream-pair ]]; then
  export GC_SM100_MATRIX_CLOCKS=0
  unset SM100_MATRIX_OVERLAY GC_MOE_ECO_TRACE
  export DG_JIT_CACHE_DIR="$root/dg-cache"
  test ! -e "$root/mega.json" && test ! -e "$root/gc.json" || exit 2
  sha256sum "$SM100_V41_MODEL/model.safetensors.index.json" > "$root/weights.sha256"
  nvidia-smi -L > "$root/gpus.txt"
  python -m pip freeze > "$root/versions.txt"
  if [[ ${2:-} == upstream-pair ]]; then
    export SM100_UPSTREAM_PAIR="$root/upstream"
    test "$(git -C "$SM100_UPSTREAM_PAIR" rev-parse HEAD)" = 7902831ec4a6691b6268b308bcff5ec0cb15ec86
    test -z "$(git -C "$SM100_UPSTREAM_PAIR" status --porcelain)"
    export PYTHONPATH="$SM100_UPSTREAM_PAIR:$repo/integration:$repo/scripts/sm100"
    export PATH="$root/dsl-env/bin:$PATH"
    test "$(command -v python)" = "$root/dsl-env/bin/python"
    unset GC_MOE_DISPATCH_SMS GC_MOE_PRIMARY_CTX GC_MOE_BALANCED_WORKQUEUES
    python -c 'import gc_moe,cutlass,os; from pathlib import Path; from importlib.metadata import version; assert Path(os.environ["SM100_UPSTREAM_PAIR"]).resolve() in Path(gc_moe.__file__).resolve().parents; assert version("nvidia-cutlass-dsl") == "4.8.1"; print(gc_moe.__file__,cutlass.__file__)' > "$root/imports.txt"
    python -m pip freeze > "$root/versions.txt"
    evidence=docs/sm100_megamoe/v41_native_pair_results
    sha256sum -c "$evidence/weights.sha256"
    sha256sum -c "$evidence/measurements.sha256" > "$root/fixture-check.txt"
    while read -r digest fixture; do
      ln -s "$fixture" "$root/${fixture##*/}"
    done < "$evidence/measurements.sha256"
    test "$(wc -l < "$root/fixture-check.txt")" -eq 24
  fi
  for arm in mega gc; do
    wait_idle
    code=0
    python -u scripts/sm100/native_pair.py "$root" "$arm" > "$root/$arm.log" 2>&1 || code=$?
    echo "$code" > "$root/$arm.exit-code"
    if [[ $code != 0 ]]; then tail -40 "$root/$arm.log"; exit "$code"; fi
  done
  exit 0
fi
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
batches=(256 512 1024 2048)
case ${SM100_MATRIX_BS:-} in
  '') ;;
  256|512|1024|2048) batches=("$SM100_MATRIX_BS") ;;
  *) exit 2 ;;
esac
for bs in "${batches[@]}"; do
  group="$root/bs$bs"
  mkdir -p "$group"
  arms=(mega gc mega_rr gc_rr gc_eco)
  if [[ -n ${SM100_PHASE_TRACE:-} ]]; then arms=(mega gc); fi
  for arm in "${arms[@]}"; do
    dest="$group/$arm"
    mkdir -p "$dest"
    if [[ -e "$dest/result.json" && -f "$dest/exit-code" && $(cat "$dest/exit-code") == 0 ]]; then continue; fi
    if [[ ${SM100_NATIVE_GREEDY:-} != 1 && "$arm" != mega && ! -e "$group/mega/result.json" ]]; then break; fi
    if [[ -e "$dest/exit-code" ]]; then rc=1; continue; fi
    wait_idle
    code=0
    date -Is > "$dest/start"
    arm_path="$PYTHONPATH"
    if [[ ${SM100_FORMAL_512:-} == 1 && $arm == gc* ]]; then
      arm_path="$SM100_CLOCKED_UPSTREAM:$arm_path"
    fi
    PYTHONPATH="$arm_path" python -u scripts/sm100/five_matrix.py "$group" "$arm" "$bs" > "$dest/model.log" 2>&1 || code=$?
    date -Is > "$dest/end"
    echo "$code" > "$dest/exit-code"
    if [[ "$code" != 0 ]]; then rc=1; tail -30 "$dest/model.log"; fi
  done
  if [[ -n ${SM100_PHASE_TRACE:-} ]]; then
    if [[ $rc == 0 ]]; then
      python scripts/sm100/native_phase_perfetto.py "$group" || rc=1
      if [[ ${SM100_NATIVE_GREEDY:-} == 1 ]]; then
        python scripts/sm100/native_greedy_report.py "$group" || rc=1
      elif [[ ${SM100_FULL_DECODE:-} == 1 ]]; then
        python scripts/sm100/full_decode_report.py "$group" || rc=1
      fi
      if [[ -n ${SM100_QUIET_TRACE:-} ]]; then
        python scripts/sm100/native_skew_check.py "$group" "$repo/docs/sm100_megamoe"
      fi
    fi
    exit "$rc"
  fi
  PYTHONPATH="${SM100_CLOCKED_UPSTREAM:+$SM100_CLOCKED_UPSTREAM:}$PYTHONPATH" python scripts/sm100/five_report.py "$group" "$bs" || rc=1
done
python scripts/sm100/five_collect.py "$root" "${batches[@]}"
exit "$rc"
