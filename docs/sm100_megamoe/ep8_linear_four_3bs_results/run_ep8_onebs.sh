#!/usr/bin/env bash
# Keep setup and the CPU reports outside the shared eight-GPU lock.
set -eo pipefail
root=$(realpath -m "${1:?private run directory required}")
repo=$(git rev-parse --show-toplevel)
cd "$repo"
mkdir -p "$root"
exec 8>"$root/launcher.lock"
flock -n 8 || { echo "Another launcher owns this run" >&2; exit 1; }
rev=$(git rev-parse HEAD)
if [[ -e $root/revision ]]; then test "$(cat "$root/revision")" = "$rev"; else echo "$rev" > "$root/revision"; fi
test -z "$(git status --porcelain --untracked-files=no)"
if [[ -v CUDA_VISIBLE_DEVICES && -z $CUDA_VISIBLE_DEVICES ]]; then
  echo "The scheduler device mask is empty" >&2
  exit 1
fi
mask=${CUDA_VISIBLE_DEVICES-}
source /home/admin/workspace/megamoe/env.sh
base=/data0/home/admin/workspace/megamoe/native-v41-stsFS4w7/venv
source "$base/bin/activate"
set -u
export SM100_V41_MODEL=${SM100_EP8_MODEL:-/data0/home/admin/workspace/megamoe/native-v41-KO1hTbTj/model}
for entry in '8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879 config.json' '74b0686a3d2891980d5e303251b075a3bccae2c2ff650747db2620a649b98fa8 model.safetensors.index.json'; do
  (cd "$SM100_V41_MODEL" && sha256sum -c <<< "$entry")
done
export GC_MOE_PTXAS="$base/lib/python3.12/site-packages/nvidia/cu13/bin/ptxas"
export GC_MOE_NVCC="${GC_MOE_NVCC:-$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)}"
test -x "$GC_MOE_PTXAS" && test -x "$GC_MOE_NVCC"
data=${SM100_EP8_DATA:-/mnt/dsv41/nanjun.cp/MoE-Bench-Experiments/datasets}
if [[ ! -r $data/dsv4-flash_online_input_ids_26-10-08_1.jsonl && -z ${SM100_EP8_DATA:-} ]]; then
  data=/mnt/nas2/nanjun.cp/MoE-Bench-Experiments/datasets
fi
if [[ -n ${SM100_EP8_PROMPTS:-} ]]; then
  python scripts/sm100/ep8_inputs.py --prepared "$SM100_EP8_PROMPTS" "$root"
else
  python scripts/sm100/ep8_inputs.py "$data" "$root"
fi
upstream=fb8b65fa829c97962eb2dbc9a4ad8b759b9cd736
export SM100_CLOCKED_UPSTREAM="$root/upstream"
if [[ ! -f $root/prepared ]]; then
  if [[ ! -d $root/upstream ]]; then git clone --branch eco https://github.com/Natureal/gc-moe.git "$root/upstream"; fi
  git -C "$root/upstream" checkout --detach "$upstream"
  test -z "$(git -C "$root/upstream" status --porcelain)"
  "$base/bin/python" -m venv "$root/dsl-env"
  py="$root/dsl-env/bin/python"
  mkdir -p "$root/wheels"
  packages=(nvidia-cutlass-dsl nvidia-cutlass-dsl-libs-base nvidia-cutlass-dsl-libs-core nvidia-cutlass-dsl-libs-cu12 nvidia-cutlass-dsl-libs-cu13)
  wheel_args=()
  if [[ -n ${SM100_DSL_WHEELS:-} ]]; then wheel_args=(--no-index --find-links "$SM100_DSL_WHEELS"); fi
  "$py" -m pip download "${wheel_args[@]}" --no-deps -d "$root/wheels" "${packages[@]/%/==4.8.1}"
  (cd "$root/wheels" && sha256sum -c "$repo/docs/sm100_megamoe/v41_native_phase_results/dsl-wheelhouse.sha256")
  "$py" -m pip install --no-deps --no-index --find-links "$root/wheels" "${packages[@]/%/==4.8.1}"
  lib=$("$py" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
  blib=$("$base/bin/python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
  printf "import site; site.addsitedir('%s')\n" "$blib" > "$lib/zz_v41_common.pth"
  cp docs/sm100_megamoe/v41_full_decode_pair_results/compat/sitecustomize.py "$lib/sitecustomize.py"
  for patch in scripts/sm100/natureal_matrix_audit.patch docs/sm100_megamoe/v41_full_decode_pair_results/compat/gc_capture_fix.diff scripts/sm100/vllm_combine_order.patch scripts/sm100/ep8_linear_roles.patch; do
    git -C "$root/upstream" apply "$repo/$patch"
  done
  cp -R source/gc_moe/eplb "$root/upstream/gc_moe/"
  cp source/gc_moe/eco/meta_sm{90,100}.py "$root/upstream/gc_moe/eco/"
  git -C "$root/upstream" diff > "$root/upstream.diff"
  touch "$root/prepared"
fi
test "$(git -C "$root/upstream" rev-parse HEAD)" = "$upstream"
git -C "$root/upstream" diff | cmp - "$root/upstream.diff"
export PATH="$root/dsl-env/bin:$PATH" PYTHONPATH="$root/upstream:$repo/integration:$repo/scripts/sm100"
export SM100_EP8_LINEAR_FOUR=1 GC_SM100_MATRIX_CLOCKS=1 GC_MOE_ECO_UNIT_CFG=26,2,1,6,0
unset SM100_FORMAL_512 SM100_DIRECT_512 SM100_PHASE_TRACE SM100_FULL_DECODE SM100_FIRST_DIFF SM100_FAIR_BOUNDARY SM100_NATIVE_GREEDY
unset GC_MOE_ECO_BLOCK_M GC_MOE_ECO_OBJECTIVE GC_MOE_ECO_HELPER GC_MOE_ECO_TRACE GC_MOE_DISPATCH_SMS
unset GC_MOE_PREFIX_DIAG GC_MOE_VANILLA_STATS_REPLAY GC_MOE_PRIMARY_CTX GC_MOE_BALANCED_WORKQUEUES
unset SM90_SIX_ARM SM90_EIGHT_ARM
export GC_MOE_ECO_CACHE_DIR="$root/eco-cache" TRITON_CACHE_DIR="$root/triton" DG_JIT_CACHE_DIR="$root/dg-cache"
export TORCHINDUCTOR_CACHE_DIR="$root/inductor" VLLM_CACHE_ROOT="$root/vllm-cache" HF_HUB_OFFLINE=1
export VLLM_USE_V2_MODEL_RUNNER=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1 PYTHONHASHSEED=0
python -m pip freeze > "$root/versions.txt"
expected=$(python -c 'import json,re; print(",".join(re.findall(r"GPU-[0-9A-Fa-f-]{36}",json.load(open("docs/sm100_megamoe/ENV.json"))["commands"]["gpu_list"]["output"])))')
test "$(tr , '\n' <<< "$expected" | sort -u | wc -l)" = 8
export CUDA_VISIBLE_DEVICES="${mask:-$expected}"
command -v fuser >/dev/null
exec 9>>/tmp/gcmoe-sm100-gpu-group.lock
acquire() {
  local busy
  while true; do
    flock 9
    if busy=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader) && [[ -z $busy ]] && ! fuser /dev/nvidia{0..7} >/dev/null 2>&1; then break; fi
    flock -u 9
    sleep 60
  done
  date -Is >> "$root/acquired.log"
  nvidia-smi -L > "$root/gpus.txt"
  python - "$expected" <<'PY'
import sys,torch
want={s.lower().removeprefix("gpu-") for s in sys.argv[1].split(",")}
got={str(torch.cuda.get_device_properties(i).uuid).lower().removeprefix("gpu-") for i in range(torch.cuda.device_count())}
assert len(want)==8 and want==got,(want,got)
PY
}
failed=0
for bs in ${SM100_ONLY_BS:-256 512 1024}; do
  group="$root/bs$bs"
  if [[ -f "$group/verified.sha256" ]]; then (cd "$group" && sha256sum --quiet -c verified.sha256); continue; fi
  if [[ ! -f "$group/vanilla/result.json" ]]; then
    python - "$group" "$bs" <<'PY'
import shutil, sys
required=2*8*(32<<30) + 2*40000*int(sys.argv[2])*6*8 + (16<<30)
if shutil.disk_usage(sys.argv[1]).free < required:
    raise SystemExit(f"Need {required/(1<<30):.1f} free GiB for this BS")
PY
  fi
  complete=1
  for arm in vanilla random rr eco; do
    dest="$group/$arm"
    if [[ -f "$dest/result.json" && $(cat "$dest/exit-code" 2>/dev/null) == 0 ]]; then continue; fi
    if [[ -e "$dest" ]]; then
      echo "Preserving incomplete $dest" >&2
      complete=0
      if [[ $arm == vanilla ]]; then break; fi
      continue
    fi
    mkdir "$dest"
    acquire
    python - "$SM100_V41_MODEL" <<'PY'
import json, os, pathlib, sys
model = pathlib.Path(sys.argv[1])
for shard in sorted(set(json.loads((model / "model.safetensors.index.json").read_text())["weight_map"].values())):
    with (model / shard).open("rb") as f:
        os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_WILLNEED)
PY
    code=0
    python -u scripts/sm100/ep8_matrix.py "$group" "$arm" "$bs" > "$dest/model.log" 2>&1 || code=$?
    echo "$code" > "$dest/exit-code"
    date -Is >> "$root/released.log"
    flock -u 9
    if (( code )); then
      tail -35 "$dest/model.log"
      complete=0
      if [[ $arm == vanilla ]]; then break; fi
    fi
  done
  if (( ! complete )); then failed=1; continue; fi
  if ! python scripts/sm100/ep8_report.py "$group" > "$group/report.log" 2>&1; then failed=1; cat "$group/report.log"; continue; fi
  (cd "$group" && sha256sum *.csv */result.json */*.npz */*.npy */exit-code kv-*/rank*.json plan.npz random.npz eco-initial.npz > verified.sha256)
  if [[ ${SM100_KEEP_REFERENCE:-0} != 1 ]]; then
    # Only this successful group's reproducible reference banks and KV payloads.
    (cd "$group" && sha256sum *-routes-*.pt kv-{stats,measure}/rank*.bin > reference.sha256)
    rm -- "$group"/*-routes-*.pt "$group"/kv-{stats,measure}/rank*.bin
  else
    (cd "$group" && sha256sum *-routes-*.pt kv-{stats,measure}/rank*.bin > reference.sha256)
  fi
done
python - "$root" <<'PYREPORT'
import csv, pathlib, sys
root = pathlib.Path(sys.argv[1])
for name, expected in (("main",12), ("layers",480), ("auction",3), ("pairs",18)):
    paths = [root/f"bs{bs}"/f"{name}.csv" for bs in (256,512,1024)]
    if all(p.exists() for p in paths):
        rows = []
        for p in paths:
            with p.open() as src:
                rows.extend(csv.DictReader(src))
        assert len(rows) == expected
        with (root/f"{name}.csv").open("w", newline="") as dst:
            out = csv.DictWriter(dst, fieldnames=rows[0])
            out.writeheader()
            out.writerows(rows)
PYREPORT
exit "$failed"
