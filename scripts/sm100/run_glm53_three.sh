#!/usr/bin/env bash
# One GLM cohort. Paths under the caller's private archive only.
set -euo pipefail
repo=$(git rev-parse --show-toplevel)
root=$(realpath "${1:?pass an existing empty private run directory}")
if [[ -e $root/revision ]]; then echo "use an empty run directory" >&2; exit 2; fi
cd "$repo"
git rev-parse HEAD > "$root/revision"
base=${GLM_RUNTIME_PYTHON:-/data0/home/admin/workspace/glm53-work/glm53-runtime-qgmxfwva/venv/bin/python}
export GLM_MODEL=/mnt/hf3fs/3fs/models/GLM-5.3-FP4Moe-FP8ue8m0
mask=${CUDA_VISIBLE_DEVICES-${SM100_GPU_UUIDS-}}
if [[ ${CUDA_VISIBLE_DEVICES+x} && -z $mask ]]; then echo "empty allocation mask" >&2; exit 2; fi
export CUDA_VISIBLE_DEVICES=''
test -x "$base"
(cd "$GLM_MODEL" && sha256sum -c <<'DIGEST'
82bd28e0ac294868b134211748cbe27a166621928231607a5510fced5a86cb2b  config.json
fbf6cbce2835c7002fcf62a0e9bcf57016323d25083f4f9d011ac30ccc1aac0c  model.safetensors.index.json
DIGEST
)
"$base" - "$root" <<'PY'
import shutil, sys
# Two independent fixed-address 48 GiB per-rank KV boundary images, route
# banks, small results and compressed Kineto artifacts.
if shutil.disk_usage(sys.argv[1]).free < 800*(1<<30):
    raise SystemExit("insufficient disk for two eight-rank KV snapshots; choose a private volume with 800 GiB free")
PY
git clone --quiet https://github.com/Bruce-Lee-LY/gc-moe.git "$root/upstream"
git -C "$root/upstream" checkout --quiet --detach af5a7a750800954f52b9ff28cc55d945ead48da6
git -C "$root/upstream" apply "$repo/scripts/sm100/bruce_sm100_eco.patch"
cp -R source/gc_moe/eplb "$root/upstream/gc_moe/"
cp source/gc_moe/eco/meta_sm{90,100}.py "$root/upstream/gc_moe/eco/"
"$base" -m venv "$root/dsl"
py="$root/dsl/bin/python"
lib=$("$py" -c 'import sysconfig;print(sysconfig.get_path("purelib"))')
blib=$("$base" -c 'import sysconfig;print(sysconfig.get_path("purelib"))')
pkgs=(nvidia-cutlass-dsl nvidia-cutlass-dsl-libs-base nvidia-cutlass-dsl-libs-core nvidia-cutlass-dsl-libs-cu12 nvidia-cutlass-dsl-libs-cu13)
mkdir "$root/wheels"
"$py" -m pip download --no-deps -d "$root/wheels" "${pkgs[@]/%/==4.8.1}"
(cd "$root/wheels" && sha256sum -c "$repo/docs/sm100_megamoe/v41_native_phase_results/dsl-wheelhouse.sha256")
"$py" -m pip install --no-deps --no-index --find-links "$root/wheels" "${pkgs[@]/%/==4.8.1}"
printf "import site; site.addsitedir('%s')\n" "$blib" > "$lib/zz_glm_runtime.pth"
cp docs/sm100_megamoe/v41_full_decode_pair_results/compat/sitecustomize.py "$lib/sitecustomize.py"
export PATH="$root/dsl/bin:$blib/nvidia/cu13/bin:/usr/local/cuda/bin:$PATH"
if ! command -v nvcc >/dev/null; then
  "$py" -m pip install 'nvidia-cuda-nvcc>=13,<14'
  export PATH="$lib/nvidia/cu13/bin:$PATH"
fi
export GC_MOE_PTXAS="$(command -v ptxas)" GC_MOE_NVCC="$(command -v nvcc)"
test -x "$GC_MOE_PTXAS" && test -x "$GC_MOE_NVCC"
export PYTHONPATH="$root/upstream:$repo/scripts/sm100:$repo/integration"
if ! "$py" -c 'from cuda.bindings import driver as cu; assert hasattr(cu.CUdevWorkqueueConfigScope, "CU_WORKQUEUE_SCOPE_GREEN_CTX_BALANCED")' >/dev/null 2>&1; then
  "$py" -m pip install --ignore-installed 'cuda-bindings==13.2.0' 'cuda-python==13.2.0'
fi
"$py" -c 'import gc_moe; from gc_moe.eco.meta_sm100 import build_eco_meta_sm100; from gc_moe.eco.measurement import snapshot'
export GC_MOE_SM100_TRACE=2 GC_MOE_ECO_PLAN_DISPATCH=1 GC_MOE_SM100_TRACE_EVENTS=65536
unset GC_MOE_ECO_UNIT_CFG GC_MOE_ECO_OBJECTIVE GC_MOE_ECO_BLOCK_M GC_MOE_ECO_HELPER GC_MOE_DISPATCH_SMS
unset GC_MOE_PREFIX_DIAG GC_MOE_VANILLA_STATS_REPLAY SM90_SIX_ARM SM90_EIGHT_ARM
export GC_MOE_ECO_CACHE_DIR="$root/eco-cache" TRITON_CACHE_DIR="$root/triton" VLLM_CACHE_ROOT="$root/vllm-cache"
export HF_HUB_OFFLINE=1 TORCHINDUCTOR_CACHE_DIR="$root/inductor" PYTHONHASHSEED=0
export VLLM_USE_V2_MODEL_RUNNER=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1
"$py" scripts/sm100/glm53_matrix.py prepare "$root"
# Resolve only the assigned mask; a dedicated 8-device grant may expose all eight.
if [[ -n $mask ]]; then
  # CUDA ordinals can differ from nvidia-smi indices. Resolve the original
  # allocation in the driver's device order without retaining a CUDA context.
  grant_ids=$(CUDA_VISIBLE_DEVICES="$mask" "$base" - <<'ENUM'
import ctypes as c
import uuid
cu=c.CDLL("libcuda.so.1")
def check(status):
    if status:
        raise RuntimeError(f"CUDA device enumeration failed: {status}")
check(cu.cuInit(0))
count=c.c_int()
check(cu.cuDeviceGetCount(c.byref(count)))
if count.value!=8:
    raise SystemExit("expected exactly eight devices in the authorized mask")
ids=[]
for ordinal in range(count.value):
    dev=c.c_int()
    check(cu.cuDeviceGet(c.byref(dev),ordinal))
    buf=(c.c_ubyte*16)()
    check(cu.cuDeviceGetUuid_v2(c.byref(buf),dev))
    ids.append("GPU-"+str(uuid.UUID(bytes=bytes(buf))))
print(",".join(ids))
ENUM
  )
  gpu_args=(-i "$grant_ids")
else
  gpu_args=()
fi
list=$(nvidia-smi "${gpu_args[@]}" --query-gpu=index,uuid --format=csv,noheader,nounits)
mapfile -t ids < <(awk -F', ' '{print $2}' <<< "$list")
mapfile -t devs < <(awk -F', ' '{print "/dev/nvidia"$1}' <<< "$list")
[[ ${#ids[@]} == 8 && $(printf '%s\n' "${ids[@]}" | sort -u | wc -l) == 8 ]]
grant=$(IFS=,; echo "${ids[*]}")
command -v fuser >/dev/null
lock=/tmp/gcmoe-sm100-gpu-group.lock
if [[ ! -e $lock ]]; then (umask 022; set -C; : > "$lock") 2>/dev/null || test -e "$lock"; fi
exec 9<"$lock"
acquire() {
  while true; do
    flock 9
    local procs util busy=0
    procs=$(nvidia-smi -i "$grant" --query-compute-apps=pid --format=csv,noheader,nounits) || busy=1
    util=$(nvidia-smi -i "$grant" --query-gpu=utilization.gpu --format=csv,noheader,nounits) || busy=1
    [[ -z $procs ]] || busy=1
    [[ $(awk '$1=="0"{n++} END{print n+0}' <<< "$util") == 8 ]] || busy=1
    if fuser "${devs[@]}" >/dev/null 2>&1; then busy=1; fi
    if ((busy == 0)); then return; fi
    flock -u 9
    echo "$(date -Is) eight-GPU group busy, waiting"
    sleep 60
  done
}
for arm in vanilla rr eco; do
  acquire
  export CUDA_VISIBLE_DEVICES="$grant"
  echo "$(date -Is) starting $arm $grant"
  "$py" -u scripts/sm100/glm53_matrix.py "$arm" "$root"
  export CUDA_VISIBLE_DEVICES=''
  flock -u 9
  echo "$(date -Is) released $arm"
  if [[ $arm == vanilla ]]; then "$py" scripts/sm100/glm53_matrix.py plan "$root"; fi
done
"$py" scripts/sm100/glm53_matrix.py report "$root"
