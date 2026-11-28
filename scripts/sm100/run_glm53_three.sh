#!/usr/bin/env bash
# One GLM cohort. Paths under the caller's private archive only.
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
root=$(realpath "${1:?pass an existing empty private run directory}")
if [[ -e $root/revision ]]; then echo "use an empty run directory" >&2; exit 2; fi
cd "$repo"
if [[ -f $repo/SOURCE_REVISION ]]; then
  cat "$repo/SOURCE_REVISION" > "$root/revision"
else
  git rev-parse HEAD > "$root/revision"
fi
base=${GLM_RUNTIME_PYTHON:-/data0/home/admin/workspace/glm53-work/glm53-runtime-qgmxfwva/venv/bin/python}
export GLM_MODEL=/mnt/hf3fs/3fs/models/GLM-5.3-FP4Moe-FP8ue8m0
mask=${CUDA_VISIBLE_DEVICES-${SM100_GPU_UUIDS-}}
if [[ ${CUDA_VISIBLE_DEVICES+x} && -z $mask ]]; then echo "empty allocation mask" >&2; exit 2; fi
export CUDA_VISIBLE_DEVICES=''
export PIP_DISABLE_PIP_VERSION_CHECK=1
if [[ ! -x $base && -z ${GLM_RUNTIME_PYTHON:-} ]]; then
  for prev in "$(dirname "$(dirname "$root")")"/formal.*/results/base/bin/python; do
    [[ -x $prev ]] || continue
    if "$prev" -c 'import torch,vllm; assert vllm.__version__=="0.31.0" and torch.__version__.split("+")[0]=="2.13.0" and torch.version.cuda=="13.0"' >/dev/null 2>&1; then
      base=$prev
      break
    fi
  done
  if [[ ! -x $base ]]; then
    echo "[prepare] installing pinned runtime in private storage"
    python3.12 -m venv "$root/base"
    base="$root/base/bin/python"
    "$base" -m pip install --only-binary=vllm,torch 'vllm==0.31.0' 'torch==2.13.0' 'safetensors==0.8.0'
  fi
fi
"$base" -c 'import torch,vllm,safetensors; assert vllm.__version__=="0.31.0" and torch.__version__.split("+")[0]=="2.13.0" and torch.version.cuda=="13.0" and safetensors.__version__=="0.8.0", (vllm.__version__, torch.__version__, torch.version.cuda, safetensors.__version__)'
echo "[prepare] verified runtime $base"
(cd "$GLM_MODEL" && sha256sum -c <<'DIGEST'
82bd28e0ac294868b134211748cbe27a166621928231607a5510fced5a86cb2b  config.json
fbf6cbce2835c7002fcf62a0e9bcf57016323d25083f4f9d011ac30ccc1aac0c  model.safetensors.index.json
DIGEST
)
echo "[prepare] KV snapshots retain their full logical size; zero slabs can use sparse extents"
df -h "$root"
echo "[prepare] unpacking pinned upstream GC source"
echo "67105182741957269ed1f3a321a091ea78a5ee363ba7707eb814baac8048370d  scripts/sm100/bruce-af5a7a7.tar.gz" | sha256sum -c -
mkdir "$root/upstream"
tar -xzf scripts/sm100/bruce-af5a7a7.tar.gz -C "$root/upstream"
(cd "$root/upstream" && git apply --no-index "$repo/scripts/sm100/bruce_sm100_eco.patch")
cp -R source/gc_moe/eplb "$root/upstream/gc_moe/"
cp source/gc_moe/eco/meta_sm{90,100}.py "$root/upstream/gc_moe/eco/"
blib=$("$base" -c 'import sysconfig;print(sysconfig.get_path("purelib"))')
storage=$(dirname "$(dirname "$root")")
py=''
for prev in "$storage"/formal.*/results/dsl/bin/python; do
  [[ -x $prev ]] || continue
  prev_lib=$("$prev" -c 'import sysconfig;print(sysconfig.get_path("purelib"))' 2>/dev/null) || continue
  cmp -s "$prev_lib/sitecustomize.py" docs/sm100_megamoe/v41_full_decode_pair_results/compat/sitecustomize.py || continue
  grep -Fxq "import site; site.addsitedir('$blib')" "$prev_lib/zz_glm_runtime.pth" || continue
  if "$prev" -c 'import torch, vllm, cutlass; from importlib.metadata import version; assert version("nvidia-cutlass-dsl")=="4.8.1" and vllm.__version__=="0.31.0"' >/dev/null 2>&1; then
    py=$prev
    lib=$prev_lib
    echo "[prepare] using existing GLM CuTe environment $py"
    break
  fi
done
if [[ -z $py ]]; then
  "$base" -m venv "$root/dsl"
  py="$root/dsl/bin/python"
  lib=$("$py" -c 'import sysconfig;print(sysconfig.get_path("purelib"))')
  pkgs=(nvidia-cutlass-dsl nvidia-cutlass-dsl-libs-base nvidia-cutlass-dsl-libs-core nvidia-cutlass-dsl-libs-cu12 nvidia-cutlass-dsl-libs-cu13)
  manifest="$repo/docs/sm100_megamoe/v41_native_phase_results/dsl-wheelhouse.sha256"
  wheels=""
  for candidate in "${SM100_DSL_WHEELS:-}" "$storage/dsl-wheels-4.8.1" "$storage"/formal.*/results/wheels; do
    if [[ -d $candidate ]] && (cd "$candidate" && sha256sum --status -c "$manifest"); then
      wheels=$candidate
      echo "[prepare] reusing verified wheels $candidate"
      break
    fi
  done
  if [[ -z $wheels ]]; then
    wheels="$root/wheels"
    mkdir "$wheels"
    "$py" -m pip download --no-deps -d "$wheels" "${pkgs[@]/%/==4.8.1}"
    (cd "$wheels" && sha256sum -c "$manifest")
  fi
  "$py" -m pip install --no-deps --no-index --find-links "$wheels" "${pkgs[@]/%/==4.8.1}"
  printf "import site; site.addsitedir('%s')\n" "$blib" > "$lib/zz_glm_runtime.pth"
  cp docs/sm100_megamoe/v41_full_decode_pair_results/compat/sitecustomize.py "$lib/sitecustomize.py"
fi
# DeepJIT discovers CUDA_HOME and its DG-prefixed nvcc override at import.
# GC assembles newer CuTe PTX separately from C++ compilation.
export CUDA_HOME=/usr/local/cuda
export CUDA_PATH="$CUDA_HOME"
export DG_JIT_NVCC_COMPILER="$CUDA_HOME/bin/nvcc"
export GC_MOE_NVCC="$DG_JIT_NVCC_COMPILER"
export PATH="$CUDA_HOME/bin:$(dirname "$py"):$lib/nvidia/cu13/bin:$blib/nvidia/cu13/bin:$PATH"
"$DG_JIT_NVCC_COMPILER" --version
"$py" - <<'HEADERS'
import sysconfig
from pathlib import Path
includes = [Path(sysconfig.get_path(key)) / "Python.h" for key in ("include", "platinclude")]
if not any(path.is_file() for path in includes):
    raise SystemExit(f"Inductor needs the Python development header: {includes}")
HEADERS
ptx="$repo/scripts/sm100/glm53_probe.ptx"
GC_MOE_PTXAS=""
for candidate in "$lib/nvidia/cu13/bin/ptxas" "$blib/nvidia/cu13/bin/ptxas" "$CUDA_HOME/bin/ptxas"; do
  [[ -x $candidate ]] || continue
  version=$("$candidate" --version) || continue
  [[ $version =~ release[[:space:]]+([0-9]+)\.([0-9]+) ]] || continue
  (( BASH_REMATCH[1] > 13 || (BASH_REMATCH[1] == 13 && BASH_REMATCH[2] >= 2) )) || continue
  if "$candidate" -arch=sm_103f "$ptx" -o "$root/sm103f.cubin"; then
    GC_MOE_PTXAS=$candidate
    break
  fi
done
if [[ -z $GC_MOE_PTXAS ]]; then
  "$py" -m pip install --no-deps --target "$root/assembler" 'nvidia-cuda-nvcc==13.5.37'
  GC_MOE_PTXAS="$root/assembler/nvidia/cu13/bin/ptxas"
  "$GC_MOE_PTXAS" -arch=sm_103f "$ptx" -o "$root/sm103f.cubin"
fi
export GC_MOE_PTXAS
echo "[prepare] DeepGEMM nvcc=$DG_JIT_NVCC_COMPILER GC ptxas=$GC_MOE_PTXAS"
"$GC_MOE_PTXAS" --version
export PYTHONPATH="$root/upstream:$repo/scripts/sm100:$repo/integration"
if ! "$py" -c 'from cuda.bindings import driver as cu; assert hasattr(cu.CUdevWorkqueueConfigScope, "CU_WORKQUEUE_SCOPE_GREEN_CTX_BALANCED")' >/dev/null 2>&1; then
  "$py" -m pip install --ignore-installed 'cuda-bindings==13.2.0' 'cuda-python==13.2.0'
fi
"$py" -c 'import gc_moe; from gc_moe.eco.meta_sm100 import build_eco_meta_sm100; from gc_moe.eco.measurement import snapshot'
export GC_MOE_SM100_TRACE=2 GC_MOE_ECO_PLAN_DISPATCH=1 GC_MOE_SM100_TRACE_EVENTS=65536
unset GC_MOE_ECO_UNIT_CFG GC_MOE_ECO_OBJECTIVE GC_MOE_ECO_BLOCK_M GC_MOE_ECO_HELPER GC_MOE_DISPATCH_SMS
unset GC_MOE_PREFIX_DIAG GC_MOE_VANILLA_STATS_REPLAY SM90_SIX_ARM SM90_EIGHT_ARM
export GC_MOE_ECO_CACHE_DIR="$root/eco-cache" TRITON_CACHE_DIR="$root/triton" VLLM_CACHE_ROOT="$root/vllm-cache"
export DG_JIT_CACHE_DIR="$VLLM_CACHE_ROOT/deep_gemm"
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
    if ! flock -n 9; then
      echo "$(date -Is) waiting for shared group lock $lock"
      sleep 60
      continue
    fi
    local procs util handles="" busy=0
    procs=$(nvidia-smi -i "$grant" --query-compute-apps=pid --format=csv,noheader,nounits) || busy=1
    util=$(nvidia-smi -i "$grant" --query-gpu=utilization.gpu --format=csv,noheader,nounits) || busy=1
    [[ -z $procs ]] || busy=1
    [[ $(awk '$1=="0"{n++} END{print n+0}' <<< "$util") == 8 ]] || busy=1
    if handles=$(fuser "${devs[@]}" 2>/dev/null); then busy=1; fi
    if ((busy == 0)); then return; fi
    flock -u 9
    echo "$(date -Is) waiting on eight GPUs: compute_pids=[$procs] utilization=[$util] device_handles=[$handles]"
    sleep 60
  done
}
for arm in vanilla rr eco; do
  echo "$(date -Is) $arm checkpoint readahead"
  "$base" - "$GLM_MODEL" <<'WARM'
import errno, json, os, sys
from pathlib import Path
root = Path(sys.argv[1])
files = sorted(set(json.loads((root/"model.safetensors.index.json").read_text())["weight_map"].values()))
queued = 0
for name in files:
    with (root/name).open("rb") as f:
        try:
            os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_WILLNEED)
        except OSError as e:
            if e.errno not in (errno.ENOSYS, errno.EOPNOTSUPP, errno.EINVAL):
                raise
            print(f"filesystem declined readahead ({e}); proceeding with local expert mmap")
            break
        queued += 1
print(f"queued {queued}/{len(files)} shards; vLLM EP weight filter enabled")
WARM
  acquire
  export CUDA_VISIBLE_DEVICES="$grant"
  if [[ $arm == vanilla ]]; then
    DG_JIT_PRINT_COMPILER_COMMAND=1 "$py" -u "$repo/scripts/sm100/glm53_probe.py" "$root"
  fi
  echo "$(date -Is) starting $arm $grant"
  "$py" -u scripts/sm100/glm53_matrix.py "$arm" "$root"
  export CUDA_VISIBLE_DEVICES=''
  flock -u 9
  echo "$(date -Is) released $arm"
  if [[ $arm == vanilla ]]; then "$py" scripts/sm100/glm53_matrix.py plan "$root"; fi
done
"$py" scripts/sm100/glm53_matrix.py report "$root"
