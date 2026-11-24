# Sourced before allocation from run_eight.sh; exports only the DeepJIT compiler selection.
export SM90_MEGA_PROBE_DIR
SM90_MEGA_PROBE_DIR=$(mktemp -d "$out/mega-jit-XXXXXXXX")
export SM90_MEGA_NVCC=${SM90_MEGA_NVCC:-/usr/local/cuda/bin/nvcc}
export DG_JIT_NVCC_COMPILER="$launcher_dir/mega_nvcc.sh"
export DG_JIT_CACHE_DIR="$SM90_MEGA_PROBE_DIR/cache"
"$SM90_MEGA_NVCC" --version > "$SM90_MEGA_PROBE_DIR/nvcc.txt"
hosts=("${SM90_MEGA_HOST_CXX:-}" "$TOOLCHAIN/bin/x86_64-conda-linux-gnu-g++")
for version in 14 13 12 11; do
  hosts+=("/usr/bin/g++-$version" "/opt/rh/gcc-toolset-$version/root/usr/bin/g++")
done
hosts+=("/usr/bin/g++")
found=0
for host in "${hosts[@]}"; do
  [[ -x $host ]] || continue
  export SM90_MEGA_HOST_CXX="$host"
  attempt=$(mktemp -d "$SM90_MEGA_PROBE_DIR/compiler-XXXXXXXX")
  echo "$host" > "$attempt/cxx.txt"
  "$host" --version >> "$attempt/cxx.txt"
  for phase in 1 2; do
    if ! CUDA_VISIBLE_DEVICES='' "$DG_JIT_NVCC_COMPILER" "$launcher_dir/mega_jit_probe.cu" \
        --cubin -arch=sm_90a -std=c++20 -O3 -Xcompiler=-fPIC,-fconcepts \
        --expt-relaxed-constexpr --expt-extended-lambda --ptxas-options=--register-usage-level=10 \
        -I"$SM90_DG_SITE/deep_gemm/include" "-DMEGA_PHASE=$phase" -o "$attempt/l$phase.cubin" \
        > "$attempt/l$phase.log" 2>&1; then break; fi
    if [[ $phase == 2 ]]; then found=1; fi
  done
  if (( found )); then
    echo "$host" > "$SM90_MEGA_PROBE_DIR/selected-cxx"
    echo "Mega JIT preflight passed: $SM90_MEGA_NVCC with $host"
    break
  fi
done
if (( ! found )); then
  echo "Both phases must compile before allocation; see $SM90_MEGA_PROBE_DIR" >&2
  exit 1
fi
