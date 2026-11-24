#!/usr/bin/env bash
# Selected through private DeepJIT's DG_JIT_NVCC_COMPILER.
set -eu
for arg in "$@"; do
  if [[ $arg == *.cu && -f $arg ]] && grep -Fq '<deep_gemm/impls/sm90_fp8_mega_moe.cuh>' "$arg"; then
    if "$SM90_MEGA_NVCC" -ccbin "$SM90_MEGA_HOST_CXX" "$@"; then exit 0; else code=$?; fi
    cp -- "$arg" "$(mktemp "$SM90_MEGA_PROBE_DIR/failed-kernel-XXXXXXXX.cu")"
    exit "$code"
  fi
done
exec "$SM90_MEGA_NVCC" "$@"
