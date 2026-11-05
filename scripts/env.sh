#!/bin/bash
# Runtime environment for both arms.
# NOTE: do NOT put DEEPEP_BUILD on PYTHONPATH. That tree contains a standalone
# `deep_gemm` package that imports fine but lacks the APIs vLLM's kernels need;
# once it wins the import, vLLM reports "DeepGEMM backend is unavailable" only
# much later (inside profile_run). The working configuration lets vLLM fall
# back to its vendored `vllm.third_party.deep_gemm` (log line: "deep_gemm not
# found in site-packages, trying vendored ...").
W=/home/admin/workspace/aop_lab/app_source/gc_moe_rtp_ep8_20261031
export GC_ENV=$W/env3
export GC_SHIM=$W/shim
TOOLCHAIN=/home/admin/workspace/aop_lab/app_source/deep_ep25_ar_probe_20261025/toolchain
export TOOLCHAIN
export NCCL_ROOT=$GC_ENV/lib/python3.10/site-packages/nvidia/nccl
export NVSHMEM_ROOT=$GC_ENV/lib/python3.10/site-packages/nvidia/nvshmem

export LD_PRELOAD=$TOOLCHAIN/lib/libstdc++.so.6:$TOOLCHAIN/lib/libgcc_s.so.1
export LD_LIBRARY_PATH=$NCCL_ROOT/lib:$TOOLCHAIN/lib:/usr/local/cuda/lib64:$GC_ENV/lib/python3.10/site-packages/torch/lib
# [why] GC_SHIM also contains a *stub* `deep_gemm` package (only the pure-torch
# quantisation helpers). If it is on PYTHONPATH it shadows
# vllm.third_party.deep_gemm, so vllm's `_lazy_init()` resolves every GEMM
# symbol to None and the run dies in profile_run with "DeepGEMM backend is
# unavailable". The GC MoE operator needs `deep_gemm.utils` (per_token_cast_*),
# so the driver injects the shim at the END of sys.path itself.
export PYTHONPATH=

# Cluster-validated NVSHMEM/IB settings.
export NVSHMEM_SYMMETRIC_SIZE=1073741824 NVSHMEM_IB_ENABLE_IBGDA=1 NVSHMEM_IBGDA_NIC_HANDLER=cpu
export NVSHMEM_IBGDA_HOST_SIDEWING=0 NVSHMEM_IBGDA_NUM_RC_PER_PE=17 NVSHMEM_QP_DEPTH=1024
export NVSHMEM_MAX_TEAMS=7 NVSHMEM_DISABLE_NVLS=1 NVSHMEM_IB_GID_INDEX=3 NVSHMEM_DISABLE_P2P=0
export NVSHMEM_CUMEM_GRANULARITY=536870912 NVSHMEM_FCOLLECT_LL_THRESHOLD=4096
export NCCL_IB_DISABLE=0 NCCL_IB_GID_INDEX=3 NCCL_NET=IB
export NCCL_SOCKET_IFNAME=eth0 GLOO_SOCKET_IFNAME=eth0
export NCCL_IB_HCA='mlx5_2:1,mlx5_8:1,mlx5_20:1,mlx5_24:1'
export NVSHMEM_HCA_LIST='mlx5_2:1,mlx5_8:1,mlx5_20:1,mlx5_24:1'
export OMP_NUM_THREADS=8
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export PYTHONFAULTHANDLER=1
