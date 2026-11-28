# SM100 GLM-5.3 EP8 三臂 · 失败回执（formal.20261128193913）

- 状态: **0/3 arms**，exit 1；GPU 已全部释放（八卡 utilization 0%）
- 节点: e01-cn-il24rbk1805，8× NVIDIA L20D，torch 报 capability 10.3，sm_103f 探针八卡全通过（见 arch-proof.json，observed=259=0x103）
- 任务书: 12f21e5937d9 `docs/sm100_megamoe/SM100_GLM53_FORMAL.md`
- 执行包: 444969ae13dd `glm53_executor.tar.gz`（Contents sha 8e08c43d…，sha256 43d9ff69… 校验通过）
- 私有目录: /data0/home/admin/workspace/glm53-work/formal.20261128193913

## 三臂进度

| 臂 | 结果 |
|---|---|
| vanilla | 失败：加载权重后 `process_weights_after_loading` → deep_gemm UE8M0 scale 布局 JIT 编译失败 |
| rr | 未启动（前一臂失败即停） |
| eco | 未启动 |

## 首个错误（保留原文）

```
deep_jit command failed (exit 1):
nvcc .../kernel.cu --cubin --gpu-architecture=sm_103f ...
<venv>/nvidia/cu13/include/cccl/cuda/std/__cccl/cuda_toolkit.h:41:8:
error: #error "CUDA compiler and CUDA toolkit headers are incompatible, please check your include paths"
```

调用栈: `fp8.py process_weights_after_loading` → `deepgemm_post_process_fp8_weight_block` → `transform_sf_into_required_layout` → `get_mn_major_tma_aligned_packed_ue8m0_tensor`。

## 根因

venv `/data0/home/admin/workspace/glm53-work/glm53-runtime-qgmxfwva/venv` 内部 CUDA 组件版本不一致：

- `nvidia-cuda-nvcc` **13.5.37**（deep_gemm 用 `CUDA_HOME/bin/nvcc` 找到的编译器）
- `nvidia-cuda-runtime` **13.0.96** / CUDART_VERSION **13000** / `nvidia-cuda-crt` 13.5.37
- `nvidia-cuda-cccl` 13.3.5.0.6

CCCL 的 `cuda_toolkit.h` 要求编译器与 toolkit 头同版本，13.5 编译器 × 13.0 头 → `#error`。

## 已验证的最小修复（未改仓库代码、未改语义）

用与头文件同版的系统 `nvcc 13.0.48`（`/usr/local/cuda/bin/nvcc`）编译同一 deep_gemm 内核可成功。已实测：
`CUDA_VISIBLE_DEVICES=0 JIT_NVCC_COMPILER=/usr/local/cuda/bin/nvcc python -c "dg.get_mn_major_tma_aligned_packed_ue8m0_tensor(...)"` → 正常返回。

但整机执行时需要 deep_gemm 的 `CUDA_HOME` 指向 `/usr/local/cuda`（或把 venv 内 nvcc 降回 13.0.88），否则 deep_gemm 仍用 venv 的 13.5 nvcc。建议构建者二选一：
1. `export CUDA_HOME=/usr/local/cuda`（推荐，零安装）；
2. `pip install nvidia-cuda-nvcc==13.0.88 nvidia-cuda-crt==13.0.88` 使 venv 内版本对齐。

## 其它环境处置（本轮已做，属环境非语义）

- 安装系统 `python3.12-devel`（补齐 `Python.h`，Inductor 编译用）
- 预取并校验 `nvidia-cutlass-dsl 4.8.1` 五件 wheel 到 `glm53-work/dsl-wheels-4.8.1`（sha256 全对），`SM100_DSL_WHEELS` 复用后省去慢速 pypi 下载

## 前序失败回执（同一根因链）

- formal.20261128192017：`Python.h` 缺失（已修）
- formal.20261128193159：deep_gemm nvcc/toolkit 不匹配（首次暴露）

## 交接

GPU 已空闲，可立即复跑。修好 `CUDA_HOME` 后重跑同一启动命令即可，无需改动任务书任何 BS/步数/放置/量化/trace 配置。
