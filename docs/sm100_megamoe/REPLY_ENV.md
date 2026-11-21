# 【SM100】MegaMoE 环境回执

> 执行者身份：**SM100 执行者**。本回执只对应 SM100 任务线；没有执行 SM90 任务书，也没有把 SM90 结果混入本回执。
>
> 探针执行仓库：`ehdbdns/gcmoe-sm90`，探针源码 SHA = `c6c970122fac461d8ab1b1b41b08f5c3dfa22b15`；回执写回前 rebase 到随后最新的 SM90-only main 提交。
>
> 执行时间：2026-11-21 14:38--14:40（Asia/Shanghai）。探针退出码 0，`ENV.json` 通过 `json.tool` 校验。

## 1. 分配与释放

- 机器：`e01-cn-v8n4s5sfd01`（`mega-a`），`11.163.39.114`，单机 8 卡。
- 访问方式：从现有 Codex/平台终端进入；当前 shell 已是目标机器，不需要额外容器启动。
- 调度与预约：没有发现 Slurm、Kubernetes、`srun`、`sbatch` 等作业调度器，也没有发现独立的 GPU 预约命令或服务。`nvidia-smi` 的 MPS 进程均为 defunct。
- 本轮使用本地 advisory `flock`（`/tmp/gcmoe-sm100-gpu-group.lock`）整组登记。它不是硬件隔离机制，不能阻止其他用户绕过锁直接启动任务。
- 启动前检查：`nvidia-smi --query-compute-apps=gpu_uuid,pid` 为空；`fuser /dev/nvidia0..7` 无输出；未发现他人 CUDA 计算进程。
- 整组 UUID（两次探针相同，顺序对应物理 GPU 0--7）：

```text
GPU-3aad6312-0e20-d01b-4249-71cb090c2127
GPU-b0ac7f5d-738f-cdad-423f-456f4c26947e
GPU-c5a93c78-be0c-2648-564a-4b4666671960
GPU-77463de9-85e0-a6f3-2c7e-f1147d6f7f49
GPU-4a493131-84b9-4c88-18ee-50fc6f213a36
GPU-045e62f4-d04d-2f9e-e456-1fb82810029f
GPU-6b44066c-6712-30f2-02d0-fa0cb91733a3
GPU-31a509fd-e741-5ab8-b837-443ad986c404
```

- 正式探针占用：`2026-11-21T14:38:33+08:00` 开始，`14:38:45` 释放；因补齐跨用户 Git 元数据重跑一次，`14:39:29` 开始，`14:39:40` 释放。两次均在启动 CUDA 前设置 `CUDA_VISIBLE_DEVICES` 为上述 8 个 UUID，结束后释放 advisory lock。
- 完整性说明：在第一次正式占用之前，环境发现阶段曾用当前 venv 做过一次 `torch.cuda` 设备属性只读查询，当时尚未显式设置 `CUDA_VISIBLE_DEVICES`。该查询只读取 8 张空闲卡的属性，没有分配显存、没有启动算子或模型。之后所有正式探针均按上述 UUID 整组绑定执行。

## 2. 运行入口

- Python：`/data0/home/admin/workspace/megamoe/venv/bin/python`，Python 3.12.7。
- CUDA toolkit：`/usr/local/cuda -> /usr/local/cuda-13.0`；`nvcc`/`ptxas` 均为 CUDA 13.0，V13.0.48。
- 编译器：GCC 13.3.1；CMake 3.26.5；Ninja 1.8.2。
- Torch：`2.14.1+cu130`，`torch.version.cuda=13.0`，`_GLIBCXX_USE_CXX11_ABI=True`，NCCL `2.30.7`。
- Torch 编译架构列表：`sm_75, sm_80, sm_86, sm_90, sm_100, sm_120`。系统 `nvcc --list-gpu-arch` 支持 `sm_100/sm_103/sm_110/sm_120/sm_121`，但 Torch 自身 arch 列表没有列出 `sm_103`；实机 capability 仍由驱动报告为 10.3。
- 模型：`/data0/data3/DeepSeekV4-Flash`，可读。`config.json`：DeepSeek-V4、43 层、hidden 4096、256 routed experts、top-k 6、1 shared expert、MoE intermediate 2048、BF16，权重 FP8 per-block `[128,128]`，activation scheme dynamic，scale ue8m0。
- 可写工作盘：`/data0/home/admin/workspace/megamoe`，约 5.75 TiB 可用；`/tmp` 在根盘上，约 125 GiB 可用。
- 网络：本机 IPv4 `11.163.39.114/24`，16 个 `reth*` 端口；`ibv_devices` 可见 12 个 HCA（`mlx5_0/2/3/4` 与 `mlx5_bond_0..7`），RDMA link 均为 ACTIVE/LINK_UP。GitHub API 与 Git 拉取可用；`raw.githubusercontent.com` 在本次环境中出现 TLS 断连，因此任务书改用 GitHub API 获取。

### 已跑过的命令

当前 admin 环境没有可确认的端到端模型推理命令。可确认实际跑过的是 2026-10-28 的单机 8 卡 DeepGEMM MegaMoE 合成 workload 测试，日志 `/home/admin/workspace/megamoe/megamoe_test.log`，`TEST_EXIT=0`：

```bash
source /home/admin/workspace/megamoe/env.sh
cd /home/admin/workspace/megamoe/DeepGEMM
export PYTHONPATH=/home/admin/workspace/megamoe/DeepGEMM:$PYTHONPATH
export MASTER_ADDR=127.0.0.1 MASTER_PORT=8361
python tests/test_mega_moe.py
```

该测试使用 8192 token/rank、hidden 7168、intermediate 3072、384 experts、top-k 6、fp8xfp4，不是 DeepSeek-V4-Flash 的端到端推理，也没有加载模型权重。日志明确显示 `No module named 'deep_ep'`，所以 DeepEP baseline 被跳过。

RTP-LLM 源码存在于其他用户工作区，但本轮没有启动服务，也没有得到可由 admin 环境复现的端到端命令。

## 3. 探针

工作树由最新 `origin/main` 独立创建，未修改现有实验 checkout：

```bash
repo=/data0/home/admin/workspace/gcmoe-sm90
git -C "$repo" fetch origin main
work=$(mktemp -d /tmp/sm100-probe.XXXXXX)
git -C "$repo" worktree add --detach "$work/code" origin/main
cd "$work/code"

export CUDA_VISIBLE_DEVICES=GPU-3aad6312-0e20-d01b-4249-71cb090c2127,GPU-b0ac7f5d-738f-cdad-423f-456f4c26947e,GPU-c5a93c78-be0c-2648-564a-4b4666671960,GPU-77463de9-85e0-a6f3-2c7e-f1147d6f7f49,GPU-4a493131-84b9-4c88-18ee-50fc6f213a36,GPU-045e62f4-d04d-2f9e-e456-1fb82810029f,GPU-6b44066c-6712-30f2-02d0-fa0cb91733a3,GPU-31a509fd-e741-5ab8-b837-443ad986c404
source /home/admin/workspace/megamoe/env.sh

GIT_CONFIG_COUNT=4 \
GIT_CONFIG_KEY_0=safe.directory GIT_CONFIG_VALUE_0=/data0/data3/wjh528431/work/RTP-LLM \
GIT_CONFIG_KEY_1=safe.directory GIT_CONFIG_VALUE_1=/data0/brucelee.ly/open_source/gc-moe \
GIT_CONFIG_KEY_2=safe.directory GIT_CONFIG_VALUE_2=/data0/home/nanjun.cp/gc-moe \
GIT_CONFIG_KEY_3=safe.directory GIT_CONFIG_VALUE_3=/data0/tianpeng.wtp/workspace/gc-moe \
python docs/sm100_megamoe/probe.py \
  --repo "$PWD" \
  --repo /home/admin/workspace/megamoe/DeepGEMM \
  --repo /data0/data3/wjh528431/work/RTP-LLM \
  --repo /data0/brucelee.ly/open_source/gc-moe \
  --repo /data0/home/nanjun.cp/gc-moe \
  --model /data0/data3/DeepSeekV4-Flash \
  --scratch /home/admin/workspace/megamoe \
  > docs/sm100_megamoe/ENV.json
python -m json.tool docs/sm100_megamoe/ENV.json > /dev/null
```

- 探针执行 SHA：`c6c970122fac461d8ab1b1b41b08f5c3dfa22b15`。
- `ENV.json` SHA-256：`dd0e2592b94dddd0d6b4d4d8bc6e1f8f60efd34c6a37b178324bca8ecd1c9d15`。
- 探针退出码：0。
- 未提供 `--dataset`：当前没有选定本轮数据集文件；不伪造路径。

### 探针结果要点

| 项目 | 结果 |
|---|---|
| `nvidia-smi` 型号/CC | `NVIDIA L20D`, `compute_cap=8.9`（伪装值） |
| `torch` 实际 capability | 8 卡全部 `10.3` |
| 单卡显存 | Torch 报告 `287,428,771,840` bytes（约 267.7 GiB）；`nvidia-smi` 报告 275040 MiB |
| 驱动 | 580.105.08；CUDA 13.0 |
| MIG | Disabled |
| GPU peer | 8x8 全部可互相访问 |
| GPU 拓扑 | 8 卡彼此 `NV18`；分属两个 NUMA domain（GPU 0--3 NUMA 0，GPU 4--7 NUMA 1） |
| Torch/NCCL | Torch 2.14.1+cu130；NCCL 2.30.7 |
| NVSHMEM | 已安装 `nvidia-nvshmem-cu13 3.4.5` 运行时 wheel；没有可导入的顶层 `nvshmem` 模块，transport 未验证 |
| CuTe DSL | 当前 admin venv 未安装 `nvidia-cutlass-dsl` |
| DeepEP | 当前 admin venv 未安装 `deep_ep` |
| 推理框架 | 当前 admin venv 未安装 vLLM、SGLang、FlashInfer |
| MegaMoE Python 模块 | 当前 admin venv 未安装；DeepGEMM 仅以源码 + 本地 `_C` 扩展形式存在 |
| GCMoE Python 模块 | 当前 admin venv 未安装 |
| RDMA | 12 个 HCA，RDMA link ACTIVE/LINK_UP |

## 4. 源码与安装包

### MegaMoE：实际实现是 DeepGEMM

- 仓库：`/data0/home/admin/workspace/megamoe/DeepGEMM`，公开上游 `https://github.com/deepseek-ai/DeepGEMM.git`。
- commit：`5081f8513af57e61696de4f4f04b0e707ffb503e`（2026-10-20，`Fix mega_mhc split barriers under CUDA graph replay (#478)`），工作树干净。
- 不是独立 wheel 安装；当前 venv 没有 `deep-gemm` 发行包元数据。源码目录有本地编译的 `deep_gemm/_C.cpython-312-x86_64-linux-gnu.so`，指向 `build/lib.linux-x86_64-cpython-312/deep_gemm/_C...so`。
- Python 入口在 `deep_gemm/__init__.py` 与 `deep_gemm/mega/__init__.py`：
  - `deep_gemm.get_symm_buffer_for_mega_moe(group, num_experts, num_max_tokens_per_rank, num_topk, hidden, intermediate_hidden, num_shared_experts=0, use_fp8_dispatch=None, mma_type='fp8xfp4', activation='swiglu')`
  - `deep_gemm.transform_weights_for_mega_moe(l1_weights, l2_weights, activation='swiglu')`
  - `deep_gemm.fp8_fp4_mega_moe(y, l1_weights, l2_weights, sym_buffer, shared_l1_weights=None, shared_l2_weights=None, cumulative_local_expert_recv_stats=None, recipe=(1,1,32), activation='swiglu', activation_clamp=None, fast_math=True)`
  - `deep_gemm.bf16_mega_moe(...)`，参数同 BF16 routed 路径。
- 支持的 `mma_type`：`fp8xfp4`、`fp8xfp8`、`bf16xbf16`；激活只支持 `swiglu`。
- 测试入口：`tests/test_mega_moe.py`，默认 `--num-processes 8`、`--num-experts 384`、`--num-topk 6`、`--hidden 7168`、`--intermediate-hidden 3072`、`--num-max-tokens-per-rank 8192`。它通过 `torch.multiprocessing.spawn` 启动单机多 rank，是现有 8 卡用例。
- dispatch/combine 入口：MegaMoE 把 EP dispatch、L1/L2 GEMM、SwiGLU 和 combine 融合在 `fp8_fp4_mega_moe`/`bf16_mega_moe` 内核里；DeepEP 的 `dispatch()`/`combine()` 只用于可选 legacy baseline。
- 图捕获：测试的基准计时使用 `tilelang_bench(... backend='cudagraph')`；没有发现模型级 CUDA Graph 捕获验证。`--dump-profile-traces` 是性能 trace，不是正确性验证。
- 正确性基准：只有安装 `deep_ep` 时才会与 DeepEP legacy 路径逐元素比较；当前环境没有 `deep_ep`，因此现有日志没有完成这项比较。
- 量化/权重布局：`transform_weights_for_mega_moe` 对 FP8/FP4 的 L1 gate/up 做 interleave 并转置 SF，对 L2 转置 SF；BF16 仅处理 L1 interleave。

### GCMoE：仓库源码与本机已安装版本不同

本任务仓库 `source/gc_moe`：

- commit `c6c970122fac461d8ab1b1b41b08f5c3dfa22b15`；源码 `__version__ = '0.3.1'`。
- 包含 SM100 与 SM90 两套接口；Blackwell 入口是 `source/gc_moe/apis/gc_moe.py::fp8_fp4_gc_moe`，buffer 入口是 `source/gc_moe/gc/__init__.py::get_symm_buffer_for_gc_moe`，权重入口是 `transform_weights_for_gc_moe`。
- 约束：当前实现只接受 `mma_type='fp8xfp4'` 和 `swiglu`；routed recipe 固定 `(1,1,32)`；`TOKEN_ALIGNMENT=1920`；`MAX_RANKS=72`、`MAX_EXPERTS=2048`、`MAX_EXPERTS_PER_RANK=512`；token 上限为每 rank capacity。
- 源码含 `source/gc_moe/eplb/` 与 `source/gc_moe/eco/`：`eco_unit_kernel.cu`、`eco_unit_auction.cuh`、`plan.py`、`unit_ref.py`、`meta_sm90.py`。`fp8_fp4_gc_moe` 还有可选参数 `eco_meta`、`eco_state`、`eco_logical_experts`。
- 按仓库自己的 `SOURCE.md`，`source/` 上游为 `Bruce-Lee-LY/gc-moe` 的 `3d520f2`，并叠加 ECO 分支和 DeepInfer/RTP-LLM EPLB 移植；这些是源码事实，不代表本轮已运行。

机器上已安装的 GCMoE（另一个用户的 Python 3.10 site-packages）：

- `gc_moe 0.6.2+4c0641d.cu132`，路径 `/data0/brucelee.ly/.local/lib/python3.10/site-packages/gc_moe`。
- METADATA 要求 `torch>=2.11,<2.12`、`nvidia-cutlass-dsl==4.8.1`、`cuda-python>=13.2`。
- 该安装导出 `GCSymmBuffer`、`fp8_fp4_gc_moe`、`get_symm_buffer_for_gc_moe`、`transform_weights_for_gc_moe`，以及 SM90 和 SM120 接口；没有 `eco/` 或 `eplb/` 子树，也没有 `eco_meta` 参数。
- 对应源码 `/data0/brucelee.ly/open_source/gc-moe`，commit `c5d93d15aa5c9e64579b71d6434ce6fde72c6f12`，上游 `git@github.com:Bruce-Lee-LY/gc-moe.git`，有一个未跟踪的 `docs/qoder_review.md`。
- 另有 `/data0/home/nanjun.cp/gc-moe`，commit `fb8b65fa829c97962eb2dbc9a4ad8b759b9cd736`，工作树有 11 个修改文件，含 ECO/EPLB 与 SM100 kernel；这是实验工作树，不是当前 admin venv，也未在本轮运行。

当前 admin venv 既没有安装本任务仓库的 `source/gc_moe`，也没有安装上述 0.6.2 wheel；因此 GCMoE 在本环境是**源码存在、可导入性未验证、未运行**。

### 推理框架：RTP-LLM

- 候选源码：`/data0/data3/wjh528431/work/RTP-LLM`，顶层 commit `c207b52b2ddce5acf964514968717a748053973b`，干净；远程为内部 GitLab `foundation_models/RTP-LLM`。该目录的 `githubopensource` 子树为 commit `e59977f84d34dbdb9695ebf4a4afa8886ef8e364`，远程指向一个 GitHub fork；仓库公开上游按项目记录为 `alibaba/rtp-llm`。
- RTP-LLM 存在 MegaMoE 集成：`rtp_llm/models_py/modules/factory/fused_moe/impl/cuda/executors/mega_moe.py::MegaMoeExecutor`，以及 `mega_moe_se.py`。
- 策略选择源码写明：`auto` 对带 shared experts 的 EP 模型优先 `mega_moe_se`；显式 `MOE_STRATEGY=mega_moe` 可保留独立 shared-expert 计算。
- 源码还有 `rtp_llm/eplb/eplb.py` 与 `rtp_llm/eplb/ep_balancer.py`；本轮没有启动 RTP-LLM，也没有验证专家副本、EPLB 或端到端入口。
- 其他用户的 RTP-LLM checkout 也存在（例如 huzetao `eb641bc6...`、luxiaojin `be4aa847...`），但都不是当前 admin 环境，不能当作本轮已验证基线。

## 5. 最自然的原生配置（待设计端冻结）

按本机已确认事实，最自然的候选是单机 8 卡 EP8：

- RTP-LLM + DeepGEMM MegaMoE/MegaMoE-SE，单节点 8 个 SM103 物理卡，TP=1、EP=8。
- DeepSeek-V4-Flash，hidden 4096、43 层、256 routed experts、top-k 6、1 shared expert、MoE intermediate 2048、FP8 per-block `[128,128]`。
- 权重与激活走 FP8/FP4 routed kernel；输出 BF16；同一组 8 个 UUID 串行运行所有臂。
- 但这条路径**尚未由本轮实际启动或验证**，因此不能把它当作已验收的原生基线。设计端若需要冻结，至少要先补端到端 smoke、模型输出与 DeepEP/参考路径的正确性对照。

## 6. 缺失与未验证

- 没有作业调度器/硬件预约服务；advisory lock 只能做协作式互斥。
- 当前 admin 环境没有可运行的端到端推理命令或服务日志；只有 DeepGEMM 合成 workload 的实际通过记录。
- 当前 admin venv 缺 `deep_ep`、`nvidia-cutlass-dsl`、`gc_moe`，也没有 vLLM/SGLang/FlashInfer。
- GCMoE 的源码要求与当前 venv（Torch 2.14、无 CuTe DSL）不一致；未做安装，符合本轮“安装预算为零”的限制。
- DeepGEMM MegaMoE 的 DeepEP 对照未运行；现有 `megamoe_test.log` 明确跳过了 baseline。
- 没有选定数据集，未填 `--dataset`。
- 没有在本轮启动模型、编译新 kernel 或做性能实验。
- `nvidia-smi` 的 `L20D/8.9/275040 MiB` 与 Torch 的 `10.3/267.7 GiB` 不一致；后续应以 `cudaGetDeviceProperties`/Torch 结果为准，并在每轮重新核对 UUID。

## 7. 回执结论

SM100 探针本身成功：8 张实际 capability 10.3 的卡对 Torch 可见，peer 全通，CUDA/Torch/NCCL/RDMA 版本已记录，源码入口和两个版本的 GCMoE 已区分。真正缺失的是经过 admin 环境验证的端到端推理基线、GCMoE 可运行环境和 DeepEP 正确性对照；这些留给后续任务，不在本轮用安装、编译或性能实验补做。
