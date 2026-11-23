# SM90 执行任务 · 8×H20 · BS256 · MegaMoE / GCMoE 八臂

使用 DeepSeek-V4.1-Flash、TP8/EP8 和当前 H20 主表的 BS256。每臂一次独立进程，前 1000 步统计，层级放置生效后再测 1000 步。共八次正式启动。设计端负责集成、计时和验收代码；执行端负责环境核验、运行和回执。当前收到即可在 CPU 上执行第一阶段。

## 八臂清单

| ID | Routed 后端 | Shared expert | 放置 | 副本分流 |
|---|---|---|---|---|
| 1 | DeepSeek MegaMoE SM90 | unfused | 原生 | 无副本 |
| 2 | GCMoE SM90 | fused | 原生 | 无副本 |
| 3 | GCMoE SM90 | unfused | 原生 | 无副本 |
| 4 | GCMoE SM90 | fused | 逐层 EPLB | RR |
| 5 | DeepSeek MegaMoE SM90 | unfused | 逐层 EPLB | RR |
| 6 | GCMoE SM90 | unfused | 逐层 EPLB | RR |
| 7 | GCMoE SM90 | fused | 逐层 EPLB | ECO |
| 8 | GCMoE SM90 | unfused | 逐层 EPLB | ECO |

fused 表示 GC 内核计算 shared expert。unfused 表示 routed 内核只处理 top-6 专家；模型在核外对同批输入执行同一份 shared MLP，并把结果加回一次。三条 GC fused/unfused 配对分别为 2/3、4/6、7/8。Mega 和 GC 在相同 shared 执行方式下的对比分别为 1/3 和 5/6。RR 与 ECO 在相同后端及融合方式下的对比分别为 4/7 和 6/8。

## 第一阶段 · 立即执行的 CPU 核验

在 SM90 仓库执行以下命令。无需预约或初始化 GPU。在本机上次成功的 SM90 Python 环境中核对包；若项目源码和运行环境位于不同目录，记录各自绝对路径。可以在私有源码目录中准备下述官方提交。保留共享软件环境。

```bash
set -euo pipefail
git fetch origin main
book=docs/trace_executor_20261119/SM90_H20_BS256_MEGA_GC_EIGHT.md
rev=$(git log -1 --format=%H origin/main -- "$book")
area=$(mktemp -d /tmp/sm90-h20-eight-XXXXXXXX)
git worktree add --detach "$area/code" "$rev"
git clone --single-branch --branch nv_dev https://github.com/deepseek-ai/DeepGEMM.git "$area/DeepGEMM"
git -C "$area/DeepGEMM" checkout --detach 7292e7bd0ef3b497ca948a7239d55f3f01539048
(
  echo "task=$rev"
  git -C "$area/DeepGEMM" rev-parse HEAD
  git -C "$area/DeepGEMM" grep -nE 'fp8_mega_moe|get_symm_buffer_for_sm90_mega_moe|transform_weights_for_mega_moe_sm90' -- deep_gemm tests/test_mega_moe_sm90.py
  cd "$area/code"
  source scripts/env.sh
  CUDA_VISIBLE_DEVICES='' "$GC_ENV/bin/python" - <<'PY'
import importlib.metadata as md
from importlib.util import find_spec
import sys
print("python", sys.executable, sys.version)
for module in ("deep_gemm", "gc_moe", "torch", "vllm"):
    spec = find_spec(module)
    print(module, "origin", spec.origin if spec else None)
for package in ("deep-gemm", "torch", "vllm", "nvidia-cutlass-dsl"):
    try:
        print(package, md.version(package))
    except md.PackageNotFoundError:
        print(package, "no installed distribution metadata")
PY
) > "$area/cpu-preflight.log" 2>&1
echo "$area/cpu-preflight.log"
```

回执附 CPU 日志、原模型配置与权重索引 SHA256、保留的 BS256 KV/回放目录、可用私有磁盘空间，以及本机 DeepGEMM 源码来源和 revision（如可查）。若已安装包另外暴露 sm90_fp8_fp4_mega_moe，记录签名和源码出处。这与上述官方 nv_dev 的 fp8_mega_moe 是两个入口，矩阵 1/5 采用后者。无需上卡验证安装。

## 正式运行的集成约定

- **精度。** 官方 Mega SM90 routed L1/L2 要求 E4M3 权重、FP32 的 128×128 block scale、SwiGLU；现有 GC 使用 checkpoint 的块缩放 FP4。设计端从同一 checkpoint 按原 scale 解码，再离线生成 Mega 所需 FP8 权重及 scale，记录转换算法、逐层校验和及数值误差，副本复制相同的已转换权重。加载、转换、打包、跨卡放置均在计时窗口之外。1/3 和 5/6 的结果标为“后端及权重格式共同变化”。同一 GC 配对保持 routed 权重逐位一致。
- **负载。** 用本轮 GC 原生 fused（2）发布输入参考，建议顺序 2、3、1、4、6、5、7、8。复用 H20 五臂已验证的稳定 prefill/decode top-k、固定请求、token、KV、逐步逐层逻辑路由和 gate。保存覆盖前原生输出及覆盖后实际工作量，两套分别校验。性能主指标基于相同逻辑路由。若 Mega 的权重转换引起原生输出差异，报告首差和数量，并单列固定工作量结果。
- **放置。** 384 个逻辑专家，原生每 rank 48 个槽；EPLB 使用同一逐层 DeepInfer 放置，448 个物理槽，每 rank 56 个。三条 RR 臂使用同一逻辑到物理映射，并逐步逐层核对物理槽计数。两条 ECO 臂沿用 H20 主表的 fine cost、slack 11469、通信分区 1 个 plan SM、stream 开启、SM 复用关闭；核对 plan、配额、成交数和物理槽计数。GC 采用 M64，完整记录 Mega 实际 M 和调度；按每臂实际 M 统计物理块数。
- **核外 shared。** 在 1/3/5/6/8 共用相同实现、权重、输入及 TP/EP 语义。预检用真实层分别核对 routed 输出、shared 输出、最终 BF16 输出和 TP 汇合位置。记录图中实际 shared kernel 与最终加法；fused 臂验证内核 shared 角色执行一次。计时取每层共同的 MoE 入口到下游可消费最终结果的设备时间，包含 RR 路由或 ECO plan、dispatch、routed、shared、combine 和必要的加法/同步。若两条路径并行，取真实跨度。
- **结果。** 按同一步同层八 rank 的最大完整 MoE 时间，逐步求 40 层之和，再求测量窗口均值，单位 ms/步。主表给出相对 2 号臂的有符号耗时变化，并列指定配对的同轮百分比。同步记录 token IR、总/峰值 M 块、活跃物理槽、M 块 IR、routed/graph 预测误差、ECO 成交数；缺失预测器的臂填 N/A。归档逐 rank、逐层、逐步原始计数及归约公式、模型/后端/配置 SHA、图捕获证据、显存与退出码。原生一致性、固定工作量一致性、完整 shared 工作量及权重精度分别报告。

## 用卡、交付和执行边界

SM90 为共享机。只在指定八张 H20 全部空闲、预约允许并取得整组使用权后启动任何 GPU 预检或模型进程；持锁再次核对 UUID 和占用。任一卡被使用或预约则在 CPU 侧等待，禁止终止他人进程。两臂之间串行；成功、失败或中止均及时退卡，离线整理期间释放组锁。

本仓当前 run_main24.sh 实现的是既有 GC 矩阵，不能执行以上八臂。八臂启动器、Mega 的实际模型适配和核外 shared 计时接好后，由设计端在本任务书补充固定 SHA 和正式命令。第一阶段完成后先交回执并释放现场，避免在准备阶段占用八卡。执行者仅可修复不改变语义的编译错误，附完整 diff；不自行移植后端或改精度、路由、验收口径。本轮无需启动 SM100 工作。

回执写入 docs/trace_handover_20261119/REPLY_SM90_H20_BS256_MEGA_GC_EIGHT.md；产物放 docs/trace_executor_20261119/sm90_h20_bs256_mega_gc_eight_results/。首份回执注明 CPU 环境核验，正式回执附八臂逐项状态。大型数据保留路径、大小和 SHA256，小型证据推送 main 并核对远端提交。

## 09118c3 回执后的 CPU 准备

执行者可立即编译官方 nv_dev 到私有持久目录。使用上轮 env3 的 Python 和已安装 Torch ABI，在无可见 GPU 的子进程内构建；共享 env3 只读。此步骤独立于模型和数据挂载，可先行完成。软件包加载和八卡算例的硬件验证留在正式预约内。

```bash
set -euo pipefail
persist=/home/admin/workspace/aop_lab/app_source/sm90_persist_20261123
area=$(mktemp -d "$persist/dg-sm90-build-XXXXXXXX")
py=/home/admin/workspace/aop_lab/app_source/gc_moe_rtp_ep8_20261031/env3/bin/python
export CUDA_VISIBLE_DEVICES='' TORCH_CUDA_ARCH_LIST=9.0 DG_FORCE_BUILD=1 MAX_JOBS=4
git clone --branch nv_dev https://github.com/deepseek-ai/DeepGEMM.git "$area/src"
git -C "$area/src" checkout --detach 7292e7bd0ef3b497ca948a7239d55f3f01539048
git -C "$area/src" submodule update --init --recursive
mkdir -p "$area/wheels"
"$py" -m pip wheel --no-deps --no-build-isolation --wheel-dir "$area/wheels" "$area/src" 2>&1 | tee "$area/build.log"
"$py" -m pip install --no-deps --no-index --target "$area/site" "$area"/wheels/*.whl
sha256sum "$area"/wheels/*.whl
echo "$area/site"
```

CPU 回执追加 wheel 路径及 SHA、编译日志、子模块版本或首错。正式运行使用单进程私有包路径。重启后的 HCA 按实际 ACTIVE 端口及 GPU/NIC 拓扑核对，启动器须显式传入 NCCL_IB_HCA 与 NVSHMEM_HCA_LIST。194709d 已找到模型和请求文件。正式启动器使用sm90_h20_bs256_mega_gc_eight_results/model_and_dataset.sha256 所列的新挂载路径与摘要。
