# SM100 执行者 · MegaMoE 环境回执

本轮只调查目标机器。请在实际用于推理的容器、GPU 作业和 Python 环境内执行。设计端负责选择 MegaMoE 接入点、编写代码、定义指标和发布后续命令。此次模型启动、编译、安装和性能实验预算均为零。回执与 SM90 项目独立，写入本目录。

## 1. 确定运行入口

记录能进入该环境的现有方式、Python 解释器、CUDA toolkit 路径、实际可分配 GPU 数、单机还是跨节点，以及访问 GPU、网络和临时磁盘的限制。若已有可工作的推理命令，给出去掉凭证的命令、模型和数据集路径、TP/EP、backend 和已有日志的位置。标明命令是否在这台机器实际跑通过。

找到本机已有的 MegaMoE、GCMoE、ECO 及推理框架源码或安装包。针对每项给出仓库名、公开上游地址（若已知）、commit、工作树改动、调用入口的文件和函数。尤其要指明 MegaMoE 实际指哪个实现，例如是否来自 DeepGEMM；若仅有 wheel，给出版本、模块路径和可见的 Python 入口。找不到就写缺失。本仓库也带有 SM100 GCMoE 源码，应与机器上已安装的版本分别记录。

## 2. 执行环境探针

在本仓库获取最新 main，使用独立工作树，避免改变现有实验 checkout。下面的 python 应换成上一步确认的推理解释器。按实际存在的路径补充任意多个 `--repo /源码目录`，以及 `--model /模型目录 --dataset /数据文件 --scratch /可写工作盘`。没有对应资源就省略该参数并在回执说明。探针只查询版本、文件元数据、设备可见性与 CUDA peer 能力。

```bash
set -eo pipefail
repo=$(git rev-parse --show-toplevel)
git -C "$repo" fetch origin main
work=$(mktemp -d)
git -C "$repo" worktree add --detach "$work/code" origin/main
cd "$work/code"
dir=docs/sm100_megamoe
python "$dir/probe.py" --repo "$PWD" > "$dir/ENV.json"
python -m json.tool "$dir/ENV.json" > /dev/null
```

在输出里检查 GPU 型号和实际 compute capability、显存、拓扑、Torch CUDA/ABI、NCCL、nvcc/ptxas、CuTe DSL、框架及通信库版本。补充 NVSHMEM 是否安装、其版本和可用 transport，记录 MegaMoE 已有示例支持的量化格式、专家维度、top-k、图捕获、dispatch/combine 入口及已有多卡用例文件。只写可以从本机确认的事实，未运行的用例注明“未验证”。无需复制权重、数据、环境变量全集或认证信息。

## 3. 提交回执

在 `docs/sm100_megamoe/REPLY_ENV.md` 写清探针实际命令、执行 SHA、上述结果、缺失项和能供设计端读取的源码版本。若实现只有本机可读，给出相关入口签名、张量布局及已有示例调用；不要自行移植。特别注明这台机器目前最自然的“原生”配置，包括框架默认 MoE backend、是否有专家副本及已有正确性基准。原生基线由设计端收到回执后冻结。探针若部分失败，保留已得结果、错误与返回码，照常回执。

```bash
git add -- docs/sm100_megamoe/ENV.json docs/sm100_megamoe/REPLY_ENV.md
git commit -m 'Report SM100 MegaMoE execution environment'
git fetch origin main
git rebase origin/main
git push origin HEAD:main
```

## 后续五臂

计划为原生、MegaMoE + EPLB + RR、MegaMoE + EPLB + ECO、GCMoE + EPLB + RR、GCMoE + EPLB + ECO。四个 EPLB 臂对齐逐层放置，两个 RR 臂对齐副本选择，两个 ECO 臂对齐价格、slack 与拍卖规则。拟覆盖 BS256、512、1024、2048，实际容量待本机验证。端到端、完整 MoE、token IR、总 M 块、峰值 rank M 块、活跃物理槽、M 块 IR、routed/graph 预测误差与 ECO 成交数沿用主表口径。适配阶段需确认两种内核的实际 M 块粒度和计时边界，避免把不同单位直接比较。严格回放及原生输出仍需独立验收。[当前主表](../../README.md)
