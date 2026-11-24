# SM90 H20 · BS256 补跑 Mega unfused 第 1、5 臂

执行顺序为 1 原生 Mega、5 Mega + EPLB + RR。第 3、6 臂分别为对应的现有 GC unfused 对照。沿用已完成六臂的轨迹、KV、放置、CUDA Graph、40 层与 1000＋1000 步。按 H20 站点预约规则执行，只用八张同时空闲的 H20，有其他用户时等待。CPU 准备在申请 GPU 前完成。

## 编译故障与修复

回看 arm1-runner.log 第 524 行起，三个 float 模板参数声明首先报错，kernel.cu 第 7 行是第四个诊断。官方 7292e7b 的生成器显式传入 swap_ab 和 bf16_scaled_accum，之后 L1 再传一个布尔值，L2 再传三个，正好对应两个入口。沿用 7c848494 开头的原 wheel 和原始头文件。私有 DeepJIT 通过 DG_JIT_NVCC_COMPILER 选择包装器，包装器只给含 SM90 Mega 头文件的编译添加经过实测的 -ccbin。L1/L2 真模板以首个失败形状离线生成 cubin 后才开始实验。

## 操作

在此次 SM90 执行仓库运行以下命令。任务脚本取自本任务提交，测量工作树钉在 14d0e09。它与原测量 c1a176d 的唯一差异是加载时期每次批量转换四组 128 行权重。启动器逐字节比较本机旧版保存的三个投影的转换摘要及误差，同时校验原六臂产物。

~~~bash
set -eo pipefail
git fetch origin main
task_sha=$(git log -1 --format=%H origin/main -- docs/trace_executor_20261119/SM90_H20_RETRY_MEGA_1_5.md)
area=/mnt/fuse/sm90-h20-eight-CF3A1Zkp
out="$area/results"
test "$(cat "$out/measurement.sha")" = c1a176dbf45f1fd8d433189c1f24533aa44a7212
job=$(mktemp -d "$area/mega-retry-XXXXXXXX")
git worktree add --detach "$job/task" "$task_sha"
git worktree add --detach "$job/measure" 14d0e097ea8bfd6438f8fd3a6e47b45cd220d552
cd "$job/measure"
bash "$job/task/scripts/sm90/run_eight.sh" --resume-mega "$out" 2>&1 | tee "$job/retry.log"
~~~

脚本先检查模型与数据摘要、原 wheel、真实权重转换及六个 complete 清单，再在 CPU 上探测宿主 C++ 编译器。随后检查网卡、锁和 GPU 空闲状态，只调度 1 和 5。上次失败目录自动归档。本次的两臂共用私有 JIT 缓存。若中断，从同一个 measure 目录重新执行最后一条命令，会校验并跳过已完成的臂。

若所有宿主编译器的预检都失败，先看最新 mega-jit-*/compiler-*/l1.log、l2.log 的第一条错误。可将 SM90_MEGA_HOST_CXX 指向现场与 CUDA 兼容的 GCC 13 或 14 的 g++ 绝对路径再执行。若现场只有系统 GCC 10 且已有 GCC 15 超出 NVCC 支持范围，可以在 job 目录使用站点提供的 conda 或 micromamba 创建私有 conda-forge gxx_linux-64=13 环境，并指向该环境 bin/x86_64-conda-linux-gnu-g++。取得两个真实 cubin 才进入正式执行。允许处理保持语义的编译环境问题，保留具体命令及版本；测量代码、模型、模板、阈值和计时口径的变更交回设计端。若运行时 JIT 仍失败，包装器自动保留 failed-kernel-*.cu，请回传该源文件、完整编译命令和首错。保留原始回执与六臂产物。

## 回执和退卡

1. 两臂实际 exit、execution.sha、8 rank NPZ、run.json、installed-layout、完整 runner.log、wall-seconds 和 sha256。记录选中的 NVCC/g++ 路径和版本、两阶段预检记录、转换对照。单列每臂模型加载秒数、正式测量秒数及总墙钟秒数。
2. 用原 report_eight.py 重算 summary.csv、verification.json，逐项查看 failures，报告 1 对 3 和 5 对 6 的完整 FFN 毫秒/步和相对变化。注明原 reference SHA 和两个新臂的 execution SHA，提供实际输出差异、轨迹及物理工作量校验结果。若仅 CPU 报告失败，保留模型运行证据并报告首个失败字段。
3. GPU 工作结束即释放站点预约与本任务进程，检查八卡进程列表。压缩保存编译证据和失败源码后清理本次私有 JIT 缓存、临时构建环境。保留 replay、KV、六臂原件及新臂证据；勿清理共享缓存或他人进程。回执推送到 docs/trace_handover_20261119/REPLY_SM90_H20_RETRY_MEGA_1_5.md，产物放原八臂结果树单独的 mega_retry 子目录。
