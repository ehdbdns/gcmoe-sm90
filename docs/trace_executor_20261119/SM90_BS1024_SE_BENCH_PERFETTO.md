# SM90 · GC shared expert 算子 benchmark · EP8 BS1024

本任务优先。搁置 BS32 和 Mega，空闲时仅运行下述两臂。8×H20，各源 rank 独立输入 128 个 token，总计 1024，固定 top-6，合计 6144 个路由项。输入、权重使用 benchmark 的确定性种子。采用仓库已有 BS1024 历史需求中第 20 层的 1000 步统计，按 DeepInfer 重建并核对已发布的放置。采样的是该放置下同一组 benchmark 输入的三次 graph replay。

| 配置 | RR | ECO |
|---|---|---|
| 放置与专家数 | EPLB，384 逻辑、448 物理 | 相同 rank 所有权，单副本槽按 ECO 要求前置 |
| 执行 | DeepInfer RR | fine cost，slack 11469，通信分区 plan，禁用 SM 复用 |
| 公共几何 | H5120、I2304、M64、1 个 fused shared expert、stream、6 个 dispatch SM | 相同 |

使用站点预约，八卡同时空闲才能运行；保留调度器的设备掩码。启动器负责本机协作锁及进程检查。新建独立工作树运行：

~~~bash
set -eo pipefail
git fetch origin main
rev=$(git log -1 --format=%H origin/main -- docs/trace_executor_20261119/SM90_BS1024_SE_BENCH_PERFETTO.md)
job=$(mktemp -d /mnt/fuse/sm90-bs1024-se-XXXXXXXX)
git worktree add --detach "$job/code" "$rev"
cd "$job/code"
bash scripts/sm90/run_bs1024_se_profile.sh "$job/results"
~~~

每臂先预热，再用 torch.profiler 记录八个 rank。退卡后可单独重做 CPU 导出。保留原始 Kineto JSON 和设备戳。合并图在每个 rank 分轨显示 dispatch（含 count exchange 和 pull）、shared expert L1/L2、routed kernel、wait token、从首个 CTA 开工到最后 CTA 完工的 routed GEMM，以及 combine；ECO 另外显示 plan。每一段带 sample、epoch、rank token 数和 M 块数。

将 rr.perfetto.json 和 eco.perfetto.json **实际提交**至 docs/trace_executor_20261119/sm90_bs1024_se_bench_results/，连同 execution.sha、十六个 rank*-device.json（按臂分目录）和两臂日志。回执放 docs/trace_handover_20261119/REPLY_SM90_BS1024_SE_BENCH_PERFETTO.md，给出两个 GitHub 文件链接、原始 torch trace 的归档位置、三次样本的各 rank 数据。若编译失败，只允许保持语义的编译环境修复；运行或校验失败保留原始证据，禁止补画区间或改动实验参数。结束后释放预约、核查本任务进程退出并清理本任务临时编译产物。
