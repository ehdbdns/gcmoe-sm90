# SM90 · 8×H20 · Mega 与 GC 的完整 decode step 和完整 FFN

运行 DeepSeek-V4.1-Flash 全部 40 层。正式结果包含四行：Mega 原生、GC 原生、Mega EPLB＋RR、GC EPLB＋RR。四臂都由 vLLM 计算 unfused shared。第 2 号进程专用于生成参考状态，采用 GC unfused，独立于四个计时进程。后续 3、1、6、5 均为消费者。历史 BS256 的成功臂与 OOM 记录继续保存在原目录。

## 运行

使用站点预约机制，只在八张 H20 都空闲时使用，忙时等待。沿用上一轮私有 wheel、工具链和模型数据目录。使用独立工作树及新结果目录，确保结果盘至少有 90 GB 可用空间。执行以下命令。

~~~bash
set -eo pipefail
git fetch origin main
rev=$(git log -1 --format=%H origin/main -- docs/trace_executor_20261119/SM90_H20_BS32_MEGA_FULL_STEP.md)
job=$(mktemp -d /mnt/fuse/sm90-h20-bs32-XXXXXXXX)
git worktree add --detach "$job/code" "$rev"
cd "$job/code"
bash scripts/sm90/run_eight.sh --paired-bs32 "$job/results" 2>&1 | tee "$job/launch.log"
~~~

如本任务被外部中断，回到该工作树，执行以下命令续跑。

~~~bash
bash scripts/sm90/run_eight.sh --paired-bs32 --resume "$job/results" 2>&1 | tee "$job/resume.log"
~~~

## 规模和公平条件

- 全组 BS32、TP8/EP8、每卡 4 GiB KV、prefill chunk 和通信容量均为 256，只捕获 BS32 的完整 decode graph。路由专家数、权重精度和 replica 规则保留各臂的实际实现，原生与 EPLB 分别保留 48 和 56 个本地物理槽。两阶段各 1000 步，同一批请求贯穿全程。
- 发布者生成此 BS 的 prompt/token、两阶段逐层路由和两个 KV 边界。四个消费者校验请求顺序、实际反馈 token、路由 ID 和权重、边界 KV 读回。两个 RR 臂分别计算并校验同一放置。报告逐配对检查逐步逐层逐 rank 的物理槽 counts。量化路径造成的原生 top-1 差异单列。
- 初始化失败时保留错误阶段及各 rank 显存。允许修复保持语义的编译环境问题。模型、批量、缓存、槽位、步数及运行时代码的调整交回设计端。不复用 BS256 产物作为本轮对照。

## 回执与退卡

1. 回传 summary.csv、verification.json 和 failures，每臂 exit、wall-seconds、execution.sha、run.json、runner.log 及八 rank NPZ 的归档位置与摘要。列出实测的 stats/measure 起点各卡最小空闲显存，保存 JIT 编译器路径及版本。
2. 四行均报告完整 step 的 ms/step、完整 FFN 的 ms/step。前者取后 1000 个真实 engine.step() 的同步墙钟均值，后者逐层取 rank 最大 FFN 跨度，40 层求和再对 1000 步取均值。报告两组配对各自的两项百分比，统一为 100 × (GC / Mega − 1)，负值表示 GC 耗时较低。另列运行窗口所有 worker 的 FULL graph replay 次数及各层 shared 模块。
3. 只有两端均通过公平性校验的配对可作性能结论。单一消费者失败时保留该臂失败栈，完成独立的其余消费者。GPU 工作结束即退卡，确认本任务进程退出，CPU 汇总可以随后执行。清理本次私有缓存，保留恢复实验所需的结果、参考状态和校验摘要。
4. 通过 GitHub 推送 docs/trace_handover_20261119/REPLY_SM90_H20_BS32_MEGA_FULL_STEP.md 与对应的小型汇总文件。
