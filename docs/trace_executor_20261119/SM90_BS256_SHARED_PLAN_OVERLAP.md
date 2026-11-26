# SM90 · BS256 shared L1 与 ECO plan 启动和重叠验证

**任务已撤回。SM90 执行者接续 BS256 Mega/GC 八臂。以下是历史实验说明。**

在 H20 × 8 上复测 BS256 整模，shared L1 先入图，ECO plan 和 dispatch 入场 CTA 可在通信分区独立启动。要看 shared L1 能否领先几微秒并与 plan 大部分重叠，以及正式窗口的完整 MoE 是否缩短。旧 shared-first 的 replay 2 三层共 24 个样本中，plan 与 L1 重叠全部为 0 µs。该图的 wa=1，供查看原先的时间线现象。正式耗时对照使用同价的已归档线性 wa=2（测量 SHA 2f70314，BS256 为 872.49 µs/层）。

## 执行

在已授权且同组八卡全部空闲时运行。保留设备掩码、节点预约和八卡协作锁，遇到他人使用则等待。只执行 BS256 的 EPLB + ECO，沿用完整模型、TP8/EP8、每 rank 八个冗余槽、M64、split shared、stream、wb=26、wa=2、slack=92。正式和独立 profiler 进程分别使用 1000 步统计、1000 步测量。脚本优先复用归档路径的回放输入及两套 KV；缺失时先由 vanilla 生成一次。执行者仅可修复保持语义的编译错误，回报原始报错、diff 和运行 SHA。

~~~bash
set -eo pipefail
git fetch origin main
rev=3a64b348612ab85a86d6d4491d4644f1cfcaa315
job=$(mktemp -d /mnt/fuse/sm90-bs256-shared-plan-overlap-XXXXXXXX)
echo "$job"
git worktree add --detach "$job/code" "$rev"
cd "$job/code"
rc=0
bash scripts/sm90/run_full_model_four.sh --shared-plan-overlap "$job" > "$job/launcher.log" 2>&1 || rc=$?
echo "$rc" > "$job/exit-code.txt"
exit "$rc"
~~~

## 回执

1. 核对正式和采样日志中的角色顺序 (4,5,7,1,8,2,3)、independent_plan=True、每 rank 的 mode=0 和 slack=92。plan 使用通信分区的一个 SM，dispatch 使用其余五个 SM。采样取重排后第二次完整 graph replay，层号按零起计。保留原始 Kineto、capture.json、rank NPZ 和所有 CSV。
2. 首行给第 20 层 Perfetto 的 GitHub blob 链接。附第 8、32 层、launch-order.csv、phase-times.csv 与 overlap.csv。逐层、逐 rank 报 plan 起点减 shared L1 起点（带符号，正值表示 shared 先开始）、两者交集的微秒数及其占较短 kernel 的百分比。同时给出 24 个样本中 shared 早于 plan、实际重叠、shared 领先 0 到 10 µs 且交集超过较短 kernel 一半的样本数。这个时间区间仅用来描述“领先几个微秒”。列出方向相反、大幅滞后和重叠不足的 rank、层号及原始数值，包含零和负值。各时间均取同一 rank 上的 GPU kernel 区间。
3. 对正式无 profiler 运行，用 comparison.csv 列完整 MoE 均值、相对归档 wa=2 的有符号差（µs/层和百分比），用 slow_rank_phases.csv 列 graph 到首个 routed CTA、routed 跨度及其后的 combine 收尾。脚本逐 rank 对照相同输入、路由、边界 KV、放置、native top-1、物理槽计数与拍卖。若配对失败，交付时间线及失败证据，并注明该次耗时的可比性。单独列启动领先与实际性能变化。
4. 将回执推到 docs/trace_handover_20261119/REPLY_SM90_BS256_SHARED_PLAN_OVERLAP.md，数据归档到 docs/trace_executor_20261119/sm90_bs256_shared_plan_overlap_results/。保留失败现场、校验和以及输入归档，结束后释放本任务的八卡并核实 worker 已退出，清理本任务临时缓存。
