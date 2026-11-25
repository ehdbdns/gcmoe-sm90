# SM90 执行任务 · shared L1 优先启动，线性 ECO 取 wa=1

在 H20 × 8、DeepSeek-V4.1-Flash、TP8/EP8 上，对上轮全部六档 BS 各重跑一次整模 ECO。参照组是测量提交 `2f70314` 的同批量 shared-first、wa=2 归档。只把活跃物理槽的价格从 2 改成 1，相应的满 M 块进入空槽的 slack 从 92 改成 91。BS256 另做一次独立的 torch.profiler 采样。本版替代上一版三档任务，按本页提交执行。

| 全局 BS | 32 | 64 | 128 | 256 | 512 | 1024 |
|---|---:|---:|---:|---:|---:|---:|
| EP8 均摊请求数/rank | 4 | 8 | 16 | 32 | 64 | 128 |
| 每档正式 native top-1 比较数 | 32,000 | 64,000 | 128,000 | 256,000 | 512,000 | 1,024,000 |
| 上轮 wa=2 完整 MoE（µs/层） | 355.40 | 437.55 | 586.67 | 872.49 | 1475.12 | 2742.03 |

按 rank 计算 `T + 26M + A`。T 是各物理槽实际接收的路由行数之和，M 是逐槽物理行数除以 64 向上取整再求和，A 是非空物理槽数。沿用原窗口、每 rank 8 个冗余槽、逐层 EPLB、warm/cold 轮数 1/6、stream 按块发号、pull、fused split shared、FULL_DECODE_ONLY。plan 使用通信分区的 1 个 SM，dispatch 使用其余 5 个 SM 上的 20 个 CTA，compute 分区保留 72 个 SM。主指标为同一步同一层八 rank 各自从 graph 起点到 combine 完成的跨度取最大，再对 40,000 次层调用求均值。

## BS256 已有测量

对每次层调用选出完整 MoE 最慢的 rank，在该 rank 上拆分再对 40,000 次取均值。时间单位均为 µs。旧 wa=1 与上轮 wa=2 的输入核对已通过。

| BS256 ECO | Graph 起点到首个 routed CTA 开算 | 首个到最后一个 routed CTA | 最后一个 CTA 到 combine 完成 | 完整 MoE |
|---|---:|---:|---:|---:|
| 原启动顺序，wa=1 | 88.02 | 678.57 | 57.56 | 824.15 |
| shared L1 先启动，wa=2 | 149.15 | 668.27 | 55.07 | 872.49 |

两次运行中首个 routed CTA 前的均值相差 +61.13 µs，整层相差 +48.34 µs。启动顺序和 wa 同时改变。新回执加上 shared-first、wa=1 这一行，按同一规则输出 `slow_rank_phases.csv`，同时列清两次跨轮对比各自固定的变量。六档判断 wa 优劣统一使用 shared-first 两组的完整 MoE。

## 执行

保留调度器分配的八卡掩码，先确认是获授权且空闲的同组八张 H20。已有进程或预约时等待。已经启动的 Mega/GC 实验照常完成。本任务与它共用八卡锁。执行者按下列命令运行；可自行修复保持语义的编译问题，回传原始报错、补丁和实际执行 SHA。设置、数据、CUDA Graph 依赖或校验发生不符时保留现场并说明具体差异。

~~~bash
set -e
git fetch origin main
book=docs/trace_executor_20261119/SM90_SHARED_FIRST_LINEAR_WA1.md
rev=$(git log -1 --format=%H origin/main -- "$book")
test -n "$rev"
job=$(mktemp -d /mnt/fuse/sm90-shared-first-wa1-XXXXXXXX)
echo "$job"
git worktree add --detach "$job/code" "$rev"
cd "$job/code"
rc=0
bash scripts/sm90/run_full_model_four.sh --shared-first-wa1 "$job" > "$job/launcher.log" 2>&1 || rc=$?
echo "$rc" > "$job/exit-code.txt"
exit "$rc"
~~~

1. 每档先检查历史 route bank、token、两套边界 KV。文件齐全时链接到独立回放目录，核验历史摘要并逐 rank 读回 KV。缺失时运行一次原生 publisher 重建本档的参考，使用归档 DeepInfer 放置，ECO 根据新统计独立计算并逐项验证该放置。权重从已校验的本地副本预热。每个正式 ECO 先统计 1000 步，再连续测量 1000 步 × 40 层。执行脚本选择数据集与显存预算。
2. 八 rank 核对 prepared mode=0、slack=91、每次调用实际导出的价格。核对 prompt、逻辑路由 ID/权重/需求、逻辑权重、反馈顺序、两套 KV、40 层物理放置、预热 native 摘要及 1000 步每个请求的实际 native top-1。保存相同 GPU UUID、驱动、运行时、完整图回放数。允许价格改变成交、各 rank 的行数和 M 块数。正式耗时表用同档历史 wa=2 基线；报告跨轮有符号变化与同期环境，跨轮波动单列。
3. BS256 的采样使用同一份回放输入和放置，另起一个 ECO 进程。只在重排后的第 2 次正式 decode graph replay（零起编号 1）启动八个 worker 的 torch.profiler。脚本逐次核对采样运行与正式 wa=1 的 epoch、物理槽计数、成交和原生输出。按真实 epoch、从已编译图读取的角色几何和 Kineto 时间基准导出层 8、20、32（三层均零起）。每个 rank 展示 shared L1、shared L2、ECO plan、同轨 count exchange 与 pull token、各占一轨的 wait token 和 routed GEMM，以及 combine。routed 取首个实际 tile 执行起点，到最后一个 tile 完成（含末端 push）。图中事件携带该 rank 的物理路由行数、M 块、活跃槽。图的阶段、轨名、合并规则与 Word 所附 ECO replay 2 JSON 一致。这里采整模真实 BS256 的第二次 replay，每源 rank 的 TP 路由张量为 [256, 6]，八卡物理行数合计 12,288。正式性能表使用独立的无 profiler 运行。
4. 查看 `bs256/perfetto/launch-order.csv` 的 24 行。按层统计 8 个 rank 的 plan 起点减 L1 起点、dispatch 起点减 L1 起点、plan 起点减 L1 终点、L1 与 plan 重叠、首个 routed 计算 CTA 起点减 L1 终点；回执列出范围和任何反例的 rank，并列 shared L2 与 routed、pull 与 routed 的实测重叠。每层的八个阶段起止、持续时间及各 rank token/M 块数另见 `phase-times.csv`。图角色应为 `(4,5,7,1,8,2,3)`。源码中 L1 到 plan 和 dispatch 入场门为 launch-order 边（L1 的 CTA 全部开始即可放行），L1 到 routed 是完成依赖。端口与图配置只是配置证据，观察实际顺序使用每卡的 profiler 行。
5. `main_table.csv` 将 6 档 × 2 个权重并排，wa=2 来自上轮已归档的八 rank 原始记录，脚本重算并逐列核对归档表。列出完整 MoE µs/层、相对同档原生及 RR 的有符号变化、token IR、八卡总 M 块、峰值 rank M 块、八卡活跃物理槽、M 块 IR、routed MAE（百分点）、每次层调用成交和峰值 rank 活跃槽，另列慢 rank 对各项峰值的命中次数。完整 MoE 和 routed MAE 使用上轮相同的计时边界。拍卖表列轮数、报价、成交、接受率和零成交调用。保留各 rank 份额、逐层误差和每次调用原始计数。
6. `comparison.csv` 逐档写两种权重的 µs/层、有符号差、`100 × (wa1 / wa2 − 1)`、本次均值较低的权重、40 层中改善的层数以及 20 个配对 50 步段中改善的段数。负值表示 wa=1 更快。两种价格的各 rank 分配和拍卖结果可以不同。运行时、请求与反馈、prefill/测量 KV 清单、物理放置及 native top-1 等公平性有差异时，保留原始数据并在回执对应 BS 标注差异，该档不判胜。跨轮计时的小幅差异结合配对步段解读。
7. 每档在 `delivery-$bs` 执行 `sha256sum --quiet -c SHA256SUMS`。把六档 delivery、启动日志、验收结果、测量 SHA 与根目录的汇总表推送至 `docs/trace_executor_20261119/sm90_shared_first_wa1_results/`，回执写入 `docs/trace_handover_20261119/REPLY_SM90_SHARED_FIRST_LINEAR_WA1.md`。汇总验收行数为主表 12、逐层 480、rank 份额 96、拍卖 12、公平性 6、耗时对比 6。回执首行放 BS256 ECO 第 20 层的 GitHub blob 地址，同列层 8、32 以及 24 行的启动顺序 CSV 和 192 行的逐阶段 CSV。文件名中的 `step001` 表示第 2 次 replay。原始 Kineto 保存为每 rank 的压缩 JSON；KV、模型权重与路由 bank 留在节点。GPU 工作完成即释放预约，并核实本任务的 worker 已退出。回执确认校验后清理本次任务专属的可重建文件，保留历史共享输入。
