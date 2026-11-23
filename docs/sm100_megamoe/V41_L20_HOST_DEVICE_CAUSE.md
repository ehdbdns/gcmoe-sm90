# SM100 执行任务 · 定位 BS512 第 20 层 MoE 入口错峰的原因

针对 c909c64 的 V4.1 Flash 原生 Mega、原生 GC。回答三个问题：第 19 层 combine 结束后，晚 rank 的第一个额外等待发生在哪次调用？最后一次 reduce-scatter 已重新对齐后，哪些调用重新造成第 20 层入口差？早 rank 的 count exchange 有多少时间发生在晚 rank 进入之前？按臂独立回答。

## 1. 先查现存记录

按 V41_RESIDUAL_L20.md 校验 dense-submit 原始 SHA。在 step 1 的原始 profile-rank*.json 清点 cuda_runtime、cuda_driver、cpu_op、GPU kernel、flow、PID/TID、时钟锚点；quiet 模式仍可能有 CUDA runtime。按同进程 correlation 试配 all-gather 后首个 cublas、q_kv_norm 后首个 dense、最后一次 reduce-scatter 后的 mhc、gate、prepare 及 L20 图提交。列出已有覆盖，旧的非 dense-replay CPU trace 仅作导航。只有两臂的归因、无 profiler 复核及下述证据均已齐备才跳过采集。

## 2. 在同一轮采主机、设备、调度证据

从本任务书提交建立临时 worktree。严格复用同目录 V41_BS512_DENSE_SUBMIT_FIX.md 的“执行”代码块，将其中 book 设为 docs/sm100_megamoe/V41_L20_HOST_DEVICE_CAUSE.md；保留准备命令，在最后一条 bash scripts/sm100/run_five.sh 前执行下面的替代启动段。每次都使用新 run 目录。两臂各启动一次，Mega 后 GC。沿用同组八卡 UUID、组锁和每臂空闲检查；共享机有人使用时等待。原先的输入、模型、上游 SHA、dense replay 开关序列、warmup、输出校验和各层计时全部沿用。执行者仅可做不改变语义的编译修复并附 diff。

```bash
set -euo pipefail
export SM100_L20_HOST_TRACE=1
# 仅记录本次启动树。用单调时钟，保留调度切换、用户态栈样本与丢样情况。
if command -v perf >/dev/null && perf record -o "$run/perf-probe.data" --clockid mono --switch-events -e cpu-clock:u -F 199 -g -- true >"$run/perf-probe.log" 2>&1; then
  perf record -o "$run/worker-sched.data" --clockid mono --switch-events -e cpu-clock:u -F 199 -g -- bash scripts/sm100/run_five.sh "$run" quiet-native-bs512 2>&1 | tee "$run/launcher.log"
  perf script -i "$run/worker-sched.data" --show-switch-events --ns >"$run/worker-sched.txt"
else
  bash scripts/sm100/run_five.sh "$run" quiet-native-bs512 2>&1 | tee "$run/launcher.log"
fi
```

本次开关只给 step 1 的现有 Kineto 会话加入 CPU activity；step 4/6 关闭 Kineto 的 forward 用于复核量级。外部 perf 如开启则覆盖整个运行，须同时对照 c909c64 的无 perf 数据。核实实际 worker PID/TID 均在记录中，尤其注意复用或独立启动的 worker；将缺失线程或调度权限显式列出。校验 runtime/API/kernel 覆盖、capture 开销、路由与输出。出现采集缺口先查原始数据和工具权限，不动模型或内核。

## 3. 分解实际时间线

1. 每个 arm/step 重新选最早和最晚进入 L20 的 rank，保留八 rank。使用同一步设备锚点和公共时钟，给出映射误差；perf 的 monotonic 与 Kineto 时间用同一进程可识别的公共事件或实测时钟映射验证，误差过大时只做同一时钟域的结论。
2. 列出 L19 出口、最后 collective 起止、mhc、gate、prepare、L20 graph launch、设备入口、交换结束及 MoE 出口的跨 rank 差。分别按同 rank 的窗口长度和公共时钟的跨 rank 差计算。设备 busy 取所有流并集，保留部分流空闲的区别。给出两臂各自早 rank 的交换区间与“最后一个 rank 尚未进入 L20”的时间交集；这个交集只表示可能受错峰影响的上界。
3. 为每个 ≥10 µs 的设备空档匹配前后 kernel、stream、提交线程、runtime/driver API enter/exit、enclosing CPU op、correlation。图内节点对应同一次 graph launch，禁止虚构逐节点 host 提交。找不到关联就标 unmatched。按共同调用序号对齐早晚 rank；给出晚 rank 首次新增差、最大贡献与最终净差。collective 内重新对齐造成的负贡献也入表，正负项加残差应闭合到最终入口差。
4. 区分前序设备活动结束时后继调用尚未进入 API、API 执行中、已经返回但设备尚未开始三段，按实际边界裁剪。GPU 可以在 API 返回前启动。对于晚提交，定位具体 Python/C++ op、测量包装或 CUDA 同步调用；用 TID 切换与调用栈区分线程在运行、阻塞或可运行而未获 CPU，给出直接证据。对于早已提交的工作，查 stream 顺序、event record/wait、graph 前驱、同 stream 拷贝与 collective peer 到达。没有依赖边时只记 pending。kernel 的长 duration 和空白栈样本都不能单独证明原因。
5. 以无 Kineto 的 step 4/6 核对入口、交换跨度和 rank 是否轮换。分别量化可解释部分、未覆盖部分、时钟不确定度和 profiler 引入的变化。只有 step 1 出现的调用级现象须标为采样步结论。

## 4. 有缺口时在本任务内补证

如 Kineto 已定位具体调用，但仍有足以改变主因判断的未解释段，优先复用已有 perf 调度和栈；先筛出对应 worker 和调用区间，避免把整机调度混入。若 perf 无权限、时钟不齐或 worker 未被继承，记录原始报错，检查本机获准的进程级采集手段。允许只复采受影响的证据，仍由同一受锁 launcher 运行；如果必须串行重跑两臂则记录原因。可以用获准的外部线程调度工具围绕启动器采集；避免让两个 CUPTI 工具抢同一会话。禁止为了“证明”猜想引入 rank barrier、设备同步、CPU 绑核或改动生产路径。如证据已能定位主因，跳过复采。设备/权限确实不支持时，交付已证实区间及缺口大小和缺少的具体事件。

## 5. 回执与清理

提交 docs/sm100_megamoe/REPLY_V41_L20_HOST_DEVICE_CAUSE.md 及同级 v41_l20_host_device_results/。交付两张第 1 步公共时轴 Perfetto（8 rank，CPU op/API、设备 stream、collective、MoE、可验证的调度轨）、逐边界及 gap 归因 CSV、每 rank 的 L20 到达及交换区间、step 4/6 对照、分析脚本和命令。回执开头用微秒回答上述三个问题，列至少一个早晚 rank 的可复查调用链、两臂是否同因、未解释的净差。附原始文件路径、SHA、环境、工具权限、退出码及 GPU 释放证据。推 main 并验证远端提交后，清理本次私有环境、worktree、缓存和锁；保留可复查原始记录及原有共享环境。
