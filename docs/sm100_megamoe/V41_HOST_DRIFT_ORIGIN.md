# SM100 · 找出 CPU 提交队列从哪一层开始落后

只读 6495cf5 已保留的两个原生臂、同一轮全部 40 层的 CPU/CUDA trace 和 perf。无需 GPU 或模型启动。GC 在 L19 图提交时已有约 9.12 ms 的早晚 rank 差，Mega 约 1.99 ms；直接对 L20 设备空档归因会把早 rank 预先排队的工作也计入晚 rank 的额外耗时。

从本任务书提交建临时 checkout，使用原 NumPy 环境执行：

```bash
set -euo pipefail
raw=/data0/home/admin/workspace/megamoe/v41-l20-host-device-T2qg4jm8/run
out=docs/sm100_megamoe/v41_host_drift_origin_results
python scripts/sm100/host_layer_drift.py "$raw/bs512" "$out"
```

1. 核对原始 SHA。脚本只接受实际 MoE 设备 kernel 与提交线程 CUDA graph launch 的唯一 correlation。八 rank 分别列出 L0、首次明显分叉、最大新增差所在层、L19、L20。对晚/早固定 rank 对，L0 分开报 anchor 自身跨 rank 偏移与 anchor→提交差。对 L1–20 的 host API 时差按相邻层有符号增量闭合，打印 L20 的 inherited + new，以及 GPU 入口差 = 提交差 + 两端各自排队时间之差。
2. 按 exclusive.csv 分开累计 L1–19 和 L20 的独占时间差，保留负值和 unobserved；取累计贡献最大和首次明显分叉的具体层、op、重复调用序号。回到原始 Kineto 用 PID/TID、邻居和 correlation 区分 graph.replay 包装中的 copy/zero/output audit、模型 forward、CUDA API 以及无包围事件。按实际安装的模型源码/overlay 定位到文件及函数，附路径、SHA 和最短可复查摘录。
3. 复用归档 perf 在**这些 rank 的 CPU 提交窗口**匹配 SWITCH 和用户栈，累计 L1–19 反复出现的同类区间取样。分别给出每个 rank 的墙钟、running/off-CPU、样本总数与热点栈占比，标记时钟拟合误差，按算子实例数归一化。重点排查开销是否普遍分散到所有 CPU 操作，或集中于分配器、审计、Python 调度、同步、线程迁移等具体路径。统计覆盖不足时给出边界，不用无样本作归因。核对 step 4/6 的原始 monotonic 提交曲线；使用各步自己的早晚 rank，区分固定机器位置和轮换现象。
4. 回执 docs/sm100_megamoe/REPLY_V41_HOST_DRIFT_ORIGIN.md，附上述 CSV、原始证据 SHA、源码位置、证据支持的原因及剩余微秒。无需重复导出 L20 设备图。仅可修复保持语义的编译问题；分析工具若遇缺数据或 association 错误请附具体诊断，不做最近时间替代。推 main 并验证，清理本次临时 checkout，保留原始归档。

## 对照 SM90 的图覆盖范围

SM90 的已交付 baseline/run.json 记录 ENFORCE_EAGER=0、CUDAGRAPH_MODE=FULL_DECODE_ONLY；本轮 SM100 的 five_matrix.py 固定 enforce_eager=True，five_worker.py 只捕获单层 MoE 调用，dense 另有局部图。把 L19/L20 两次 MoE 图提交之间实际发生在主机上的 kernel/graph launch 数和 copy/fill 次数列出来，区分模型与计数审计；复核两臂 graph replay 的提前量。查已安装 vLLM 及 V4.1 Flash 代码中 full decode capture 的入口，并列出当前按层变化的 Python current/call、host_submit、动态 scratch/审计地址等在整图回放下哪些需要调整。附源码路径和版本，作为精确实现整图对照的依据。本轮由执行者只读检查，不自行翻转 enforce_eager 或修改模型。
