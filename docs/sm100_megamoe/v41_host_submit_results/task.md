# SM100 执行任务 · 查明 BS512 的主机提交延迟

用保留的 CPU/CUDA Kineto 记录，回答第 2 步的 GC rank 2 和 Mega rank 4 从哪一次 CPU 调用开始落后，落后多少，以及那次调用属于模型、测量代码、CUDA API，还是现有记录没有覆盖的时间。优先解释第 20 层，同时复核 8、32、39 层。仅做 CPU 离线诊断，模型和 kernel 均保持原样。

## 命令

在 SM100 仓库执行。使用已有 Python 3 标准库，保持原始文件只读。

```bash
set -euo pipefail
git fetch origin main
book=docs/sm100_megamoe/V41_HOST_SUBMIT_CAUSE.md
sha=$(git log -1 --format=%H origin/main -- "$book")
wt=$(mktemp -d /tmp/sm100-host-submit-XXXXXXXX)
git worktree add --detach "$wt" "$sha"
cd "$wt"
raw=/data0/home/admin/workspace/megamoe/v41-phase-bs512-ku9Hv473
docs=$PWD/docs/sm100_megamoe
(cd "$raw" && sha256sum --quiet -c "$docs/v41_native_phase_results/raw-evidence.sha256")
out=$docs/v41_host_submit_results
mkdir -p "$out"
python3 scripts/sm100/native_host_submit.py "$raw/bs512" "$docs/v41_entry_skew_results/intervals.csv" "$out/host_segments.csv.gz"
```

## 分析与验收

1. 确认 2 臂 × 8 rank × 40 个 CPU 窗口齐全。窗口为前一次图提交开始至本次提交开始；第 0 层含启动锚点到首次提交，单独列出。表内嵌套的 CPU op 与 runtime 调用按最内层归属，每段只计一次。对层 1–39 确认段长总和与相邻 launch_host_ns 之差一致（容差 0.1 µs）。
2. 第 20 层 GC 比较 0/2，Mega 比较 6/4。其他层用上轮 intervals.csv 的最早和最晚设备入口 rank。为每对 rank 从第 0 层列出历次 CPU graph 提交时差。区分上一层已继承的提交差和本窗口新增的差；核对二者相加等于本次提交差。第 0 层同时列出两个 rank 的首锚点差。必要时沿前面的层找，不能把先前积累的差归给本层最后一个调用。
3. 对时差首次显著扩大及后续扩大最多的窗口，按 category/name/enclosing_cpu_op 求两端独占时间并做差，列贡献最大的 5 项以及 unobserved。沿同一线程按调用顺序匹配同名调用的序号、前后邻居和 GPU correlation；重复出现的同名 op 要定位到具体一次。记录关键证据的 arm/layer/rank/pid/tid、相对时间、调用名、duration、对应 kernel 及两侧的事件摘要。两端调用序列不一致时，从共同锚点逐段对齐，保留分歧。
4. 若发现同步 API 较慢，用关联的 GPU 事件和 stream 依赖核对是否在等设备。若是 dense GEMM 前的长空档，核对 CPU 是否已提交对应 kernel；已提交的 pending 工作与尚未提交分别报告。核对是否落在 five_worker 的拷贝、审计或正常模型 forward；只有明确的包围事件或源码对应关系才能指定函数。只有无记录区间时，写清其两端的具体 CPU/CUDA 调用与可解释的微秒数，不能单凭 Kineto 空白推断线程被操作系统抢占。对照两个臂，说明结论能覆盖哪些层。
5. 回执写 docs/sm100_megamoe/REPLY_V41_HOST_SUBMIT_CAUSE.md，提交关键区间 CSV、累计提交差表、证据摘录、脚本实跑 SHA、输入校验和及结论。host_segments.csv.gz 如过大可留在机器并交付 SHA 和路径。推送 main；核验推送后清理本次临时 worktree。执行者只执行和分析，不修改模型、校验或测量语义。如具体原因仍超出现有 trace，报告未覆盖的调用区间和缺失的观测量，不擅自重跑 GPU 实验。
