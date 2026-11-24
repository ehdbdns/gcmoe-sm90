# SM90 · 八臂运行期间离线核查 shared 融合耗时

现在在另一 CPU shell 执行，只读 c1a176d 本轮已完成的 arm2 和 arm3。正在运行的启动器、环境、工作树、卡和预约交给原实验继续使用。此任务 GPU 用量为零，新增实验次数为零。执行者交付单独的诊断回执。

在提交回执的仓库执行以下命令。results 取当前八臂启动器已记录的输出目录，核对 measurement.sha 以 c1a176d 开头、arm2/complete 和 arm3/complete 存在。历史 formal/state-path.txt 属于早先轮次。脚本只需 Python 和 NumPy，写入独立临时目录；从 Git 对象提取脚本可避免切换运行中的 checkout。

```bash
set -euo pipefail
: "${results:?先设为当前八臂启动器的绝对结果目录}"
grep -q '^c1a176d' "$results/measurement.sha"
git fetch origin main
job=$(mktemp -d /tmp/sm90-shared-pair-XXXXXXXX)
git show origin/main:scripts/sm90/shared_pair_cpu.py > "$job/shared_pair_cpu.py"
py=/home/admin/workspace/aop_lab/app_source/gc_moe_rtp_ep8_20261031/env3/bin/python
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 nice -n 15 \
  "$py" "$job/shared_pair_cpu.py" "$results" > "$job/pair.csv" 2> "$job/audit.log"
```

1. 报告正式 1000 步的整步 decode 和完整 FFN 两组均值、差值、有符号百分比。差值均为 fused 减 unfused，百分比分母为 unfused。附原生 top-1 差异数、固定反馈与门权重、逐槽计数、KV 和逻辑权重的核验结果。原生差异和固定工作量分别表述。
2. 对每一步每一层在各臂选本地 FFN 跨度最大的 rank。用该 rank 的时间戳拆成 FFN 起点→GC graph 起点、GC 起点→首个 routed CTA、首个→末个 routed CTA、末个 CTA→combine 结束、combine 结束→FFN 结束。每项单位为 ms/步；五项差值之和应等于完整 FFN 差值。CSV 后半部分给出同层同 rank 配对，以检查最慢 rank 切换造成的差异。
3. 给出差值贡献最大的阶段及代表层/rank，核对已有日志中的实际 shared 模块、kernel 配置和已有 profiler 证据。前置段含门控与输入准备，graph→first 也含 dispatch 等待。缺少 shared kernel 或 stream 的独立起止证据时，将原因定位到已测阶段，明确进一步归因所缺的证据。若 FFN 已持平或更快，直接报告整步差值和 FFN 差值的方向。
4. 核验失败时保留错误、输入 SHA、字段形状和失败 rank，只回传可证实的结论。即使诊断失败，原八臂继续。把 pair.csv、audit.log 放入本轮 formal/shared_pair_cpu/，结论写在 docs/trace_handover_20261119/REPLY_SM90_H20_SHARED_PAIR_CPU.md，从独立提交用工作树推送，附分析脚本 revision。无需等待后六臂。
