# SM100 · dense 回放后的 L19→L20 剩余错峰

只用 CPU 分析 779eabe 保留的证据。现有 NPZ 显示 GC step 4/6 中，L20 最早、最晚进入的两个 rank 在 L19 combine 结束时只差约 1 µs，层间跨度却差 270–282 µs。step 1 也出现相同现象，且仅此步有 dense 同图回放后的真实 Kineto kernel 记录。

在干净的任务书提交 checkout 中运行，使用已有含 NumPy 的 Python。
```bash
set -euo pipefail
archive=docs/sm100_megamoe/v41_dense_submit_results
raw=/data0/home/admin/workspace/megamoe/v41-dense-submit-5u4m9Asy/run/bs512
out=$(mktemp -d /tmp/sm100-residual-XXXXXXXX)
sha256sum --quiet -c "$archive/raw-evidence.sha256"
python scripts/sm100/residual_l20.py "$raw" "$archive" "$out/window.csv"
```

1. 对 step 1 GC rank 2/0、Mega rank 3/4（早/晚）核对同 rank 窗口长度及跨 rank 到达差。逐 rank 统计 gap 总量、设备活动并集，闭合到窗口长度。列最大的空档及两侧真实 kernel、stream、correlation。设备活动时长按所有流取并集。
2. 在同一流按 kernel 名、形状、调用次序对齐两 rank 的 collective、attention/indexer、dense、路由和最后一次 MoE 前操作，列每个共同边界相对 L19 结束的累计差和新增差。特别检查最后一个 collective 前后差值是否收敛、随后哪一段再次扩大；量化耗时是某个 kernel 自身增长、提交前空闲，还是已有依赖未结束。重叠流和不匹配调用标明。无对应 host API 或依赖证据时只报设备空档，不猜线程抢占。
3. 交付两臂各一张该窗口的八 rank Perfetto，保留真实跨 rank 时差（使用公共 Kineto 基准和两锚点，勿按各 rank 的 L19 出口分别置零）。用不带 profiler 的 step 4/6 核对时间量级及 rank 变化，禁止将 step 1 的 kernel 移植到其他步。
4. 回执 docs/sm100_megamoe/REPLY_V41_RESIDUAL_L20.md，附 window.csv、逐边界 CSV、图、原始 SHA 和首个有证据解释的差值；仍无法归因时列具体残差与缺少的观测。仅离线分析，不改语义，不占 GPU。推 main 后清理本次临时 checkout，保留原始证据。
