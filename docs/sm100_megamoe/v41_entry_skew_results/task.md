# SM100 · BS512 层间到达偏差的 CPU 复核

只读取上轮留存的 16 份 Kineto 文件及 16 份时钟锚点。无需占用或申请 GPU，也不启动模型。两臂的计时归档取自同一份 Git 回执。

```bash
set -eu
git fetch origin main
book=docs/sm100_megamoe/V41_ENTRY_SKEW_CPU.md
sha=$(git log -1 --format=%H origin/main -- "$book")
checkout=$(mktemp -d /tmp/sm100-skew-code-XXXXXXXX)
git worktree add --detach "$checkout" "$sha"
cd "$checkout"
archive=$PWD/docs/sm100_megamoe/v41_native_phase_results
raw=/data0/home/admin/workspace/megamoe/v41-phase-bs512-ku9Hv473
(cd "$raw" && sha256sum --quiet -c "$archive/raw-evidence.sha256")
out=$PWD/docs/sm100_megamoe/v41_entry_skew_results
python scripts/sm100/native_entry_skew.py "$raw/bs512" "$archive" "$out"
```

使用有 numpy 的 Python。应该有 624 行区间数据（两臂各 8 rank × 39 个层间区间）。按 arm/layer 选择 entry_host_ns 最大和最小的 rank。针对层 8/20/32/39 报告相减后的 gap_us、device_active_union_us、uncovered_us；后二者之和应为 gap_us 差。再比较这两个 rank 的 kernel 分组耗时和最长空档边界。单项 kernel 可能重叠，勿把分组总和当作 GPU 活动并集。如果识别到 40 次有序的图提交，比较各 rank 上图提交相对上一层结束的时间以及提交到 GPU 入口的时间。显示负数代表 CPU 提前排队。若列为空，交付原始 trace 中相应 cudaGraphLaunch、CPU op、GPU stream 和 correlation 的片段进行复核。只在看到证据时归因到某个算子或 CPU 提交；空档本身不能证明 CPU 调度原因。同步/collective kernel 的运行也可能含等待。

推送 intervals.csv、kernels.csv、版本、核验结果和 docs/sm100_megamoe/REPLY_V41_ENTRY_SKEW_CPU.md。出错原样回报首个错误。保留上轮原始证据；交付后移除本次临时 worktree。
