# V41 BS512 层间到达偏差 CPU 复核结果

本目录只含 **SM100** 的 CPU 分析，未申请 GPU、未启动模型。

- `intervals.csv`：624 行（2 臂 × 8 rank × 39 个层间区间）。
- `kernels.csv`：每个区间内按 kernel 名称分组的累计设备时间。
- `skew_summary.csv`：层 8/20/32/39 的 entry 最小/最大 rank 对比。
- `kernel_top.csv`：上述 16 个 rank 的 top 5 kernel 分组。
- `validation.txt`：行数、几何、恒等式、图提交完整性及极值结果。
- `versions.txt`：任务、源码、Python/numpy 和执行环境。

全部 624 个区间都识别到 40 次有序 `cudaGraphLaunch`，提交两列均非空。
kernel 分组可能重叠，不能把 `summed_us` 当作 GPU 活动并集。空档边界只说明区间
内 GPU 无活动，不能单独证明某个 CPU 算子或调度原因；同步/collective kernel
本身也可能包含等待。
