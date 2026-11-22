# V41 BS512 主机提交延迟 CPU 结果

本目录只含 **SM100** 的离线 CPU 分析，不申请 GPU、不启动模型、不修改原始 profile 或计时语义。

- `host_segments.csv.gz`：640 个 arm/rank/layer 窗口、274,995 个最内层独占时间片段。
- `cumulative_submission_diffs.csv`：层 8/20/32/39 极值 rank 对从第 0 层到目标层的逐层累计提交差与新增差。
- `expansion_windows.csv`：首次达到 100 us 的窗口、其后新增差最大窗口、目标层新增差和累计差。
- `contributions.csv`：上述两类窗口的 category 总差、真实调用实例 top 5、unobserved 片段 top 5。
- `call_evidence.csv`：关键调用的 arm/rank/pid/tid、相对时间、独占时间、父调用时长、内嵌 CUDA correlation/kernel、前后调用。
- `key_intervals.csv`：第 20 层首次显著扩大和最大扩大窗口的类别分解与恒等式误差。
- `launch_gap_evidence.csv`：第 20 层 dense GEMM 前空档的“尚未提交”和“已提交待执行”拆分。
- `sequence_alignment.txt`：两端片段序列与真实调用序列的对齐情况。
- `task.md`/`task.sha256`、`source-raw-evidence.sha256`、`versions.txt`、`validation.txt`、`command.txt`、`run.log`：任务、输入、环境和实跑记录。

`unobserved` 是 Kineto 没有任何 span 覆盖的主机时间，只能给出两端调用和微秒数，不能据此认定操作系统抢占或某个具体函数。重复同名调用按同一线程的调用序号匹配；一个调用的多个嵌套独占片段先汇总到该调用实例，再作两端差。
