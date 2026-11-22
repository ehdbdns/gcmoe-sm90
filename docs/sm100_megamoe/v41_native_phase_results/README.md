# V41 原生阶段 Perfetto 结果

本目录只包含 **SM100** 执行结果。任务书为 `../V41_NATIVE_PHASE_PERFETTO.md`，
SHA256 `86b8b2a1eea3a1c566af27bfbe48c7637437b048e9d95f3987ce90f46d4fcf41`。

## 范围

- `mega`、`gc` 各启动一次，BS512，TP8/EP8，40 层，原生 greedy decode。
- 每臂 1000 个统计步，随后 3 个测量步；只追踪零基 step 2。
- 每臂五层 `0,8,20,32,39`，每层一张八 rank Perfetto 与对应 counts。
- 每层八 rank token 合计 3072；关键 rank 按完整 MoE 设备跨度最大者选取。

## 文件

- `phase_table.csv`：五层关键 rank 的 10 个重叠阶段，单位 us。
- `validation.txt`：几何、token、M 块、漂移与文件数量检查。
- `mega/`、`gc/`：Perfetto、counts、8 个 measure NPZ、result 与逐臂日志。
- `raw-evidence.sha256`：执行机原始 Torch profile 与 anchor 的相对路径哈希。
- `gpu_release.txt`、`processes-stopped.txt`、`lock-release.txt`：释放证据。
- `SHA256SUMS`：本目录提交文件的校验清单。

原始 profile 约 3.8 MiB/rank，保留在
`/data0/home/admin/workspace/megamoe/v41-phase-bs512-ku9Hv473/bs512/{mega,gc}/`，不提交 Git。阶段区间可重叠，不能相加当作 MoE
总时；带 profiler 的 `result.json` 墙钟不作为速度结论。
