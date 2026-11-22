# 【SM100】BS1024 原生 MegaMoE 与原生 GCMoE 同输入对照回执

本回执仅由 SM100 执行；未执行 SM90 任务书。

## 结论

三个样本均完成了同输入、同 rank、同 40 层图捕获与计时，但 **没有样本满足严格逐元素性能判定**：Mega 对实时参考均为 0 差异，GC 在三个样本分别有 21、4、11 个元素不等。因此下列耗时只作本次观测，不作严格性能结论。观测方向一致：GC 比 Mega 慢 5.2907% 至 6.0233%。

任务书：`docs/sm100_megamoe/V41_NATIVE_PAIR.md`，SHA256 `41826b5bb360e94696c2af17a93812e2b05f123d3d128c4dfb79601f2964471b`。执行基线 `e477f5a752aeedcf29e759f7b330b92bbda1643c`，GC 捕获修复提交 `3e24814b272aaef0cc4c84c14ca1b2a48d2e9961`。

## 配置与执行

- 模型：`/data0/home/admin/workspace/megamoe/native-v41-KO1hTbTj/model`；TP8 / EP8，每个样本 40 个真实专家层，输入 top-6、每 rank 48 个本地专家及 fused shared expert。
- 输入：`docs/trace_executor_20261119/sm90_main24_stable_decode_bs256_results/delivery/bs256/vanilla/rank0.npz`，SHA256 `ceaa52dc3f3b09fa95e08c59b83b56b97d90b8335b3437a506cb5dbd42fe09f2`。以其中 8 条 prompt 复制扩展为 BS1024。
- 采集步：0、8、16。每步每个 rank 保存实际层输入与实时输出；两臂从同一 `step{0,8,16}-rank{0..7}.pt` 重新加载。
- 计时：40 层按层序合为一张图；每组重放 20 次，共 15 组 CUDA event；每组取最慢 rank，再取 15 组中位数。图外完成采集、存盘、验证和输出比较。
- 首次执行：`bash scripts/sm100/run_five.sh "$work" native-pair`。Mega 成功；GC 首次在 `pair_time` 捕获时报 `CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED`，未产生计时样本。
- 修复后重试：`python -u scripts/sm100/native_pair.py "$work" gc`，使用相同 8 卡 UUID、`GC_SM100_MATRIX_CLOCKS=0`、私有 `DG_JIT_CACHE_DIR`，退出码 0。
- 成功测量轮次：Mega 1 次，GC 1 次（首次捕获尝试不含任何可用计时，失败堆栈完整保留）。

## 三步绝对耗时

| Step | Mega (ms) | GC (ms) | GC 相对 Mega | 严格判定 |
|---:|---:|---:|---:|:---:|
| 0 | 7.887664 | 8.362762 | +6.0233% | 否 |
| 8 | 7.852147 | 8.279971 | +5.4485% | 否 |
| 16 | 7.898392 | 8.316273 | +5.2907% | 否 |

正号表示 GC 更慢。严格判定要求同一实时参考逐元素无差异；三个样本的 GC 均有差异，故不宣称严格性能胜出或失败。

## 15 组分布

表中每组值是最慢 rank 的单次重放耗时；中位数即上表结果。

| Arm | Step | Min (ms) | Median (ms) | Max (ms) | Max/Min-1 |
|---|---:|---:|---:|---:|---:|
| Mega | 0 | 7.883768 | 7.887664 | 7.903784 | 0.2539% |
| Mega | 8 | 7.849867 | 7.852147 | 7.854722 | 0.0618% |
| Mega | 16 | 7.895248 | 7.898392 | 7.903971 | 0.1105% |
| GC | 0 | 8.358704 | 8.362762 | 8.369374 | 0.1276% |
| GC | 8 | 8.271320 | 8.279971 | 8.286448 | 0.1829% |
| GC | 16 | 8.311121 | 8.316273 | 8.323008 | 0.1430% |

全部 90 个逐组值见 `v41_native_pair_results/groups.csv`。

## 输入摘要与输出差异

- 输入/参考文件摘要：Mega 与 GC 的 24/24 个 step-rank `.pt` SHA256 完全相等；两臂对同一 rank、同一步加载同一文件。
- Eager/replay 一致性：两臂各 960/960 个层输出（8 rank x 3 step x 40 layer）通过 `torch.equal`，即各自图输出与同后端 eager 输出逐位相等。
- Mega 图对实时 Mega 参考：三步均为 0 个不同元素。
- GC 图对实时 Mega 参考（每步总元素 8 x 40 x 655360 = 209,715,200）：

| Step | 不同元素 | Max abs | 各 rank 不同元素数 | 各 rank 最大绝对误差 |
|---:|---:|---:|---|---|
| 0 | 21 | 2.0 | `[1,0,1,1,9,4,4,1]` | `[2.98e-8,0,1.49e-8,4.88e-4,1,2,2,3.73e-9]` |
| 8 | 4 | 0.015625 | `[0,0,1,0,0,0,0,3]` | `[0,0,6.10e-5,0,0,0,0,1.5625e-2]` |
| 16 | 11 | 8.0 | `[0,2,0,9,0,0,0,0]` | `[0,4.88e-4,0,8,0,0,0,0]` |

GC 合计 36 / 629,145,600 个元素不同（5.72e-8，比例 5.72e-6%）。逐 rank 与逐层完整记录在 `gc.json`、`output_diff_by_rank.csv`、`output_diff_by_layer.csv`。

## 后端、失败与语义修复

- Mega 8/8 rank 后端：`vllm.third_party.deep_gemm.mega`。
- GC 8/8 rank 后端：`gc_moe.gc`。
- 首个 GC 失败：`_ChildGraph.__init__` 在父图 stream capture 内调用 `cuGraphInstantiate`，返回 `CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED (900)`。完整首栈在 `v41_native_pair_results/gc-attempt0.log`。
- 修复：提交 `3e24814` 移除构造函数中的无条件 `self.instantiate()`；普通 launch 仍在 `cuGraphLaunch` 前实例化，捕获路径交给父图实例化。仅改变实例化时机，不改变计算、布局或数值语义。diff 为 `gc_capture_fix.diff`。
- 退出码：`mega.exit-code=0`；`gc-attempt0.exit-code=1`；`gc-retry1.exit-code=0`。`gc-launcher.exit-code=1` 是首次 launcher 退出，正式测量以重试 0 为准。

## 释放与清理

- 使用 `gpus.txt` 记录的 8 张 L20D UUID；预约开始 `2026-11-22T15:29:19+08:00`。
- 首次运行于 `15:36:14+08:00` 释放；GC 重试于 `15:37:13` 开始、`15:40:08` 结束。
- 重试后检查：`nvidia-smi --query-compute-apps` 为空，`/dev/nvidia0..7` 无持有者；证据为 `gpu_release_after_retry.txt`。
- 已删除本轮私有 `dg-cache`（1.5M，路径检查后删除）；无 overlay、无 `package-*` 目录。记录见 `cleanup.txt`。未清理或终止他人 GPU 作业。

## 提交内容

结果目录：`docs/sm100_megamoe/v41_native_pair_results/`，包括 `comparison.json`、`mega.json`、`gc.json`、两臂日志、首次失败与重试日志、`versions.txt`、`weights.sha256`、`gpus.txt`、`revision`、输入/任务/24 份测量 SHA、退出码、修复 diff、逐组与逐层差异 CSV、释放/清理记录及 `SHA256SUMS`。

24 份 `.pt` 输入/参考文件各 105,272,703 字节，超过 GitHub 单文件 100 MiB 限制，未复制进 Git；保留在 `/data0/home/admin/workspace/megamoe/v41-native-pair-bs1024-01/step{0,8,16}-rank{0..7}.pt`，其 SHA256 已推送在 `measurements.sha256`。输入归档另保留持久副本 `/data0/home/admin/workspace/megamoe/v41-native-pair-bs1024-01/input-rank0.npz`，SHA256 同上。
