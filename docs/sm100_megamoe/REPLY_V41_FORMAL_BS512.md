# 【SM100】V4.1 Flash 正式 BS512 五臂中止回执

执行者：**SM100**。本轮未执行任何 SM90 任务书。

## 结论

按指挥官要求，因正式 BS512 公平性流程太慢，本轮实验已停止并释放八卡。
后两臂和跨臂公平性验证均未完成；当前**没有任何五臂有效计时或严格对比结果**。

指挥官给出的后续方向是：**去掉公平性验证，直接串行跑五臂**。本轮只完成停机、
释放、证据归档和回执，没有在该指令下启动新的五臂运行。

任务书：`docs/sm100_megamoe/V41_FORMAL_BS512.md`，
SHA256 `c11cbf86e95fc57af6bb78b737c958d10ed9c140753253d380e4f4771afaef98`。
执行仓 SHA：`0eaeec98d885d6ef0d521473daede8fb0c1fd62d`。上游 GCMoE 固定为
`fb8b65fa829c97962eb2dbc9a4ad8b759b9cd736`。BS 为 **512**：每步 512 个请求、
40 个专家层、TP8/EP8。

## 中止时进度

- 只有 `mega` 进入正式流程，启动时间 `2026-11-22T17:59:03+08:00`。
- `mega` 已完成全部 1000 个统计步，并生成 `bs512/plan.npz`。
- 八个 rank 的统计计数 `stats-rank*.npz` 已落盘。
- stats 边界的八个 KV checkpoint 已完成，逐 rank JSON 中记录了成功 SHA256。
- 停止时正在 measure 边界写 KV，`kv-measure/rank*.bin` 是未完成的残留文件。
- `mega/result.json` 未生成，因此 `mega` 也没有可用计时。
- `gc`、`mega_rr`、`gc_rr`、`gc_eco` 尚未启动。
- 任何跨臂 native top-1、controlled replay 或 10 对严格判定均未执行。

## 前置失败与修复

1. 首次启动在模型加载完成后发现输入归档不在 sparse worktree：
   `docs/trace_executor_20261119/sm90_main24_stable_decode_bs256_results/delivery/bs256/vanilla/rank0.npz`。
   该文件实际由 Git 跟踪；将目录加入 sparse checkout 后恢复，SHA256
   `ceaa52dc3f3b09fa95e08c59b83b56b97d90b8335b3437a506cb5dbd42fe09f2`。
2. 第二次启动因私有上游已应用 `natureal_matrix_audit.patch`，重复 `git apply`
   被拒绝。将上游恢复到固定 SHA，并移除上次复制产生的未跟踪文件后重跑。

两次失败的 launcher 与模型日志均保留。该过程未修改模型、kernel、输入、参考或
计时语义。

## 停机与释放

- 终止的是本轮进程组 `1027075`，包括 `run_five.sh`、`five_matrix.py` 和八个
  VLLM worker；没有终止其他用户的进程。
- 释放记录：`2026-11-22T18:21:21+08:00`。
- 释放后于 `2026-11-22T18:21:58+08:00` 检查：
  `nvidia-smi --query-compute-apps` 为空，`/dev/nvidia0..7` 无持有者。
- 证据：`released.log`、`processes-stopped.txt`、
  `gpu_release_after_stop.txt`。

## 清理

本轮私有目录从 263 GiB 清到 208 KiB。已删除：

- `bs512/` 内 256 GiB stats KV、部分 measure KV、route 张量和临时结果；
- `overlay/` 39 MiB；
- `dg-cache/` 1.5 MiB；
- 私有 `dsl-env/` 498 MiB；
- 私有 `upstream/` 18 MiB。

清理前已将轻量证据复制到
`docs/sm100_megamoe/v41_formal_bs512_results/`。详见 `cleanup.txt`。

## 后续运行口径

按指挥官要求，下一轮不再走公平性关键路径，直接串行执行五臂：

- 移除共享 KV backing store 的写入、恢复和边界比较；
- 移除跨臂 native top-1、controlled replay 与 10 对严格判定；
- 直接运行 `mega`、`gc`、`mega_rr`、`gc_rr`、`gc_eco` 并报告各自完整 MoE
  计时。

各臂自身的 eager/replay、输出有限性检查仍应保留。该替代运行尚未开始。

## 证据

结果目录：`docs/sm100_megamoe/v41_formal_bs512_results/`。包含任务与 SHA、
执行/上游 SHA、两轮失败日志、最终 launcher、`mega` 模型日志、1000 步统计
计数、`plan.npz`、8 个 KV 统计 JSON、KV/route 摘要、部分 measure 清单、
获取/释放记录、停机进程清单、释放后 GPU 检查、清理记录和 `SHA256SUMS`。
