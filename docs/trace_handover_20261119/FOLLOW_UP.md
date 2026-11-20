# 当前八卡任务入口

已收到 R2 回执 `16baf33`。请拉取最新 `main`，执行任务一 R3。先在 CPU 核对 R2 保存的负载并复算首步 40 层缺口，然后在同一修复版工作树上，使用同一份 token、路由和布局，依次运行 baseline 和 stream 各一次。保留全部 105 步的完整性校验，新回执写到本目录的 `REPLY_FIXED_STREAM_R3.md`。执行入口见[两次运行任务书](../trace_executor_20261119/MULTILAYER_AB_TASK.md)。

任务二暂缓。发布者先复核任务一的输出一致性、E2E 和整步 MoE 实测收益，再更新此入口。解锁后只比较 stream 上的旧、新耗时函数，各一次。见[耗时函数任务书](../trace_executor_20261119/VALID_COST_TASK.md)。

执行者只可处理语义等价的最小编译修复。其余失败保留日志和已完成结果，按任务书回执。

## 回执 92788ab 后续

旧采样中八份 torch trace 都有 40 层 × 5 角色，每 rank 一次 CUDA Graph Launch。设备快照推进量为 120，具体是哪一侧快照边界导致的偏差仍需设备证据。修订版把设备快照当候选范围，用每层设备打点和整步层间 dispatch 间隔与 torch trace 对照。只有八卡共同匹配唯一一组连续 40 个带标签 epoch 时才写入正式 `capture.json`，并保留原始快照边界及候选 epoch。匹配失败时保留 `capture.invalid.json` 供诊断。

历史执行说明。此前按 [任务书](../trace_executor_20261119/README.md) 的单一入口运行，在步骤 2 通过后执行原定 BS256 采样。成功或失败按任务书回传。原始失败 run 保留作证据。
