# 当前八卡任务入口

请执行[多层双臂对照任务书](../trace_executor_20261119/MULTILAYER_AB_TASK.md)，在本目录提交 `REPLY_MULTILAYER_AB.md`。

执行者的代码修复权限仅限保持原语义的最小编译兼容性修改。运行、采集或验收失败时，请按当前任务书回传证据。

## 回执 92788ab 后续

旧采样中八份 torch trace 都有 40 层 × 5 角色，每 rank 一次 CUDA Graph Launch。设备快照推进量为 120，具体是哪一侧快照边界导致的偏差仍需设备证据。修订版把设备快照当候选范围，用每层设备打点和整步层间 dispatch 间隔与 torch trace 对照。只有八卡共同匹配唯一一组连续 40 个带标签 epoch 时才写入正式 `capture.json`，并保留原始快照边界及候选 epoch。匹配失败时保留 `capture.invalid.json` 供诊断。

本轮恢复八卡采集。请按 [任务书](../trace_executor_20261119/README.md) 的单一入口运行，在步骤 2 通过后执行原定 BS256 采样。成功或失败按任务书回传。原始失败 run 保留作证据。
