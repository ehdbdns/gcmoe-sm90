# 当前八卡任务

已收到 R3 回执 `ccd28c9`。固定负载下 E2E 下降 2.07%，整步 MoE 下降 0.73%，采样步 60/256 个原生 top‑1 有差异。执行 R4，两臂使用此次修复的 ECO、旧线性价格和同一参考负载。两次都用 baseline 预热，重排后才切换 stream。回执写入本目录的 `REPLY_FIXED_STREAM_R4.md`。[R4 任务书](../trace_executor_20261119/MULTILAYER_AB_TASK.md)

执行者仅可处理语义等价的编译问题。每臂限启动模型一次，其余情况保留证据直接回执。valid 耗时函数任务继续暂缓。
