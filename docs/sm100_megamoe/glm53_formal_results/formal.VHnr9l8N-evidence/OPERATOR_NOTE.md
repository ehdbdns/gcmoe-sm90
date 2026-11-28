# 执行者补充说明 · formal.VHnr9l8N

- 21:28:33 起在八卡确认整组空闲后按任务书 `--resume` 启动（第一次尝试 21:09 因外部 root `rtp_llm_rank` 抢占导致 vLLM 启动显存预检失败，退出码 1，未改语义）。
- `rr` 臂完整成功：`stats-rank*.npz`、`measure-rank*.npz`、`perfetto/` 共 41 个文件。
- `eco` 臂在 21:49:19 触发内核 `device_trap`：`GC MoE combine readiness timeout` 反复出现（rank/peer 组合覆盖 0 与 7），随后 7 张卡报 `Xid 43`，只剩 GPU 0/7 空转 100%，日志停滞。
- 已按协议停止本任务进程、释放八卡；`Xid 43` 属内核主动 trap，非外部进程抢占。
- 证据：`evidence/dmesg-xid.txt`、`evidence/eco-tail.txt`（对应 result 目录内 `eco-extra8-try0.log` 末尾）。
- 未做任何修改语义的补丁；`8` 槽设置保持任务书默认，未触发 4 槽回退（回退条件要求 CUDA OOM，本次是内核 barrier 死锁）。
