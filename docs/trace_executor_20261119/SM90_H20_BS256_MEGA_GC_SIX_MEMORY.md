# SM90 · H20 · BS256 Mega/GC 六臂显存重试

排在 `SM90_BS256_LINEAR_WA_PAIR.md` 之后。先让该任务完成并释放预约，再从本任务提交启动。只预约同组八张空闲 H20，任一卡有人使用就等待，保留调度器分配的设备掩码。旧回执 `243072d`、已交付数据和节点目录 `/mnt/fuse/sm90-bs256-six-v2-Y9tGZbel` 用于追溯。本轮用新目录。

## 上次失败与本轮显存预算

上次 arm5 已完成 Mega MoE kernel 预热。随后 vLLM 在采样器预热的 `torch.softmax` 申请 128 MiB，GPU 7 当时仅余 35.31 MiB。该臂 10 GiB/rank KV 配置下，引擎报告 1,300,510 token 容量，本批保守需求为 692,480。优先减少预留 KV，保持完整步数；运行时以引擎实际容量、满批和逐层审计为准。

| 优先级 | 每臂统计步 + 正式步 | 每 rank KV | 最大上下文 | 固定请求的保守 KV 需求 | 每对正式原生 top-1 样本 |
|---|---:|---:|---:|---:|---:|
| 首选 | 1000 + 1000 | 7 GiB | 2705 | 692,480 token | 256,000 |
| 显存失败后 | 500 + 500 | 5 GiB | 1705 | 436,480 token | 128,000 |

固定 DeepSeek‑V4.1‑Flash、TP8/EP8、BS256（每 rank 32）、同一批 256 条真实 256-token 前缀、512-token prefill chunk、同一 BS256 FULL_DECODE_ONLY 图、greedy 和 40 层。发布者和六个消费者都使用当档预算。图预热完成后、提交请求前校验引擎实际 KV token 容量及并发能否容纳 256 条最大长度请求。每个窗口再校验每步都是 256 个纯 decode token。两档分别生成 KV、反馈和逐步路由，按档独立汇总。

## 执行

| 策略 | Mega · vLLM shared | GC · fused split shared | GC · vLLM shared |
|---|---:|---:|---:|
| 原生，48 槽/rank | 1 | 2 | 3 |
| 逐层 DeepInfer EPLB + RR，56 槽/rank | 5 | 4 | 6 |

顺序为 **0 → 5 → 1 → 2 → 3 → 4 → 6**。arm0 只发布统一参考；先测峰值显存最高的 Mega + EPLB。每个正式臂独立加载，GC 采用 M64 和 stream，fused 开 split shared，六臂关闭 ECO 与 SM-reuse。Mega routed 使用官方 SM90 FP8×FP8，GC routed 使用 FP8×FP4。使用旧六臂同一模型、数据、私有 wheel 和已验证工具链；本地权重副本按原清单验哈希。

在原仓库执行以下命令。记录打印的 job 路径，保留预约与启动器会话直到当前进程退出。

~~~bash
set -eo pipefail
git fetch origin main
book=docs/trace_executor_20261119/SM90_H20_BS256_MEGA_GC_SIX_MEMORY.md
rev=$(git log -1 --format=%H origin/main -- "$book")
job=$(mktemp -d /mnt/fuse/sm90-bs256-six-memory-XXXXXXXX)
echo "$job"
git worktree add --detach "$job/code" "$rev"
cd "$job/code"
rc=0
bash scripts/sm90/run_six_memory_retry.sh "$job/results" > "$job/launcher.log" 2>&1 || rc=$?
echo "$rc" > "$job/exit-code.txt"
exit "$rc"
~~~

- 首选档成功即结束。发现明确的 CUDA OOM 或引擎实际 KV 容量不足，启动器保存失败臂日志、退出码与档位，然后自动进入备用档，从 arm0 开始。备用档再出现显存失败就结束并交回两档证据。编译、设备忙、数据审计、原生输出差异或其他退出原因均按原档处理。
- 若断线先确认原作业是否仍在。需要恢复时，取得同组空闲卡预约，进入同一 `$job/code`，再次运行 `bash scripts/sm90/run_six_memory_retry.sh "$job/results"`。启动器验证测量 SHA、档位与已完成臂文件哈希。工作树和结果目录分开。执行者仅处理保持语义的编译环境问题并附工具链与 diff，运行参数由脚本统一设定。
- 完成标记以 `armN/complete` 和 `artifacts.sha256` 为准，最后一步 decode 打印之后仍有跨 rank 审计与落盘。若收到“当前臂完成后停止”的新指令，等该臂完成归档再停止后续调度，避免组杀覆盖尚在写盘的 worker。

## 回执

1. 在 `docs/trace_handover_20261119/REPLY_SM90_H20_BS256_MEGA_GC_SIX_MEMORY.md` 写明档位、测量 SHA、六臂状态及 GPU 释放。数据放在 `docs/trace_executor_20261119/sm90_h20_bs256_mega_gc_six_memory_results/`，保留实际采用的 `window-1000` 或 `window-500` 名称。若发生回退，两档分别归档。
2. 交付每臂 `run.json`、八 rank NPZ、`artifacts.sha256`、退出码、每臂耗时与日志证据，以及 `summary.csv`、`pairs.csv`、`verification.json`。列出每臂本地 KV 配置、引擎实际 KV 容量、最大并发、worker 的显存记录。失败臂附首错、所处阶段、申请量与当时可用显存；发布者和失败臂单列。需要只重建报告时，使用实验 Python 运行 `scripts/sm90/report_eight.py "$job/results/window-实际步数"`。
3. 主表六行同时给出完整 step ms/步、完整 FFN ms/步和 µs/层、token IR、总 M 块、峰值 rank M 块、M 块 IR、活跃物理槽和实测 block M。整步取正式窗口的同步墙钟；FFN 按每层最慢 rank 的完整 FFN 入口至返回计时，再对 40 层求和。按同一档比较请求、反馈、路由、KV、放置、完整图与 shared owner。报告全部 15 对的原生 top-1 差异、预热哈希与 strict 标志；差异对只报告固定反馈轨迹下的实测值。性能变化使用 `100 × (B/A − 1)`，负值表示 B 更快。
4. GPU 阶段结束立即释放八卡预约，CPU 汇总可随后完成。确认回执、原始产物路径和校验和可读，再清理本轮专属可重建临时文件。历史回执、原始数据、共享环境与其他用户文件继续保留。
