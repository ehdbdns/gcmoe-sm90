# SM100 · 在两个整模后端上迁移真实的逐层专家权重

两臂各启动一次。目标是验证 40 层同步换槽、DeepInfer 副本顺序与换槽后的 RR decode。共用 SM90 BS512 已归档的前八步逻辑需求作为迁移用例；正式五臂的放置需要由 SM100 自身的统计窗口产生。

1. 拉取 main 并记录 SHA。沿用 [RR 任务书](V41_DEEPINFER_RR.md) 的同一 checkpoint、venv、CUDA 13.5 ptxas、八卡绑定及锁。先在仓库根目录、已激活同一 venv 时，运行一次以下 CPU 命令。把打印的 plan 绝对路径传给两次启动。
   ```bash
   plan_dir=$(mktemp -d /tmp/sm100-placement-XXXXXXXX)
   plan="$plan_dir/plan.npz"
   PYTHONPATH="$PWD/source" OMP_NUM_THREADS=1 python scripts/sm100/v41_placement.py docs/trace_executor_20261119/sm90_main24_stable_decode_bs512_results/delivery/bs512/stats-demand.npz "$plan"
   realpath "$plan"
   sha256sum "$plan"
   ```
2. 仅在指定八卡无人占用且无人预约时持锁运行，有占用就等待。在上一轮两条模型命令上均追加 `--placement-file /上一步打印的绝对路径/plan.npz`（使用实际文件绝对路径）。依次运行 Mega、GC，各保留新的工作目录、RR 预检、版本、首错和退出码。仅允许有记录的等价编译修复，其他失败原样回执。共同迁移路径失败时先停止。
3. 核对两臂使用同一 plan SHA，报告每层实际换槽数、所有 rank 的迁移后权重抽样不符数、8 个 rank 的 RR 调用数、40 层最终布局哈希及逐步输出。与原始放置下的四个 ID [75,200,18,200] 对比，记录首个偏差。输出不一致仍交付证据。每个槽每个权重组件抽查 32 个字节；该计数属于抽样检查。
4. 将计划文件、两臂完整模型日志、结构化结果和八卡释放证据放入 docs/sm100_megamoe/v41_deepinfer_placement_results/，回执写入 docs/sm100_megamoe/REPLY_V41_DEEPINFER_PLACEMENT.md 并推送 main。本轮不做性能计时。
