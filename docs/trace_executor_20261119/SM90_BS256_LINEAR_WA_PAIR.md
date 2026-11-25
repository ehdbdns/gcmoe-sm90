# SM90 · BS256 线性 ECO 激活专家权重对照

**当前执行**　Mega 六臂作业已按指令停止，执行者在回执 `243072d` 中确认八卡释放。本轮从新的私有目录开始，保留上一轮现场 `/mnt/fuse/sm90-bs256-six-v2-Y9tGZbel` 的 KV、route bank 和未完成臂日志。先进入站点预约队列。使用同组八张 H20，保留调度器分配的设备掩码。任一卡有其他进程时等待。启动器在每次模型启动前取得与 Mega 相同的八卡协作锁，并在锁内复查占用。

## 实验

在 DeepSeek‑V4.1‑Flash 上固定 BS256、TP8/EP8、逐层 EPLB、M64、stream、pull、fused split shared、关闭 SM-reuse、FULL_DECODE_ONLY CUDA Graph。通信分区共 6 个 SM，plan 取其中 1 个，compute 保有 72 个。每臂先跑 1000 个满批纯 decode 步统计，再重排并跑 1000 个满批步测量 40 层。每 rank 均摊 32 个请求；价格中的 token 数取该 rank 各物理槽实际收到的路由行数。

| 正式臂 | Rank 线性价格 | 实际 role 5 slack |
|---|---|---:|
| eco_linear | token 数 + 26 × M 块数 | 90 |
| eco_linear_wa1 | token 数 + 26 × M 块数 + 激活专家槽数 | 91 |

同一次层调用里，一个物理槽只要收到至少一行就计为活跃。线性模式为 0，token 权重为 1。slack 由同一个满 M 块在空槽上的价格自动导出。冷、暖拍卖分别为 6、1 轮。同时比较八卡中的激活槽数峰值和完整 MoE 耗时。BS256 是前次离线拟合中信号最大的一档。

启动顺序为 **vanilla 参考 → wa=0 → wa=1**。共三次模型加载，两条正式性能记录。参考进程在本轮的独立目录生成输入 tokens、统计与正式两套 route bank 和两次边界 KV。两个测量臂均以消费者身份读取同一份数据，分别从完整统计历史求逐层 EPLB，使用相同权重放置与槽序，重排后 ECO 从空的 quota 状态开始。本轮使用 BS256 主表第一份原始 JSONL，固定请求顺序，KV 为每卡 20 GiB，加载前预热本地 48 个权重分片。

## 启动和验收

1. 核对 Mega 六臂最终退出状态、站点授权和八卡占用。用单独的任务目录、checkout 和 checkpoint 盘。保留 Mega 的进程、文件、环境和结果。在节点既有实验环境运行下列命令，模型默认取本地副本 `/mnt/fuse/models/DeepSeek-V4.1-Flash`。若副本在另一块本地盘，提前设置 `SM90_LOCAL_MODEL`。启动器按权重索引、模型配置、数据集哈希和分片数量预检输入。启动器按两份八卡 KV、两套路由和工作区检查可用空间，BS256 约需 335 GiB。环境变量 `GC_MOE_CHECKPOINT_ROOT` 可指定独立的充足空间。

~~~bash
set -eo pipefail
git fetch origin main
book=docs/trace_executor_20261119/SM90_BS256_LINEAR_WA_PAIR.md
rev=$(git log -1 --format=%H origin/main -- "$book")
test -n "$rev"
job=$(mktemp -d /mnt/fuse/sm90-bs256-linear-wa-XXXXXXXX)
echo "$job"
git worktree add --detach "$job/code" "$rev"
cd "$job/code"
rc=0
bash scripts/sm90/run_linear_wa_pair.sh "$job" > "$job/launcher.log" 2>&1 || rc=$?
echo "$rc" > "$job/exit-code.txt"
exit "$rc"
~~~

2. 三个进程均须使用同一执行 SHA。记录各臂 `runner.log`、`execution.sha`、`status.txt`、硬件、空闲记录、图预热、统计与正式首 epoch。两个窗口各有 40,000 个连续层调用，首 epoch 相差 40,000。八 rank 的路由 ID、权重、源行序、逻辑需求、两次边界 KV 读回、逻辑专家权重及请求顺序通过现有校验。两正式臂安装的逐层物理槽序逐元素一致。正式窗口比较 256,000 个原生 top-1，要求零差异，同时要求统计窗口原生输出摘要相同。拍卖后各槽的 token 数、M 块和激活槽数是被测结果，允许随权重变化。
3. `linear_wa_report.py` 核验八 rank 实际准备的价格模式与 slack，并对每次调用每个 rank 按物理计数重算在线价格，与 GPU 写回值逐项相等。仅 CPU 汇总失败时，保留现场，用实验 Python 重跑 `scripts/trace_layer/linear_wa_report.py "$job/delivery"` 即可定位；保留退出状态、公平性和校验失败信息。运行中仅处理能保持计时、分配及实验语义的编译错误，附原始报错、修复 diff、实际 SHA。若需更换代码版本，正式对照须使用同一修复版本。其他错误或公平性失败直接回执，附首错栈和完成臂清单。
4. GPU 进程结束即释放预约。成功时在 `$job/delivery` 运行 `sha256sum --quiet -c SHA256SUMS`，离线汇总。若仅重新生成了报告，须从该 delivery 重新生成并复核 SHA256SUMS，保留原失败记录。

## 回执

- 打印两行正式主表。主指标是同一步同层八 rank 的最大本地 graph 起点至 combine 结束跨度，再平均 40,000 次，单位 µs/层。列出 wa=1 相对 wa=0 的绝对差和有符号变化百分比，负值表示更快。把 `blocks.csv` 中相同步段的 20 对 50 步均值配对，报告多少段变快、多少段变慢。两臂各一次运行，若差异很小或各段方向不一致，说明观测幅度与波动。
- 同表列出 token IR、八卡总 M 块、峰值 rank M 块、M 块 IR、八卡活跃槽总数和**峰值 rank 激活槽数**。激活峰值先逐调用取八卡最大，再取平均。另列逐调用的理想下界 `ceil(全局活跃逻辑专家数 / 8)` 的均值。给出峰值下降、持平、上升各多少次，三项合计应为 40,000。
- 列出 routed 负载份额 MAE（百分点），范围从首个实际 routed CTA 开始计算到最后一个 routed CTA 完成，含末尾 push。交付拍卖总报价、总成交、接受率、每层调用平均成交和零成交调用数。分别回答专家峰值、routed MAE、完整 MoE 三项如何变化。
- 结果提交到 `docs/trace_executor_20261119/sm90_bs256_linear_wa_pair_results/`，回执提交到 `docs/trace_handover_20261119/REPLY_SM90_BS256_LINEAR_WA_PAIR.md`。归档 delivery（含逐 rank NPZ、输入校验、`main_table.csv` 两行、`layers.csv` 80 行、`auction.csv` 两行、`blocks.csv` 40 行、`pair.csv` 一行）、启动日志、退出码与执行 SHA。默认在验收通过后释放本轮 KV 二进制，保留两份 KV manifest 和输入摘要。route bank 保留在本轮私有现场至归档完成。确认远端归档可读后清理本轮可重建的私有文件与 worktree，记录清理范围；失败现场保留到协调端复查。
