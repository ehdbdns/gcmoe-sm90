# SM90 · H20 EP8 六档批量整模四臂复测

在 DeepSeek‑V4.1‑Flash 上测量 shared L1 优先启动、ECO 使用线性价格之后的完整 MoE 耗时。六档各跑四臂，共 24 次整模运行。每臂先统计 1000 个满批纯 decode 步，再连续测量 1000 步，每步覆盖 40 层。BS256、512、1024 全部重新测量。启动顺序直接随本版本上线，本任务只运行下表四臂。

| 全局 BS | 32 | 64 | 128 | 256 | 512 | 1024 |
|---|---:|---:|---:|---:|---:|---:|
| EP8 均摊请求数/rank | 4 | 8 | 16 | 32 | 64 | 128 |
| 每对比较的正式原生 top‑1 数 | 32,000 | 64,000 | 128,000 | 256,000 | 512,000 | 1,024,000 |

| 顺序 | 标识 | 专家放置与分流 |
|---:|---|---|
| 1 | vanilla | 无副本、48 槽/rank，生成本档统一参考，同时测量原生 |
| 2 | random | 固定随机种子 0 的随机放置、56 槽/rank，RR |
| 3 | rr | 逐层 DeepInfer EPLB、56 槽/rank，RR，发布本档放置 |
| 4 | eco | 同一逐层 EPLB rank 成员，按线性价格拍卖 |

## 算法和计时

沿用 Word 小结的 TP8/EP8、FULL_DECODE_ONLY CUDA Graph、M64、按 M 块就绪发号、pull、fused split shared、20 GiB KV/rank、关闭 SM-reuse。plan 独占通信分区 6 个 SM 中的 1 个，dispatch 使用 20 个 CTA，compute 分区保留 72 个 SM。每臂加载前预热本地权重文件，权重索引、模型配置和两份数据清单按哈希校验。

四臂均先提交 shared L1。CUDA Graph 的 launch-order 边在 L1 所有 CTA 开始执行后放行后继。ECO 上 plan 和 dispatch 入场门可与正在计算的 L1 重叠，dispatch 继续等 plan 驻留信号。其余臂放行 dispatch。routed 仍需等待 shared L1 完成，shared L2 继续按块消费 L1 的结果。记录测量批量对应的 `[ep8]` 图角色与依赖日志。本轮不增加启动顺序 A/B 或 profiler 测量臂。

每个 rank 的价格为 **T + 26M + 2A**。T 是接收的路由行数，M 是各物理槽行数分别除以 64 向上取整后的和，A 是非空物理槽数。模式 0，暖轮 1、冷轮 6，满 M 块进入空槽的 slack 为 92。八 rank 的 prepared mode/slack 以及每步每层的设备 plan 分数由报告逐项核验。BS256 回执 `d29002c` 实测 wa=0/1 分别为 830.91/824.15 µs/层，19/20 个配对步段改善。固定这两份分配，wa 从 1 提到 2 的 routed MAE 分别从 0.14858 到 0.14825、0.15271 到 0.15238 个百分点。旧版线性整模归档在大 BS 上也出现过 wa=1 的误差更低，本轮统一使用 2，实际效果取完整 MoE 实测。

## 执行一次

1. 使用站点授权的同组八张空闲 H20，保留调度器设备掩码。任一卡在用就等待。若 Mega 六臂任务已在运行，先让该任务按原计划结束；两个任务使用同一八卡协作锁及各自独立的工作目录。执行者只处理保持语义的编译问题，回执附原始错误、diff、版本及复验。改变代码版本后，参与同组比较的四臂须使用同一版本。
2. 从原仓库执行以下命令。默认本地模型为 `/mnt/fuse/models/DeepSeek-V4.1-Flash`，可通过 `SM90_LOCAL_MODEL` 指向已校验的节点本地副本。把打印的 job 路径写入执行日志。结果盘使用原整模实验的存储路径；脚本按 BS 在每组开始时核算两份八卡 KV 和路由所需磁盘空间。

~~~bash
set -eo pipefail
git fetch origin main
book=docs/trace_executor_20261119/SM90_EP8_LINEAR_FOUR_6BS.md
rev=$(git log -1 --format=%H origin/main -- "$book")
test -n "$rev"
job=$(mktemp -d /mnt/fuse/sm90-ep8-linear-four-XXXXXXXX)
echo "$job"
git worktree add --detach "$job/code" "$rev"
cd "$job/code"
rc=0
bash scripts/sm90/run_full_model_four.sh --linear-ep8 "$job" > "$job/launcher.log" 2>&1 || rc=$?
echo "$rc" > "$job/exit-code.txt"
exit "$rc"
~~~

3. 四个小批量使用第一份原始 JSONL。BS512 使用原两份 JSONL，BS1024 按历史规则重复同一批 BS512 请求并校验摘要。每档在模型加载前对照历史同档 prompt 哈希。每档重新生成自己的统一参考 token、统计与测量 route bank 及两次边界 KV。四臂使用相同输入和测量窗口。原生与三个消费者均执行同一套设备路由选择和审计，文件读写位于计时窗口外。
4. 每臂每 rank 核验 40,000 次连续层调用、完整图的 1000 次正式回放、路由 ID/权重、行序、逻辑需求、逻辑权重、请求/反馈顺序、prefill 与测量边界 KV 读回。RR 另逐次核对 DeepInfer 物理槽计数，ECO 核对每个 rank 的放置成员。每档六对均要求预热原生输出摘要相同、正式原生 top‑1 零差异。原生 FFN、随机 RR 和 EPLB RR 均使用真实 split shared；`[ep8]` 应打印 M64 和角色 `(4,1,8,2,3)`，ECO 为 `(4,5,7,1,8,2,3)`。
5. 脚本逐档落盘，保留首错并继续其他可运行的 BS。断线先确认原进程是否仍在。相同 checkout、相同 job 和上述命令可检查并跳过校验和完整的组；保留未完整交付的组和现场。若仅报告失败，使用该节点实验 Python 重跑 `scripts/trace_layer/main20_report.py "$job/delivery-$bs" --linear-ep8 --bs "$bs"`，附原始失败和复验结果。请求数、步数、价格、显存预算和输入固定。

## 指标与交付

- 主表按 Word 小结列出每层 MoE µs、相对原生及相对 RR 的有符号变化、token IR、八卡总 M 块、峰值 rank M 块、八卡活跃物理槽、M 块 IR。routed MAE（百分点）和每层调用成交次数只展示 ECO。另列峰值 rank 激活槽、报价数、总成交、接受率、零成交调用数和逐层明细。MoE 按同一步同一层八个 rank 的 graph 起点至 combine 结束取最大，再平均 40,000 次。变化使用 `100 × (本臂/参考臂 − 1)`，负值表示耗时下降。
- 对每次调用分别按 routed 跨度和完整 MoE 跨度确定慢 rank，统计慢 rank 命中 M 块峰值、激活槽峰值、同时命中两项的次数。时间或指标并列时，只要有同一个最慢 rank 满足条件就计该调用一次。routed 跨度从第一个 CTA 真正开始计算到最后一个计算 CTA 结束，包含末尾 push。
- 汇总检查主表 24 行、层表 960 行、拍卖表 6 行、公平性对照 36 对。高三档另有 `historical.csv` 12 行，逐臂列出 Word 原值、重测值、相对变化，以及相对同期 RR 的变化差（百分点）。BS256 ECO 同时列出相对最新 wa=1 实测的变化和两份运行的工作负载核对结果。`controlled_inputs_match` 核对跨版本输入、反馈、路由、KV 和放置摘要。六对同轮公平性和跨轮可比性分别填写。新旧两轮同时改变了启动顺序和 ECO 价格，这个比较衡量组合效果。任何失败或未通过的对照写明具体原因。
- 结果交付到 `docs/trace_executor_20261119/sm90_ep8_linear_four_6bs_results/`，回执写入 `docs/trace_handover_20261119/REPLY_SM90_EP8_LINEAR_FOUR_6BS.md`。附每档原始八 rank NPZ、run.json、设备与空闲记录、测量 SHA、提交日志、checkpoint manifest、放置、工作负载摘要及校验和。对每档 delivery 执行 `sha256sum --quiet -c SHA256SUMS`。GPU 工作完成后立即释放预约，核实本任务 worker 全部退出。交付复核后清理本轮专属可重建临时缓存，共享模型、历史档案与其他任务目录继续保留。
