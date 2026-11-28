# SM90 · 8×H20 · 每个源 rank 独立请求的六档整模四臂复测

跑 DeepSeek-V4.1-Flash 的 24 个正式实验。**BS 表示每个源 rank 在一个纯 decode 步交给 MoE 的独立请求数**。一个请求在该步提供一个输入 token，产生一个输出 token，经过每层时选 6 个 routed 专家。以 BS8 为例，8 个 rank 各拥有 8 个不同请求，整步生成 64 个 token，每个 MoE 层应收到 384 个 routed 行。单个目的 rank 的接收量由路由和放置决定。

| 主表 BS（每源 rank） | 全局独立流／每步生成 token | 每源每层 top-6 路由行 | 全局每层实际接收行 | routed M | ECO 小额候选（token） | ECO slack |
|---:|---:|---:|---:|---:|:---|---:|
| 8 | 64 | 48 | 384 | 8 | 1、2、4 | 35 |
| 16 | 128 | 96 | 768 | 8 | 1、2、4 | 35 |
| 32 | 256 | 192 | 1,536 | 16 | 2、4、8 | 43 |
| 64 | 512 | 384 | 3,072 | 16 | 2、4、8 | 43 |
| 128 | 1,024 | 768 | 6,144 | 32 | 4、8、16 | 59 |
| 256 | 2,048 | 1,536 | 12,288 | 64 | 8、16、32 | 91 |

## 四臂和一次运行的边界

| 运行次序 | 名称 | 放置、分发 |
|---|---|---|
| 1 | vanilla，原生 GC | 384 个逻辑专家，每卡 48 个物理槽。发布本档输入、路由和两处 KV 边界。 |
| 2 | random，随机 + RR | 每卡 56 槽，随机种子 0，副本间 RR。 |
| 3 | rr，EPLB + RR | 每卡 56 槽，前 1000 个满批步的逐层需求送入 DeepInfer EPLB，发布 40 层放置。 |
| 4 | eco，EPLB + ECO | 逐层专家所在 rank 和副本数与 rr 一致。开启细粒度拍卖、线性价格。 |

每臂加载一次真实模型，TP8/EP8。先做 1000 个满批 decode 步统计，完成本臂的放置操作和权重、KV 核验，再测 1000 个满批步，每步覆盖 40 个 MoE 层。首臂本身是正式测量。六档各自建立新参考，使用本次同档的 vanilla、rr 作分母。旧表的源路由重复，旧数据不能改列名或除以 8 后并入本表。

TP 仍共同执行全局 batch 的 attention 和模型前段。进入 GC 时，rank r 取全局行序的区间 [rB,(r+1)B)，同时切 hidden、top-k ID 和 weight，shared 与 routed 共同处理这 B 个源行。combine 后按源 rank 顺序 all-gather，交回下一层需要的完整 TP 张量。主表沿用 GC graph 起点至 combine 结束的设备区间。外侧 TP all-gather 和模型前段不计入该列。预填充及非满批准备不计入两个窗口。

四臂统一 FULL_DECODE_ONLY CUDA Graph、fused split shared、shared M32、TMA pull、按 M 块就绪发布 N tile、MB-CLAIM、关闭 SM-reuse。使用仓库回退后的原图提交顺序。ECO plan 使用通信分区 6 个 SM 中的 1 个，保留 72 个 compute SM。模型、量化、数据来源、权重页缓存预热、20 GiB KV/rank、固定 top-k 顺序等沿用整模主表。正式窗口的 8 个源 rank 共用同一有序请求清单，各自只提交本段。为大批量建立不同请求时，优先取原 JSONL 的不同 token 序列，数量不足时使用同一语料中不同的真实前缀，长度保留在 256–2048。本档四臂使用相同 prompt 哈希。

routed M 按仓库归档的上游 GC-MoE `af5a7a750800954f52b9ff28cc55d945ead48da6` 的 78-SM H20 legacy 策略取值。传给正式 decode 选块器的 bound 是 B。384 为逻辑专家数，6 为 top-k。关闭 M 强制覆盖，2048 的预填充容量单独处理。M8 使用支持按块发号的 384 线程 routed 实现。每个 rank 从实际 prepared graph 读回 M；同档四臂必须匹配表中值。

ECO 的 rank 价格为 **T + 26 × M块数 + A**。T 是该 rank 接收的路由行数，A 是非空物理槽数，M块数按各槽行数分别向上取整后求和。使用 mode 0、wa=1、wb=26，slack 为 routed M + 27。冷启动最多 6 轮，暖启动 1 轮。暖轮在既有整块和尾块候选上增加 M/8、M/4、M/2 三种 token 出价，保留既有增量计价、复核和提交条件。细粒度开关只在 ECO 臂开启。

## 执行命令

使用已授权的同一台 8×H20 节点，保留调度器给的设备掩码。每臂取得站点预约及八卡协作锁后检查空闲，任一卡有别人的任务就等待。权重预热使用节点本地模型，默认路径为 `/mnt/fuse/models/DeepSeek-V4.1-Flash`。站点路径不同可指定 `SM90_LOCAL_MODEL` 指向同一校验版本；依赖沿用节点已有的 env3。结果盘需要容纳两套八卡 KV 快照、路由参考和归档。启动器逐档核对空间。

从现有 Git 仓库运行以下命令。工作树固定在本任务书提交，作业目录建在代码树外。

~~~bash
set -eo pipefail
git fetch origin main
book=docs/trace_executor_20261119/SM90_EP8_PER_RANK_FOUR_6BS.md
rev=$(git log -1 --format=%H origin/main -- "$book")
test -n "$rev"
job=$(mktemp -d /mnt/fuse/sm90-ep8-per-rank-XXXXXXXX)
echo "$job"
git worktree add --detach "$job/code" "$rev"
cd "$job/code"
rc=0
bash scripts/sm90/run_full_model_four.sh --per-rank "$job" > "$job/launcher.log" 2>&1 || rc=$?
echo "$rc" > "$job/exit-code.txt"
exit "$rc"
~~~

日志和 delivery 的路径沿用驱动的全局请求数。例如主表 BS8 对应 `delivery-64/bs64/`，主表 BS256 对应 `delivery-2048/bs2048/`。汇总 CSV 使用每卡 BS，同时给出 `global_streams`、`routed_block_m`。成功的同一 SHA delivery 可用校验和复核。中断后保留部分结果，记录首个错误和已经成功的臂。执行者只可修保持语义的编译问题，回执附修复 diff、前后 SHA 和复验；同档四臂必须同一测量版本。遇到 OOM、调度抢占、GPU 挂起或一致性失败时记录对应 BS、臂、阶段、rank、最早错误、KV 容量与显存状况，释放本任务的 GPU，回执到下述位置。窗口长度、活跃流数、并发、M、冗余度、拍卖粒度及输出判断均按本任务书执行。

## 验收与交付

- **每步、每层、每个源 rank。** 两个窗口各 1000 步，每个请求每步生成一个 token，零 prefill token。检查八份 `source_requests` 的并集恰为全局请求清单，互不重叠，每个源 rank 的请求顺序和成员在两个窗口内及四臂间一致。每个源的 route tape 形状为 [40000, B, 6]。逐调用把物理槽接收计数按逻辑专家还原，与源 top-6 直方图逐专家相等，总数必须为 8B×6。记录每个 rank 的实际 M。正式测量每 rank 须有 1000 次全图 replay。
- **公平性。** 每档复用同一输入、参考反馈 token、每个源的逐步逐层路由 ID/weight、KV 边界与逻辑权重。核验实际消费的路由序号、RR 的物理槽计数及 rr/eco 的放置成员。逐请求对照模型自身产生的 greedy top-1；六对预热一致、正式窗口零差异，另核对参考反馈。教师强制送回相同 token 本身不能证明输出一致。失败档标记失败，保留证据。
- **主表。** 每档四行，列出 MoE µs/层、相对原生、相对 EPLB+RR、token IR、总 M 块、峰值 rank M 块、活跃物理槽、M 块 IR、ECO routed MAE（百分点）和成交/层调用。每次层调用先对八 rank 的本地 graph 起点至 combine 结束跨度取最大，再对 40,000 次调用等权平均。所有 M 块都按实际 M 逐槽取整；IR 逐调用取峰值/八卡均值再平均。变化为 100 × (本臂耗时/参考耗时 − 1)，负值表示更快。ECO routed MAE 按各 rank 的预测价格占比和实际 routed 跨度占比计算；实际边界从第一个 compute CTA 开工至最后一个结束，含 L2 末尾 push。报告整轮报价、接受、小额接受、零成交调用；八份同一拍卖只计一次。
- **文件。** 成功时交付 `main_table.csv` 24 行、`layers.csv` 960 行、`auction.csv` 6 行、`pairs.csv` 36 行。保留逐臂八 rank 的原始 counts/clocks/epochs/native、ECO 计数、run.json、rank_shares、测量 SHA、放置、设备/空闲记录、两处 checkpoint manifest 和 SHA256SUMS。检查每档 delivery 的 SHA256SUMS 后，将结果归档到 `docs/trace_executor_20261119/sm90_ep8_per_rank_four_6bs_results/`，回执推送到 `docs/trace_handover_20261119/REPLY_SM90_EP8_PER_RANK_FOUR_6BS.md`。清理仅限本作业私有的可再生临时文件，确认本作业进程退出。原主表保留作历史记录，复核后再更新。
