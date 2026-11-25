# SM100 · B300 EP8 整模四臂复测

执行者为 SM100。用同一份 DeepSeek-V4.1-Flash、同一批请求，在 B300 上比较四种 GC MoE 放置和分流策略。主指标是完整 MoE µs/层。另逐步逐层回答，按线性价格预测的最重 rank 是否也是实际 routed 最慢的 rank。

| 全局 BS | 每 rank 的 decode token | 每档正式运行 | 每臂统计步 | 每臂测量步 |
|---:|---:|---:|---:|---:|
| 256 | 32 | 4 | 1000 | 1000 |
| 512 | 64 | 4 | 1000 | 1000 |
| 1024 | 128 | 4 | 1000 | 1000 |

| 臂 | 工作方式 |
|---|---|
| vanilla | 原生 GC、384 个逻辑专家、每 rank 48 个物理槽，生成本档统一参考 |
| random | 随机种子 0，448 个物理槽，固定随机放置，DeepInfer RR 分流 |
| rr | 448 槽，统计窗口结束后安装逐层 DeepInfer EPLB 放置，RR 分流 |
| eco | 安装与 rr 相同的逐层放置，使用在线线性 ECO 分流 |

共 12 次整模运行。TP8/EP8，真实权重，sequence parallel，top-k 6，每步覆盖全部 40 层。两阶段连续推进同一批请求。同档四臂使用同一 prefill 顺序、原生 greedy 参考反馈、逻辑路由 ID 和 gate 权重，两处边界 KV 均写入原图地址并读回校验。EPLB 从本档 vanilla 的完整 1000 × 40 × 384 需求生成逐层放置，保留同时峰值。随机臂从 prefill 开始就使用其冻结放置。四臂都使用 vLLM V2 的 FULL_DECODE_ONLY 图，统一使用 torch sparse indexer、相同的 combine 求和顺序和每卡 32 GiB KV。两阶段各须在每个 rank 观测到 1000 次整图 replay 和 40,000 次有序的层调用。

## B300 的内核和价格

固定上游 Natureal/gc-moe 的 eco 提交 `fb8b65fa829c97962eb2dbc9a4ad8b759b9cd736` 及本任务脚本施加的补丁。四臂都启用原生 fused shared。B300 的 role 2 同时承载 shared 和 routed，本轮优先提交 role 2，以 CUDA Graph 的 launch-order 边放行通信，使两者可以重叠。finalize 等待所有前驱完成。ECO 的 plan 是通信分区内的一个 CTA，dispatch 使用该分区 SM 数减一，compute 保留整分区的双 CTA cluster。关闭 helper，使用设备查询出的上游默认 Green Context 分区。回执附每个 rank 的实际角色、context 和 grid。H20 的独立 shared L1/L2 角色只用于解释架构差别。

价格为 **P = T + 26M + 2A**。T 为 rank 实收路由行，M 为该 rank 各槽分别按实际 BM 向上取整后的块数之和，A 为非空物理槽数。冷启动最多 6 轮、暖启动最多 1 轮、单跳，slack 为 BM + 28。B300 原生选择器在 BS256、512、1024 分别选 M16、M16、M32，对应 slack 44、44、60。回执核对各 rank 实际 BM、prepared 单元配置、设备最终价格和接收配额。跨架构比较使用同一价格系数，解释 M 块数量时注明实际块高。

## 执行

1. 使用站点授权的同组八张 B300。任一卡被其他人预约、占用或有设备句柄，整组等待。已有 SM100 任务运行时留给原任务完成。使用站点预约和现有八卡协作锁，保留调度器的设备掩码。每臂单独启动，结束即释放协作锁。CPU 汇总在锁外进行。
2. 从原仓库执行下面一段。脚本会先核对模型配置、权重索引、两份原始 JSONL 和三档有序 prompt 的 SHA-256，再使用八卡。本轮沿用 SM90 的 256–2048 token 真实提示词，BS1024 的请求顺序为 BS512 的两份串接。JSONL 在同站点不同路径时，设置 `SM100_EP8_DATA` 为其目录。若 B300 只挂载导出的提示词，在具有原始文件的已授权存储环境运行 `python scripts/sm100/ep8_inputs.py 数据目录 私有导出目录`，将该导出目录通过已授权的共享存储提供给 B300，并设置 `SM100_EP8_PROMPTS`。这一步只用 CPU。Git 交付输入的摘要；原始提示词留在授权存储。无法获取历史输入时，在预约前回报缺少的路径。
3. 复用本机已校验的 DSL 4.8.1 wheelhouse，可设置 `SM100_DSL_WHEELS`。实验使用私有 venv、上游 checkout、缓存及结果目录。选择能容纳本档两份八卡 KV 快照及完整 route banks 的私有盘。脚本在每档开始前按字节检查可用空间。

~~~bash
set -eo pipefail
git fetch origin main
book=docs/sm100_megamoe/SM100_EP8_LINEAR_FOUR_3BS.md
rev=$(git log -1 --format=%H origin/main -- "$book")
test -n "$rev"
job=$(mktemp -d /data0/home/admin/workspace/megamoe/ep8-linear-four-XXXXXXXX)
echo "$job"
git worktree add --detach "$job/code" "$rev"
cd "$job/code"
if [[ $(git config --bool core.sparseCheckout || true) == true ]]; then
  git sparse-checkout add scripts/sm100 source/gc_moe integration docs/sm100_megamoe
fi
rc=0
bash scripts/sm100/run_ep8_four.sh "$job/run" > "$job/launcher.log" 2>&1 || rc=$?
echo "$rc" > "$job/exit-code"
exit "$rc"
~~~

4. 使用本任务的专用四臂入口。旧五臂脚本的 full-decode 分支只覆盖 BS512 原生双臂。运行期间每步必须恰好生成全局 BS 个 token。每次调用八 rank 的实收路由行合计应为 BS × 6。RR 按源 rank、token 局部序号及每层逆映射重新计算逐槽预期接收数。ECO 核对每个逻辑专家的 token 守恒、每个物理槽的实收数与设备 quota，并校验八份拍卖副本的成交和报价计数相同。
5. 断线先查看同一 job 的 launcher、进程、exit-code 和日志。已完成的臂及有校验和的组按同一 revision 续查。回执保留失败现场和首错。执行者只可修不改变语义的编译问题，附原编译错误、完整 diff、实际运行 SHA 和复验。某档变更了测量代码，须用同一构建比较该档的四臂。语义、输入或资源条件无法满足时，保存证据并释放资源。报告失败可在 CPU 上对保有原始 bank 的组运行 `python scripts/sm100/ep8_report.py "$job/run/bs512"`，按实际 BS 替换目录。

## 指标和判读

- 每次调用先计算各 rank 自身的设备 graph 起点至 combine 完成的跨度，取八个跨度的最大值，再对 40,000 次调用求平均。主表给出 µs/层、相对同档 vanilla 和 rr 的有符号变化，公式为 100 ×（本臂耗时 / 参考臂耗时 − 1），负值表示更快。
- 主表同时给出 token IR、八卡总 M 块、峰值 rank M 块、八卡活跃物理槽、峰值 rank 活跃物理槽、M 块 IR。IR 是每次调用的 rank 最大值除以八 rank 均值，然后在测量窗口取平均。ECO 另列 routed MAE（百分点）、平均每层调用成交次数、报价总数、成交总数、接受率、零成交调用数。同一个拍卖只计一份。
- 首要诊断是**价格峰值命中率**。对同一步同一层的八个 rank，取价格最大集合与实测 routed 最慢集合，两者相交即命中一次。汇总命中数 / 40,000 和百分比，另列价格并列和实测时间并列次数。routed 从首个 CTA 拿到数据后真正开始 routed 矩阵乘计到最后的 routed L2 epilogue/push。用整段 graph 至 combine 的实测最慢 rank 同样计算一次。ECO 以设备分数为准，并逐次对照公式；其余三臂的价格来自对实收分配的离线计算。routed MAE 只列 ECO，按每次调用八 rank 的价格占比和 routed 时长占比计算绝对百分点差，再取平均。
- 每档四臂共有六组配对。逐一核对两阶段的反馈、行序、KV 读回、覆盖前原生 token 和逻辑 gate、回放 ID 和权重。回执区分同一工作量和原生输出一致，写明失败的步、层、rank、字段和首差。相对变化只对通过的配对填写。旧 BS512 Mega/GC 回执中的原生差异属于另一组实验，本轮四臂各自验收。

## 交付和释放

把回执推到 `docs/sm100_megamoe/REPLY_SM100_EP8_LINEAR_FOUR_3BS.md`，结果目录为 `docs/sm100_megamoe/ep8_linear_four_3bs_results/`。完整交付含 `main.csv` 12 行、`layers.csv` 480 行、`pairs.csv` 18 行、`auction.csv` 3 行。主表列出三档 ECO 对同期 RR 的 MoE 变化、routed 价格峰值命中率和对应公平性状态。保留逐档结果、实际放置、各 rank 原始 counts/clocks/拍卖记录、模型及提示词摘要、设备 UUID、运行版本、实际补丁、起止日志、退出码、reference 和交付校验和。大文件保存在可复核的私有归档，回执列绝对位置、字节数和 SHA-256。接收数据中各臂放置不同，四臂应共享同一逻辑工作量。

GPU 工作结束立即确认本任务的 worker 全部退出并释放预约。核过每档 `verified.sha256` 后只清理本任务可重建的缓存、私有环境与临时 checkout；保留原始测量、失败现场和共享模型。最后核对回执和汇总文件已经进入 GitHub main。
