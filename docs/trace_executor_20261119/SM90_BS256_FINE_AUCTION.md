# SM90 执行任务书 · H20 EP8 · BS256 细粒度 ECO 拍卖

在同一组八张 H20 上跑两次完整的 DeepSeek-V4.1-Flash decode，测量允许小批 token 竞拍后，八 rank 的 routed 耗时是否更均衡，以及完整 MoE 是否更快。两次均使用本任务书所在提交的代码。本轮只有以下两臂。

| 顺序 | 输出目录 | GC_MOE_ECO_FINE_AUCTION | 拍卖规则 |
|---|---|---:|---|
| 1 | eco | 0 | 主表线性 ECO，报价单位沿用整块或专家原有尾块 |
| 2 | eco_small | 1 | 在暖启动的最后一轮同时比较原报价和 8、16、32 行报价 |

## 配置和公平性

- BS256、TP8/EP8，平均每 rank 32 个解码请求。路由行数按实际 top-k 扇出记录。40 层，每个进程先统计 1,000 个满批纯 decode 步，再基于完整历史重排逐层 EPLB，清空逐层 ECO 状态，测量随后 1,000 步，共 40,000 个步层。
- 两臂均采用每卡 8 个冗余槽、M64、stream、TMA pull、fused split shared、原生 launch 顺序、无 SM-reuse、FULL_DECODE_ONLY CUDA Graph。dispatch 分区 6 个 SM，plan 使用其中 1 个。线性价格为 token 数 + 26 × M 块数 + 2 × 活跃物理槽数，每次成交重估涉及的槽，role 5 的 slack 为 92。
- 冷启动延用现有拍卖，最多 6 轮。稳态每层每步只有 1 轮。细粒度臂在同一轮向同一专家的有效副本尝试小批报价，按两个 rank 成交后较高的价格选候选。成交前检查实时配额、接收上限和两槽总 M 块数。派发使用实际配额，下一次暖启动使用可还原的整块和单尾块状态。CTA 的 GEMM 领取粒度仍为 M 块 × N tile。
- 启动器先在 CPU 上查验历史 BS256 的统计与测量两套 route bank，以及输入反馈文件的 SHA256。原件完整时，两臂都重放。原件已清理时，粗粒度 ECO 在自己这次运行中录制，细粒度 ECO 逐项重放。两种路径都只启动两个模型进程。两臂都从本次同一对 KV 检查点读回，使用相同的捕获操作和时钟边界。回执注明实际使用了哪条输入路径。
- 本地模型默认位于 `/mnt/fuse/models/DeepSeek-V4.1-Flash`。启动器检查模型索引、48 个权重分片、主表输入及可用磁盘空间，在每次加载前预热权重页缓存。KV 每卡 20 GiB。给本次私有 checkpoint 目录预留至少 350 GiB。

## 执行

先取得站点预约，同一组八卡全部空闲后运行。保留调度器分配的设备掩码，有其他用户、任务或预约时继续等待。启动器在每一臂开始时获取 H20 共用锁并复查进程。已有其他作业时不要终止其进程或修改其环境。

在现有 SM90 仓库和节点的实验环境执行以下完整命令。权重副本在其他本地盘时可先设置 `SM90_LOCAL_MODEL`。私有目录每次由 mktemp 新建。

~~~bash
set -eo pipefail
git fetch origin main
book=docs/trace_executor_20261119/SM90_BS256_FINE_AUCTION.md
rev=$(git log -1 --format=%H origin/main -- "$book")
test -n "$rev"
job=$(mktemp -d /mnt/fuse/sm90-bs256-auction-XXXXXXXX)
echo "$job"
git worktree add --detach "$job/code" "$rev"
cd "$job/code"
rc=0
bash scripts/sm90/run_full_model_four.sh --auction-bs256 "$job" > "$job/launcher.log" 2>&1 || rc=$?
echo "$rc" > "$job/exit-code.txt"
if ((rc)); then tail -80 "$job/bs256.log"; exit "$rc"; fi
(cd "$job/delivery-256" && sha256sum --quiet -c SHA256SUMS)
cat "$job/delivery-256/bs256/"{main_table,auction}.csv
~~~

执行者仅可修复保持语义、计时及配置一致的编译问题。发生编译修复时保留首错、diff 和实际 SHA，已测量的两臂必须来自同一修复提交。运行、输出或公平性校验失败时保存现场，回执写明首个失败的步骤、步层、rank 和完成臂。CPU 汇总问题可在已有数据上复查。只需在代码 checkout 内用实验环境的 Python 执行 `scripts/sm90/auction_pair.py report "$job/delivery-256"`。

## 验收与回执

1. 打印两行主表。主指标为逐步逐层取八 rank 的最大本地 graph 起点至 combine 结束跨度，再对 40,000 次取均值，单位 µs/层。给出细粒度相对粗粒度的绝对变化和带正负号的百分比，负值表示耗时下降。报告 20 对连续 50 步分段的变快、持平、变慢段数。
2. 同列 token IR、八 rank 总 M 块数、单 rank M 块数峰值、M 块 IR、活跃物理槽总数与单 rank 峰值、预测峰值、routed MAE，以及 routed、graph 的实际峰值超均值百分比。routed 取首个 CTA 真正开始 routed GEMM 到最后一个 CTA 完成的本地跨度，含 L2 末尾 push。比例先在同一步同层算，再取均值。
3. 列出报价数、总成交数、接受率、小批成交数、有小批成交的步层数，以及八 rank 拍卖跨度取最大后的均值。八份 plan 是同一次拍卖的复制，成交只计一次。区分价格改善、实际 routed 均衡、完整 MoE 耗时三项。若小批成交为零或时间变化与分段波动相近，原样回报。
4. 启动器逐 rank 检查统计及正式路由 ID、权重、源行顺序、逻辑需求、边界 KV、原生输出、逐层槽序、同一执行 SHA 与运行配置。全部 256,000 个正式原生 top-1 按请求检查。根据实际槽配额逐 rank 重算价格与设备值核对。逐步逐层逐逻辑专家核对两臂 token 守恒、物理 M 块之和等于该专家总行数向上整除 64。两臂物理槽配额可不同。
5. 结果推到 `docs/trace_executor_20261119/sm90_bs256_fine_auction_results/`，回执推到 `docs/trace_handover_20261119/REPLY_SM90_BS256_FINE_AUCTION.md`。归档 delivery-256、两臂原始八 rank NPZ、参考来源、公平性证据、执行 SHA、启动日志、退出码、四个汇总 CSV。两臂 GPU 进程退出后立即释放预约，离线整理回执。归档校验成功后清理本作业私有的 KV、临时路由和 checkout。历史引用文件由原作业持有，清理时只移除本作业创建的链接。
