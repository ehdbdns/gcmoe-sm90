# SM90 · BS512、BS1024 整模四臂复测

目标是在共享节点的同一组 8×H20 上，用真实 DeepSeek‑V4.1‑Flash decode 测 EPLB + fine ECO 相对 EPLB + RR 和原生的完整 MoE 耗时变化。共八次模型启动，逐臂独立加载。每臂 1000 个满批纯 decode 步统计需求，随后连续测量 1000 个正式步，覆盖全部 40 个 MoE 层。

| BS | 第 1 臂 | 第 2 臂 | 第 3 臂 | 第 4 臂 |
|---:|---|---|---|---|
| 512 | 原生 vanilla | 随机放置 + RR | 逐层 EPLB + RR | 逐层 EPLB + fine ECO |
| 1024 | 原生 vanilla | 随机放置 + RR | 逐层 EPLB + RR | 逐层 EPLB + fine ECO |

- 每档 vanilla 发布该档新采的 tokens、路由和双边界 KV，其余三臂复用。两档分别验收。EPLB 在整模的 rank 0 GPU 上根据完整 1000 步逐层需求生成布局和 RR 副本顺序；ECO 使用相同 rank 副本归属，并在 rank 内执行单副本优先排列。
- TP8/EP8、FULL_DECODE_ONLY CUDA Graph、M64、stream、fused split shared、6 dispatch SM，四臂均关闭 SM-reuse。ECO 使用 fine cost、slack 11469，plan 占通信分区一个 SM。预热和正式窗口均启用稳定 top-k、固定请求顺序。两条 JSONL 的顺序固定，BS1024 使用已验证的 BS512 请求序列重复两遍。每 rank 20 GiB KV。
- 直接使用已验证的本地权重副本 /mnt/fuse/models/DeepSeek-V4.1-Flash。启动器检查真实路径、挂载类型、config 与 safetensors 索引哈希、48 个索引分片及数据集 SHA256。每臂在启动模型前用 warm_page_cache.sh 对本地分片发起页缓存预热，设置 GC_MOE_NO_PREFETCH=1。回执逐臂列实际挂载点、48/48 加载用时、从模型初始化到进入 prefill 的可得日志时间；记录实际值。若默认路径已迁移，可通过 SM90_LOCAL_MODEL 指向站点已有的同版本已校验本地副本。
- 只在同一组八张 H20 全部空闲且有站点预约时运行，保留调度器设备掩码。每臂启动前在协作锁内复查，遇到其他作业则等待。正常完成只运行表中八臂；直接调用整模入口，无单层 benchmark、trace 或额外的模型预跑。

## 一次执行

在原仓库执行以下命令，job 放持久盘，checkpoints 默认位于本次 job 的独立子目录。可在启动前指定容量足够的本地 GC_MOE_CHECKPOINT_ROOT。入口在每档前检查两份 KV 和两套路由的可用空间，执行 SHA 随两档归档。

~~~bash
set -eo pipefail
git fetch origin main
book=docs/trace_executor_20261119/SM90_BS512_BS1024_FULL_MODEL_FOUR.md
rev=$(git log -1 --format=%H origin/main -- "$book")
job=$(mktemp -d /mnt/fuse/sm90-four-full-XXXXXXXX)
echo "$job"
git worktree add --detach "$job/code" "$rev"
cd "$job/code"
rc=0
bash scripts/sm90/run_full_model_four.sh "$job" > "$job/launcher.log" 2>&1 || rc=$?
echo "$rc" > "$job/exit-code.txt"
exit "$rc"
~~~

每档在独立 delivery 运行。输出从 bs512.log、bs1024.log 和各臂 runner.log 查看。完整 delivery 可以按 SHA256SUMS 校验并跳过；部分完成目录保留首错、已完成臂和 state-path.txt。分析阶段若单独失败，先保存现场，使用同一版本在已完成的 delivery 上运行 scripts/trace_layer/main20_report.py DELIVERY --bs BS --four-arms 复查。执行者只可处理保持语义的编译故障并附 diff。其他故障交付具体阶段、原始报错、已启动次数和已完成结果。

## 公平性与主表

1. 每档四臂、六对两两对照。每对必须通过 warm_native_equal、strict_pair_comparable，正式原生 top‑1 差异为 0。每对正式窗口检查 512,000 或 1,024,000 个生成 token。验证 prompt 哈希、模型配置/硬件、两个阶段的逐 rank 路由 ID、权重、顺序、逻辑需求、KV 快照读回、逻辑专家权重和正式输出一致。记录 prefill 步数、满批条件和抢占情况。报告器核对 RR 物理目的槽、ECO 实际价格及八 rank 拍卖计数。严格校验失败时标出具体臂对，保留观察值与原因。
2. 主表按每次层调用先取八 rank 本地 graph 起点至 combine 结束跨度的最大值，再对 40 层 × 1000 步等权平均，单位 µs/层。原始 ms/步保存在 CSV。列出对同档 vanilla、EPLB + RR 的 MoE 百分比变化，负为快，正为慢。端到端时间作为原始审计字段保留。
3. 每行同时交付 token IR、八卡总 M 块、峰值 rank M 块、活跃物理槽、M 块 IR、routed 与 graph 的各 rank 耗时份额预测 MAE（百分点）。routed 从首个实际计算 CTA 至最后一个完成，含末尾 push。ECO 交付成交总数、每层调用均值、报价数与接受率、零成交调用数。保留 rank 原始计数/时钟、rank_shares.csv 与逐层结果，以便复核。
4. 两档均通过后，汇总目录应有 main_table.csv 八行、pairs.csv 十二行、layers.csv 三百二十行、auction.csv 两行。回执打印八行主表，给出各档 ECO 相对 RR 的 µs 差值和百分比；附 SHA、两档六对核验、两档退出码和权重加载日志证据。

## 交付与清理

结果推到 docs/trace_executor_20261119/sm90_bs512_bs1024_full_model_four_results/。提交两个 delivery、汇总 CSV、加载与启动日志、输入校验、测量 SHA 和退出码；回执提交 docs/trace_handover_20261119/REPLY_SM90_BS512_BS1024_FULL_MODEL_FOUR.md。原始大型 KV、route bank 留在现场或按任务私有数据清理，记录位置；保留现场可复核的 manifest 和必要证据。每臂结束检查本任务进程，GPU 作业完成后释放协作锁和站点预约，再做 CPU 汇总和 GitHub 提交。核验远端归档后清理本轮临时 worktree 与可重建的私有文件。沿用已有 README 主表至结果复核。
