# SM90 执行任务 · BS512 / BS1024 五臂三层 replay2

使用同一组 8×H20，运行 GC shared-expert 算子 benchmark。两个 BS 各跑五臂，共十次八卡启动。每次启动在同一进程组内完成 L8、L20、L32（层号从零起），复用逻辑权重、shared 权重和通信缓冲区。整轮无需启动 vLLM、加载模型 checkpoint 或构建 KV。每层安装仓库中该 BS、该臂的真实布局，预热后用 torch.profiler 记录三个 CUDA Graph replay，预先指定第二个（sample 1）作图和取数。

| 臂 | 放置与执行 |
|---|---|
| vanilla | 384 个逻辑专家，384 个物理槽，原生放置 |
| random | 448 槽，历史主表固定随机放置，RR |
| rr | 448 槽，DeepInfer 逐层 EPLB，按原有副本顺序 RR |
| eco | 同 rr 的 rank 归属，单副本前置，fine cost、slack 11469、通信分区 plan |
| eco_smr | 同 eco，额外开启 role 6 复用 dispatch SM |

公共配置沿用上一轮算子 benchmark。EP8、M64、stream、TMA pull、6 个 dispatch SM、H5120 / I2304、一个 fused shared expert、top-6。BS512 每源 rank 输入 64 个 token，八卡合计 3072 个路由项；BS1024 为 128 和 6144。三个层使用各自固定的 benchmark 输入，五臂同层输入及逻辑权重相同，L20 沿用上一轮种子。输入激活、top-k 和权重为算子 benchmark 生成，逐层放置使用对应 BS 的历史统计；回执须注明此口径。

## 执行

按站点预约同一组八张空闲 H20，保留调度器设备掩码，有其他任务就等待。启动器在每臂开始前持协作锁复查空闲状态。各臂串行；测量 SHA 为本任务书和代码所在的同一提交。在执行机仓库运行下列命令，记下打印的 job 路径。

~~~bash
set -eo pipefail
git fetch origin main
book=docs/trace_executor_20261119/SM90_BS512_BS1024_FIVE_REPLAY2.md
rev=$(git log -1 --format=%H origin/main -- "$book")
job=$(mktemp -d /mnt/fuse/sm90-five-replay2-XXXXXXXX)
echo "$job"
git worktree add --detach "$job/code" "$rev"
cd "$job/code"
base=docs/trace_executor_20261119
for bs in 512 1024; do
  group="$base/sm90_main24_stable_decode_bs${bs}_results/delivery/bs$bs"
  bash scripts/sm90/run_bs1024_se_profile.sh "$job/bs$bs" "$bs" "$group"
done
~~~

在每个 BS 的 `layer{8,20,32}/` 下可看到五份 `<arm>.replay2.perfetto.json`。每份包括 8 个 rank，只展示同一层第二次 replay。保留 count exchange 和 pull token 同轨、shared expert L1/L2、wait token、routed GEMM、combine；两个 ECO 臂另外展示 ECO plan。routed 起止取实际领到任务的首个 CTA 和完成的末个 CTA，eco_smr 包括 role 6 参与的计算。保留原始三次 Kineto trace、设备戳、物理槽计数和拍卖计数，便于核对图上的事件。

## 主表与验收

每个 BS 交付五行 `main_table.csv`、15 行 `layers.csv`、120 行 `ranks.csv`（表头另计）、一张比较图 `moe.svg`。每层先独立计算，再等权平均三层。明确使用 **µs/层**。完整 MoE 取同层八 rank 中最大的本地 graph 起点到 combine 结束跨度。相对变化以同 BS 的 vanilla 三层均值为分母，按 `100 × (本臂 / vanilla − 1)` 报告，快为负，慢为正。保留 token IR、总 M 块、峰值 rank M 块、活跃物理槽、M 块 IR、routed 预测误差和 graph 预测误差。IR 取同层八 rank 的最大值/均值。误差按同层八 rank 预测与实测耗时份额之差的绝对值取均值，单位为百分点。

ECO 价格以在线 role-5 得分核对逐槽 fine cost。两臂逐槽分配、拍卖计数须一致。拍卖成交数同时给三层合计与每层均值，八 rank 的同一份计数只记一次。其他三臂填空。每层核对五臂逐 rank 输入哈希、replay2 输出哈希、八 rank epoch 及总路由项；每臂逐逻辑专家核对实际槽计数。若输出不一致，报告具体 BS、臂、层和 rank 及两端哈希，该组标注“输出差异”，保留计时供排查。三个 replay 全部存档，展示及三层均值固定使用 sample 1。

## 交付与收尾

- 将两档 BS 的目录原样提交至 `docs/trace_executor_20261119/sm90_bs512_bs1024_five_replay2_results/{bs512,bs1024}/`。回执放 `docs/trace_handover_20261119/REPLY_SM90_BS512_BS1024_FIVE_REPLAY2.md`，打印十行主表、两张比较图链接，按 BS 和臂列出三个层的 30 个 GitHub Perfetto 链接，并注明输出一致性、执行 SHA。README 的整模主表这次不改。
- 原始八卡文件已齐时，导出或画图失败可在 CPU 上重试。进入测量工作树、加载 `scripts/env.sh` 后，对该 BS 的三个层分别执行 `"$GC_ENV/bin/python" scripts/sm90/se_perfetto.py "$job/bs$bs/layer$layer" --bs "$bs" --matrix`，再执行 `"$GC_ENV/bin/python" scripts/sm90/se_replay_report.py "$job/bs$bs"`。
- 执行者仅可修复不改变语义的编译失败，记录 diff 并保证比较使用同一版本。遇到运行或验收异常保留首错日志、已有结果和现场路径。最后一臂退出后尽快释放八卡，核查本任务 worker 全部退出；完成 CPU 归档、提交、推送及远端核验后，清理本任务私有临时 worktree 和可重建缓存。在回执末尾登记释放情况及原始数据位置。
