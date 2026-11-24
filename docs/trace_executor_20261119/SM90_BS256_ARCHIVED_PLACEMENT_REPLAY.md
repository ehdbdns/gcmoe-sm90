# SM90 · BS256 按整模归档放置补测 RR 与 ECO

上轮四臂回放已完成。同版本 vanilla 与新整模差 +0.77%，历史 vanilla/random 也已配对。剩余问题是历史 RR/ECO 的 L20、L32 放置与 CPU 逐层重算值不同。复用上轮两条对照的原始 trace，并按相同的 L8、L20、L32 顺序各启动一次 RR 和 fine ECO 算子；无需启动语言模型。保留 stream、split shared、原通信分区 plan、全部预热和 replay2 采样参数。

RR 权重按归档 installed-layout 安装，副本顺序按整模的 rank 0 CUDA 路径用完整 [1000,40,384] int32 需求一次恢复。三层正向映射必须逐槽等于归档，显式传入的逆映射保持原策略副本顺序。ECO 使用其归档的单副本优先排列。路由由已知 BS256 bank 的 SHA256 验证后抽取。报告器核对四臂逐 rank 输入与输出、三步物理槽分配与拍卖，成功才计算历史配对差值。

## 执行

沿用站点协作预约与已有卡掩码，只在同一节点八张 H20 全部空闲时运行；启动器在每臂前复查。以下是一轮完整执行。环境沿用上轮。只允许保持语义的编译修复；若归档 SHA、放置、逆表或配对核验失败，交付首错和已完成产物。

~~~bash
set -eo pipefail
git fetch origin main
book=docs/trace_executor_20261119/SM90_BS256_ARCHIVED_PLACEMENT_REPLAY.md
rev=$(git log -1 --format=%H origin/main -- "$book")
job=$(mktemp -d /mnt/fuse/sm90-bs256-layout-XXXXXXXX)
echo "$job"
git worktree add --detach "$job/code" "$rev"
cd "$job/code"
hist=docs/trace_executor_20261119/sm90_bs256_main_five_plan_comm_results/delivery/bs256
bank=/home/admin/workspace/aop_lab/app_source/sm90_persist_20261123/retained_bs256/replay
if [[ ! -f "$bank/measure-rank0.pt" ]]; then bank=/mnt/fuse/sm90-h20-eight-CF3A1Zkp/results/replay; fi
export NCCL_IB_HCA=mlx5_4:1,mlx5_5:1,mlx5_9:1,mlx5_17:1,mlx5_21:1,mlx5_29:1
export NVSHMEM_HCA_LIST="$NCCL_IB_HCA"
export SM90_BS256_ARCHIVE_REPAIR=1
rc=0
bash scripts/sm90/run_bs1024_se_profile.sh "$job/replay" 256 "$hist" "$bank" > "$job/run.log" 2>&1 || rc=$?
echo "$rc" > "$job/run.exit"
if [[ $rc == 0 ]]; then
  source scripts/env.sh
  "$GC_ENV/bin/python" scripts/sm90/compare_bs256_replay.py "$job/replay" --archived-replay > "$job/compare.log" 2>&1 || rc=$?
  echo "$rc" > "$job/compare.exit"
fi
exit "$rc"
~~~

## 回执

归档到 docs/trace_executor_20261119/sm90_bs256_archived_placement_results/。回执放 docs/trace_handover_20261119/REPLY_SM90_BS256_ARCHIVED_PLACEMENT_REPLAY.md。交付 run/compare 日志、退出码、执行 SHA 和复用对照 SHA、原始 trace、Perfetto、两张 timing CSV、history_comparison.csv、主表、层表和 rank 表。标明哪两臂复用上轮结果。核对 288/288 条分配及拍卖、12 行逐层配对、四臂输出、最大 rank 与 µs/层差值。单列本次新版回放与旧版整模的差（旧版 shared 单 kernel），同版本 vanilla 的实测误差引用上一轮。保留 SHA、失败现场或成功归档，GPU 任务退出后释放预约，再完成 GitHub 交付和本次私有临时文件清理。
