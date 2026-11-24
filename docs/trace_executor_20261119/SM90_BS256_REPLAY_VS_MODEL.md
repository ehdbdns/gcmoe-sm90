# SM90 执行任务 · BS256 真实路由回放与整模计时对照

在空闲的 8×H20 上，对 DeepSeek-V4.1-Flash 使用与 BS512/1024 相同的真实路由采集 → 算子回放流程。正常工作量为一次 vanilla 整模采集和四次算子进程组启动，每组连续跑 L8、L20、L32。四臂为 vanilla、random+RR、逐层 EPLB+RR、逐层 EPLB+fine ECO。当前代码为 split shared、M64、stream、6 个 dispatch SM，ECO 为通信分区 plan、slack 11469。采集使用与 BS256 主表相同的第一份 JSONL；记录真实 token 顺序、top‑6 ID 与权重，每源 rank 256 行。四臂复用同一路由和相同种子生成的算子激活、权重。

## 执行

沿用站点预约和设备掩码。只在同节点八张 H20 全部空闲时使用，已有作业时等待；launcher 在每次 GPU 运行前持协作锁复查进程。不改变共享环境。命令在干净 worktree 中执行，所有输出放持久卷。归档 bank 是只读输入。

~~~bash
set -eo pipefail
git fetch origin main
book=docs/trace_executor_20261119/SM90_BS256_REPLAY_VS_MODEL.md
rev=$(git log -1 --format=%H origin/main -- "$book")
job=$(mktemp -d /mnt/fuse/sm90-bs256-replay-model-XXXXXXXX)
echo "$job"
git worktree add --detach "$job/code" "$rev"
cd "$job/code"
out="$job/bs256"
bash scripts/sm90/collect_and_replay_se.sh "$out" 256 2>&1 | tee "$job/launch.log"
source scripts/env.sh
bank_args=()
for bank in /home/admin/workspace/aop_lab/app_source/sm90_persist_20261123/retained_bs256/replay /mnt/fuse/sm90-h20-eight-CF3A1Zkp/results/replay; do
  if [[ -f "$bank/measure-rank0.pt" ]]; then bank_args=(--bank "$bank"); break; fi
done
rc=0
"$GC_ENV/bin/python" scripts/sm90/compare_bs256_replay.py "$out/replay" --capture "$out/collection" "${bank_args[@]}" > "$job/compare.log" 2>&1 || rc=$?
cat "$job/compare.log"
echo "$rc" > "$job/comparison.exit"
~~~

采集经历 1000 个满批 decode 统计步，随后连续三个正式步。按统计窗口的逐层需求重算放置；回放预热后重置 ECO、依次输入三步，第二步继承第一步状态。每次层调用八卡共有 12,288 条路由项。保留 24 份路由切片、源需求及布局、SHA、各 rank 输出一致性。每档只报告第二步，另外两步用于校验状态与分配。若同 SHA 中断，已有完整 capture 和已完成臂可按上述入口恢复，先备份失败臂首错。不覆盖历史采集。

## 判读

每层先分别取整模和回放八个 rank 各自 graph 起点至 combine 结束的本地跨度最大值，再对三层等权平均。主指标为（回放均值 − 同样本整模均值）/ 同样本整模均值，正值表示回放偏慢。逐层同时给出 µs 差值和百分比。

- **本次同版本误差**用 timing_summary.csv 的 live_split / vanilla 行。来自本次 vanilla 整模直接记录的时钟；核对 capture_comparison.csv 全部 72 项逐槽相等、第二步八卡时钟完整。此项无需依赖历史 bank。
- **与旧主表同样本差距**用 archive_pre_split 四行。脚本先核验历史 bank 的 SHA256，再逐字节比对新旧 24 份切片的三步 ID 与权重，并核对四臂各层的放置、全部逐 rank 物理槽和 ECO 七项拍卖。第二步三层旧整模均值应为 vanilla 1127.691、random 1082.219、rr 889.931、eco 834.304 µs/层。旧版仅一个 shared role，新版拆为 shared L1/L2，即使样本完全匹配，历史差值也包含版本影响。对后三臂同版本整模时钟留空。
- **代表性**单列旧主表 40 层 × 1000 步均值，注明旧三层样本相对全窗口的差以及回放相对全窗口的差。独立报告四臂排序、相对 vanilla 的变化、各层最长 rank；绝对耗时、排名均照实报告，保留慢样本。

比较退出码 3 表示旧样本未全部配对，仍交付新采回放、本次 vanilla 真值和上述两张 CSV，明确失败字段。若旧 bank 已通过 SHA 核验、但与新采样本不同，可在同次预约空闲期单独回放该历史 bank 配对四臂，不再启动整模。以独立目录归档，不混用两套输入或耗时。仅此情况执行以下命令。其他异常归档首错；执行者仅可修复保持语义的编译错误并附 diff。

~~~bash
hist=docs/trace_executor_20261119/sm90_bs256_main_five_plan_comm_results/delivery/bs256
export NCCL_IB_HCA=mlx5_4:1,mlx5_5:1,mlx5_9:1,mlx5_17:1,mlx5_21:1,mlx5_29:1
export NVSHMEM_HCA_LIST="$NCCL_IB_HCA"
bash scripts/sm90/run_bs1024_se_profile.sh "$out/archive-replay" 256 "$hist" "$bank" 2>&1 | tee "$job/archive-replay.log"
"$GC_ENV/bin/python" scripts/sm90/compare_bs256_replay.py "$out/archive-replay" --archived-replay
~~~

## 交付

GPU 进程退出后释放预约，再汇总和推送。提交到 docs/trace_executor_20261119/sm90_bs256_replay_vs_model_results/，含采集切片和 manifest、原始 Kineto、设备戳、12 个 replay2 Perfetto、4 行主表、12 行层表、96 行 rank 表、timing_summary.csv、timing_layers.csv、两份逐 rank 核验 CSV、日志和退出码。回执放 docs/trace_handover_20261119/REPLY_SM90_BS256_REPLAY_VS_MODEL.md，列出同版本实测误差、四臂与历史样本及长窗口的差距、所有 GitHub 链接、实际启动次数与释放状态。保留现行 README 主表，等待设计端核验。推送后核验远端文件可读，清理本次私有 worktree 与可重建的临时文件，保留原始归档。
