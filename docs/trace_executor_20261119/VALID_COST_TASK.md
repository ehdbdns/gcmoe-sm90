# H20 逐 valid 的 plan 耗时模型与八卡验收

**暂缓执行。** 第一份任务拿到可比的实测收益、且发布者更新交接入口后，再执行本文件的两次运行。两次均用同一提交的逐块 stream。先测原线性耗时函数（`GC_MOE_ECO_COSTFN=0`），再测按 valid 定价的耗时函数（`GC_MOE_ECO_COSTFN=2`）。

## 模型依据

旧文档描述的线性拟合与当前模式 1 不同。实际 role 5 使用六个固定价，相同 padding 档内不区分 valid，满块常量与原表共用。源码也把平均交接空档折入每块定价。原始 absolute trace 显示同一 M 块的 N-tile 可分配到多个 CTA。把 37 次换活都归到这个块缺乏逐票依据。原有 prepared-cubin 缓存键还缺少 cost 模式；本轮把模式纳入不可变的调用配置。

训练使用回执 `ebddb05` 中 after/stream 的 step 2、layer 8，八个 rank 共 269 个 M 块；同一 capture 的 layer 20 单独保留验证，共 234 块。两层实际出现的 valid 是 8 的倍数。未观测行数按硬件 padding 档和档内斜率估算。

- 对每个 M 块，按 CTA、slot、M 块、N-tile 和 valid 配对区间。L1 使用完整 tile 时间，L2 分成完整 tile 扣除嵌套 push 后的时间，以及 push 自身的时间。三项分别拟合，组合权重依次为 18、20、20。
- L1、L2 的六档截距对应 8、16、32、40、48、64 的计算形状。在相同 padding 档内根据观察到的行数差估计一个非负行斜率。push 用实际 valid 拟合截距和非负行斜率。拟合时每个实测 valid 等权，避免占多数的满块支配尾块系数。
- 合并系数后，非空块的整数价格是 `截距[档] + 17 × valid`，每单位为 1/8 CTA·µs，截距依次为 4642、4841、5588、6727、7940、10381。空块为 0。每个槽分别计满块和尾块。例如 135 行计为 64 + 64 + 7。冷启动从已有估值中扣除暂分尾块的实际成本，warm-start 单位、auction 的迁入迁出差分、slack 和最终预测使用相同量纲。这套系数校准于 H20、H5120、I2304、block M=64、384 线程 routed。
- 该函数估计 plan 可以分配的本地工作量。GPU 整层耗时还包含并行执行、pull、shared 和 combine 等待其他 rank 的时间。验收同时看工作量拟合和完整 MoE graph 的占比误差。

| layer 20 留出验证 | 每块绝对误差均值（CTA·µs） | 八卡 tile 工作量占比 MAE（百分点） | 八卡 graph 耗时占比 MAE（百分点） |
|---|---:|---:|---:|
| 模式 0，26 × 块数 + routed rows | 分数无时间单位 | 0.140 | 1.314 |
| 模式 1，原样回放 | 68.83 | 0.079 | 1.236 |
| 模式 1，分析时扣除旧代码注释标称的每块 37 × 1.821 µs 空档 | 15.67 | 0.072 | 1.242 |
| 模式 2，实际 valid | 10.31 | 0.071 | 1.253 |

这些是固定旧 quota 的离线回放。整层占比分母是同一调用八个 rank 从最早根节点到 combine 结束的跨度之和。前两轮采集和本轮在线 A/B 分别保存。

## 执行步骤

在同一台空闲的八卡 H20 上运行。每个耗时函数各启动一次模型。工作负载、step 2、全 40 层与 11 层细采沿用任务一。环境路径可以指向同一个模型、数据集和已验证的 HCA。

```bash
set -eo pipefail
repo=$(git rev-parse --show-toplevel)
cd "$repo"
test -z "$(git status --porcelain)"
git fetch origin main
git merge --ff-only origin/main
revision=$(git rev-parse HEAD)
work=$(mktemp -d /tmp/gcmoe-valid-cost-XXXXXXXX)
echo "$work"
git worktree add --detach "$work/code" "$revision"
cd "$work/code"
(
  source scripts/env.sh
  "$GC_ENV/bin/python" tests/test_h20_valid_cost_cpu.py
  "$GC_ENV/bin/python" tests/test_multilayer_study_cpu.py
  data=docs/trace_executor_20261119/batch_claim_results/after/stream
  "$GC_ENV/bin/python" scripts/trace_layer/fit_valid_cost.py \
    --train "$data"/*layer08*.detail.json --holdout "$data"/*layer20*.detail.json
)
bash scripts/trace_layer/run_multilayer_study.sh "$work/code" "$work/delivery" --valid-cost \
  2>&1 | tee "$work/progress.log"
(cd "$work/delivery" && sha256sum --quiet -c SHA256SUMS)
```

输出目录名沿用比较工具。`baseline` 为 stream + 线性价格，`stream` 为 stream + valid 价格。两次各在同一进程采集 step 2 并在关闭 profiler 和 fine 写入后计时 100 步。启动入口先检查已推送的任务一 `fixed_stream_results_r4/comparison.json` 的 `scheduling_gate_passed`。保存所有 rank 的 token、M 块、预测分、实测跨度、占比、误差，以及总览和 absolute Perfetto JSON。程序核验两臂执行 SHA、stream 开关和 cost 模式，并逐 rank 验证候选的设备 plan 分等于按该次实际逐槽 owned 重算的价格之和。

## 回执

1. 在 `docs/trace_handover_20261119/REPLY_VALID_COST.md` 列出测量 SHA、两臂的开关、GPU 和软件版本、两次输入及输出摘要、原始目录与校验文件。附 `comparison.json`、`paired.csv`、`layer_pairs.csv`、逐 rank 表和完整 Perfetto 目录。
2. 并列报告这一对的端到端 ms/step、40 层 MoE 耗时、plan 占比误差及差值。对变化最大的两层列出逐 rank valid 分布、token、M 块、预测和实测比例，列出 plan 自身的 `ECO plan` 段耗时。区分配额变化、调度收益和模型计算开销。
3. 执行者只可在首个编译器诊断处作语义等价的最小编译修复，独立提交并保存原始日志。回执附补丁及等价性说明。最多两次模型启动，保留已完成的臂。运行、校验、解析或数值异常在首个失败步骤回传证据。不得更改模型系数、调度、测量口径和验收条件。若持平或退化，按实测结果回报。

```bash
cd "$repo"
target=docs/trace_executor_20261119/valid_cost_results
test ! -e "$target"
cp -a "$work/delivery" "$target"
test -s docs/trace_handover_20261119/REPLY_VALID_COST.md
git add "$target" docs/trace_handover_20261119/REPLY_VALID_COST.md
git commit -m 'Report paired H20 valid-cost planner measurements'
git push origin HEAD:main
```
