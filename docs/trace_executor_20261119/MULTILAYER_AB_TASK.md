# 八卡执行任务书 · 多层 baseline / stream 对照

只运行两臂。baseline 使用 `98b6194a56e7468653bb6230bf0b3887c9466dce`，STREAM=0；stream 使用 `6617ee976fa86eb77eb3a34b8b6df3e60059a586`，STREAM=1。两臂均使用本任务书同版本的测量驱动、解析和汇总代码。工作树选择内核和依赖，入口在启动前检查执行 SHA。两臂都使用 BS256、8×H20、逐层 EPLB 重排、block M=64、MB-CLAIM、TMA、shared expert 和 CUDA Graph。

## 测什么

| 窗口 | 每臂采样 | 交付 |
|---|---|---|
| e2e | 4 个独立进程，各预热 200 步，重排后跳过指定步数，再连续测 100 个 decode step | 无 profiler、无设备 trace 的八卡同步窗口耗时和平均 ms/step |
| coarse | 重排后 step 2、5、8、11，各独立采一次，一次覆盖全部 40 层 | 同次采集的 E2E、逐 rank 的 MoE 跨度、plan 偏差、counts 及总览 Perfetto |
| fine | 同样四个 step，另起进程；层号 0、4、8、12、16、20、24、28、32、36、39 | 每层一个含全部八个 rank 的 absolute Perfetto JSON、总览 Perfetto、counts，以及逐 rank 表 |

四对执行顺序是 AB、BA、BA、AB。两臂共 24 次进程启动。step 从重排后 0 开始。同一对的模型、数据集、提示词、预热、图设置和 GPU 必须一致。两臂 100 步窗口是评估端到端速度的主指标；coarse 单步同时保留 host wall time 和该步 GPU MoE 数据。fine 用来解释已观察到的差异。每次重排可能改变逐槽分布，因此保留每轮原始负载和生成结果摘要。

## 计时与预测口径

- 每个 rank、每个采样层的 `moe_us = combine.end − min(dispatch.ts, shared.ts, plan.ts)`，时间戳取同次 torch profiler 的 GPU kernel。三个起点是本配置下该层 MoE 子图的根节点。combine 的结束包含本层最后的同步和收尾。窗口内各阶段存在重叠。
- `plan_share[r] = plan_cost[r] / sum(plan_cost)`，`actual_share[r] = moe_us[r] / sum(moe_us)`。分母只含同一调用的八个 rank。`share_error_pp = 100 × (plan_share − actual_share)`，正数表示高估。每层报八 rank 平均绝对误差和最大绝对误差，单位为百分点。权重取设备记录的当次 plan 分数；八个 rank 的计划表要一致。
- 每个 coarse step 给出各 rank 的 40 个 MoE 层跨度之和（ms），以及这八个和的最大值 `moe_max_rank_sum_ms`。每一层另报最慢 rank 及其耗时。端到端以 driver 开始测量到八卡完成为界，在 profiler 导出和设备快照前停表。回执使用新包里的 `run.json`、`ranks.csv`、`layers.csv` 及配对报告。
- absolute JSON 保留同一次 torch/Kineto 时间轴、每张卡的真实 pull/compute CTA、L2 push 和领取事件。总览包含 count exchange、pull token、wait token、首次有效领取后的 GEMM、combine，并增加完整 MoE graph 轨。每个文件携带 capture ID、step、layer 和 epoch。本轮图形产物全部用 Perfetto JSON。

## 执行

在上轮同一台已分配的空闲八卡机器上，沿用已验证的 GPU、IB、模型、数据集和工具链路径。仓库根目录开启 Bash，在同一会话执行下列命令。工作区和产物目录独立。长任务可在 tmux 中运行。

```bash
set -eo pipefail
repo=$(git rev-parse --show-toplevel)
cd "$repo"
test -z "$(git status --porcelain)"
git fetch origin main
git merge --ff-only origin/main
measurement=$(git log -1 --format=%H --diff-filter=A -- docs/trace_executor_20261119/MULTILAYER_AB_TASK.md)
work=$(mktemp -d /tmp/gcmoe-multilayer-XXXXXXXX)
echo "$work"
git worktree add --detach "$work/tools" "$measurement"
git worktree add --detach "$work/baseline" 98b6194a56e7468653bb6230bf0b3887c9466dce
git worktree add --detach "$work/stream" 6617ee976fa86eb77eb3a34b8b6df3e60059a586
(
  cd "$work/tools"
  source scripts/env.sh
  "$GC_ENV/bin/python" tests/test_aligned_step_cpu.py
  "$GC_ENV/bin/python" tests/test_pull_gemm_detail_cpu.py
  "$GC_ENV/bin/python" tests/test_multilayer_study_cpu.py
)
bash "$work/tools/scripts/trace_layer/run_multilayer_study.sh" \
  "$work/baseline" "$work/stream" "$work/delivery" 2>&1 | tee "$work/progress.log"
(cd "$work/delivery" && sha256sum --quiet -c SHA256SUMS)
```

执行者权限只含运行和语义等价的最小编译修复。保留首个编译器报错。若新版 stream 在八卡工具链编译失败，只在 stream 工作树改编译器点名的语法或类型，单独提交，记录原 SHA、修后 SHA、完整诊断与等价性依据。用该 commit 作为 `GC_MOE_STUDY_STREAM_SHA`，在新输出目录重跑完整矩阵并将修复提交一并交付。baseline SHA 固定。运行、性能、解析、计数、缺数据等问题在首次失败时停下回执，保留该步骤的 `runner.log` 和其中指向的原始目录。禁止调整协议、门限、采样点、调度和解析口径。速度持平、变慢或精度下降也须如实回报。

## 验收与回执

1. 核对 24 份 `run.json`，其中 16 次 profile 具有独立 capture ID。两臂合计有 320 份 coarse 层总览、88 份 fine 层总览、88 份 absolute JSON，408 份 counts。每个 absolute JSON 包括 8 个 rank。总计 3,264 行逐 rank 负载和预测数据，保留模式、step、层号、epoch、rank、token、routed rows、M 块、plan 分和实际时间。
2. 回执 `docs/trace_handover_20261119/REPLY_MULTILAYER_AB.md` 列出设备、模型、数据路径、内核和测量 SHA、原始目录、每对 step 的输出 hash 是否一致。按配对列出两臂 100 步 E2E ms/step、40 层 MoE 汇总 ms、差值及变化百分比，列四组中获益次数。层与 rank 属于同一捕获，勿将它们当成独立的端到端重复。
3. 列出每对 40 层 MoE 更快／持平／更慢的层数、逐层最慢 rank、plan MAE 及最差 rank 的预测比例、实测比例、带符号百分点误差。附一处改善和一处退化（若有）的 absolute JSON，结合实际负载与 CTA 区间解释。若生成摘要或负载不同，标明对比限制。以采集结果判断本轮是否更优。
4. 附 `summary.csv`、`paired.csv`、`layer_pairs.csv` 和完整 JSON/CSV 包链接。原始 torch 八卡 trace、fine epoch 分片与日志保存在执行机，随包提交其路径与校验值。

```bash
cd "$repo"
target=docs/trace_executor_20261119/multilayer_ab_results
test ! -e "$target"
cp -a "$work/delivery" "$target"
test -s docs/trace_handover_20261119/REPLY_MULTILAYER_AB.md
git add "$target" docs/trace_handover_20261119/REPLY_MULTILAYER_AB.md
git commit -m 'Report paired multilayer EP8 measurements and absolute Perfetto traces'
git push origin HEAD:main
```
