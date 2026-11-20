# 任务一 · 相同 token、路由和专家布局下的 stream 对照

**现在只执行这份任务书。** 同一台 8×H20，同一代码提交和工作树，先 `GC_MOE_SM90_STREAM_M_BLOCKS=0`，再设为 `1`。每臂启动模型一次。两臂均用线性 plan、WB=26、block M=64。上一份自由生成回执保存在 `REPLY_MULTILAYER_AB_PAIR.md`。

## R3 · 收到 `16baf33` 后的执行起点

R2 的 baseline 完成 100 个计时步，固定负载 E2E 为 86.186 ms/步，step 2 的 40 层 MoE 为 37.525 ms。重排后 step 0 的 39 层共少 54,016 个 routed rows，等于 844 个满 M 块。这 39 层各有专家占用 5–8 个活跃副本。其余 104 步逐层守恒。执行者在首个验收失败处停止，stream 尚未启动。[R2 回执](../trace_handover_20261119/REPLY_FIXED_STREAM_R2.md)

旧 plan 用实际副本数做除法，但冷启动发放配额只遍历前四个副本。用旧代码在 CPU 上复现，单专家 8 个活跃副本、1,024 行时只发出 768 行，修复后发出 1,024 行。R3 的两臂统一使用修复后的 plan，重新计时。R2 时间只作历史记录。请使用 R2 原八卡机器，保留 `/tmp/gcmoe-route-replay-Q6W0KywL` 和两次旧失败目录。按下列入口执行，模型启动上限为 baseline 一次、stream 一次。

## 如何固定负载

1. 入口读取失败回执附带的 `fixed_stream_results_failed/tokens.json`，将其 prompt 摘要及 256 条完整生成轨迹的 SHA256 与原 baseline 的 `run.json` 逐项核对，通过后供两臂共用。
2. 两臂使用同一批真实 prompt，自 prefill 起对每个请求强制反馈同一条参考 token 轨迹。vLLM V2 仍做原生 greedy 采样，随后将参考 token 送进下一步。step 2 另报 256 个请求在替换前的原生 top‑1 差异数。
3. 入口复制 R2 已冻结的八份逐层、逐行 top-k 专家 ID、路由权重和一份 EPLB 布局到新目录。CPU 先检查八个 rank 的路由、权重、层序列及最终布局 SHA 与 R2 回执一致，并用真实路由重算旧四副本循环对 step 0 各层预测的缺口。保留并汇报 `cold-replay-check.log` 中预测与实测相符的层数和所有不相符的层。缺文件或摘要不符时在启动模型前回执。
4. 两臂各预热 200 步，均从已核对的文件安装逐层布局、搬移权重、清空 ECO warm state，并从同一批路由的开头回放 2 步 flush、1 步 profiler、2 步恢复和 100 步计时。前向和 router 照常执行，固定路由在 dispatch 入口注入。检查每步请求顺序、query 长度、路由和全部 105 × 40 次逐槽 owned 行数。每一步每层八卡应合计 12,288 行，包含首个 flush。逐槽行数决定 M 块和尾块 valid。step 2 的每 rank 每槽还须与同一 epoch 的 Perfetto counts 匹配。两臂的 E2E 均为含相同回放和记录开销的**固定负载回放 E2E**。

## 每臂一次模型运行

- 同一模型和数据集，固定 BS256、200 步 warmup、逐层 EPLB 重排和 CUDA Graph。重排后的 step 从 0 开始计，跳过 0、1 后对 step 2 启动 torch profiler 一次。驱动使用 vLLM 0.31 的进程内 EngineCore，显式推进一步，保留八个 GPU worker。上一份失败日志已经显示 `async_scheduling=False`，并且另有一个独立 EngineCore 进程。采集仍严格要求每个 rank 各有 40 次 dispatch、shared、plan、routed、combine，并把设备 epoch 唯一对应到原始 trace。
- 同一次采集导出 0–39 层的总览 Perfetto、counts 和逐 rank 表。其中 0、4、8、12、16、20、24、28、32、36、39 层另导出 absolute Perfetto JSON，每份合并八个 rank。保留 count exchange、pull token、wait token、首个有效 CTA 领票起算的 GEMM，以及详细 pull / L1 / L2 / push 区间。图形交付全部用 JSON。
- 每层每 rank 的 MoE 耗时取同一调用的 `combine.end − min(dispatch.ts, shared.ts, plan.ts)`。逐层汇报最慢 rank。全步指标 `moe_max_rank_sum_ms` 先对每个 rank 求 40 层耗时之和，再取八个和的最大值。
- 同层八个 rank 分别归一化，`plan_share = plan_cost / Σplan_cost`，`actual_share = moe_us / Σmoe_us`，`share_error_pp = 100 × (plan_share − actual_share)`。正值为高估。保留每 rank 的 token、routed rows、M 块、各槽 owned、预测分、实测耗时与比例，列每层 MAE 及最差 rank。
- trace 导出后在原模型进程关闭 profiler 和 fine 写入，再过两步，随后连续计时 100 个纯 decode step。E2E 以八个 worker 同步的墙钟时间计，检查每步生成 256 个 token 且无 prefill，检查 fine 写指针保持一致。主指标是这个窗口的平均 ms/step。另列 step 2 开 profiler 时的墙钟时间。两臂都保留同一套采样版内核。

## 执行命令

在上轮同一台空闲八卡机器使用已经验证的模型、数据、HCA 和软件环境。从主仓库开始，在同一个 Bash 会话执行。使用最新测量脚本，历史失败目录保留。旧任务中 step 5、8、11 和分开启动 e2e/coarse/fine 的入口已撤销。

```bash
set -eo pipefail
repo=$(git rev-parse --show-toplevel)
cd "$repo"
test -z "$(git status --porcelain)"
git fetch origin main
git merge --ff-only origin/main
measurement=$(git rev-parse HEAD)
work=$(mktemp -d /tmp/gcmoe-two-arm-XXXXXXXX)
echo "$work"
git worktree add --detach "$work/tools" "$measurement"
(
  cd "$work/tools"
  source scripts/env.sh
  "$GC_ENV/bin/python" tests/test_aligned_step_cpu.py
  "$GC_ENV/bin/python" tests/test_pull_gemm_detail_cpu.py
  "$GC_ENV/bin/python" tests/test_multilayer_study_cpu.py
  "$GC_ENV/bin/python" tests/test_fixed_workload_cpu.py
  "$GC_ENV/bin/python" tests/test_eco_copies_cpu.py
)
if bash "$work/tools/scripts/trace_layer/run_multilayer_study.sh" \
  "$work/tools" "$work/delivery" --replay-from /tmp/gcmoe-route-replay-Q6W0KywL \
  2>&1 | tee "$work/progress.log"; then
  (cd "$work/delivery" && sha256sum --quiet -c SHA256SUMS)
else
  echo "采集已停止，请按失败路径回传已有文件和首条错误。"
fi
```

执行者只可对编译器点名的语法或类型做语义等价的最小修改。若需修复，单独提交补丁并回传首个诊断、受影响的 SHA、完整日志与等价性说明。本轮最多两次模型启动，部分失败时保留已完成的臂和原始目录，直接回执。额外启动、重复整对或更改调度、模型、采样、计数、解析和判据都超出本次任务。

## 回执与判定

1. 在 `docs/trace_handover_20261119/REPLY_FIXED_STREAM_R3.md` 写新回执。记录测量与内核 SHA、环境、输入及生成摘要、两次 capture ID、原始目录、共享路由文件路径和 R2 首步缺口复算结果。失败时附首条异常、已完成步骤、原始日志与实际启动次数，把现有 delivery 存入新的 `fixed_stream_results_r3_blocked` 目录，并推送回执。
2. 成功交付两份 `run.json`、80 份层总览、80 份 counts、22 份 absolute JSON 和 640 行逐 rank 数据。附两臂的 `workload.json`、`placement.json`、`workload.sha256`、`paired.csv`、`layer_pairs.csv`、`comparison.json` 及产物的 GitHub 页面链接。两臂分别附上 step 2 layer 20 的 absolute JSON 链接。原始八卡 Kineto、fine 分片及日志留在执行机，交付路径与 SHA256。
3. 并排列出 E2E、40 层 MoE、预测 MAE 和最大绝对误差，报差值和百分比。列出逐层结果及变快／持平／变慢层数，结合实际负载和 CTA 轨解释变化最大的层。若输入或生成 token 不一致、主指标方向相反或收益接近计时波动，写明限制。每臂只有一次独立运行。
4. 并列报告 `fixed_workload_faster` 及 `native_top1_different_of_256`。前者要求反馈序列、路由、布局、ECO 状态和实际逐槽工作量一致，且 E2E 与 40 层 MoE 均下降。另要求采样步的原生 top‑1 对齐，才设置 `scheduling_gate_passed`。此核对只覆盖给定回放负载的一个采样步。提交后停在此处，等待发布者复核回执并更新交接入口。其余结果也应原样提交。

```bash
cd "$repo"
target=docs/trace_executor_20261119/fixed_stream_results
if [[ ! -s "$work/delivery/SHA256SUMS" ]]; then
  target=docs/trace_executor_20261119/fixed_stream_results_r3_blocked
fi
test ! -e "$target"
mkdir "$target"
if [[ -d "$work/delivery" ]]; then cp -a "$work/delivery/." "$target/"; fi
if [[ -f "$work/progress.log" && ! -s "$target/SHA256SUMS" ]]; then
  cp "$work/progress.log" "$target/"
fi
test -s docs/trace_handover_20261119/REPLY_FIXED_STREAM_R3.md
git add "$target" docs/trace_handover_20261119/REPLY_FIXED_STREAM_R3.md
git commit -m 'Report the fixed-workload EP8 stream comparison'
git push origin HEAD:main
```
