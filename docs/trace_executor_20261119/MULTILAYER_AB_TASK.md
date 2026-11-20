# 任务一 · 相同 token、路由和专家布局下的 stream 对照

**现在只执行这份任务书。** 同一台 8×H20，同一代码提交和工作树，先 `GC_MOE_SM90_STREAM_M_BLOCKS=0`，再设为 `1`。每臂启动模型一次。两臂均用线性 plan、WB=26、block M=64。上一份自由生成回执保存在 `REPLY_MULTILAYER_AB_PAIR.md`。

## 收到 `cb55f4e` 失败回执后的重跑

上轮 baseline 在 vLLM 构造时因重复注册 token 回放入口报错，尚未加载权重。stream 未启动，两臂都没有可复用的性能数据。本次从更新后的 `origin/main` 建工作树，复用已回传的参考轨迹，按下面的命令分别测量两臂。启动测试会走真实驱动入口、模拟模型构造和首次请求，并检查 worker 的采样后回填路径。全套 CPU 检查在模型启动前执行。

## 如何固定负载

1. 入口读取失败回执附带的 `fixed_stream_results_failed/tokens.json`，将其 prompt 摘要及 256 条完整生成轨迹的 SHA256 与原 baseline 的 `run.json` 逐项核对，通过后供两臂共用。
2. 两臂使用同一批真实 prompt，自 prefill 起对每个请求强制反馈同一条参考 token 轨迹。vLLM V2 仍做原生 greedy 采样，随后将参考 token 送进下一步。step 2 另报 256 个请求在替换前的原生 top‑1 差异数。
3. baseline 在原有 200 步预热内，保留最后 105 个完整 decode step 的真实逐层、逐行 top-k 专家 ID 和路由权重，并落盘八份。两臂都在重排之后从头回放这同一组路由，依次覆盖 2 步 flush、1 步 profiler、2 步恢复和 100 步计时。两臂照常执行模型前向、router 和 ECO 计划，并在 dispatch 入口注入相同路由。预热期的真实路由提供可复现的负载分布；表中的 E2E 是**固定负载回放 E2E**，包括两臂相同的回放与核对开销。
4. baseline 的逐层 EPLB 布局在广播前保存，stream 复用该文件。两臂都执行实际权重搬移，安装同一布局并清空 ECO warm state。回执对比最终布局、初始 ECO 状态、每一步请求排序与 query 长度、八个 rank 的路由和所有 105 × 40 次调用的实际逐物理槽 owned 行数。该行数决定每个槽的 M 块数和尾块 valid。采样步的 owned 还须与同一 epoch 下 Perfetto 对应的 counts 逐 rank 匹配。

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
)
bash "$work/tools/scripts/trace_layer/run_multilayer_study.sh" \
  "$work/tools" "$work/delivery" 2>&1 | tee "$work/progress.log"
(cd "$work/delivery" && sha256sum --quiet -c SHA256SUMS)
```

执行者只可对编译器点名的语法或类型做语义等价的最小修改。若需修复，单独提交补丁并回传首个诊断、受影响的 SHA、完整日志与等价性说明。本轮最多两次模型启动，部分失败时保留已完成的臂和原始目录，直接回执。额外启动、重复整对或更改调度、模型、采样、计数、解析和判据都超出本次任务。

## 回执与判定

1. 在 `docs/trace_handover_20261119/REPLY_FIXED_STREAM_R2.md` 写新回执。记录测量与内核 SHA、环境、输入及生成摘要、两次 capture ID、原始目录及共享路由文件路径。之前 `11d031e` 的回执继续保留。
2. 成功交付两份 `run.json`、80 份层总览、80 份 counts、22 份 absolute JSON 和 640 行逐 rank 数据。附两臂的 `workload.json`、`placement.json`、`workload.sha256`、`paired.csv`、`layer_pairs.csv`、`comparison.json` 及产物的 GitHub 页面链接。两臂分别附上 step 2 layer 20 的 absolute JSON 链接。原始八卡 Kineto、fine 分片及日志留在执行机，交付路径与 SHA256。
3. 并排列出 E2E、40 层 MoE、预测 MAE 和最大绝对误差，报差值和百分比。列出逐层结果及变快／持平／变慢层数，结合实际负载和 CTA 轨解释变化最大的层。若输入或生成 token 不一致、主指标方向相反或收益接近计时波动，写明限制。每臂只有一次独立运行。
4. 并列报告 `fixed_workload_faster` 及 `native_top1_different_of_256`。前者要求反馈序列、路由、布局、ECO 状态和实际逐槽工作量一致，且 E2E 与 40 层 MoE 均下降。另要求采样步的原生 top‑1 对齐，才设置 `scheduling_gate_passed`。此核对只覆盖给定回放负载的一个采样步。提交后停在此处，等待发布者复核回执并更新交接入口。其余结果也应原样提交。

```bash
cd "$repo"
target=docs/trace_executor_20261119/fixed_stream_results
test ! -e "$target"
cp -a "$work/delivery" "$target"
test -s docs/trace_handover_20261119/REPLY_FIXED_STREAM_R2.md
git add "$target" docs/trace_handover_20261119/REPLY_FIXED_STREAM_R2.md
git commit -m 'Report the fixed-workload EP8 stream comparison'
git push origin HEAD:main
```
