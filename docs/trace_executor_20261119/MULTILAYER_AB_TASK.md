# 任务一 · 逐 M 块发号的两次八卡对照

**现在只执行本任务。** 同一台 8×H20 先跑改前 baseline 一次，再跑改后 stream 一次。每臂只启动一次模型，两臂都用线性 plan（`GC_MOE_ECO_COSTFN=0`）。收到本轮回执并核实收益后，由任务发布者更新交接入口，再开始耗时函数实验。

| 顺序 | 内核提交 | 逐块 stream | 采样 |
|---|---|---:|---|
| 1 · baseline | `98b6194a56e7468653bb6230bf0b3887c9466dce` | 0 | 重排后 step 2 |
| 2 · stream | `6617ee976fa86eb77eb3a34b8b6df3e60059a586` | 1 | 同一步、同一组层 |

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

执行者只可对编译器点名的语法或类型做语义等价的最小修改。若需修复，单独提交补丁并回传首个诊断、受影响的 SHA、完整日志与等价性说明。本轮最多两次模型启动，部分失败时保留已完成的臂和原始目录，直接回执。额外启动、重复整对或更改调度、模型、采样、计数、解析和判据都超出本次任务。

## 回执与判定

1. 在 `docs/trace_handover_20261119/REPLY_MULTILAYER_AB_PAIR.md` 写新回执。记录测量与内核 SHA、环境、输入及生成摘要、两次 capture ID、原始目录。之前 `11d031e` 的回执继续保留。
2. 成功交付两份 `run.json`、80 份层总览、80 份 counts、22 份 absolute JSON 和 640 行逐 rank 数据。附 `paired.csv`、`layer_pairs.csv`、`comparison.json` 及产物的 GitHub 页面链接。两臂分别附上 step 2 layer 20 的 absolute JSON 链接。原始八卡 Kineto、fine 分片及日志留在执行机，交付路径与 SHA256。
3. 并排列出 E2E、40 层 MoE、预测 MAE 和最大绝对误差，报差值和百分比。列出逐层结果及变快／持平／变慢层数，结合实际负载和 CTA 轨解释变化最大的层。若输入或生成 token 不一致、主指标方向相反或收益接近计时波动，写明限制。每臂只有一次独立运行。
4. 输入及生成摘要匹配、E2E 与 40 层 MoE 均下降时，`scheduling_gate_passed` 为真。提交后停在此处，等待发布者复核回执并更新交接入口。其余结果也应原样提交。

```bash
cd "$repo"
target=docs/trace_executor_20261119/multilayer_ab_results
test ! -e "$target"
cp -a "$work/delivery" "$target"
test -s docs/trace_handover_20261119/REPLY_MULTILAYER_AB_PAIR.md
git add "$target" docs/trace_handover_20261119/REPLY_MULTILAYER_AB_PAIR.md
git commit -m 'Report the two EP8 scheduling runs'
git push origin HEAD:main
```
