# 八卡 H20 任务书 · 1000 步统计、重排、1000 步对照

本轮要回答：固定逐 M 块 stream 后，逐 valid 的 ECO 价格能否缩短完整 MoE 层耗时？同一台空闲八卡 H20，总共启动两次模型。先跑线性价格，再跑 fine cost。每臂独立完成一次真实 prefill、1000 步统计、一次逐层 EPLB 重排和 1000 步正式测量。正式步从重排后第一个 decode 编为 0，最后一步为 999。

| 臂 | ECO 模式 | rank 价格 |
|---|---:|---|
| linear | 0 | routed rows + 26 × M 块数。每个物理槽分别向上取整，M=64 |
| fine | 2 | 各非空 M 块累加「padding 档截距 + 17 × valid」。六个档上界为 8、16、32、40、48、64，截距为 4642、4841、5588、6727、7940、10381 |

两臂的 TP/EP=8、BS=256、top‑k=6、40 层、stream=1、逐块领票、模型、数据和测量代码相同。fine 价格包含 L1、L2 和 L2 尾部 push 的工作量，维持本轮约定的系数，空档单独定价留待后续。

## 两个窗口和固定工作量

1. 全部请求完成 prefill 后，清空真正用于 EPLB 重排的逐层统计。前 1000 步均为 256 个请求各生成一个 token 的纯 decode。逐层验证逻辑需求在八卡合计为 1000 × 256 × 6 × 8 条路由行。fine 臂核对完整逐层逐专家需求，并使用同一份 placement。
2. 线性臂记录本轮 256 个请求的真实反馈 token、前后两个窗口各自的逐步逐层 router ID 和权重。fine 臂按请求及对应步号回放；两臂使用相同的槽内源行排序和压缩 attention 索引排序。两臂在正式步 0 前装入并读回同一份 KV 快照，检查请求位置、物理块表、ECO 起始状态、placement、打包权重。各臂按自己的价格拍卖，所得逐槽配额、rank 行数和 M 块数是实验结果。表中 routed_rows 是实际物理路由行，tokens 沿用历史口径，按八份输入复制折算为 routed_rows / 8。
3. 正式窗口覆盖每一步的全部 40 层和 8 个 rank，每臂 320,000 条 rank 记录。各层校验 epoch 连续、12,288 条物理路由行守恒，以及设备输出的 plan 分数等于按实际逐槽行数复算的结果。rank 快照在正式计时结束后导出。本轮使用全窗设备时间戳，不另外启动模型采集 Perfetto。

## 执行步骤

在同一 bash 会话运行。沿用上轮成功回执的模型、数据、HCA 和运行时。为 KV、路由文件和交付物选择八个本机 worker 均可访问、至少有 180 GiB 可用空间的位置。大文件留在执行机。若 HCA 不同，按执行机原有映射设置环境变量。独立工作树固定测量 SHA。

```bash
set -eo pipefail
repo=$(git rev-parse --show-toplevel)
cd "$repo"
test -z "$(git status --porcelain)"
git fetch origin main
git merge --ff-only origin/main
revision=$(git rev-parse HEAD)
work=$(mktemp -d /tmp/gcmoe-cost-1000-XXXXXXXX)
git worktree add --detach "$work/code" "$revision"
cd "$work/code"
export GC_MOE_CHECKPOINT_ROOT="${GC_MOE_CHECKPOINT_ROOT:-/tmp}"
(
  source scripts/env.sh
  df -h "$work" "$GC_MOE_CHECKPOINT_ROOT"
  nvidia-smi --query-gpu=index,uuid,name,memory.used --format=csv
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name --format=csv
  for test_file in tests/test_*_cpu.py; do "$GC_ENV/bin/python" "$test_file"; done
  "$GC_ENV/bin/python" tests/test_stream_checkpoint_cpu.py --gpu-preflight
  "$GC_ENV/bin/python" tests/test_attention_order_cpu.py --cuda
  "$GC_ENV/bin/python" tests/test_long_cost_cpu.py --cuda
  for cost in 0 2; do
    GC_MOE_ECO_COSTFN="$cost" GC_MOE_LONG_COST=1 GC_MOE_FIXED_DIR="$work" \
    GC_MOE_SM90_STREAM_M_BLOCKS=1 GC_MOE_MB_CLAIM=1 GC_MOE_BLOCK_M_FORCE=64 \
    GC_MOE_ECO_MAX_COPIES=8 GC_MOE_ECO_PER_LAYER=1 GC_MOE_LAYERS=40 \
    GC_MOE_STEP_PROFILE=1 GC_MOE_ECO_TRACE=1 GC_MOE_PULL_TRACE=1 GC_MOE_FINE_TRACE=0 \
    GC_MOE_SM90_PULL_TMA=1 GC_MOE_IR_STEP=0 GC_MOE_SM90_DISPATCH_SMS=6 \
    GC_MOE_SM90_SM_REUSE=0 SMOKE_ECO_WB=26 "$GC_ENV/bin/python" scripts/smoke_compile_eco.py
  done
) 2>&1 | tee "$work/preflight.log"
test "$(grep -Fc 'compiled roles [1, 4, 2, 5, 3]' "$work/preflight.log")" -eq 2
```

确认八卡空闲。预检中若编译器报告兼容性错误，执行者只可作语义等价的最小编译修复。保留首次错误、命令和日志，独立提交补丁，重新通过预检，再用同一新 SHA 跑两臂。冻结路由、计时边界、价格、参数、解析器和验收条件。其余失败或任一模型启动后遇错，在该步停止并提交失败回执，已启动的进程计入两次预算。两臂只有下面一个入口。

```bash
bash scripts/trace_layer/run_long_cost.sh "$work/delivery" 2>&1 | tee "$work/progress.log"
(cd "$work/delivery" && sha256sum --quiet -c SHA256SUMS)
```

## 测量和回执

先报告严格对照是否成立，再列出 linear 和 fine 实测值、差值及变化百分比。启动预算只有一对，两臂的 20 个连续 50 步窗口用于观察方向随步数的变化。每个窗口来自同一对进程。

| 指标 | 计算与交付 |
|---|---|
| **完整 MoE 耗时，主指标** | 每步每层在各 rank 的本地设备时钟上计算「最早 graph 根 CTA 进入至最后 combine CTA 完成」，先取该层八卡最大跨度，再对 40 层求和。报告 1000 步的均值 ms/步、平均单层 µs、fine 减 linear 的差值及百分比、变快的步数 / 1000、层数 / 40、50 步窗口数 / 20。 |
| **端到端** | worker 全部同步后启动主机计时，1000 次真实 decode 后再次同步。报告两臂 ms/步、整体相对变化和 20 个窗口；prefill、重排、快照装载与落盘均在窗口外。两臂均带相同的路由审计和设备打点，数值对应此配置。 |
| **plan 预测准确度** | 每步每层将 rank 价格除以八卡价格总和，将 rank 的 routed 跨度除以本层八卡 routed 跨度总和。routed 从首个 compute CTA 开始有效 tile 至最后一个实际 tile 完成，包含 L1、L2 及内嵌的 push。偏差以百分点计，为 100 ×（预测占比 − 实测占比），报告平均绝对偏差及最大绝对偏差。另列以完整 MoE graph 跨度为目标的误差。每臂的实际分配都以两种价格重算，保留同配额的交叉比较。 |

- 成功采集时把回执写入 `docs/trace_handover_20261119/REPLY_LONG_COST_1000.md`。给出起始和测量 SHA、模型实际启动数、机器及软件版本、两个窗口边界、原生 top‑1 差异数（分母 256,000）、`strict_pair_comparable`、三项指标，以及原始目录和校验文件。严格对照失败时照实标注原因，保留观察差值，结论填「无法归因」。
- 上传 `comparison.json`、`step_pairs.csv`（1000 步）、`layer_pairs.csv`（40 层）、`blocks.csv`（20 个窗口）、`ranks.csv.gz`（两臂 640,000 条数据行）、两臂的 `run.json`、16 个 rank 的原始 NPZ、环境与日志、placement、快照清单、工作量哈希与 SHA256SUMS。在回执附上 GitHub 文件页。
- 失败时写相同回执，明确阶段、臂、已启动数、SHA、首条完整异常、命令和退出码。保留 `runner.log` 中的 `Capture workspace` 路径、该目录的 `status.txt` 和失败日志。若两臂均已完成但对照条件不符，归档两臂及比较报告。

填写回执后执行以下交付命令。预检失败也可使用此段；`repo` 和 `work` 沿用开始时记录的绝对路径。

```bash
cd "$repo"
git merge --ff-only "$(git -C "$work/code" rev-parse HEAD)"
target=docs/trace_executor_20261119/long_cost_1000_results
if [[ ! -s "$work/delivery/SHA256SUMS" ]]; then target=${target}_blocked; fi
test ! -e "$target"
mkdir "$target"
if [[ -d "$work/delivery" ]]; then cp -a "$work/delivery/." "$target/"; fi
cp "$work/preflight.log" "$target/"
if [[ -f "$work/progress.log" ]]; then cp "$work/progress.log" "$target/"; fi
(cd "$target" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS && sha256sum --quiet -c SHA256SUMS)
receipt=docs/trace_handover_20261119/REPLY_LONG_COST_1000.md
test -s "$receipt"
git add -- "$target" "$receipt"
git diff --cached --stat
git commit -m 'Report H20 1000-step linear and fine cost comparison'
git push origin HEAD:main
```
