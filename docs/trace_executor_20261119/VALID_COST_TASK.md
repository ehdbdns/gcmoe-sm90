# H20 逐 valid 的 plan 耗时模型与八卡验收

**当前执行。** 在同一台空闲的八卡 H20 上固定逐 M 块 stream，只更换 ECO 耗时函数。总共启动两次模型，线性价格和逐 valid 价格各一次。用户已选定保留 stream。R8 的 MoE 下降 0.37%，100 步 E2E 上升 0.13%，保留该轮原判定。[R8 指标](fixed_stream_results_r8/paired.csv)

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

## R8 旧价格基准

step 2 全 40 层共 320 个层与 rank 配对。固定 R8 stream 实际各槽的行数和 graph 起止时间重新打分，线性价格的 MAE 为 0.244221 个百分点，最大绝对偏差为 0.891692 个百分点。逐 valid 价格的两项分别为 0.271562 和 1.017430 个百分点。两者的最大偏差均在 layer 17、rank 1。R8 baseline 调度上的线性价格对应 0.245281 和 0.919729 个百分点。新实验另行测量更换价格后拍卖实际产生的分配与耗时。[R8 逐 rank 原始值](fixed_stream_results_r8/step02/stream/paired/ranks.csv)

## 两臂共同条件

使用 R8 的 256 个请求、teacher-forced token、105 步逻辑路由 ID、路由权重、源 rank 内行号及逐层 EPLB placement。沿用压缩 KV 索引排序。两次均按既有路径预热，在重排后从相同 KV 快照及冷启 ECO 状态开始；回放第 0 到 104 步均使用 stream。重排安装新 placement 时内核缓冲管理器会清零旧 quota。发布快照和首步参照摘要的进程按线性价格识别。两臂用各自的价格执行拍卖，允许每个副本槽的 owned、各 rank 的 token 数与 M 块数随拍卖改变。各层均校验总量守恒。

- 回放步 0 核验第一层输入、整步 40 层的数值指纹、共享权重与 sampler。全部 105 步核验 native top‑1、实际 stream 模式与压缩层执行覆盖。
- 在回放第 2 步采集 40 层 torch profiler Perfetto 和逐 rank counts、另采 0、4、8、12、16、20、24、28、32、36、39 层 absolute Perfetto JSON。dispatch 包含 count exchange，routed GEMM 自首个有效 CTA 领取开始。实测耗时为各 rank 的 MoE graph 起点至 combine 结束。过渡两步后，同进程关闭 profiler 与 fine trace 写入，计时 100 步 E2E。
- 两臂分别按实际逐槽行数核验设备导出的 plan 分数。每臂额外在相同的实际配额和时间上用两种价格分别打分，记录到 `fixed_quota_scores.csv`。误差（百分点）为 100 ×（rank 分数 / 本层八卡分数总和 − rank 跨度 / 本层八卡跨度总和）。分别评价预测 MAE、最大误差、40 层 MoE 与 100 步 E2E。

## 执行步骤

沿用 R8 的模型、数据和 HCA。快照位置需供八个 worker 共同读写，至少预留 88 GiB 快照空间并留出采样文件空间。以下命令在同一个 bash 会话中执行。第一次模型启动前完成预检。若 R8 回放目录已迁移，`GC_MOE_VALID_REPLAY` 指向相同文件的目录，脚本按历史回执验证内容。

```bash
set -eo pipefail
repo=$(git rev-parse --show-toplevel)
cd "$repo"
test -z "$(git status --porcelain)"
git fetch origin main
git merge --ff-only origin/main
revision=$(git rev-parse HEAD)
work=$(mktemp -d /tmp/gcmoe-valid-cost-XXXXXXXX)
git worktree add --detach "$work/code" "$revision"
cd "$work/code"
r8=docs/trace_executor_20261119/fixed_stream_results_r8
replay=${GC_MOE_VALID_REPLAY:-$(cat "$r8/workload-path.txt")}
export GC_MOE_CHECKPOINT_ROOT="${GC_MOE_CHECKPOINT_ROOT:-/tmp}"
(
  source scripts/env.sh
  while read -r digest oldpath; do
    echo "$digest  $replay/${oldpath##*/}"
  done < "$r8/workload.sha256" | sha256sum --quiet -c -
  df -h "$work" "$GC_MOE_CHECKPOINT_ROOT"
  nvidia-smi --query-gpu=index,uuid,name,memory.used --format=csv
  for t in tests/test_*_cpu.py; do "$GC_ENV/bin/python" "$t"; done
  "$GC_ENV/bin/python" tests/test_stream_checkpoint_cpu.py --gpu-preflight
  "$GC_ENV/bin/python" tests/test_first_step_cpu.py --cuda
  "$GC_ENV/bin/python" tests/test_attention_order_cpu.py --cuda
  "$GC_ENV/bin/python" tests/test_attention_probe_cpu.py --cuda
  for cost in 0 2; do
    GC_MOE_ECO_COSTFN="$cost" GC_MOE_FIXED_DIR="$replay" GC_MOE_SM90_STREAM_M_BLOCKS=1 \
    GC_MOE_MB_CLAIM=1 GC_MOE_BLOCK_M_FORCE=64 GC_MOE_ECO_MAX_COPIES=8 GC_MOE_ECO_PER_LAYER=1 \
    GC_MOE_LAYERS=40 GC_MOE_STEP_PROFILE=1 GC_MOE_PULL_TRACE=1 GC_MOE_FINE_TRACE=1 \
    GC_MOE_FINE_FLOOR=1 GC_MOE_FINE_RECS=4194304 GC_MOE_SM90_PULL_TMA=1 GC_MOE_IR_STEP=0 \
    GC_MOE_SM90_DISPATCH_SMS=6 GC_MOE_SM90_SM_REUSE=0 SMOKE_ECO_WB=26 \
      "$GC_ENV/bin/python" scripts/smoke_compile_eco.py
  done
) 2>&1 | tee "$work/preflight.log"
```

执行者仅可对编译器明确定位的语法、类型或工具链兼容问题作保持语义的最小修复。保留首个原始错误、独立提交补丁；预检修复后用同一新 SHA 开始两臂。模型一经启动即计入总预算，若运行中遇错，保留现场和已完成臂，发布者处理。禁止改路由、价格、采样、输出、验收条件或自行追加运行。

```bash
bash scripts/trace_layer/run_multilayer_study.sh "$work/code" "$work/delivery" --valid-cost-from "$replay" \
  2>&1 | tee "$work/progress.log"
(cd "$work/delivery" && sha256sum --quiet -c SHA256SUMS)
```

## 回执

1. 将回执写入 `docs/trace_handover_20261119/REPLY_VALID_COST.md`。报告测量 SHA、实际启动数、GPU 与软件版本、两臂 capture ID、原始路径、输入对照 `cost_inputs_match`、含数值一致性的 `strict_pair_comparable`、原生 top‑1 差异、配额是否改变。在旧目录标签中 `baseline` 表示 stream + 线性价格，`stream` 表示 stream + valid 价格。
2. 并列报告两臂的 E2E、40 层 MoE、逐 rank 预测 MAE/最大误差，以及 `same_quota_scores` 的两组固定配额复算。列出预测偏差变化最大的两层的各 rank 有效行数分布、token、M 块、预测和实测比例，以及 `ECO plan` 段耗时。单项退化或持平照实报告。回执附两臂的 40 层 Perfetto、11 层 absolute Perfetto、40 层逐 rank 表、比较表与校验文件的 GitHub 文件页。每臂一次，报告观察差值。
3. 首个非编译失败即结束，使用 `valid_cost_results_blocked` 并附首个错误、失败命令、日志和已经产生的证据。预检失败可直接执行以下交付步骤。KV 二进制与 route tape 留在执行机。

```bash
cd "$repo"
git merge --ff-only "$(git -C "$work/code" rev-parse HEAD)"
target=docs/trace_executor_20261119/valid_cost_results
if [[ ! -s "$work/delivery/SHA256SUMS" ]]; then target=${target}_blocked; fi
test ! -e "$target"
mkdir "$target"
if [[ -d "$work/delivery" ]]; then cp -a "$work/delivery/." "$target/"; fi
cp "$work/preflight.log" "$target/"
if [[ -f "$work/progress.log" ]]; then cp "$work/progress.log" "$target/"; fi
(cd "$target" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
receipt=docs/trace_handover_20261119/REPLY_VALID_COST.md
test -s "$receipt"
git add "$target" "$receipt"
git commit -m 'Report H20 linear and valid cost with streaming enabled'
git push origin HEAD:main
```
