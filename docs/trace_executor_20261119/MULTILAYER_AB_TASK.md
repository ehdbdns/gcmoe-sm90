# R4 八卡任务 · 修复 ECO 后只比较 stream

R3 的两臂在 105 步的每层逐槽行数一致，stream 的 E2E 从 83.124 降到 81.403 ms/步，40 层 MoE 从 33.976 降到 33.726 ms。采样步 256 个请求有 60 个原生 top‑1 不同，验收未通过。两臂各 11 层覆盖 3,066 个完整 M 块，均未观察到某块的 L2 计算早于同块最后一个 L1 tile 结束。此项只观测计算区间。源码中，旧 baseline 独立分发 M 块，却只数编号最后一块的完成信号，理论上可能提前放行 L2。本版本等待槽内所有 L1 的 (M, N) 任务完成。stream 保留每块独立发号及 L2 就绪信号。[R3 回执](../trace_handover_20261119/REPLY_FIXED_STREAM_R3.md)

## 两臂共用的 ECO 修复

对照 SM100 的 `eco_unit_kernel.cu`、`eco_unit_auction.cuh` 和 `unit_ref.py`，本版本保留整 M 块拍卖。修复完整分数及槽号的竞价排序、热启动增减单位循环、小 bound 下计数与 plan 的握手、连续逐层重排时使用的权重和统计源布局。CPU 用实际 SM90 配额与拍卖函数对照 SM100 参考，检查逐槽物理配额、前缀、被动副本与下一轮状态。高分数回归中，旧实现对 15,429 行输入有 6 个槽物理配额偏离参考，本版逐槽吻合。本次 H20 负载仍按 WB=26 计价。

## 实验条件

- 在 R2、R3 使用的同一台空闲 8×H20 上执行，模型进程共启动两次，先 baseline，再 stream。两臂用本次同一工作树。线性价格取 `GC_MOE_ECO_COSTFN=0`、`WB=26`、`WA=0`、M=64，rank 分数为 `Σ槽 (26 × ceil(owned / 64) + owned)`。两臂都开启 ECO 拍卖，冷启动 6 轮，热启动 1 轮。暂缓 valid 耗时函数实验。
- 复用 `/tmp/gcmoe-route-replay-Q6W0KywL` 的 256 条真实 token 轨迹、逐层 EPLB 布局和八份真实路由文件。CPU 入口检查 R2 摘要和首步 40 层缺口。缺输入时先回执。
- 两臂加载同一份包含两种调度路径的内核，使用相同显存布局和 CUDA Graph。prefill 和 200 步 warmup 都走 baseline。重排、权重搬移、清空 ECO 状态、装入路由参考后，在同一位置写一次设备调度开关，baseline 写 0，stream 写 1。
- 重排后的全部 105 步强制反馈同一 token。每个源 rank 和逻辑专家按原始 token 及 top-k 列顺序确定行号，dispatch 实际使用的行号必须逐个等于参考。各源 rank 依序拼接，同一份逐槽配额按前缀切分，因而每个物理副本获得相同的 token、权重、M 块和尾块 valid。逐层检查八卡行数总和为 12,288，并验证 packed 权重与 ECO 元数据层号对齐。
- 第 0、1 步 flush，step 2 采集 torch profiler。导出 40 层的总览、counts、逐 rank 数据，同时导出层 0、4、8、12、16、20、24、28、32、36、39 的 absolute Perfetto JSON。保留 count exchange、pull、wait token、首个 CTA 成功领到可用票时起算的 routed GEMM、逐 CTA 的 L1/L2/push 与 combine。全部图形交付 JSON。
- 采集后关闭 profiler 和 fine 写入，同一进程过两步，再测 100 个同步 decode step 的平均 E2E。step 2 的 MoE 从同一调用最早 graph 根（dispatch、shared、plan）开始，到 combine 结束；整步取每个 rank 的 40 层耗时之和，再取八卡最大值。预测和实测各自按同层八卡总和归一化；逐 rank 报比例误差（百分点）。这是带同等回放和审计开销的测量，每臂只有一次。

## 执行步骤

使用已经验证过的模型、数据集、HCA 和软件环境，在同一 Bash 会话执行。第一段在独立工作树进行 CPU 回归和一次纯编译，编译不加载模型。

```bash
set -eo pipefail
repo=$(git rev-parse --show-toplevel)
cd "$repo"
test -z "$(git status --porcelain)"
git fetch origin main
git merge --ff-only origin/main
measurement=$(git rev-parse HEAD)
work=$(mktemp -d /tmp/gcmoe-r4-XXXXXXXX)
git worktree add --detach "$work/tools" "$measurement"
echo "$work"
(
  cd "$work/tools"
  source scripts/env.sh
  for name in aligned_step pull_gemm_detail multilayer_study fixed_workload eco_port eco_copies stream_mblock; do
    "$GC_ENV/bin/python" "tests/test_${name}_cpu.py"
  done
  GC_MOE_FIXED_DIR=/tmp/gcmoe-route-replay-Q6W0KywL GC_MOE_MB_CLAIM=1 \
  GC_MOE_BLOCK_M_FORCE=64 GC_MOE_ECO_MAX_COPIES=8 GC_MOE_LAYERS=40 \
  GC_MOE_STEP_PROFILE=1 GC_MOE_PULL_TRACE=1 GC_MOE_FINE_TRACE=1 \
  GC_MOE_FINE_FLOOR=1 GC_MOE_FINE_RECS=4194304 GC_MOE_IR_STEP=0 \
  GC_MOE_SM90_DISPATCH_SMS=6 GC_MOE_SM90_SM_REUSE=0 SMOKE_ECO_WB=26 \
    "$GC_ENV/bin/python" scripts/smoke_compile_eco.py
) 2>&1 | tee "$work/preflight.log"
```

预检通过后只执行下面的单一入口。若执行器返回非零，保留原始文件，直接写回执。若在模型启动前遇到编译失败，仅可对编译器点名的语法或类型作语义等价的最小修复。修复单独提交，在两臂开始前固定同一修复 SHA 并附差异。模型一旦已启动，就计入该臂的一次预算。调度、拍卖、耗时函数、采样、判据及解析均由发布者维护。

```bash
if bash "$work/tools/scripts/trace_layer/run_multilayer_study.sh" \
    "$work/tools" "$work/delivery" --replay-from /tmp/gcmoe-route-replay-Q6W0KywL \
    2>&1 | tee "$work/progress.log"; then
  (cd "$work/delivery" && sha256sum --quiet -c SHA256SUMS)
fi
```

## 交付和判定

1. 新建 `docs/trace_handover_20261119/REPLY_FIXED_STREAM_R4.md`，记录实测 SHA、实际启动次数、GPU、驱动和软件版本、原始目录、捕获 ID、共享路由 SHA、预检日志及首条错误（如有）。完整输出放到 `docs/trace_executor_20261119/fixed_stream_results_r4/`。采集中断或校验抛错时请放入 `fixed_stream_results_r4_blocked/`，附两臂各自进行到哪一步。保留旧回执和产物。
2. 完整交付包括 2 份 run、80 份总览 Perfetto、80 份 counts、22 份 absolute Perfetto、640 行逐 rank 数据、两份 workload、placement、workload.sha256、paired.csv、layer_pairs.csv、comparison.json 及 SHA256SUMS。回执贴两臂 step 2 layer 20 的 GitHub 文件页面。
3. 并列 E2E、整步 MoE、计划占比误差、变快层数和变化最大层。报告 `same_run_controls`、`placement_and_initial_state_match`、`routes_and_all_slot_counts_match`、`same_physical_routes`、`last_warm_input_match`、`warm_native_match`、采样步 top‑1 差异数、全 105 步差异计数及最早分歧步。预热激活摘要覆盖八卡最后一次 warmup MoE 输入和量化尺度。
4. `fixed_workload_faster` 表示逐槽负载与路由身份等控制项相同，且两个耗时指标都下降。`scheduling_gate_passed` 还要求预热观测一致、回放全部 26,880 个原生预测一致。只在此项通过时，报告“本次固定工作负载两项实测均改善”。两臂各一次没有重复运行误差界。原生输出有差异时仍回传完整产物，列出首个分歧步与请求；两个指标方向相反时分别报告。交付后等待发布者复核。

```bash
cd "$repo"
target=docs/trace_executor_20261119/fixed_stream_results_r4
if [[ ! -s "$work/delivery/SHA256SUMS" ]]; then target=${target}_blocked; fi
test ! -e "$target"
mkdir "$target"
if [[ -d "$work/delivery" ]]; then cp -a "$work/delivery/." "$target/"; fi
cp "$work/preflight.log" "$target/"
if [[ -f "$work/progress.log" && ! -s "$target/SHA256SUMS" ]]; then cp "$work/progress.log" "$target/"; fi
(cd "$target" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
receipt=docs/trace_handover_20261119/REPLY_FIXED_STREAM_R4.md
test -s "$receipt"
git add "$target" "$receipt"
git commit -m 'Report the R4 fixed-order EP8 stream comparison'
git push origin HEAD:main
```
