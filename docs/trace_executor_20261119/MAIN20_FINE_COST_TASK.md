# 历史任务 · 八卡 H20 主表 20 臂

当前执行四档 BS 各六臂的 24 臂版本。新的执行入口、预算和回执路径见[24 臂任务书](MAIN24_COST_TASK.md)。

这轮回答两个问题。在真实的逐步 EPLB 统计窗口、正确的 RR 副本映射和逐 valid 的 ECO 价格下，五种配置各自的完整 MoE 和端到端延迟是多少？大 batch 下，当 EPLB 已经比较均衡时，整 M 块拍卖还有多少机会、出价和成交？

## 启动预算和配置

同一台空闲的 8×H20，四档全局 BS 依次为 32、64、128、256。每档依下表顺序各启动一次 DeepSeek-V4.1-Flash，共 **20 次模型启动**，每臂 1000 步纯 decode 统计和 1000 步正式测量。固定 TP=EP=8、top-k=6、40 层、M64、逐 M 块 stream、shared expert、CUDA graph、随机种子和 6 个 dispatch SM。后四臂均为每 rank 56 个物理槽，其中 8 个冗余。测量从共同边界后的第 0 步至第 999 步。

|臂|placement 与分配|附加 compute role|
|---|---|---|
|vanilla|384 个原生槽，逐专家唯一映射。发布本 BS 的工作量|无|
|random|448 个槽，种子 0 的随机初始布局，固定布局，RR|无|
|rr|448 个槽，逐层 EPLB 重排一次，DeepInfer 顺序的 RR|无|
|eco|同一逐层 EPLB 分配，修正后的 ECO，fine cost 模式 2|无|
|eco_smr|与 eco 相同的放置、价格和拍卖|role 6 在 dispatch 结束后复用其 6 个 SM|

## 开跑前的算法核对

- ECO 的 M64 满块价格在线性模式是 26+64=90，在模式 2 是 10381+17×64=11469。slack 取本模式的满块价格。冷热启动、迁出和迁入的工作量差、接收后的 rank 价及导出的最终分数均使用本模式价格。接收要求迁入后严格低于原发送方，且不超过全卡均值加 slack。没有给 fine 臂单独搜索阈值。拍卖仍移动一个满 M 块或已有尾块。上一轮两臂在共同旧 placement 下测得 BS256 完整 MoE 35.7168 → 35.5798 ms/步。切换到修正后的 EPLB 后，旧表中的绝对数值需要重新测量。 [上一轮严格回执](../trace_handover_20261119/REPLY_LONG_COST_1000.md)
- DeepInfer 的放置目标使用各步峰值之和。现按 epoch 保留前 1000 步的物理槽行数，按当时的槽映射还原到逻辑专家，保留 step、layer、expert 三维并跨卡求和。CPU 用例中两步负载为 [10,0,6,4] 和 [0,10,4,6] 时，原策略复制专家 0、1；先把两步相加会复制 2、3。 [DeepInfer 统计窗口](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/eplb/__init__.py) · [上游放置策略](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/eplb/policy.py)
- RR 保留该策略返回的副本顺序，副本表覆盖最多 65 个物理槽/专家；上游 gate 按源 rank 和本地 token 行号取副本。ECO 的 active copy 仍是每专家每 rank 最多一个，同 rank 的冗余物理槽可以是 passive。三个自适应臂独立算 placement，检查每个专家在每个 rank 的物理副本数一致。ECO 的 rank 内 single-copy-first 排序同时应用于权重和路由。 [上游 RR gate](https://github.com/deepseek-ai/TileKernels/blob/c1cf914a82e6901c679445f872051ce44ac2cc63/tile_kernels/moe/moe_topk_gate_forward_cuda.py)

## 执行命令

登录机器后在干净的 gcmoe-sm90 仓库运行以下命令。沿用该机已验证的 HCA、模型、数据和 env3 位置。选择本机有空间的文件系统作为 GC_MOE_CHECKPOINT_ROOT；每档需要一套 8×20 GiB 的临时 KV 和两份 route bank，脚本按 BS 检查剩余空间。原始 route bank 留在执行机。该档五臂的快照读回全部验证后脚本释放本档的临时 KV 二进制，并保留清单及哈希。需要保留 KV 且有足够空间时可设置 GC_MAIN20_KEEP_KV=1。

```bash
set -eo pipefail
repo=$(git rev-parse --show-toplevel)
cd "$repo"
test -z "$(git status --porcelain)"
git fetch origin main
git merge --ff-only origin/main
revision=$(git rev-parse HEAD)
work=$(mktemp -d /tmp/gcmoe-main20-XXXXXXXX)
git worktree add --detach "$work/code" "$revision"
cd "$work/code"
export GC_MOE_CHECKPOINT_ROOT="${GC_MOE_CHECKPOINT_ROOT:-/tmp}"
bash scripts/trace_layer/preflight_main20.sh "$work/preflight" 2>&1 | tee "$work/preflight.log"
bash scripts/trace_layer/run_main20.sh "$work/delivery" 2>&1 | tee "$work/progress.log"
(cd "$work/delivery" && sha256sum --quiet -c SHA256SUMS)
```

预检用 CPU 回归、五种真实角色组合的 CUDA 编译和八卡合成热槽 CUDA Graph 回放来验收，SM-reuse 开关的 8 个 rank 输出必须各自相等。模型只有 run_main20.sh 一个入口。执行者只可在**第一个模型启动前**修复编译失败，限语义等价的最小改动。保存首次错误、补丁、起始和修复 SHA，将补丁单独提交并在新目录重做完整预检。指标、价格、拍卖、路由、输入、输出门槛、模型启动预算及解析器按本提交执行。任何其他预检错误、正式运行失败或结构校验失败即交失败回执，附失败阶段、首条完整异常、退出码、已启动的臂及尚未启动的臂，由设计端处理。原生 top-1 比较有差异时，脚本可继续收集其余臂，回执逐对明确可归因范围。每种配置一次。

## 固定工作量和验收

1. 每档 vanilla 发布真实 prompt、反馈 token、两个窗口的逐步逐层源 router ID 和权重。其他臂回放同一工作量。保留源行顺序、batch 顺序和压缩 attention 索引排序。两个窗口各自的每一步、每一层都要满足全卡物理路由行数 = BS × 6 × 8；还要将实际接收行按逻辑专家还原并与该步的源路由直方图逐项相等。
2. 正式步 0 前按请求检查进度、历史、物理块表，五臂装载同一份 KV 并校验读回。按逻辑专家规范化六个打包权重张量并校验所有副本。原生 top-1 在反馈替换前独立记录，回执同时给出预热摘要和正式窗口精确差异数（分母 BS×1000）。每档给出全部 10 对的 strict_pair_comparable。严格标记要求全部共同控制通过、预热原生摘要相同且正式步原生差异为零。
3. 各自的物理槽行数、M 块数、激活数和调度耗时均保留。两种 ECO 臂还须逐调用核对 installed layout、实际配额和拍卖结果。20 行主表均给出实测值及严格判定，对于输出不同的臂对，耗时差列为观察值。

## 回执中的表

|项目|口径和交付|
|---|---|
|README 主表|20 行。端到端 ms/步、相对同 BS 原生变化，以及 token IR、总 M 块均值、最重 rank 的 M 块均值、活跃物理槽均值、M 块 IR。对每一步每一层先在八卡间计算，再平均 40,000 次。每槽块数取 ceil(实际行数/64)。活跃物理槽按副本分别计数。|
|完整 MoE 主指标|rank 本地 graph 根 CTA 进入至最后 combine 完成。每层取八卡最大跨度，逐步合计 40 层，报告正式 1000 步均值 ms/步；报告每 BS 的 ECO 对 EPLB+RR 和 SM-reuse 对 ECO 的成对差值。端到端使用设备同步包围真实 decode 窗口。|
|新 plan 误差|主目标是各个 rank 首个有效 GEMM CTA 至最后一个实际 tile 完成的跨度，包括 L1、L2 和 L2 末尾 push。每个调用分别归一化 rank 预测分数和 rank 实测跨度，按百分点报告 MAE 和最大绝对偏差。另列针对完整 graph 的误差。两个 ECO 臂逐 rank 核验在线 role 5 分数与实际配额复算一致；其余三臂按实际分配离线套用同一函数，标明来源。rank_shares.csv 给出各 rank 的预测/实测份额。|
|拍卖|八个 ECO 臂各给出实际轮数、合格迁出槽数、至少存在一个单独可行目的地的槽数、实际提交数、接受并执行的迁移数、接受率、每调用平均成交数、零成交调用数、无出价轮数、有出价但全拒轮数。八卡重复求解同一全局 plan，逐项相等后只计一次。auction_calls.csv.gz 保留 step、layer，auction_phases.csv 分别列第 0 步冷启动和第 1–999 步稳态，结合 EPLB+RR 的 token IR/M 块 IR 解读大 BS。零出价、出价被拒、每笔成功迁移含尾块分别有不同含义。|

拍卖仍使用原有粒度。本轮可以判断无成交发生在哪一步，也可以观察证据是否支持尝试更小的 token 批次；粒度改变后的提速需独立对照。ECO 的拍卖观测有额外设备指令，报告中的耗时对应当前采集配置。

## 提交到 GitHub

填写 `docs/trace_handover_20261119/REPLY_MAIN20_FINE_COST.md`，写明机器、软件、两个 SHA、预检、实际模型启动数、严格判定、主表及上述指标。给出原始目录和校验清单的 GitHub 文件页。成功交付应有 20 个 run.json、160 个 rank NPZ、4 档窗口需求和快照清单、20 份实际布局、全局 main_table.csv、layers.csv、pairs.csv 和 auction.csv。保留原始设备计数、环境、失败日志及 workload.sha256，设计端将独立复算后更新历史主表。失败交付亦保留所有已取得的原始结果；如 runner.log 指向的 Capture workspace 中还有首个异常，附上该日志。

```bash
cd "$repo"
git merge --ff-only "$(git -C "$work/code" rev-parse HEAD)"
target=docs/trace_executor_20261119/main20_fine_cost_results
if [[ ! -s "$work/delivery/SHA256SUMS" ]]; then target=${target}_blocked; fi
test ! -e "$target"
mkdir "$target"
if [[ -d "$work/delivery" ]]; then cp -a "$work/delivery/." "$target/"; fi
if [[ -d "$work/preflight" ]]; then cp -a "$work/preflight" "$target/"; fi
for log in preflight progress; do
  if [[ -f "$work/$log.log" ]]; then cp "$work/$log.log" "$target/"; fi
done
(cd "$target" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS && sha256sum --quiet -c SHA256SUMS)
receipt=docs/trace_handover_20261119/REPLY_MAIN20_FINE_COST.md
test -s "$receipt"
git add -- "$target" "$receipt"
git diff --cached --stat
git commit -m 'Report controlled H20 EPLB and ECO 20-arm measurements'
git push origin HEAD:main
```
