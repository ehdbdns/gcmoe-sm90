# 八卡 H20 主表重测 · 24 臂执行任务书

本任务替代 20 臂任务。主指标是完整 MoE 的毫秒/步。对 BS 32、64、128、256 各运行六臂，分别检查 EPLB 放置、ECO 分配、线性与 fine 价格及 SM-reuse 的效果。每臂各一次模型启动，前 1000 步纯 decode 统计，正式测量重排边界后的 1000 步。模型启动预算合计 24 次。若旧任务已启动模型，保留已经启动的进程及其工作目录，在回执中先报实际 SHA、BS、臂、已启动数和 KV/replay 路径，由设计端核对可复用结果后续排。仅做过旧预检的机器可直接执行本任务。

## 六臂与主要对比

同一台空闲的 8×H20，DeepSeek-V4.1-Flash，TP=EP=8、top-k=6、40 层；M64、stream、shared expert、CUDA Graph 和 6 个 dispatch SM 在各臂相同。每档 BS 按表中顺序运行，每档的 vanilla 发布独立工作量。BS 之间分别评估。

|臂|物理槽数|放置与分配|ECO 价格模式|拍卖 slack|SM-reuse|
|---|---:|---|---|---:|---|
|vanilla|384|原生，每专家一槽|—|—|关|
|random|448|种子 0 的固定随机放置，RR|—|—|关|
|rr|448|前 1000 步逐层 DeepInfer 放置，上游 RR 副本选择|—|—|关|
|eco_linear|448|同一 DeepInfer 放置，修正后的 ECO|0，原始线性|90|关|
|eco|448|同一 DeepInfer 放置，修正后的 ECO|2，逐 valid 的 fine cost|11,469|关|
|eco_smr|448|相同放置与 fine ECO 分配|2，逐 valid 的 fine cost|11,469|开，role 6|

**价格主对照取 eco_linear → eco**。两臂都是同一版 ECO，拍卖单位都是一个满 M 块或已有尾块。允许价格改变实际配额和成交数。独立比较 eco → eco_smr 时，要求实际配额与拍卖逐调用一致。

- 线性价格对物理槽的行数 q 取 `q + 26 × ceil(q/64)`，wa=0。fine 价格对每个非空块取 `intercept(bucket(valid)) + 17 × valid`，六档上界为 8、16、32、40、48、64，截距为 4642、4841、5588、6727、7940、10381。各 rank 的价格是本地各槽之和。slack 在两种模式下均取一个满块的本模式价格。接收方迁入后的价格须严格低于原发送方，且不高于全卡均值加 slack。预检检查实际 role 5 的编译常量；正式窗口前每个 rank 还须从该 batch 已准备的 role 5 读取模式和 slack，记录于 run.json 的 measured.initial[].checkpoint.plan_price。
- 放置直接调用随仓库保存的 DeepInfer 策略原文件。预检检查其 SHA-256 与上游提交一致。策略输入为 [1000 步, 40 层, 384 专家]，输出每层的物理放置和有序副本逆映射。四个自适应臂独立重排一次，检查每层的专家到 rank 分配相等；三个 ECO 臂再检查 rank 内物理槽顺序相等。rr 在两个窗口结束后，按上游公式 `(ep_rank + token_index × 23333) % replica_count` 和该层逆映射复算每一步、每层、每个物理槽的接收行数，并与设备计数逐项核对。模型和 kernel 由本项目的 vLLM/SM90 后端运行。[DeepInfer 放置](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/eplb/policy.py) · [上游 RR gate](https://github.com/deepseek-ai/TileKernels/blob/c1cf914a82e6901c679445f872051ce44ac2cc63/tile_kernels/moe/moe_topk_gate_forward_cuda.py)

## 执行

沿用该机已验证的 HCA、模型、数据、CUDA 和 env3。GC_MOE_CHECKPOINT_ROOT 指向可容纳单档 8×20 GiB KV、两份 route bank 的本地文件系统。脚本按 BS 检查可用空间，保留 route bank、快照清单和哈希，待该档六臂通过结构校验后释放本档临时 KV。需要保留 KV 且空间足够时设置 GC_MAIN20_KEEP_KV=1。以下都在同一 shell 中执行。

```bash
set -eo pipefail
repo=$(git rev-parse --show-toplevel)
cd "$repo"
test -z "$(git status --porcelain)"
git fetch origin main
git merge --ff-only origin/main
revision=$(git rev-parse HEAD)
work=$(mktemp -d /tmp/gcmoe-main24-XXXXXXXX)
git worktree add --detach "$work/code" "$revision"
cd "$work/code"
export GC_MOE_CHECKPOINT_ROOT="${GC_MOE_CHECKPOINT_ROOT:-/tmp}"
bash scripts/trace_layer/run_main24.sh "$work"
```

入口先运行 CPU 测试、策略文件校验、六种配置的 CUDA 编译，以及八卡合成热槽的 CUDA Graph 回放；检查 SM-reuse 开关两侧每个 rank 的输出相同。通过后按矩阵启动正式模型。

执行者只可在**首次模型启动前**做语义等价的最小编译修复。记录编译命令、首次完整错误和补丁，提交单独的修复 SHA，再用干净目录重做完整预检。冻结价格、放置、分配、stream、采样、输入输出校验、计时和解析逻辑。遇其他预检错误、正式运行失败或结构校验失败，保留现场，提交失败回执，列出首条完整异常、退出码、已启动和未启动的配置。已有原生 top‑1 差异时，脚本可继续采集，回执逐对标注可归因范围。每种配置一次。

## 工作量与输出一致性

1. 同一 BS 的六臂共享 prompt、实际反馈 token 和两个窗口各自的逐步逐层源 router ID、权重、batch 顺序以及压缩 attention 索引排序规则。记录实际消费的源行顺序。每一步每层都检查物理行总数等于 BS × 6 × 8，并将实际分配映射回逻辑专家后逐项对照源需求。
2. 正式步 0 之前按请求检查进度、历史和物理块表，装载同一份 KV 后逐 rank 读回核验，逐逻辑专家检查六个打包权重张量及所有副本。三个 ECO 臂均从已清空的 ECO 状态开始。所有臂使用同一测量 SHA、设备和运行时。
3. 同一 BS 输出全部 15 对比较。记录预热原生 top‑1 摘要，逐项比较反馈替换前的 1000×BS 个正式 top‑1。共同控制通过、预热相同且正式差异为零，才将该对的 strict_pair_comparable 记为 true。保留有差异的观察值并给出分母。优先汇报每档 eco_linear → eco 的判定。

## 正式窗口指标与回执

|交付|计算口径|
|---|---|
|main_table.csv，24 行|完整保留历史主表列。端到端 ms/步、相对同档 vanilla 的变化、token IR、全卡总 M 块均值、最重 rank 的 M 块均值、激活物理槽均值、M 块 IR。后五项均按每步每层先在八卡间计算，再平均 40,000 次。每个物理槽的 M 块数分别向上取整。|
|完整 MoE 与端到端|每层每个 rank 取 graph 根 CTA 进入到最后 combine 完成的本地跨度，再取八卡最大值，逐步合计 40 层后报告 1000 步均值 ms/步。端到端使用 worker 同步包围真实 decode。两种耗时都给出 fine 减 linear 的毫秒差及相对 linear 的百分比。计时值对应统一采集配置。|
|cost_pairs.csv、pairs.csv|分别含 4 个价格主对照、全部 60 个臂对。回执逐 BS 摘出两臂的 MoE、端到端、耗时差、严格判定，另列 fine ECO 对 RR、SM-reuse 对 fine ECO。|
|plan 误差|主目标跨度为各 rank 首个有效 GEMM CTA 至最后实际 tile 完成，含 L1、L2 及尾部 push。每步每层将预测分数和各 rank 实测跨度分别归一化，再求占比的平均和最大绝对偏差，单位为百分点。另列完整 graph 的同口径误差。三个 ECO 臂核验本臂设备 role 5 导出的分数；三条对照臂按实际配额离线计价。每一臂都用两种函数在同一实际分配上复算误差，主表以 fine_ 和 linear_ 列标识，rank_shares.csv 记录主价格的各 rank 份额。|
|auction.csv，12 行|每个 ECO 臂给出实际轮数、合格迁出槽数、有单独可行目的地的槽数、实际提交数、接受并已执行的迁移数、接受率、每调用平均成交数、零成交调用数、无出价轮数、有出价但全拒轮数。八个 rank 独立求解同一问题，核对相等后只计一次。auction_calls.csv.gz 留逐步逐层数，auction_phases.csv 分第 0 步冷启动和第 1–999 步稳态。回执并列各 BS 两种价格的出价、成交及 EPLB+RR 的 token/M 块 IR，说明大 BS 的成交受限出现在哪个环节。|

## GitHub 交付

回执写入 `docs/trace_handover_20261119/REPLY_MAIN24_COST.md`。给出起始及测量 SHA、实际模型启动数、机器和软件版本、预检结果、主表、四个价格主对照、拍卖表、原始结果与校验清单的 GitHub 文件页。完整交付包含 24 个 run.json、192 个 rank NPZ、24 份 installed-layout.json、四档逐步专家需求、原始日志、快照清单、workload.sha256、全局 CSV 及 SHA256SUMS。失败也交同名回执和已产生的文件，注明 Capture workspace 路径和首个失败阶段。主表由设计端独立复算后更新。

```bash
cd "$repo"
git merge --ff-only "$(git -C "$work/code" rev-parse HEAD)"
target=docs/trace_executor_20261119/main24_cost_results
if [[ ! -s "$work/delivery/SHA256SUMS" ]]; then target=${target}_blocked; fi
test ! -e "$target"
mkdir "$target"
if [[ -d "$work/delivery" ]]; then cp -a "$work/delivery/." "$target/"; fi
if [[ -d "$work/preflight" ]]; then cp -a "$work/preflight" "$target/"; fi
for log in preflight progress; do
  if [[ -f "$work/$log.log" ]]; then cp "$work/$log.log" "$target/"; fi
done
(cd "$target" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS && sha256sum --quiet -c SHA256SUMS)
receipt=docs/trace_handover_20261119/REPLY_MAIN24_COST.md
test -s "$receipt"
git add -- "$target" "$receipt"
git diff --cached --stat
git commit -m 'Report 24 controlled H20 EPLB and ECO cost measurements'
git push origin HEAD:main
```
