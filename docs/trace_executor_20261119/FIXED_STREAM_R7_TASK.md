# R7 八卡任务 · 先验证 baseline 能否复现，再测 stream

R6 的 105 步物理路由和逐槽负载通过校验。100 步 E2E 为 82.506 → 82.243 ms/步，step 2 的 40 层 MoE 为 34.170 → 34.203 ms。第 0 步有 6/256 个原生 top‑1 不同，105 步合计 563/26,880。首步在 layer 0、1 的 MoE 探针逐位一致。到 layer 2 的 MoE 入口，每个 rank 的 BF16 输入都有 108 行不同。648 是同层六路 routed 返回值不同的槽行数。现有观测尚不能把分歧归因到 sparse attention、计算调度或某个算子。[R6 两臂明细](fixed_stream_results_r6/paired.csv) · [首步各层](fixed_stream_results_r6/first_step_layers.csv)

## 实验

总共启动两次模型，按 baseline、stream 顺序各一次，同一提交、同一组空闲 8×H20、原 256 个请求和原 replay 文件，沿用修好的 ECO 和旧价格，每槽为 `26 × ceil(owned / 64) + owned`。两臂均用 baseline 预热。第 0 步和第 1 步沿用 R6 已有的两个过渡步。

| 回放步 | baseline 进程 | stream 进程 | 取数 |
|---|---|---|---|
| 0 | baseline | baseline | 从同一份重排后 KV 恢复。记录 A/A 的全部 40 层、原生 sampler、layer 2 FlashMLA，计算控制步结束时八个 rank 的完整 KV backing 摘要。 |
| 1 | baseline | stream | 两边都从本轮 baseline 第 0 步结束后的第二份 KV 快照恢复。请求历史、物理页表、权重、ECO 和 epoch 校验成功后切换。再采一遍上述前向数据。 |
| 2 | baseline | stream | 40 层 torch profiler Perfetto、逐 rank token/M 块及 plan 与实际占比，两臂各 11 层 absolute Perfetto JSON。dispatch 包含 count exchange，routed 从首个 CTA 领到有效计算任务起计，MoE 从 graph 起点到 combine 结束。 |
| 3–4、5–104 | baseline | stream | 两个过渡步，随后连续 100 个无 profiler 步测 E2E。第 1 步的 MoE 张量在设备上封存，计时结束后读取。 |

layer 2 探针在实际 FlashMLA decode 边界按请求记录 Q、SWA 和压缩 KV 的有效索引长度、原顺序和保留重复项的排序摘要，以及原始 MLA 输出的有效头。排序仅用于主机比对。两臂均把原张量交给原生 backend；索引尾部 padding 不参与摘要。两次恢复都逐 rank 从 GPU 回读核验。A/A 结束时的原始 KV 单独记录，第二次恢复不能掩盖这项差异。完整 KV 也包含预留字节，进一步定位时需要结合活跃行和索引证据。

## 执行步骤

回执源目录是上轮同一台机器上的原 replay。已有产物需保留。两份 KV 快照共需约 176 GiB，`GC_MOE_CHECKPOINT_ROOT` 选择 8 个 worker 可读写、有额外 trace 与行摘要空间的本机磁盘。沿用 R6 的模型、环境和数据。以下命令在同一个 bash 会话执行。执行者仅可修复编译器明确报告的语法、类型或版本兼容编译问题，须保留语义、单独提交，并记录错误与补丁。数据、代码逻辑、调度、采样、门禁和运行次数由发布者维护。

```bash
set -eo pipefail
repo=$(git rev-parse --show-toplevel)
cd "$repo"
test -z "$(git status --porcelain)"
git fetch origin main
git merge --ff-only origin/main
measurement=$(git rev-parse HEAD)
work=$(mktemp -d /tmp/gcmoe-r7-XXXXXXXX)
git worktree add --detach "$work/tools" "$measurement"
replay=/tmp/gcmoe-route-replay-Q6W0KywL
export GC_MOE_CHECKPOINT_ROOT="${GC_MOE_CHECKPOINT_ROOT:-/tmp}"
(
  cd "$work/tools"
  source scripts/env.sh
  for f in "$replay"/{placement.json,tokens.json,routes-rank{0..7}.pt}; do test -s "$f"; done
  df -h "$work" "$GC_MOE_CHECKPOINT_ROOT"
  for t in tests/test_*_cpu.py; do "$GC_ENV/bin/python" "$t"; done
  "$GC_ENV/bin/python" tests/test_stream_checkpoint_cpu.py --gpu-preflight
  "$GC_ENV/bin/python" tests/test_first_step_cpu.py --cuda
  "$GC_ENV/bin/python" tests/test_attention_probe_cpu.py --cuda
  GC_MOE_FIXED_DIR="$replay" GC_MOE_MB_CLAIM=1 GC_MOE_BLOCK_M_FORCE=64 \
  GC_MOE_ECO_MAX_COPIES=8 GC_MOE_LAYERS=40 GC_MOE_STEP_PROFILE=1 \
  GC_MOE_PULL_TRACE=1 GC_MOE_FINE_TRACE=1 GC_MOE_FINE_FLOOR=1 \
  GC_MOE_FINE_RECS=4194304 GC_MOE_IR_STEP=0 GC_MOE_SM90_DISPATCH_SMS=6 \
  GC_MOE_SM90_SM_REUSE=0 SMOKE_ECO_WB=26 "$GC_ENV/bin/python" scripts/smoke_compile_eco.py
) 2>&1 | tee "$work/preflight.log"
```

上述预检均无需加载模型。GPU attention 预检经过已安装 vLLM 的真实 `_forward_decode`、真实索引转换和 CUDA Graph，用小张量替身校验 FlashMLA 调用边界。预检通过后本轮运行入口仅执行一次。失败保留原始目录并按下面步骤交付，已经启动的模型计入两次预算；若预检后修了编译问题，在测量 worktree 单独提交补丁，用 `git -C "$repo" merge --ff-only "$(git -C "$work/tools" rev-parse HEAD)"` 把修复纳入交付分支。两臂从同一修复提交完成全部预检后才开始第一次模型启动。

```bash
if bash "$work/tools/scripts/trace_layer/run_multilayer_study.sh" \
     "$work/tools" "$work/delivery" --negative-control-from "$replay" \
     2>&1 | tee "$work/progress.log"; then
  (cd "$work/delivery" && sha256sum --quiet -c SHA256SUMS)
fi
```

## 回执

写入 `docs/trace_handover_20261119/REPLY_FIXED_STREAM_R7.md`。完整结果写入 `docs/trace_executor_20261119/fixed_stream_results_r7/`，中断写入同名加 `_blocked` 的目录。保留两组各八份 checkpoint 清单、原始 workload、40 层 counts 与两类 Perfetto、rank 表、`first_step_layers.csv`、`treatment_step_layers.csv`、`flashmla_rows.csv`、比较结果、校验和。KV 二进制和逐行 MoE 参考摘要保留在执行机。请附 layer 20 的 GitHub 文件页。

- 报告实测提交、两次 capture ID、实际启动次数、A/A 结束时恢复前 KV 不同的 rank、两次回读、`negative_control_match`、`strict_pair_comparable` 与 `scheduling_gate_passed`。
- 分列 A/A、A/B 的首个可见分歧。写明层、字段、rank、请求示例及原生 token，逐项区分 FlashMLA 的 query、有效索引长度、集合含重复数、排列顺序和输出。记录 105 步原生分歧总数。共同失败的 A/A 说明分歧可在开启 stream 之前发生。Q 和索引一致而输出有差异时，还需核实本步写入的 KV 和内部实现，当前探针无法单独区分二者。
- 列两项主指标、变化百分比、40 层快慢数量、十步 E2E 分段及 plan 占比误差。严格结论要求 A/A、切换前后的既有门禁及 105 步原生输出全通过，同时 E2E 和 MoE 都下降。单次双臂只支持该负载上的观测。交付实测结果，valid cost 继续暂缓。
- 失败时记录首个错误、对应命令、已完成阶段和原始日志路径。若已经取得第 0 步，交付可用的 `control-step-mode*.json`。完整实验达到数值或性能否定结论时使用正常结果目录。

```bash
cd "$repo"
target=docs/trace_executor_20261119/fixed_stream_results_r7
if [[ ! -s "$work/delivery/SHA256SUMS" ]]; then target=${target}_blocked; fi
test ! -e "$target"
mkdir "$target"
if [[ -d "$work/delivery" ]]; then cp -a "$work/delivery/." "$target/"; fi
cp "$work/preflight.log" "$target/"
if [[ -f "$work/progress.log" ]]; then cp "$work/progress.log" "$target/"; fi
(cd "$target" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
receipt=docs/trace_handover_20261119/REPLY_FIXED_STREAM_R7.md
test -s "$receipt"
git add "$target" "$receipt"
git commit -m 'Report R7 same-mode control and EP8 stream comparison'
git push origin HEAD:main
```
