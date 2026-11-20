# R8 八卡任务 · 固定压缩 KV 索引顺序后复测 stream

R7 第 0 步两个进程都跑 baseline。layer 2 每个 rank 的 Q、SWA 索引、压缩索引有效长度及含重复项的集合全部一致。按同一请求与 rank 重算，869/2,048 行的压缩索引排列不同，564 行的 MLA 输出不同，后者全部落在前者中。第 1 步对应数字为 857 和 570，同样全部重合。R7 的 A/A、A/B、采样步原生 top‑1 差异分别为 3、10、5 个。105 步共 617/26,880，单步范围为 0–12。100 步 E2E 下降 1.58%，step 2 的 40 层 MoE 下降 0.19%，严格判据未通过。[R7 指标](fixed_stream_results_r7/paired.csv) · [索引与 MLA 输出](fixed_stream_results_r7/flashmla_rows.csv)

## 干预和量测

两臂使用同一新提交。在实际 FlashMLA decode 调用前，将每行有效前缀内的非负物理槽号升序排列，保留重复次数、负掩码的位置和值以及前缀后的每个元素。每次只传一份新索引给原 backend，原生 top‑k 选择、长度、Q、KV、SWA 索引及原生成采样仍由原模型计算。此规则覆盖模型声明使用压缩 KV 的每个 decode 层。两臂在所有压缩层记录相同的检查数据。原始排列差异单列报告，严格对照核验实际送入 FlashMLA 的序列和长度。新排序开销计入两臂 E2E，本轮性能结论限于这种确定索引顺序的负载。

| 回放步 | baseline / stream 两个进程的实际模式 | 采样 |
|---|---|---|
| 0 | baseline / baseline | 从同一份重排后快照恢复，核验 40 层及 sampler、八卡控制步结束时的原始 KV、所有压缩层逐行排序，以及 layer 2 实际传入 FlashMLA 的张量与输出。 |
| 1 | baseline / stream | 从 baseline 控制步结束后的第二份快照共同恢复，校验后切换。重复上述前向记录。 |
| 2 | baseline / stream | QPLB 重排后同一步 40 层 torch profiler Perfetto、逐 rank token 数/M 块数和预测/实际耗时占比，另取 11 层 absolute Perfetto JSON。dispatch 含 count exchange，routed 从首个 CTA 领到可计算任务开始，MoE 从 graph 起点到 combine 结束。 |
| 3–4、5–104 | baseline / stream | 两步过渡，随后连续 100 个无 profiler 步测 E2E。 |

保留同一组空闲 8×H20、同一批 256 个请求、原 replay 文件、修正后的 ECO 和旧价格，每槽为 `26 × ceil(owned/64) + owned`。只启动这两个模型，每臂一次。有效索引变化、KV 回读差异、任一步原生输出分歧均按原严格判据处理。valid cost 继续暂缓。

## 执行步骤

沿用 R7 的模型、环境和数据。两份新 KV 快照需要约 176 GiB，`GC_MOE_CHECKPOINT_ROOT` 指向八个 worker 可读写且还能容纳 trace 与行摘要的本机磁盘。保留旧快照与回执。下面的命令在同一个 bash 会话执行。执行者仅可修复编译器明确报告的语法、类型或版本兼容编译错误，须保持语义、单独提交并附原报错和补丁。采样、模型计算、判据、参数及启动预算由发布者维护。

```bash
set -eo pipefail
repo=$(git rev-parse --show-toplevel)
cd "$repo"
test -z "$(git status --porcelain)"
git fetch origin main
git merge --ff-only origin/main
measurement=$(git rev-parse HEAD)
work=$(mktemp -d /tmp/gcmoe-r8-XXXXXXXX)
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
  "$GC_ENV/bin/python" tests/test_attention_order_cpu.py --cuda
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
     "$work/tools" "$work/delivery" --ordered-decode-from "$replay" \
     2>&1 | tee "$work/progress.log"; then
  (cd "$work/delivery" && sha256sum --quiet -c SHA256SUMS)
fi
```

## 回执与验收

写入 `docs/trace_handover_20261119/REPLY_FIXED_STREAM_R8.md`，正常结果写入 `docs/trace_executor_20261119/fixed_stream_results_r8/`。中断使用同名加 `_blocked` 的目录，记录首个错误、实测提交、已启动模型数、原始路径与日志，保留已产生的控制步材料。完整负结果使用正常目录交付。

- 报告两臂 SHA、capture ID、启动次数、两项主耗时、40 层快慢数、10 个分段、plan 占比误差，以及第 0、1、2 步和 105 步合计的原生差异。从 `comparison.json` 抄录 `index_order`、`negative_control_match`、`strict_pair_comparable` 与 `scheduling_gate_passed`。
- 交付 `compressed_index_order.csv`。每个压缩层每个 rank 每步一行，覆盖第 0、1 步。分别报告两臂原顺序是否一致、各臂有多少请求实际重排、消费顺序与长度是否一致。若消费序列有差异，定位最早的步、层和 rank。第 2 层消费摘要另由真正在用的 FlashMLA 输入验证。每个压缩层的 105 步调用次数和实际模式保存在 workload，比较器逐项核验。
- 在 `flashmla_rows.csv`、`first_step_layers.csv` 和 `treatment_step_layers.csv` 中分别报告 A/A 和 A/B 的首个可见差异。若排序已经对齐但 MLA 输出仍分歧，把该项连同 Q、SWA、原始 KV 摘要和请求示例交回，如实区分观测与假设。
- 交付同一采样步的两组 40 层 counts / Perfetto、各 11 层 absolute Perfetto、rank 表、两套各八份快照清单、workload、比较 CSV/JSON 与 SHA256SUMS。附 baseline / stream 的 layer 20 普通和 absolute 四个 GitHub 文件页。KV 二进制保留在执行机。
- 两项主耗时都下降且严格对照全部通过，才认定该负载上 stream 更快。若严格对照通过但一项或两项耗时未下降，把相应观测作为正式结果交付。

```bash
cd "$repo"
target=docs/trace_executor_20261119/fixed_stream_results_r8
if [[ ! -s "$work/delivery/SHA256SUMS" ]]; then target=${target}_blocked; fi
test ! -e "$target"
mkdir "$target"
if [[ -d "$work/delivery" ]]; then cp -a "$work/delivery/." "$target/"; fi
cp "$work/preflight.log" "$target/"
if [[ -f "$work/progress.log" ]]; then cp "$work/progress.log" "$target/"; fi
(cd "$target" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
receipt=docs/trace_handover_20261119/REPLY_FIXED_STREAM_R8.md
test -s "$receipt"
git add "$target" "$receipt"
git commit -m 'Report R8 canonical sparse decode and EP8 stream comparison'
git push origin HEAD:main
```
