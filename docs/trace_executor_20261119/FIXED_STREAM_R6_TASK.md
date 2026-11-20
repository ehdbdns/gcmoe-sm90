# R6 八卡任务 · 共享专家输入修复与首个分歧定位

R5 的 100 步平均 E2E 从 81.762 降到 81.369 ms/步，step 2 的 40 层 MoE 从 34.128 降到 33.948 ms。共同 KV 快照、逐槽物理路由和 layer 0 的输入输出均通过对照。回放第 0 步已有 24 个原生 top‑1 不同，例如 req9 为 494 / 281。105 步合计 2,782/26,880 个不同。R5 的严格判据因回放期间的原生分歧未通过，快照前的预热摘要仅作诊断。[R5 结果](../trace_handover_20261119/REPLY_FIXED_STREAM_R5.md)

## 这一轮测什么

SM90 role 4 从独立的 `shared_l1_acts` 和 `shared_l1_acts_sf` 读取共享专家输入。原 vLLM 接入仅填写 routed 的 K128 输入，初始化清零的 shared 缓冲区没有相应的真实激活。R6 在两臂都按项目独立 SM90 shared 测试的口径填入每 token 每 K32 的 E4M3 数据和 UE8M0 数值的 FP32 尺度。此修复改变了被测模型的计算，跨轮耗时不可直接归因于 stream。R5 只记录到 layer 0，尚无法确定当时原生分歧的原因。[独立 shared 测试](../../source/tests/test_sm90_gc_moe_se.py)

仍只启动两个模型，顺序 baseline、stream，各一次。两臂使用同一提交、同一空闲的 8×H20、256 个请求、原路由文件、修复后的 ECO 和旧价格 `26 × ceil(owned/64) + owned`（逐槽求和）。各自以 baseline 预热。由本轮 baseline 建立新的重排后 KV 快照，两臂分别恢复并校验。保留 R5 的硬件、请求与物理页表、权重和回放验收。valid 耗时对照继续暂缓。

第 0 步的 40 层均采实际 BF16 输入、两条分支的 FP8 输入和尺度、逐 token 的六路 routed 回传、共享输出、最终 BF16 输出。另取实际送给 LM head 的 hidden 和送给 sampler 的 logits。执行机保留 baseline 的逐行哈希，stream 比较全部有效行，回执包存整张量摘要、不同记录数和请求示例。对原生分歧按请求报告。首步采集与主机读取均在计时前完成。后续 graph 中的审计分支跳过这些张量拷贝。

第 2 步保留 40 层 torch profiler Perfetto、逐 rank token/M 块及 plan 与实测占比，另存 11 层 absolute Perfetto JSON。dispatch 含 count exchange，routed 时间从第一个 CTA 领到有效票开始，MoE 从同一 graph 最早根到 combine 结束。两次过渡步后关闭 profiler 和 fine 写入，连续测 100 步 E2E。保留原有计时分段和 105 步原生预测。

## 执行顺序

使用 R5 的模型、数据、网络和软件环境。KV 快照放在八个 worker 均可访问且有至少 88 GiB 可用的目录，另留 trace 空间。已有 R5 快照和产物保留。下面的 GPU 小张量预检验证 FP8 字节和尺度、CUDA Graph 回放、首步探针开关，随后执行一次已有 SM90 编译预检。均不加载模型。

```bash
set -eo pipefail
repo=$(git rev-parse --show-toplevel)
cd "$repo"
test -z "$(git status --porcelain)"
git fetch origin main
git merge --ff-only origin/main
measurement=$(git rev-parse HEAD)
work=$(mktemp -d /tmp/gcmoe-r6-XXXXXXXX)
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
  GC_MOE_FIXED_DIR="$replay" GC_MOE_MB_CLAIM=1 GC_MOE_BLOCK_M_FORCE=64 \
  GC_MOE_ECO_MAX_COPIES=8 GC_MOE_LAYERS=40 GC_MOE_STEP_PROFILE=1 \
  GC_MOE_PULL_TRACE=1 GC_MOE_FINE_TRACE=1 GC_MOE_FINE_FLOOR=1 \
  GC_MOE_FINE_RECS=4194304 GC_MOE_IR_STEP=0 GC_MOE_SM90_DISPATCH_SMS=6 \
  GC_MOE_SM90_SM_REUSE=0 SMOKE_ECO_WB=26 "$GC_ENV/bin/python" scripts/smoke_compile_eco.py
) 2>&1 | tee "$work/preflight.log"
```

预检通过后只执行一次以下入口。执行者可在模型启动前修复编译器点名的语法或类型兼容问题，保留语义并单独提交，在回执中附补丁和实测 SHA。调度、模型语义、比较判据、数据和实验参数由发布者维护。运行或校验出错后保留现场并交付回执；已启动的模型计入两次预算。

```bash
if bash "$work/tools/scripts/trace_layer/run_multilayer_study.sh" \
     "$work/tools" "$work/delivery" --trace-first-step-from "$replay" \
     2>&1 | tee "$work/progress.log"; then
  (cd "$work/delivery" && sha256sum --quiet -c SHA256SUMS)
fi
```

## 交付与判定

- 回执写到 `docs/trace_handover_20261119/REPLY_FIXED_STREAM_R6.md`。成功执行的两臂（含数值判据未通过）交到 `docs/trace_executor_20261119/fixed_stream_results_r6/`。附两臂 40 层总览和 counts、各 11 份 absolute、逐 rank 表、checkpoint 清单、workload、两项主指标及 plan 误差、`comparison.json`、`first_step_layers.csv`、校验和及 layer 20 的 GitHub 文件页。逐行哈希和 KV 二进制保留在执行机。
- 报告两个 capture ID、启动次数、耗时及变化、十步分段差、`shared_weights_match`、`shared_branch_active`、`first_step_chain_match`、`strict_pair_comparable`、`scheduling_gate_passed`。若数值不同，照录 `first_observed_differences` 的层、阶段、rank、请求和 top‑k 槽（适用时），以及 `first_step_native_differences` 的 baseline/stream token。若权重不同，另列 `shared_weight_differences`。最早观测到差异的边界是定位线索，根因仍须相应数据支持。
- 严格通过要求 R5 所有回放门禁、第 0 步八卡全 40 层和 sampler 数值一致、各层激活与尺度均为有限数、实际共享权重一致且分支有非零输入输出，再要求两项主耗时同时下降。每臂一次，结论限于该负载。两项分别报告，无需凑正向结果。
- 中断时交到 `fixed_stream_results_r6_blocked/`，附首条报错、失败命令、已完成阶段、实际原始目录和相关日志尾部。

```bash
cd "$repo"
target=docs/trace_executor_20261119/fixed_stream_results_r6
if [[ ! -s "$work/delivery/SHA256SUMS" ]]; then target=${target}_blocked; fi
test ! -e "$target"
mkdir "$target"
if [[ -d "$work/delivery" ]]; then cp -a "$work/delivery/." "$target/"; fi
cp "$work/preflight.log" "$target/"
if [[ -f "$work/progress.log" ]]; then cp "$work/progress.log" "$target/"; fi
(cd "$target" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
receipt=docs/trace_handover_20261119/REPLY_FIXED_STREAM_R6.md
test -s "$receipt"
git add "$target" "$receipt"
git commit -m 'Report R6 EP8 stream timing and first divergent layer'
git push origin HEAD:main
```
