# R5 八卡任务 · 从同一份 KV 缓存验证逐块调度

在上一轮，两臂的全部 4,200 次 MoE 调用使用了相同的路由和逐槽行数。切换 stream 前，八个 rank 的预热激活摘要已不同。R4 的 E2E 下降 1.39%，step 2 的 40 层 MoE 上升 1.12%。R5 专门验证在共同起点、同一批真实请求和同一价格下，stream 能否同时改善这两项。优于、持平、退化和不可比都按实回报。[R4 回执](../trace_handover_20261119/REPLY_FIXED_STREAM_R4.md)

## 固定条件

同一台空闲的 8×H20，同一模型、数据集、驱动、进程绑定和提交。只启动两次完整模型，依次运行 baseline 和 stream。两臂均使用修复后的 ECO 拍卖、旧线性价格 `26 × M块数 + routed rows`、M=64、冷启动 6 轮和热启动 1 轮。valid 耗时函数继续暂缓。

1. 两臂用 baseline 完成真实 prefill 和 200 步 warmup，重排、装载原路由文件、清空 ECO 价格状态。每卡固定 11 GiB KV 预算。baseline 发布各 rank 的完整 KV 物理存储，两个进程均将该快照逐块写回原 CUDA Graph 地址，并从设备读回计算 SHA-256。运行脚本比较实际请求、轨迹位置、物理页表、缓存别名、ECO、epoch 和重排后的 40 层专家权重。八卡全部确认后，另一条 RPC 才向两臂分别写 0 和 1。
2. 第一个 flush 步对 layer 0 实际消费的 FP8 输入、FP32 尺度、BF16 输出和调用 epoch 取证。两臂每步都做相同的小量设备拷贝，主机读取只在第一次 flush 后发生。继续逐次检查 105 步 × 40 层的路由身份和逐槽行数，并比较强制反馈前的 26,880 个原生 top‑1。预热历史单独保留。
3. step 2 用 torch profiler 采 40 层。MoE 时间按同 rank 最早的 graph 根到 combine 结束；整步按每个 rank 累加 40 层后取八卡最大值。采集每层每 rank 的 token、M 块、plan 预测占比、实测占比和百分点差。层 0、4、8、12、16、20、24、28、32、36、39 加采逐 CTA absolute Perfetto JSON。沿用 count exchange、pull、wait token、首个有效计算 CTA 到 routed 结束及 combine 的标记。
4. 在两个过渡步后测连续 100 步纯 decode，窗口两端同步，profiler 和 fine 写入均关闭。新增十个连续十步的主机用时分段，最后一段含最终同步。各段共用一次模型运行，仅用于观察差值分布。

## 执行

在同一 Bash 会话执行。沿用 R4 模型和数据路径。在宿主机选一个对八个本地 worker 可见、有至少 88 GiB 可用空间的可写文件系统保存 KV。脚本支持通过 `GC_MOE_CHECKPOINT_ROOT` 指定该目录。输出和原始 trace 还需另留空间。预检运行 CPU 测试、小张量 CUDA 存取和一次 SM90 内核编译，模型启动预算仍为两次。

```bash
set -eo pipefail
repo=$(git rev-parse --show-toplevel)
cd "$repo"
test -z "$(git status --porcelain)"
git fetch origin main
git merge --ff-only origin/main
measurement=$(git rev-parse HEAD)
work=$(mktemp -d /tmp/gcmoe-r5-XXXXXXXX)
git worktree add --detach "$work/tools" "$measurement"
replay=/tmp/gcmoe-route-replay-Q6W0KywL
export GC_MOE_CHECKPOINT_ROOT="${GC_MOE_CHECKPOINT_ROOT:-/tmp}"
(
  cd "$work/tools"
  source scripts/env.sh
  for f in "$replay"/{placement.json,tokens.json,routes-rank{0..7}.pt}; do test -s "$f"; done
  df -h "$GC_MOE_CHECKPOINT_ROOT" "$work"
  for t in tests/test_*_cpu.py; do "$GC_ENV/bin/python" "$t"; done
  "$GC_ENV/bin/python" tests/test_stream_checkpoint_cpu.py --gpu-preflight
  GC_MOE_FIXED_DIR="$replay" GC_MOE_MB_CLAIM=1 GC_MOE_BLOCK_M_FORCE=64 \
  GC_MOE_ECO_MAX_COPIES=8 GC_MOE_LAYERS=40 GC_MOE_STEP_PROFILE=1 \
  GC_MOE_PULL_TRACE=1 GC_MOE_FINE_TRACE=1 GC_MOE_FINE_FLOOR=1 \
  GC_MOE_FINE_RECS=4194304 GC_MOE_IR_STEP=0 GC_MOE_SM90_DISPATCH_SMS=6 \
  GC_MOE_SM90_SM_REUSE=0 SMOKE_ECO_WB=26 "$GC_ENV/bin/python" scripts/smoke_compile_eco.py
) 2>&1 | tee "$work/preflight.log"
```

预检通过后执行一次以下入口，两个臂都由它调度。执行者只可在模型启动前修复编译器报告的语法或类型错误，并须保证语义等价，单独提交补丁，把两臂固定在同一个修复提交，在回执中附 SHA 和 diff。算法、阈值、状态校验、解析和采样由发布者维护。启动后遇运行错误或校验失败，保存原始证据并回执，到此结束本轮。两次启动预算包含已启动但中断的臂。

```bash
if bash "$work/tools/scripts/trace_layer/run_multilayer_study.sh" \
     "$work/tools" "$work/delivery" --same-state-from "$replay" \
     2>&1 | tee "$work/progress.log"; then
  (cd "$work/delivery" && sha256sum --quiet -c SHA256SUMS)
fi
```

## 回执

将回执写到 `docs/trace_handover_20261119/REPLY_FIXED_STREAM_R5.md`。写明实际启动次数、提交和软硬件、原始路径、两个 capture ID、首个失败命令或分歧步及请求。分别列出两项主耗时、绝对及百分比变化、十个 E2E 分段差、各层快慢、plan 占比误差与控制项。给出两臂 step 2 layer 20 的 GitHub 文件页。此前 R4 的预热摘要继续列为诊断。

- `checkpoint_match`、`same_hardware`、`same_run_controls`、`same_physical_routes`、`first_moe_input_match` 和 `first_moe_output_match` 均须通过，并且 105 步原生预测全部一致，才可报告 `strict_pair_comparable=true`。
- 只有整个 100 步 E2E 和 step 2 的 40 层 MoE 同时下降且对照可比，才可报告 `scheduling_gate_passed=true`。固定先后顺序，每臂一次，结论限于本次负载和测量。若方向相反，明确写出哪项改善、哪项退化。逐位的 MoE 激活比较覆盖第一调用，后续调用按路由、逐槽行数和原生预测验收。

成功目录包含 80 份总览、80 份 counts、22 份 absolute、两臂各 320 行逐 rank 数据、两个 workload、八份 checkpoint 清单、硬件记录、配对表与 SHA256SUMS。KV 二进制约 88 GiB，只在八卡机器保留，GitHub 回传清单与摘要。出现错误时在 blocked 目录追加原始 status、failed-command、日志尾部及已产生的清单。预检失败可直接按下列步骤交付。

```bash
cd "$repo"
target=docs/trace_executor_20261119/fixed_stream_results_r5
if [[ ! -s "$work/delivery/SHA256SUMS" ]]; then target=${target}_blocked; fi
test ! -e "$target"
mkdir "$target"
if [[ -d "$work/delivery" ]]; then cp -a "$work/delivery/." "$target/"; fi
cp "$work/preflight.log" "$target/"
if [[ -s "$work/progress.log" ]]; then cp "$work/progress.log" "$target/"; fi
(cd "$target" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
receipt=docs/trace_handover_20261119/REPLY_FIXED_STREAM_R5.md
test -s "$receipt"
git add "$receipt" "$target"
git commit -m 'Report the R5 common-checkpoint EP8 stream comparison'
git push origin HEAD:main
```
