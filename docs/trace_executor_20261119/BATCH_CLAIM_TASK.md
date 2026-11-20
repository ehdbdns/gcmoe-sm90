# 八卡任务书 · 验证逐 M 块领票并交付四组 absolute trace

在同一台八卡 H20 机器上完成四次独立采集。每次使用 torch profiler 和同次调用的设备打点，采样逐层 EPLB 重排后的 decode step 2、layer 8 和 20（层号从 0 开始）。每层保留八个 rank。回执提交到 [交接目录](../trace_handover_20261119/) 的 `REPLY_BATCH_CLAIM.md`，产物放在本目录的 `batch_claim_results/`。

| 目录 | 执行代码 | STREAM_M_BLOCKS |
|---|---|---:|
| before/baseline | `98b6194` | 0 |
| before/stream | `98b6194` | 1 |
| after/baseline | `6617ee9` | 0 |
| after/stream | `6617ee9` | 1 |

上一轮 layer 8 / rank 6 的旧 stream 首个非空槽在 41 µs 拉完，shared 在 192.4 µs 结束，routed 在 193.3 µs 进入，首次有效领取在 195 µs；71 个 CTA 的首个 tile 分布在 196–330 µs。旧 baseline 的首波为 193–194 µs。两轮的该 rank 各为 31 和 33 个 M 块。此次分别测量 routed 启动前的图依赖、入场后的首波展开和同一 CTA 两个 tile 之间的空档。

新版 stream 让每个满 M 块和尾块各自发布一批 L1 N-tile。本块所有 L1 tile 完成后发布本块的 L2 N-tile。CTA 可以跳过尚未发布的块；多个 CTA 同时争到一个块时，有效领取粒度始终为一个阶段的一个 M 块、一个 N-tile。保持四组的 shared → routed 图依赖。

## 1. 固定版本与编译

在执行机仓库根目录开一个 Bash，会话贯穿步骤 1–4。确认八张 H20 空闲，继承上轮验证过的模型、数据集、网络接口和工具链环境。使用两个独立工作树。原始捕获、编译日志和临时包都在仓库外。

```bash
set -eo pipefail
repo=$(git rev-parse --show-toplevel)
cd "$repo"
test -z "$(git status --porcelain)"
git fetch origin main
git merge --ff-only origin/main
old=98b6194a56e7468653bb6230bf0b3887c9466dce
new=6617ee976fa86eb77eb3a34b8b6df3e60059a586
git merge-base --is-ancestor "$new" HEAD
work=$(mktemp -d /tmp/gcmoe-claim-ab-XXXXXXXX)
bundle="$work/delivery"
mkdir -p "$bundle/preflight"
git worktree add --detach "$work/before" "$old"
git worktree add --detach "$work/after" "$new"
echo "Working directory: $work"
(
  cd "$work/after"
  source config/experiment.env
  source scripts/env.sh
  export GC_MOE_SRC="$PWD/source" PYTHONPATH="$PWD/source:$PWD/patches:$PWD/integration"
  export GC_MOE_MB_CLAIM=1 GC_MOE_SM90_PULL_TMA=1 GC_MOE_BLOCK_M_FORCE=64
  export GC_MOE_SM90_SM_REUSE=0 GC_MOE_SM90_DISPATCH_SMS=6 GC_MOE_IR_STEP=0
  export GC_MOE_PULL_TRACE=1 GC_MOE_ECO_TRACE=1 GC_MOE_FINE_FLOOR=1
  export GC_MOE_FINE_EPOCHS=1 GC_MOE_FINE_RECS=4194304 GC_MOE_FINE_MIN_EPOCH=0
  export SMOKE_ECO=1 SMOKE_NULL=0 SMOKE_NOAUC=0
  nvidia-smi --query-gpu=index,name,driver_version,memory.used --format=csv | tee "$bundle/preflight/gpu.log"
  /usr/local/cuda/bin/ptxas --version | tee "$bundle/preflight/ptxas.log"
  "$GC_ENV/bin/python" -c 'import cutlass, torch, vllm; print(getattr(cutlass, "__version__", "unknown"), torch.__version__, torch.version.cuda, vllm.__version__)' | tee "$bundle/preflight/runtime.log"
  for mode in 0 1; do
    for profile in 0 1; do
      roles='[1, 2, 5, 3]'
      wb=64
      if (( profile )); then roles='[1, 4, 2, 5, 3]'; wb=26; fi
      log="$bundle/preflight/stream-${mode}-profile-${profile}.log"
      GC_MOE_SM90_STREAM_M_BLOCKS="$mode" GC_MOE_STEP_PROFILE="$profile" GC_MOE_FINE_TRACE="$profile" \
        SMOKE_ECO_WB="$wb" "$GC_ENV/bin/python" scripts/smoke_compile_eco.py 2>&1 | tee "$log"
      grep -Fq "compiled roles $roles" "$log"
    done
  done
)
```

仅在新版 CuTe/ptxas 报编译错误时允许改代码。保留失败命令与完整诊断，在 after 工作树做单处等价语法或类型兼容修复，单独提交。先复测失败组合，再重跑四组编译，用修后同一 SHA 跑两轮 after。若其中一轮已跑，保留原轮记录并重采两轮 after。旧版 SHA 固定。其余工作严格执行下述命令。运行挂起、数值、采样、解析、绘图、验收或环境出错时，停在首个失败步骤，回传日志、SHA 和原始数据路径。不得修改调度、栅栏、内存序、trace 位置、解析器、绘图、测试、模型、参数或验收口径。

## 2. 依序运行四次采集

入口固定 BS256、block M=64、MB-CLAIM、逐层重排、shared expert、TMA、CUDA Graph 和 step 2。stream 每次先做八卡 135 行热点门禁（64、64、7），比较两种路径的逐 rank BF16 校验和。每次调用均写独立的 capture ID 和原始目录；保留这些目录至结果验收。

```bash
collect() (
  rev=$1
  mode=$2
  flag=0
  if [[ $mode == stream ]]; then flag=1; fi
  cd "$work/$rev"
  sha=$(git rev-parse HEAD)
  dest="$bundle/$rev/$mode"
  mkdir -p "$dest"
  GC_MOE_SM90_STREAM_M_BLOCKS="$flag" bash scripts/trace_layer/run_aligned_ep8.sh "$sha" --fine-trace \
    2>&1 | tee "$dest/runner.log"
  capture=$(sed -n 's/^Capture workspace: //p' "$dest/runner.log")
  grep -Fxq 'exit_code=0' "$capture/status.txt"
  test "$(cat "$capture/execution.sha")" = "$sha"
  cp "$capture"/deliver/*.json "$capture"/{execution.sha,status.txt,sha256.txt} "$dest/"
  if (( flag )); then cp "$capture"/hot-parity-*.sha "$dest/"; fi
)
collect before baseline
collect before stream
collect after baseline
collect after stream
diff -u "$bundle/before/stream/hot-parity-0.sha" "$bundle/after/stream/hot-parity-0.sha"
```

## 3. 输出八个 rank 的负载与 absolute trace

四组统一从同次采集、同 step/layer/epoch 的 detail 与 counts 生成图。绘图使用 after 版本，时间零点为该 rank 的 dispatch 入口。每张图显示真实 pull CTA 行、compute CTA 行和 L2 push；相邻 CSV 逐条保留微秒区间。汇总表中的 `tokens = routed_rows / input_replication`，`m_blocks = sum(ceil(slot_owned / block_m))`。计数源自采样层的实测 owned。

```bash
(
  cd "$work/after"
  source scripts/env.sh
  "$GC_ENV/bin/python" scripts/trace_layer/batch_claim_report.py "$bundle"
  for rev in before after; do
    for mode in baseline stream; do
      dest="$bundle/$rev/$mode"
      mkdir -p "$dest/absolute"
      for detail in "$dest"/*.detail.json; do
        stem=$(basename "$detail" .detail.json)
        for rank in {0..7}; do
          "$GC_ENV/bin/python" scripts/plot_absolute_trace.py "$detail" "$rank" \
            "$dest/absolute/$stem.rank$rank.png"
        done
      done
    done
  done
)
test "$(wc -l < "$bundle/rank_metrics.csv")" -eq 65
test "$(find "$bundle" -name '*.perfetto.json' | wc -l)" -eq 16
test "$(find "$bundle" -name '*.detail.perfetto.json' | wc -l)" -eq 8
test "$(find "$bundle" -name '*.counts.json' | wc -l)" -eq 8
test "$(find "$bundle" -name '*.detail.json' | wc -l)" -eq 8
test "$(find "$bundle" -path '*/absolute/*.png' | wc -l)" -eq 64
test "$(find "$bundle" -path '*/absolute/*.csv' | wc -l)" -eq 64
(cd "$bundle" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
(cd "$bundle" && sha256sum --quiet -c SHA256SUMS)
```

每轮每层必须有七段总览 Perfetto 和带 pull/GEMM 明细的 Perfetto。dispatch 包含 count exchange 和 pull token。routed 的 wait token 终点、GEMM 起点都取设备记录的首个有效 CTA claim。细图的 tile begin 是消费者首次执行该 tile 的打点，保留两者间隔。新版 stream 要通过发布 → 唯一逐 tile 领取 → 同 CTA 同块同 N-tile 执行的完整覆盖校验，同时保留 L1 完成至 L2 发布的顺序检查。

## 4. 回执与推送

在 `docs/trace_handover_20261119/REPLY_BATCH_CLAIM.md` 给出以下实测结果。

1. 列四组执行 SHA、capture ID、两层 epoch、运行状态、GPU/软件版本、原始目录和校验文件。链接两层 Perfetto、八卡 counts、绝对时间图目录和 `rank_metrics.csv`。逐层标出 routed 最慢的 rank 及其 tokens、routed_rows、M 块数和活跃槽数；重排可能改变分布，列出对照组相应行。
2. 对 step 2 / layer 8 / rank 6 和每层实际最慢 rank，并排给出四组的首个非空槽完成、shared 结束、routed 入场、首个有效 claim、首个 tile、最后一个 CTA 开始首个 tile、最长内部空档及 routed 退出时刻。标记时间原点和单位。另列 layer 8 / rank 6 的 CTA 25 在四组各自最长的一段内部空档。用具体时间、batch、slot、M、N 写出 after/stream 最长一段空档前后的领取与执行事件。在两层各找一个多块热槽，列它每块的 L1 发布、首次 L1 领取、最后 L1 完成、L2 发布与首次 L2 领取，统计多少个满块在整槽结束前已开始 GEMM。
3. `first_wave_spread_us` 是参加计算的 CTA 各自第一个 tile 起点的极差。新版 `backlog_before_claim_cta_us` 表示逐 CTA 的相邻 tile 空档中，在下一 tile 获领前且全 rank 有已发布未领票的累计时间；`max_backlog_in_gap_us` 为单个此类空档内的最大重叠量。报告数值和对应区间，单独说明已领取到实际开工的间隔。发布与领取打点位于原子操作附近，时间分辨率为 1 µs。前三组缺少逐票标识，对应两列留空。旧 stream 的 queue empty 可能含竞争失败，新版 ready scan empty 表示一轮扫描未见候选。性能结论限于本次插桩窗口。若并行度退化、有票仍空转或首波仍长，照实报告。
4. 成功时在同步的主工作树提交完整包和回执。若步骤 1 有允许的编译修复，先将对应提交合入交付分支，回执写清原 SHA、修复 SHA、诊断与等价性。失败时只提交失败回执和相关日志，保留其余原始目录。

```bash
cd "$repo"
target=docs/trace_executor_20261119/batch_claim_results
test ! -e "$target"
cp -a "$bundle" "$target"
test -s docs/trace_handover_20261119/REPLY_BATCH_CLAIM.md
git add -- "$target" docs/trace_handover_20261119/REPLY_BATCH_CLAIM.md
git commit -m 'Report four H20 captures for per-block SM90 tile claims'
git push origin HEAD:main
```
