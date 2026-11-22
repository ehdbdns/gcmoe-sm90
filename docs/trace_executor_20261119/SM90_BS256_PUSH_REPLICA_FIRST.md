# SM90 · BS256 单副本直推与副本拉取的竞争

目标是让 single_push 在相同输入下胜过当前 baseline。上轮 b4c0a9f 的 baseline 为 35.756925 ms/步，旧 single_push 为 37.233828 ms/步。原始 999 个非采样步的设备时钟支持这个差值；两臂各自只有一次启动。旧版的 sender 在每个源 rank 上有 8 个，并发运行。

本轮只测原 baseline 和新版 single_push。新版让每个 dispatch CTA 在自己的 grid-stride 槽序列中先办理有副本的专家，再办理单副本槽的最终计数。单副本块仍由独立 CTA 到块即发布。源端扫描、逐行 TMA、fence、远端原子计数、拍卖和模型输入均沿用旧实现。对照点是 step 2、L20、rank 0、CTA 21。旧版槽 45 的 192 行配额在 121 µs 已可用，首次 issue 在 371 µs；其中 229 µs 与同一 CTA 处理槽 21 重叠。此次检验这个具体阻塞对完整 MoE 的影响。

## 在原八卡预约内执行

仅用原八张 H20。核对预约、UUID、卡数及空闲状态；遇占用等待。复用 b4c0a9f 回执中已验证的 HCA 设置、固定 KV 与回放目录。如原目录已迁移，将 SM90_RETAINED 设置为保留了同一 KV 和 replay 的实际绝对路径。仅允许修复不改变语义的编译错误并附 diff；两臂使用同一 SHA。无需重跑历史 single_push。

```bash
set -euo pipefail
git fetch origin main
book=docs/trace_executor_20261119/SM90_BS256_PUSH_REPLICA_FIRST.md
rev=$(git log -1 --format=%H origin/main -- "$book")
area=$(mktemp -d /tmp/sm90-replica-first-XXXXXXXX)
git worktree add --detach "$area/code" "$rev"
cd "$area/code"
archive=docs/trace_executor_20261119/sm90_bs256_async_return_fix_results
fixture=docs/trace_executor_20261119/sm90_main24_stable_decode_bs256_results/delivery/bs256
if [[ $(git config --bool core.sparseCheckout || true) == true ]]; then
  git sparse-checkout add "$archive" "$fixture" scripts source integration patches
fi
until busy=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader) && [[ -z "$busy" ]]; do sleep 30; done
(
  source scripts/env.sh
  unset GC_MOE_SM90_SPLIT_EXCHANGE GC_MOE_SM90_SOURCE_INDEX_PULL GC_MOE_SM90_PLAN_DISPATCH
  export PYTHONPATH="$PWD/source:$PWD/patches:$PWD/integration" GC_MOE_SRC="$PWD/source"
  export GC_MOE_ECO=1 GC_MOE_ECO_MAX_COPIES=8 GC_MOE_MB_CLAIM=1 GC_MOE_BLOCK_M_FORCE=64
  export GC_MOE_SM90_SINGLE_PUSH=1 GC_MOE_SM90_STREAM_M_BLOCKS=1 GC_MOE_SM90_DISPATCH_SMS=6 GC_MOE_SM90_PULL_TMA=1
  export GC_MOE_FIXED_DIR="$area/synthetic-fixed" TL_PUSH_PAYLOAD=1
  export TL_CAP=128 TL_TOKENS=17 TL_HOT_BLOCKS=1 TL_CHECK_PUSH_RETURN=1
  timeout 600 "$GC_ENV/bin/torchrun" --standalone --nproc-per-node=8 scripts/verify_eco_capture.py
) 2>&1 | tee "$area/return-smoke.log"
for rows in 135 0 64 65; do grep -q "PUSH_PAYLOAD_OK hot_rows=$rows replays=3" "$area/return-smoke.log"; done
grep -q 'ECO_CAPTURE OK' "$area/return-smoke.log"
bash scripts/trace_layer/run_bs256_eco_exchange.sh single-push 2>&1 | tee "$area/pair.log"
work=$(sed -n 's/^Trial workspace: //p' "$area/pair.log")
test -f "$work/moe.csv"
python3 scripts/trace_layer/single_push_stalls.py "$archive" "$work" "$area/diagnosis"
```

## 归因和验收

1. 每臂 1000 步统计，重排后 1000 步测量。核对两窗口八 rank 的原生 token、固定路由与 gate、placement、逐槽 counts、拍卖、硬件和 launch 配置。按原 runner 检查推送行数和八个 sender CTA。正确性失败时交首错和已完成产物。
2. 主表列同轮 baseline、新 single_push 的 graph→first、first→last、last→combine 和完整 MoE ms/步。每步按层选择设备跨度最大的 rank，再累加 40 层；去掉采样步后取 999 步均值。变化率为 100 × (新 / 同轮 baseline − 1)，负值才表示本轮提速。另报相对旧 single_push、历史 baseline 的变化及新旧 baseline 漂移。区分本次实测与跨启动可复现性。
3. 用 replica_stalls.csv 检查五层八 rank 全部非空副本槽。按层报告 quota_to_issue_us 的中位数、P95、最大值，并列 L20/rank0/slot45 旧、新和同轮 baseline 的原始时间。新版有副本槽的 prior_single_slots 应为零。prior_single_after_quota_us 只表示与同一 CTA 之前的单副本区间重叠；多个槽、CTA 的重叠时间不可相加解释整层延迟。对仍有长等待的槽检查对应 CTA 是否同时是 sender 或 publisher，沿 detail trace 对齐发送结束、发布、首次领号和最后一次 routed 写回。
4. 用 singleton_senders.csv 分 rank/CTA 统计行数、开始/结束和重叠；同一负载对照旧、新发送窗口。另报单副本满块中 finished_sources < 8 的比例和尾块数。对尚未解释的耗时分清候选原因与实测归因：现有记录把路由扫描、TMA、fence 和原子更新包含在发送窗口内，不能分别给它们估算百分比。如果副本已及时拉取而完整 MoE 仍慢，列出差值最大的阶段、最慢 rank/CTA 及具体时间线证据，供下一版据此优化。
5. 在 docs/trace_handover_20261119/REPLY_SM90_BS256_PUSH_REPLICA_FIRST.md 回执。归档至 docs/trace_executor_20261119/sm90_bs256_push_replica_first_results/，含两 CSV、moe.csv、两臂原始 rank NPZ、正确性日志、step 2 五层 Perfetto、absolute detail、各 rank token/M 块 counts 及版本。附两臂 L20 的 GitHub blob 链接。回执推 main 后清理临时 checkout，确认八卡释放；保留历史 KV、replay 和本轮原始证据。
