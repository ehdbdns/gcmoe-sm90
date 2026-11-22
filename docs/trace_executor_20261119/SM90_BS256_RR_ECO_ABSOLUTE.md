# SM90 · BS256 重排后 RR 与 ECO 的八卡 absolute trace

目标为解释主表中 rr 的 35.303 与 eco 的 35.600 ms/步。采正式窗口第 3 步（零基 step 2），取零基层 0、8、20、32、39，八个 rank。两臂分别为逐层 DeepInfer EPLB + RR、相同放置策略 + ECO fine cost（mode 2、slack 11469）。两臂开启 stream，关闭 SM-reuse。将任务视为带观测开销的诊断，主表性能依据仍为原无细粒度插桩的 1000 步计时。

## 准备

在原 SM90 主仓、原模型和 venv、同一节点八张空闲 H20 的预约内执行。其他用户占用时等待。按下列顺序在同一个 bash 会话中执行。诊断补丁基于 BS256 成功轮的完整 SHA，仅改采集驱动、启动参数和解析器。保留 20 GiB/rank KV、TP8/EP8、M64、6 个 dispatch SM、FULL_DECODE_ONLY、稳定 prefill/decode top-k 和原始数据集。先确认私有磁盘可用空间至少 350 GiB；若 /tmp 空间不足，把工作目录设在空间足够的本地磁盘。

```bash
set -euo pipefail
git pull --ff-only origin main
repo=$(git rev-parse --show-toplevel)
ref="$repo/docs/trace_executor_20261119/sm90_main24_stable_decode_bs256_results/delivery/bs256"
work=$(mktemp -d "${GC_MOE_CHECKPOINT_ROOT:-/tmp}/sm90-bs256-trace-XXXXXXXX")
base=956bb3fa16deb9472a1493c265e69fbba7ff0b55
(cd "$ref/.." && sha256sum --quiet -c SHA256SUMS)
git fetch origin sm90-bs256-rr-eco-absolute
sha=10bde89166a3f93f656bb77f2f7f86981582d68a
git worktree add --detach "$work/code" "$sha"
cd "$work/code"
test "$(git rev-parse HEAD^)" = "$base"
git diff --exit-code "$base" HEAD -- source integration patches config
source scripts/env.sh
py="$GC_ENV/bin/python"
export GC_MOE_EXEC_ROOT="$PWD" GC_MOE_MATRIX_SIZE=24 GC_MOE_MATRIX_BS=256
export GC_MOE_MATRIX_PREFIX_CHECK=1 GC_MOE_LONG_COST=1 GC_MOE_ORDER_EXTRA=1
export GC_MOE_ORDER_PREFILL=1 GC_MOE_STABLE_PREFILL_TOPK=1 GC_MOE_STABLE_DECODE_TOPK=1
export GC_MOE_STUDY_STEP=0 GC_MOE_SM90_STREAM_M_BLOCKS=1 GC_MOE_KV_BYTES=21474836480
export GC_CAPTURE_TOKENS=1 VLLM_USE_V2_MODEL_RUNNER=1 VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1
export GC_MOE_FIXED_DIR="$work/replay" GC_MOE_STRICT_STATE_DIR="$work/kv"
export DATASET=/mnt/nas2/nanjun.cp/MoE-Bench-Experiments/datasets/dsv4-flash_online_input_ids_26-10-08_1.jsonl
unset GC_MOE_FIXED_REUSE GC_MOE_FIRST_STEP_AUDIT GC_MOE_AA_CONTROL GC_MOE_COST_STUDY GC_MOE_PREFIX_DIAG GC_MOE_VANILLA_STATS_REPLAY GC_MOE_DECODE_STATS_AA GC_BS256_TRACE_DIR
mkdir "$work/replay" "$work/kv"
(unset GC_MOE_FIXED_DIR GC_MOE_STRICT_STATE_DIR GC_MOE_LONG_COST GC_MOE_MATRIX_BS GC_MOE_MATRIX_PREFIX_CHECK GC_MOE_STUDY_STEP; bash scripts/trace_layer/preflight_main20.sh "$work/preflight") > "$work/preflight.log" 2>&1
```

## 重建与采样

成功轮的 KV 二进制已清理。vanilla 仅作参考状态发布者，运行原始 1000 步统计和 1000 步正式窗口，随后逐 rank 比较两个边界的历史 KV manifest 和原生 token。rr 和 eco 各启动一次，在正式 step 2 内围住原来的那次 engine.step() 采集；继续走完剩余步骤以核对原生轨迹、实际分发、权重与放置。全程不用失败轮的 KV，也不用旧 200 步 trace 启动器。

```bash
for arm in vanilla rr eco; do
  export GC_MOE_MATRIX_ARM="$arm"
  mkdir "$work/$arm"
  if [[ $arm != vanilla ]]; then export GC_BS256_TRACE_DIR="$work/$arm/raw"; fi
  nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader > "$work/$arm/preexisting.csv"
  test ! -s "$work/$arm/preexisting.csv"
  nvidia-smi --query-gpu=index,uuid,name,pci.bus_id,driver_version,power.limit,clocks.max.sm,clocks.max.memory --format=csv,noheader > "$work/$arm/hardware.csv"
  diff -u "$ref/$arm/hardware.csv" "$work/$arm/hardware.csv"
  bash scripts/trace_layer/run_aligned_ep8.sh "$sha" --study-main20 2>&1 | tee "$work/$arm/runner.log"
  raw=$(sed -n 's/^Capture workspace: //p' "$work/$arm/runner.log")
  grep -Fxq exit_code=0 "$raw/status.txt"
  "$py" "$repo/scripts/trace_layer/check_bs256_trace.py" "$ref/$arm" "$raw" "$arm" | tee "$work/$arm/historical-check.txt"
  cmp "$ref/$arm/installed-layout.json" "$work/replay/$arm-layout.json"
  if [[ $arm == vanilla ]]; then
    for rank in {0..7}; do
      for phase in '' 'prefill/'; do
        cmp "$ref/checkpoint/${phase}rank$rank.json" "$work/kv/${phase}rank$rank.json"
      done
    done
    continue
  fi
  cmp "$ref/placement.json" "$work/replay/placement.json"
  dest="$work/$arm/perfetto"
  "$py" scripts/trace_layer/aligned_step.py "$GC_BS256_TRACE_DIR" "$dest" --layers 0 8 20 32 39 --dispatch-grid 24
  "$py" scripts/trace_layer/pull_gemm_detail.py "$GC_BS256_TRACE_DIR" "$dest" --require-streamed-blocks --require-batch-claims
  "$py" "$repo/scripts/trace_layer/check_bs256_trace.py" "$ref/$arm" "$raw" "$arm" "$dest"
done
```

## 回执与诊断

提交 docs/trace_handover_20261119/REPLY_SM90_BS256_RR_ECO_ABSOLUTE.md。结果放 docs/trace_executor_20261119/sm90_bs256_rr_eco_absolute_results/{rr,eco}/。为十个臂/层样本分别链接粗粒度 Perfetto、带 CTA 轨道的 *.detail.perfetto.json（absolute trace）与 *.counts.json。保留 JSON，无需 PNG；若单件超过 GitHub 限额，使用 gzip 无损压缩并注明。保留原始 Kineto 与事件行 SHA256 和现场路径，提交可下载的产物、硬件和软件版本、两份代码 SHA、校验日志及 SHA256SUMS。

逐层检查八卡 Kineto baseTimeNanoseconds、epoch 和 step/layer。用 counts.json 的 slots_owned 对照历史同臂 rank NPZ 的 counts[2*40+layer]，列出每 rank 的物理 routed rows、M 块、活跃槽、完整 MoE 跨度与最慢 rank。图中保留 count exchange、pull token、wait token、首个有效 CTA 起算的 routed GEMM、L1/L2 tile、L2 push、combine，以及 ECO 的 plan/replica wait。逐层比较最慢 rank 的 plan wait、首块可领到首 CTA 开工的间隔、计算空挡和最后一批 push；区间重叠时分开列。结合历史无细粒度插桩的同一步/层 clocks 判读，注明时间与并发受插桩影响。

历史哈希、物理槽计数、图或校验不符时记录首个不符的臂/步/层/rank、期望与实测，提交已有证据并标明复现状态。执行者只处理保持语义的编译问题并附 diff，分流、采样和校验由设计端维护。完成或失败后释放八卡，保存诊断和清单；核实归档后清理本任务新建的 KV、回放和临时 checkout，保留旧现场。推送回执和结果到 main。
