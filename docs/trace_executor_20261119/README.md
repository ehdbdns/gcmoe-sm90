# 八卡执行任务书 · EPLB 重排后单步采样

本文件是本次执行的唯一操作入口。使用首次有效 CTA 领取打点提交 `cffe4835270f0d0bcc048b502a0b5716eac910b1` 及包含本任务书的后续提交。依次执行步骤 1 至 5。上一次的任务书、脚本命令和回执是历史记录。此次验收禁止使用 `prof_phases.py`、`FINAL_V2` 或任何旧 run。

> 回执 `ce91bb1` 后本轮暂停重新采集。当前操作见 [原始 trace 诊断](../trace_handover_20261119/FOLLOW_UP.md)。

## 执行权限

- 执行者负责检查环境、编译、采集、运行现有解析器以及回传产物。代码修改仅限步骤 2 中由编译器诊断定位的最小兼容性修复，例如 CuTe DSL 不支持的表达式写法。修复前保留原始错误，修复后按步骤 2 重测。任何编译修复单独提交，在回执中列出前后 SHA、诊断位置和修复依据。
- 冻结实验配置、内核行为、打点位置、调度、数据定义、解析器、测试、验收阈值和层选择。禁止关闭功能绕过编译、切换 graph/eager、改 batch、换层、换模型或数据集、改时间线、调整或手填 JSON。环境路径可指向执行机上的同一个模型、数据集和工具链。若必须改变实验配置或源码语义才能继续，按步骤 5 回传失败。
- 除上述编译兼容性修复外，任何一步失败都停在该步，按步骤 5 回执。保留首次失败日志和原始数据，等下一版明确指令。禁止改用旧归档交付、跨 run 拼接、补估计值或提交一半结果作为成功。

## 1. 固定版本并检查执行机

在八卡执行机的本仓库根目录开启一个 Bash，后续命令在同一个会话中执行。确保工作区干净。同步 `main` 时只快进。核对实际分配给本任务的八张卡都是 H20，检查空闲情况，记录驱动和显存。发现占用时回报，保留现有进程。

```bash
set -eo pipefail
test -z "$(git status --porcelain)"
git fetch origin main
git merge --ff-only origin/main
git merge-base --is-ancestor cffe4835270f0d0bcc048b502a0b5716eac910b1 HEAD
git rev-parse HEAD

out=$(mktemp -d /tmp/gcmoe-aligned-XXXXXXXX)
echo "Capture workspace: $out"
git rev-parse HEAD > "$out/start.sha"
nvidia-smi --query-gpu=index,name,driver_version,memory.used,memory.total --format=csv   | tee "$out/gpu.log"

export MODEL=/home/admin/local_models/DeepSeek-V4.1-Flash
export DATASET=/mnt/nas2/nanjun.cp/MoE-Bench-Experiments/datasets/dsv4-flash_online_input_ids_26-10-08_1.jsonl
test -d "$MODEL"
test -r "$DATASET"
source config/experiment.env
source scripts/env.sh
test -x "$GC_ENV/bin/python"
export GC_MOE_SRC="$PWD/source"
export PYTHONPATH="$PWD/source:$PWD/patches:$PWD/integration"
export WORLD=8 GPU_UTIL=0.72 GC_MOE_NO_PREFETCH=1
export GC_MOE_EPLB=1 GC_MOE_EPLB_NO_REARRANGE=0 GC_MOE_SLOT_PLACEMENT=identity
export GC_MOE_ECOPLB_PER_LAYER=1 GC_MOE_ECO_PER_LAYER=1 GC_MOE_EPLB_EXTRA_PER_RANK=8
export GC_MOE_ECO=1 GC_MOE_ECO_WB=26 GC_MOE_ECO_WA=0 GC_MOE_ECO_COSTFN=0
export GC_MOE_ECO_MAX_COPIES=8 GC_MOE_MB_CLAIM=1 GC_MOE_SM90_PULL_TMA=1
export GC_MOE_SHARED_IN_GC=1 GC_MOE_SM90_SM_REUSE=0 GC_MOE_SM90_DISPATCH_SMS=6
export GC_MOE_CAPACITY=2048 GC_MOE_BOUND=2048 GC_MOE_BLOCK_M_FORCE=64 MAX_BATCHED=2048
export GC_MOE_PULL_TRACE=1 GC_MOE_ECO_TRACE=1 GC_MOE_FINE_TRACE=0
export GC_MOE_IR_STEP=0 GC_MEASURE_IR_STEP=0 GC_MEASURE_IR=0 GC_MEASURE_LOAD=0
export GC_MOE_LAYERS=40 GC_MEASURE_FLUSH=2 ENFORCE_EAGER=0 CUDAGRAPH_MODE=FULL_DECODE_ONLY
"$GC_ENV/bin/python" -c 'import torch, vllm; print(torch.__version__, torch.version.cuda, vllm.__version__); assert torch.cuda.device_count() == 8; assert all(torch.cuda.get_device_capability(i)[0] == 9 for i in range(8))'   | tee "$out/runtime.log"
```

## 2. 运行 CPU 检查和两套编译冒烟

先确认 `ALIGNED_STEP_CPU PASS`。随后编译常规路径和插桩路径。以退出码和日志里的 `compiled roles` 为准。常规路径应包含角色 1、2、5、3。插桩路径应包含角色 1、4、2、5、3。

```bash
"$GC_ENV/bin/python" tests/test_aligned_step_cpu.py 2>&1 | tee "$out/cpu.log"
GC_MOE_STEP_PROFILE=0 "$GC_ENV/bin/python" scripts/smoke_compile_eco.py   2>&1 | tee "$out/compile-base.log"
GC_MOE_STEP_PROFILE=1 SMOKE_ECO_WB=26 "$GC_ENV/bin/python" scripts/smoke_compile_eco.py   2>&1 | tee "$out/compile-aligned.log"
git rev-parse HEAD > "$out/execution.sha"
```

若遇到编译失败，先保留本次目录，再修复编译器指出的最小问题。新建日志目录重跑本节三个命令，确保常规路径和插桩路径都通过，记录实际执行 SHA。如果无法在维护原语义的前提下修通，直接回传。模型运行时若出现 JIT 编译错误，也只允许这一类修复。修复后重新执行本节，再用新目录从头采集。

## 3. 重新采集一次 BS256

预热权重后，执行下面唯一的采集命令。运行顺序为 prefill、200 步 warmup、逐层 EPLB 重排、2 个未采样 decode 步、1 个正式采样步。采样步从重排后零基编号，所以报告里的 `step` 应为 2。记录八个 worker 的 trace 导出结果。采集耗时用于排错。

```bash
bash scripts/warm_page_cache.sh "$MODEL" 2>&1 | tee "$out/warm.log"
"$GC_ENV/bin/python" -u scripts/real_decode_100_100.py   --label ALIGNED_EP8 --output "$out/result.json"   --seqs-per-rank 32 --prefill-lo 256 --prefill-hi 2048   --warm-steps 200 --measure-steps 1 --profile-aligned-step   --profile-out "$out/raw" 2>&1 | tee "$out/run.log"
test -s "$out/raw/capture.json"
for rank in {0..7}; do
  test -s "$out/raw/rank$rank.json"
done
```

检查日志里的逐层重排、八个 worker 导出成功和 256 条序列各生成一个 token。缺文件、采集失败或环境异常时进入失败回执。

## 4. 解析固定的两层并验收

只用步骤 3 同一个 `raw/` 下的 `capture.json` 和八份 torch trace。选择该步零基 MoE 层序号 8 和 20，以解析器退出码为第一道验收。

```bash
"$GC_ENV/bin/python" scripts/trace_layer/aligned_step.py "$out/raw" "$out/deliver"   --layers 8 20 --dispatch-grid 24 --require-pull-overlap   2>&1 | tee "$out/parse.log"
test "$(find "$out/deliver" -maxdepth 1 -name '*.perfetto.json' | wc -l)" -eq 2
test "$(find "$out/deliver" -maxdepth 1 -name '*.counts.json' | wc -l)" -eq 2
sha256sum "$out/raw"/capture.json "$out/raw"/rank*.json "$out/deliver"/*.json   | tee "$out/sha256.txt"
```

逐份核对以下结果。任何一项失败都按步骤 5 回报。

1. 八个 rank 共享 capture ID。采样前后每个 rank 的设备 epoch 恰好推进 40 次。每层报告与对应 Perfetto 相同 step、layer、epoch，原始窗口和时钟残差校验通过。
2. 每个 Perfetto 有 56 个区段，即八个 rank 各含 count exchange、pull token、shared L1+L2、ECO plan、wait token、GEMM、combine。count exchange 到最早的 pull CTA 进入 slot 循环为止，覆盖 routing、push 与 peer readiness rendezvous。count exchange 与 pull 合计等于 dispatch，wait token 与 GEMM 合计等于 routed。
3. GEMM 起点是首个 routed CTA 取得有效 L1 任务且输入已就绪时的设备打点。每个交付层至少有一个 rank 同时满足 `pull_after_first_claim_us > 0` 和 `drawn_pull_gemm_overlap_us > 0`。在 Perfetto 中核对这些 rank 的 pull 与 GEMM 区段确实重叠。
4. 每层有 0–7 共八条负载记录。逐 rank 回报 `tokens`、`routed_rows`、`m_blocks` 和 `block_m`。`tokens` 按实测输入复制倍数折算，`m_blocks` 逐槽对 `owned / block_m` 向上取整后求和。八个 rank 的 `routed_rows` 总和等于全部输入行数乘 top-k。沿用解析器给出的 plan 代价和归一化 L1 差值。
5. 如导出 PNG，直接使用同一份已验收数据。这是可选交付。

## 5. 回传到 GitHub

- **成功**。把两份 `*.perfetto.json` 和两份 `*.counts.json` 原样放入本目录的 `results/`，提交并推送。在本目录新建 `REPLY.md`，填写起始与执行 SHA、是否有编译修复、机器与运行配置、capture ID、step、两层及 epoch、八卡负载和重叠时长。附原始 `capture.json`、八份 torch trace 和日志的可访问位置及校验值。
- **失败**。仍提交并推送本目录的 `REPLY.md`。填写首个失败步骤、执行 SHA、完整命令、退出码、错误原文和日志位置。能获得 rank、epoch、trace 行号或残差时一并附上。保留已有的原始采样目录和校验值。失败或部分结果写明缺项，留在诊断目录中。
- 提交时只暂存本任务相关文件，编译兼容性修复单独提交。推送冲突时保留本地提交，回报冲突。

成功时可用以下命令复制和暂存四份产物。失败时跳过复制，只暂存回执。推送前检查暂存文件。

```bash
task=docs/trace_executor_20261119
mkdir -p "$task/results"
cp "$out/deliver"/*.perfetto.json "$out/deliver"/*.counts.json "$task/results/"
git add -- "$task/results/" "$task/REPLY.md"
git diff --cached --stat
git commit -m "Report aligned EP8 profile results"
git push origin HEAD:refs/heads/main
```

回执同时适用于成功和失败。回复提交到本目录后，设计端会核对，并决定是否需要新的步骤。
