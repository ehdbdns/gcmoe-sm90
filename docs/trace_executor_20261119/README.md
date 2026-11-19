# 八卡执行任务书 · 逐层 EPLB 重排后的 torch profile

执行者请在开始时记录本目录任务书所在的 Git SHA。请在本目录写入 `REPLY.md`，记录执行 SHA、机器配置、实际参数、验收结果、失败日志或产物位置。验收通过后，提交并推送两个选定层的 Perfetto 和计数报告。原始 torch trace、运行日志和 `capture.json` 放在同一次运行的目录中，在回复里提供可访问位置及 SHA-256。

## 目标

用八张 H20 在 DeepSeek-V4.1-Flash 的一个纯 decode 步中采样。该步在真实逐层 EPLB 重排后执行。先取零基 MoE 调用序号 8 和 20。每层交付一份有八个 rank、每 rank 七个区段的 Perfetto JSON 和一份对应的计数 JSON。区段是 count exchange、pull token、shared L1+L2、ECO plan、wait token、GEMM、combine。

本任务用 BS32 做首次验证。保持逐层 EPLB、ECO、MB-CLAIM、TMA 和 fused shared，关闭 SM-reuse。单步 profiler 捕获用于解释事件时序，不用于估计稳态吞吐。

本地已验证解析器的 CPU 回归、CLI 产物和 PyTorch profiler 写入 capture ID。本地没有 NVIDIA 设备和 `libcuda.so.1`，SM90 编译及 GPU 采样在本任务中由八卡执行者验收。

## 八卡命令

在配好 CUDA/vLLM 的执行机上，从最新 Git checkout 根目录执行。先核对八张 GPU 空闲、CUDA 工具链版本和本机模型、数据集、HCA 路径。请选择新的输出目录，每次重跑使用新的目录。

```bash
set -eo pipefail
git rev-parse HEAD
nvidia-smi --query-gpu=index,name,memory.used --format=csv

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
export GC_MOE_CAPACITY=2048 GC_MOE_BOUND=2048 GC_MOE_BLOCK_M_FORCE=64
export GC_MOE_PULL_TRACE=1 GC_MOE_ECO_TRACE=1 GC_MOE_FINE_TRACE=0
export GC_MOE_IR_STEP=0 GC_MEASURE_IR_STEP=0 GC_MEASURE_IR=0 GC_MEASURE_LOAD=0
export GC_MOE_LAYERS=40 GC_MEASURE_FLUSH=2 ENFORCE_EAGER=0 CUDAGRAPH_MODE=FULL_DECODE_ONLY

"$GC_ENV/bin/python" tests/test_aligned_step_cpu.py
GC_MOE_STEP_PROFILE=1 "$GC_ENV/bin/python" scripts/smoke_compile_eco.py
bash scripts/warm_page_cache.sh "$MODEL"
out=$(mktemp -d /tmp/gcmoe-aligned-XXXXXXXX)
echo "$out"
"$GC_ENV/bin/python" -u scripts/real_decode_100_100.py \
  --label ALIGNED_EP8 --output "$out/result.json" \
  --seqs-per-rank 4 --prefill-lo 256 --prefill-hi 2048 \
  --warm-steps 200 --measure-steps 1 --profile-aligned-step \
  --profile-out "$out/raw" 2>&1 | tee "$out/run.log"
"$GC_ENV/bin/python" scripts/trace_layer/aligned_step.py "$out/raw" "$out/deliver" \
  --layers 8 20 --dispatch-grid 24
sha256sum "$out/raw"/capture.json "$out/raw"/rank*.json "$out/deliver"/*.json
```

如 graph 模式下 torch profiler 缺少 kernel 事件，保留首轮诊断，再用新目录、`ENFORCE_EAGER=1` 和 `CUDAGRAPH_MODE=NONE` 重跑并标注执行模式。重跑时整套 trace 和计数必须来自同一轮。如编译、重排或对齐失败，回复首个失败的步骤和报错，以及触发报错的 SHA、rank、行号、epoch 或残差。请保留原始数据及验收判据。

## 对齐和验收

- 查看重排日志，确认八个 worker 都完成真实的逐层重排。采样前后设备同步，八个 rank 都应从同一个起始 epoch 前进恰好 40 次。采样的 `step` 从重排完成后零基计数，命令中的 `step=2` 表示先走两个未采样步。报告中的 `layer` 是该步内零基 MoE 执行序号。
- 每个 worker 的 torch trace 和设备快照共享 capture ID。每个调用在 torch trace 里由同一时间窗内的完整 kernel 集合确定，并与该步内有标签的设备 epoch 对应。用 dispatch 到 routed、dispatch 到 combine 的独立时间差以及 dispatch 尾部打点复核这一对应。报告保留原始 trace 行号、时长、设备打点和残差。
- count exchange 覆盖 routing、push 和 peer readiness rendezvous，到最早的 pull CTA 进入 slot 循环。pull 循环仍有逐槽 count 读取，保留在 pull token 区段。GEMM 始于 math CTA 拿到第一个有效 tile 并进入计算路径，wait token 从 routed kernel 起点到这个边界。两个分段各自回加等于其 torch kernel 总时长。
- `tokens` 按实测输入复制倍数折算。每 rank 的 `routed_rows` 直接取该 epoch 的 per-slot owned 之和，`m_blocks` 是实际执行的 `sum(ceil(owned / block_m))`，不乘 N-tile 数。八个 rank 的 routed rows 总和应等于采样输入 token 数之和乘 top-k。
- `plan_cost` 是内核最终预测的代价分，单位为 token 等价成本。报告保留各 rank 的实测路由 kernel 时长、dispatch 到 routed 结束的时长，以及预测与后者的归一化 L1 差。请在回复中说明该步的 plan 预测是否跟得上 rank 间差异。
- 用 Perfetto 打开两份 JSON，核对 dispatch 和 pull 的连续性、shared 与 dispatch 的启动差、wait token 和 GEMM 的连续性，以及 GEMM 与 pull 的实际重叠。把两份 JSON 和计数报告提交至本目录的 `results/`。如果导出 PNG，使用同一批区段数据。
