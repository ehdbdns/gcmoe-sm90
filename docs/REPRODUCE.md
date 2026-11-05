# 复现

## 0. 环境

```bash
# NAS（重启后失效）
sudo mount -t nfs -o vers=4,minorversion=0,noresvport \
  YOUR-NAS-HOST:/ /DATA/
ls /DATA/models/DeepSeek-V4.1-Flash

cd <this-repo>
source scripts/env.sh
```

## 1. CPU 对拍（秒级，不占 GPU）

```bash
P=<env>/env3/bin/python
for f in eco_ref/test_v4_plan.py eco_ref/test_plan_port.py \
         eco_ref/test_kernel_logic.py eco_ref/test_meta_pipeline.py \
         eco_ref/test_kernel_full.py tests/test_eplb_logic_cpu.py; do
  printf '%-38s ' $f; $P $f | tail -1
done
```
期望：`V4 ... OK` / `PLAN_PORT PASS` / `KERNEL_LOGIC PASS` /
`META_PIPELINE PASS` / `total failures 0` / `CPU_TEST_FINAL PASS`。

`test_kernel_full.py` 是 kernel 控制流的 CPU 仿真（cold / warm / auction /
accept / quota_start），与 `unit_ref.UnitPlanner` 逐位对拍，含 EP8 全尺寸
（P=480, E=384）与随机小规模 fuzz。

## 2. GPU：只跑 plan（~20 s，不加载模型）

```bash
export GC_MOE_SRC=$PWD/source
export PYTHONPATH=$GC_MOE_SRC:$GC_SHIM:$PWD/integration
$GC_ENV/bin/torchrun --nproc_per_node 8 scripts/gpu_plan_only.py
```
期望：`[call 0] warm=False equal=True ...`、`[call 1] warm=True equal=True ...`、
`GPU_PLAN_V4 OK`。`PLAN_CALLS`（默认 2）、`PLAN_SEED`、`GC_MOE_ECO_WB/WA/ROUNDS/COLD_ROUNDS`
可调。

## 3. GPU：端到端正确性（~30 s，随机权重，8 卡）

```bash
$GC_ENV/bin/torchrun --nproc_per_node 8 scripts/verify_eco_ep8.py
```
期望：`cos=1.00000000 rel=0.000e+00 conserved=True quota_start_contiguous=True
plan_identical=True hot_copies_used=3/3 OK` 和 `ECO_EP8 ALL_RANKS_OK`。

## 4. vLLM 三臂（每次约 15–20 分钟，主要是加载权重）

```bash
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 $p; done
sleep 5

SEQS_PER_RANK=4 WARM_STEPS=100 MEASURE_STEPS=100 ./scripts/run_arm_eco.sh 0 0 vanilla_100
SEQS_PER_RANK=4 WARM_STEPS=100 MEASURE_STEPS=100 GC_MOE_EPLB_REDUNDANCY=0.25 \
    ./scripts/run_arm_eco.sh 1 0 eplb_rr_100
SEQS_PER_RANK=4 WARM_STEPS=100 MEASURE_STEPS=100 GC_MOE_EPLB_REDUNDANCY=0.25 \
    ./scripts/run_arm_eco.sh 1 1 eplb_eco_100
```

> 用 `setsid nohup ... &` 启动，避免会话断开带走进程。

成功标志：
```
[prefill] 32 seqs, N engine steps, X s (not timed)
[warm] 100 decode steps: X ms/step (device-synced window)
[measure] X ms/step (device-synced window, 100 steps, ...) running=32..32 gen_tokens/step=32..32
[ir] IR=... max=... mean=... min=... per_rank=[...]
```
ECO 臂还应有 `[eco] meta installed: words=6648 logical=384 physical=480 ranks=8`。

## 5. 指标含义

| 指标 | 来源 | 含义 |
|---|---|---|
| `ms_per_step_median` | 100 步 device-synced 窗口 | 时延 |
| `ir` | `expert_count`（每 slot 实际拉到的 token），按 rank 汇总 | `max_rank / mean_rank` |
| `load_detail.ir_window` | 每层每 slot 负载，层平均 | slot 级 IR（更敏感） |
| `load_detail.m_blocks_median` | `header` 计数 + block_m | 每层 M 块数 |
| `load_detail.active_experts_median` | `header[7]` | 每层激活槽数 |
| `tokens_per_rank` | 生成 token / rank | 必须是 32 |

## 常见故障

| 现象 | 原因 | 处理 |
|---|---|---|
| ECO 臂 8 卡 100% 不退出 | routed grid 占满 compute 分区，plan 永远排不上 | 确认 heuristics 里 `routed_grid = compute_sms - 1` |
| `[eco] meta installed` 没打印 | `GC_MOE_ECO` 没传给 worker | 用 `scripts/run_arm_eco.sh`，不要手抄 env |
| `eco_dbg` 之类 `AttributeError` | 用了 triage 专用代码 | 不要设 `GC_MOE_ECO_DEBUG`（该调试层已删） |
| `DeepGEMM backend is unavailable` | `$GC_SHIM` 进了 PYTHONPATH 抢 vendored 包 | `run_arm_eco.sh` 已处理 |
| 免费显存不足 | 上次进程残留 | 清 GPU 再起 |
