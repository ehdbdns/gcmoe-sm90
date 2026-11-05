# GC MoE EP8 — ECO 在 SM90 的完整移植（已完成）

> 起点：`gcmoe_exp/gcmoe_eplb_ep8_20261104`（已冻结，EPLB 放置 + host RR 分流）。
> 本轮把 SM100 的 **ECO 在线分流**（warm start / auction / accept / two-hop 关闭）
> 完整移植到 SM90 CuTe DSL，并在 vLLM 上跑通三臂。

## 实验结果

配置：8×H20（SM90），TP=8 / EP=8，DeepSeek-V4.1-Flash，100 步统计 + 100 步测量。
冗余：**每 rank 4 个**（`GC_MOE_EPLB_EXTRA_PER_RANK=4`，`physical=416`，52 slots/rank）。
ECO 权重：`wb=260, wa=0`。

### BS=32（8×4 seq/rank，32 gen tokens/step）

| 臂 | 放置 | 分流 | ms/step | IR |
|---|---|---|---:|---:|
| gcmoe + random + rr | random | host RR | 21.39 | 1.1366 |
| gcmoe + eplb + rr | EPLB | host RR | 20.98 | 1.0697 |
| **gcmoe + eplb + eco** | EPLB | **kernel plan** | **20.82** | **1.0551** |

三点结论：

1. **放置是最主要因素**：同样的 52 槽、同样的分发，random → EPLB 省 0.41 ms，IR 1.137 → 1.070。
2. **ECO 的分发比 RR 更快**：20.82 vs 20.98（−0.16 ms），且 IR 更好（1.0551 vs 1.0697）。
3. 之前记录的 21.99（+8.6%）是**测量缺陷**下的数字，已作废，原因见 `docs/RESULTS.md`：
   CUDA graph capture 被 host 侧 `.item()` 打断、重排后 `eco_ready` 令牌链死锁、
   时间轴行号串行。

### 架构（对齐 SM100 `eco_unit_kernel.cu`）

计数与广播**不在 dispatch 的关键路径上**：

```
dispatch:  wake plan ─→ push ─→ pull(单副本槽)      ← 关键路径
plan CTA:  静态表 ─→ count+broadcast ─→ 跨rank栅栏 ─→ D_e ─→ cold/warm ─→ auction
```

* `eco_counts` **双缓冲**（`gen = epoch & 1`，SM100 L124），不需要"上一轮读完"的令牌。
* 一个 plan CTA 用 256 线程并行计数，只需 CTA 内的 `sync_threads()`，**没有 grid 会合**。

逐段耗时（µs / MoE 层调用，8 rank 平均）：`plan_exec 35.4`（等唤醒 6.9 / 静态表 14.0 /
cold-warm 5.9 / auction 5.9 / 发布 2.7）、`push 8.6`、`pull_norep 69.9`、`pull_rep 39.6`。
**消费者饥饿 = 0**（8 rank × 256 层调用，副本数据总是先于消费者空转到达）。

正确性：`verify_eco_ep8.py` 在 8 卡上 **cos=1.00000000 / rel=0 / 逐位相同**、quota 守恒。

### batch 规模扫描（BS=32 / 64 / 128）

配置不变：每 rank 4 副本（`P=416`），`wb=260, wa=111`，100 步统计 + 100 步测量。

| BS | 臂 | ms/step | IR | 极差 | 块/步 |
|---:|---|---:|---:|---:|---:|
| **32** | gcmoe + random + rr | 21.46 | 1.1360 | 41786 | 15 |
| | gcmoe + eplb + rr | 20.97 | 1.1054 | 15538 | 17 |
| | **gcmoe + eplb + eco** | **20.41** | **1.0262** | **4971** | 15 |
| **64** | gcmoe + random + rr | 31.46 | 1.1399 | 84800 | 28 |
| | **gcmoe + eplb + rr** | **30.49** | **1.0302** | 15075 | 30 |
| | gcmoe + eplb + eco | 30.93 | 1.0439 | 14782 | 28 |
| **128** | gcmoe + random + rr | 49.93 | 1.1293 | 162848 | 39 |
| | **gcmoe + eplb + rr** | **47.26** | 1.0264 | 22891 | 41 |
| | gcmoe + eplb + eco | 47.41 | **1.0171** | **19943** | 34 |

**EPLB 放置的收益随 batch 放大**（random → eplb+rr）：

| BS | ms | IR |
|---:|---:|---|
| 32 | −2.31% | 1.1360 → 1.1054 |
| 64 | −3.08% | 1.1399 → **1.0302** |
| 128 | **−5.36%** | 1.1293 → 1.0264 |

**ECO 的分发在 BS=32 最优，大 batch 下与 rr 持平**：
BS=32 时 eco 比 rr 快 2.66% 且 IR 更好（1.0262 vs 1.1054）；
BS=128 时 eco 的 IR 最好（1.0171）但 ms 慢 0.3%。

逐段耗时随 batch 的变化（µs/层调用，8 rank 平均）：`pull_norep` 是绝对瓶颈且线性增长
（BS=32/64/128 → **69 / 115 / 187 µs**），而 `plan_exec` 稳定在 37–44 µs，
占比从 33% 降到 20%。ECO 的 plan 已经完全不是瓶颈。

复现：

```bash
./scripts/run_matrix.sh M          # 12 个实验串行，约 65 分钟
python3 scripts/agg_eco_stages.py runs/M_32_eco.json   # 时间轴
```

`GC_MOE_ECO_TRACE=1` 现在对**所有臂**生效（dispatch 侧的 plan_wake / push / pull
列相同），所以 RR 和 random 臂也能 dump 同一张表。

## 权威规格

`eco_ref/unit_ref.py` 是从 SM100 仓库逐字复制的 **bit-exact numpy 参考实现**。
kernel（role 5）必须与它逐位一致 —— 已完成，证据见 `docs/ECO_SM90_PORT.md` §2。

## 目录

```
source/       gc_moe（改动对象；所有新增/改动都带 [ECO-SM90] 注释）
integration/  vLLM 后端（EPLB 放置/重排 + GC_MOE_ECO 开关）
eco_ref/      unit_ref.py + plan_port.py + EcoMeta + 3 组 CPU 对拍
tests/        CPU 回归（上一轮的 EPLB 逻辑，仍全过）
scripts/
  run_arm_eco.sh       三臂启动脚本 (epbl, eco)
  gpu_plan_only.py     ★ 只跑 role 5，与 unit_ref 逐位对拍（~20 s，无模型）
  verify_eco_ep8.py    ★ 8 卡端到端正确性（随机权重，~30 s）
  eco_triage.py        握手/分片诊断（调试用）
  smoke_compile_eco.py 只编译 role cubin（秒级）
docs/         TASK_SPEC（原始施工规格备份）ECO_SM90_PORT / RESULTS / METRICS / REPRODUCE
runs/ logs/   三臂的原始 JSON 与日志
archive/      上一轮的 runs/logs（只读）
```

## 进度

| 步骤 | 状态 |
|---|---|
| runtime / layout / api / heuristics / jit 支持 role 5 | ✅ |
| EcoMeta 构建 + 无损往返 | ✅ |
| cold start（均分 + 负载感知取整 + jp） | ✅ 与 unit_ref 逐位一致 |
| warm start（congruent 反解 + δ 增减） | ✅ 同上 |
| one-hop auction（候选 / 报价 / 分组 / 排序） | ✅ 同上 |
| accept 前缀（两条 break + win2/drop 裁剪） | ✅ 同上 |
| dispatch：逻辑 id 路由 + 每个活跃副本 push | ✅ |
| pull：按 quota 切分 + passive 副本不拉 | ✅ |
| two-hop | ⬜ 按规格 EP8 不需要（SM100 在 ≤32 rank 也是 0 轮） |
| wb / wa 标定 | 🟨 当前实测用 `wb=260, wa=0`（`wa=111` 会让槽间更不均） |
| 计数/广播移入 plan + 双缓冲（对齐 SM100） | ✅ 关键路径上不再有 ECO 独有开销 |

## 快速验证（都不加载模型）

```bash
cd /home/admin/workspace/aop_lab/app_source/gcmoe_exp/gcmoe_eco-auction_20261105
P=/home/admin/workspace/aop_lab/app_source/gc_moe_rtp_ep8_20261031/env3/bin/python
for f in eco_ref/test_*.py tests/test_eplb_logic_cpu.py; do $P $f | tail -1; done

source scripts/env.sh
export GC_MOE_SRC=$PWD/source PYTHONPATH=$GC_MOE_SRC:$GC_SHIM:$PWD/integration
$GC_ENV/bin/torchrun --nproc_per_node 8 scripts/gpu_plan_only.py   # V4
$GC_ENV/bin/torchrun --nproc_per_node 8 scripts/verify_eco_ep8.py   # 端到端
```

## 三臂正式跑法

```bash
# BS=32 = 8 ranks x 4 seq/rank. For BS=64 use SEQS_PER_RANK=8, BS=128 use 16.
C="SEQS_PER_RANK=4 WARM_STEPS=100 MEASURE_STEPS=100"
# arm 1: random placement + host RR
env $C GC_MOE_SLOT_PLACEMENT=random GC_MOE_EPLB_EXTRA_PER_RANK=4 \
    ./scripts/run_arm_eco.sh 0 0 A2_random
# arm 2: EPLB placement + host RR
env $C GC_MOE_EPLB=1 GC_MOE_EPLB_EXTRA_PER_RANK=4 \
    ./scripts/run_arm_eco.sh 1 0 B2_rr
# arm 3: EPLB placement + kernel plan (add GC_MOE_ECO_TRACE=1 for the timeline)
env $C GC_MOE_EPLB=1 GC_MOE_EPLB_EXTRA_PER_RANK=4 GC_MOE_ECO=1 \
    GC_MOE_ECO_TRACE=1 GC_MOE_ECO_WB=260 GC_MOE_ECO_WA=0 \
    ./scripts/run_arm_eco.sh 1 1 C2_eco
```

`GC_MOE_EPLB_EXTRA_PER_RANK=n` sets redundant slots **per rank** directly
(`physical = experts + n*ranks`); it overrides the older fraction knob
`GC_MOE_EPLB_REDUNDANCY`. `GC_MOE_SLOT_PLACEMENT=random` scatters the same
multiset of experts over the same slots with a fixed seed -- that is the
placement control arm.

Dump the timeline: `python3 scripts/agg_eco_stages.py runs/C2_eco.json`.
