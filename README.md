# GC MoE EP8 — ECO 在 SM90 的完整移植（已完成）

> 起点：`gcmoe_exp/gcmoe_eplb_ep8_20261104`（已冻结，EPLB 放置 + host RR 分流）。
> 本轮把 SM100 的 **ECO 在线分流**（warm start / auction / accept / two-hop 关闭）
> 完整移植到 SM90 CuTe DSL，并在 vLLM 上跑通三臂。

## 实验结果

配置：8×H20（SM90），TP=8 / EP=8，DeepSeek-V4.1-Flash，100 步统计 + 100 步测量。
冗余：**每 rank 8 个**（`GC_MOE_EPLB_EXTRA_PER_RANK=8`，`physical=448`，56 slots/rank）。
ECO 权重：`wb=260, wa=111`。三臂全部走 TMA 拉 token。

完整表格见 §「最终结果：四档 batch × 三臂」和 `docs/RESULTS_TABLE.md`；
冗余度对照见 `docs/REDUNDANCY_COMPARE.md`。

### 架构（对齐 SM100 `eco_unit_kernel.cu`）

计数与广播**不在 dispatch 的关键路径上**：

```
dispatch:  wake plan ─→ push ─→ pull(单副本槽)      ← 关键路径
plan CTA:  静态表 ─→ count+broadcast ─→ 跨rank栅栏 ─→ D_e ─→ cold/warm ─→ auction
```

* `eco_counts` **双缓冲**（`gen = epoch & 1`，SM100 L124），不需要"上一轮读完"的令牌。
* plan CTA 用 **512 线程**、按 warp 分工（见下节），计数**没有 grid 会合**。

逐段耗时（BS=256，µs / MoE 层调用，8 rank 平均）：`plan_exec 40.1`（等唤醒 6.4 /
静态表 14.2 / cold-warm 7.1 / auction 8.2 / 发布 4.3）、`push 9.7`、
`pull_norep 124.3`、`pull_rep 142.0`。
**消费者饥饿 = 0**（8 rank × 256 层调用，副本数据总是先于消费者空转到达）。

正确性：`verify_eco_ep8.py` 在 8 卡上 **cos=1.00000000 / rel=0 / 逐位相同**、quota 守恒。

### warp 并行版（对齐 SM100）

plan CTA 提到 **512 线程**，照 `eco_unit_kernel.cu` 分工：

```
warp 0      -> 静态表           (bar.sync 1, 512 汇合)
warp 1..15  -> counts + 广播     (bar.sync 2, 480)
```

`plan_exec` 因此在 TMA 之前实测从 37.6 降到 34.3 µs；TMA 之后它是 40.1 µs，
但 pull 已经从 678 µs 降到 266 µs（见下节），plan 早已不是关键路径。

### TMA token 拉取（上游 SM90 的遗留瓶颈）

上游 `Bruce-Lee-LY/gc-moe` 的 SM90 里，**token 拉取仍是 16 B 同步拷贝**
（`copy_global_16`），而 SM100 用的是 TMA（`bulk_g2s_1d`）。权重两边都是 TMA ——
也就是说 SM90 移植时把数据量最大的 token 路径漏掉了。

实测该路径只有 **4.4 GB/s**。改成 SM100 的 TMA 结构后：

| 臂 | TMA 前 | TMA 后 | 变化 |
|---|---:|---:|---:|
| random+rr | 81.76 | **71.33** | −12.8% |
| eplb+rr | 73.28 | **66.56** | −9.2% |
| eplb+eco | 75.55 | **66.47** | −12.0% |

`pull_norep` 305 → 124 µs（−59%），`pull_rep` 373 → 142 µs（−62%），
带宽 4.4 → **10.7 GB/s**。

三个死锁级约束（都已解决）：dispatch 的 smem 必须 ≤ 58368 B 才能 4 CTA/SM 驻留
（双缓冲 84864 B 会导致 grid 会合永远等不齐）；`mbarrier` 的 phase 必须存在
shared memory（staged 循环里 Python 变量每次迭代都会重置）；`mbarrier_init_fence()`
不可省。

### 全部 rank 级 trace

`runs/rank_traces.json`（3.3 MB）：四档 batch × 三臂，每 rank 256 行 × 21 个槽位。

### 最终结果：四档 batch × 三臂（TMA 全开，extra=8，P=448）

所有三臂都用 TMA 拉 token（`GC_MOE_SM90_PULL_TMA=1`，默认值）。
`wb=260, wa=111`，100 步统计 + 100 步测量。

| BS | 臂 | ms/step | IR | M块数 | 激活专家 |
|---:|---|---:|---:|---:|---:|
| **32** | random+rr | 20.76 | 1.1373 | 14 | 14 |
| | eplb+rr | 20.48 | 1.0516 | 15 | 15 |
| | **eplb+eco** | **19.15** | **1.0252** | 19 | 19 |
| **64** | random+rr | 29.52 | 1.1362 | 30 | 30 |
| | eplb+rr | 28.63 | **1.0357** | 25 | 25 |
| | **eplb+eco** | **27.24** | 1.0441 | 24 | 24 |
| **128** | random+rr | 44.74 | 1.1359 | 41 | 41 |
| | eplb+rr | 43.26 | **1.0348** | 40 | 40 |
| | **eplb+eco** | **40.88** | 1.0435 | 33 | 33 |
| **256** | random+rr | 70.96 | 1.1318 | 49 | 49 |
| | eplb+rr | 67.40 | 1.0332 | 50 | 50 |
| | **eplb+eco** | **66.00** | **1.0282** | 42 | 42 |

相对加速：

| BS | rr vs random | eco vs rr | eco vs random |
|---:|---:|---:|---:|
| 32 | −1.32% | −6.51% | **−7.75%** |
| 64 | −3.03% | −4.85% | **−7.73%** |
| 128 | −3.30% | −5.51% | **−8.63%** |
| 256 | −5.02% | −2.09% | **−7.00%** |

**ECO 在四档全部最快**，相对 vanilla 级基线（random+rr）省 7.0–8.6%。

ECO 的 M 块数在大 batch 下明显更少（BS=128: 33 vs 40；BS=256: 42 vs 50）。
但**块数不是全部原因**：BS=32 时 ECO 的块数反而更多（19 vs 14）却仍然最快，
BS=128/256 与 rr 的块数也不是一一对应 —— 真正的驱动是槽上的 token 分布
（进而影响 GEMM 的 tile 形状），详见 `docs/REDUNDANCY_COMPARE.md` 的更正说明。

### 冗余度扫描（BS=256，eplb+rr，TMA）

| extra/rank | P | slots | ms/step | IR |
|---:|---:|---:|---:|---:|
| 4 | 416 | 52 | 66.95 | **1.0306** |
| **8** | **448** | **56** | **66.56** | 1.0344 |
| 16 | 512 | 64 | 66.14 | 1.0420 |
| 32 | 640 | 80 | 68.46 | **1.0269** |

IR 在四组之间只差 1.4%（1.027–1.042），而各组的最忙 rank token 数都在
775k–786k 之间波动 1.4% —— **这个差异是噪声，冗余度对 rank 级 IR 没有系统性
影响**（它优化的是槽级分布，不是 rank 级总量）。选 8 是因为 ms 最优且槽数适中。

### TMA A/B 验证（同一份代码，只切编译期开关）

| 配置 | ms/step | `pull_rep` |
|---|---:|---:|
| `GC_MOE_SM90_PULL_TMA=0` | 71.08 | 340.6 µs |
| `GC_MOE_SM90_PULL_TMA=1` | 66.56 | 143.8 µs |

三臂的 `pull_rep` 实测分别为 random 147.8 / rr 138.1 / eco 143.2 µs，
非 TMA 对照 340.6 µs ⇒ **确认三臂全部走 TMA**。

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
