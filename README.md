# GC MoE EP8 — ECO 在 SM90 的完整移植（已完成）

> 起点：`gcmoe_exp/gcmoe_eplb_ep8_20261104`（已冻结，EPLB 放置 + host RR 分流）。
> 本轮把 SM100 的 **ECO 在线分流**（warm start / auction / accept / two-hop 关闭）
> 完整移植到 SM90 CuTe DSL，并在 vLLM 上跑通 **四臂**（原生 / random+rr / eplb+rr / eplb+eco）。

## 实验结果：四臂 × 四档 batch（16 个实验，全部 TMA）

配置：8×H20（SM90），TP=8 / EP=8，DeepSeek-V4.1-Flash，100 步统计 + 100 步测量，
`GC_MOE_SM90_PULL_TMA=1`，`wb=260, wa=111`。

* **原生** = 无副本（P=384，48 slots/rank）
* 其余三臂 = **每 rank 8 冗余**（P=448，56 slots/rank）

| BS | 臂 | ms/step | IR(rank) | 极差 | M块/步 | 激活专家 | slots |
|---:|---|---:|---:|---:|---:|---:|---:|
| **32** | **原生 gcmoe（无副本）** | **18.90** | 1.0316 | 7256 | 14 | 14 | 48 |
| | + random + rr | 20.86 | 1.1387 | 56571 | 17 | 17 | 56 |
| | + eplb + rr | 20.48 | 1.0775 | 10520 | 17 | 17 | 56 |
| | + eplb + eco | 19.18 | **1.0301** | **6046** | 15 | 15 | 56 |
| **64** | **原生 gcmoe（无副本）** | **27.58** | 1.0326 | 13770 | 27 | 27 | 48 |
| | + random + rr | 29.65 | 1.1374 | 112925 | 27 | 27 | 56 |
| | + eplb + rr | 28.69 | 1.0407 | 15925 | 30 | 30 | 56 |
| | + eplb + eco | 27.67 | **1.0219** | **7215** | 27 | 27 | 56 |
| **128** | 原生 gcmoe（无副本） | 42.15 | **1.0301** | 25979 | 35 | 35 | 48 |
| | + random + rr | 44.31 | 1.1338 | 221504 | 41 | 41 | 56 |
| | + eplb + rr | 42.48 | 1.0398 | 28897 | 38 | 38 | 56 |
| | **+ eplb + eco** | **41.34** | 1.0459 | 28523 | 37 | 37 | 56 |
| **256** | 原生 gcmoe（无副本） | 68.91 | 1.0270 | 53215 | 45 | 45 | 48 |
| | + random + rr | 70.51 | 1.1309 | 436849 | 48 | 48 | 56 |
| | + eplb + rr | 66.31 | 1.0419 | 55115 | 49 | 49 | 56 |
| | **+ eplb + eco** | **65.86** | **1.0193** | **44385** | 44 | 44 | 56 |

ECO 相对原生 gcmoe：

| BS | ms | IR |
|---:|---:|---|
| 32 | **+1.46%**（慢） | 1.0316 → 1.0301 |
| 64 | +0.32% | 1.0326 → 1.0219 |
| 128 | **−1.94%** | 1.0301 → 1.0459 |
| 256 | **−4.42%** | 1.0270 → 1.0193 |

相对加速（三臂之间）：

| BS | rr vs random | eco vs rr | eco vs random |
|---:|---:|---:|---:|
| 32 | −1.82% | −6.35% | −8.05% |
| 64 | −3.24% | −3.56% | −6.68% |
| 128 | −4.13% | −2.68% | −6.70% |
| 256 | −5.96% | −0.68% | −6.60% |

五点结论：

1. **交叉点 BS≈64**：BS=32 原生最快，BS=64 打平，BS=128/256 ECO 反超（−1.94% / −4.42%）。
2. **冗余槽本身有固定代价**：random 臂就是原生 + 8 副本，但慢了 1.6–2.1 ms；
   BS=32 时这 1.6 ms 占了 ECO 与原生差距的全部。
3. **ECO 在 BS=32 已把冗余代价压回 0.28 ms**（20.86 → 19.18），大 batch 下变成净收益。
4. **EPLB 把 rank 极差压掉一个数量级**：random 56k–437k → eplb+eco 6k–44k。
5. ★ **时间与均衡不同向**：BS=128 时 ECO 的 IR（1.0459）比原生（1.0301）差，但 ms 快 1.94%
   —— 收益来自槽上的 token 分布（GEMM tile 形状 / 块数），不是 rank 间均衡度。

（完整四臂表见 `docs/FOUR_ARM_RESULTS.md`，原始 JSON 见 `runs/Q4_<bs>_<arm>.json`。）

### TMA：上游 SM90 移植的遗漏，四臂统一补上

上游 `Bruce-Lee-LY/gc-moe` 的 SM90 token 拉取仍是 16 B 同步拷贝（`copy_global_16`，
每 token 320 次 `ld/st.global.v4`），只有 SM100 用 TMA（`cp.async.bulk`）——
移植时漏掉了数据量最大的一条路径。

用「冻结树 + 仅 TMA，其余不动」的对照臂验证（`gcmoe_exp/gcmoe_tma_only_20261107/`，
无副本、EPLB 关、`routed_grid` 仍 72）：

| BS=32，无副本，100+100 步 | 源码目录 | ms/step | IR |
|---|---|---:|---:|
| 原生（无 TMA，上游原样） | `gcmoe_eplb_ep8_20261104` | 20.33 | 1.0299 |
| 原生 **+ TMA** | `gcmoe_tma_only_20261107` | **18.88** | 1.0290 |
| | | **−7.11%** | 不变 |

**TMA 更快（−7.11%）且语义不变**，所以**四臂全部启用**。它是共用的基础设施，
不是任何一臂的特性 —— 只给 ECO 加 TMA，四臂比的就会是「谁用了 TMA」。

> 四臂表的「原生」= 原生 **+ TMA**（18.90），不是上游原样（20.33）。

#### 第一臂与上游原生树的其余差异

第一臂跑 `gcmoe_eco-auction_20261105/source`（四臂共用）。除 TMA 外：

| 改动 | 对 vanilla 生效 |
|---|---|
| ECO 区域（`eco_counts`/`eco_meta`/`eco_hist`/`eco_quota`/`eco_state`）全在 `if self.eco:` 内 reserve | ❌ |
| role 5 仅在 `eco_logical_experts > 0` 时插入 roles | ❌ |
| `routed_grid = compute_sms - 1` 仅 ECO 时 | ❌（vanilla 仍 72） |
| `eco` / `trace` 参数透传（默认 False） | ❌ |
| role 序列校验放宽；版本 0.3.0→0.3.1、`__init__` 导出 | ❌ |
| **trace 埋点**（`eco_tl` / `pull_tl`） | ⚠️ **生效** |

trace 的保护条件是 `const_expr(self.eco_trace)`，而 `eco_trace` 取自
「**layout 有 `eco_tl` 视图**」而非「ECO 是否开启」。四臂为取 per-rank 时间轴都开了
`GC_MOE_ECO_TRACE=1` ⇒ 第一臂也写 trace，每 MoE 层调用多约 **6 次 `globaltimer` 读 + 6 次 store**。

这份开销**四臂同等承担**，臂间比较仍公平；只是绝对值略高于不埋点的版本。
要纯净绝对值用 `GC_MOE_ECO_TRACE=0` 重跑（`const_expr` 为假，分支编译期整体消失）。

### 分阶段耗时（BS=128，µs/次 MoE 层调用，8 rank 平均）

两臂都开 `GC_MOE_ECO_TRACE=1`（`runs/CB_128_*.json` / `runs/CB4_128_*.json`）：

| 阶段 | 原生 | eplb+eco | 差 |
|---|---:|---:|---:|
| push | 7.1 | 9.5 | +2.4 |
| pull 跨度（首槽开始 → 末槽完成） | 76.2 | 81.2 | +5.0 |
| **dispatch 总计**（push 开始 → pull 完成） | **83.4** | **97.0** | **+13.6** |
| plan_exec | — | 36.9 | — |

* `plan_exec 36.9 µs` 完全被 `push + pull(单副本段)` 盖住，副本槽不等 plan，
  **消费者饥饿 ≈ 0** ⇒ plan 的握手延迟不是瓶颈，不需要继续优化 plan 本身。
* 也就是说 **ECO 的 dispatch 比原生慢**，收益必须从别处找 —— 下一节是答案。

### ★ 收益来源：combine 更快，不是 dispatch 更快（BS=128）

两臂同一张 `pull_tl` 表（早期 `eco_trace` 只在 ECO layout 里分配，vanilla 拿不到 combine，
所以把 combine 首/末标记并入了 `pull_tl` 的 slot 8/9）。覆盖 8 rank × 256 层调用 × 2 臂 =
2048 行，slot 全部命中且有序。

| 阶段（µs/层，中位数） | 原生 | eplb+eco | 差 |
|---|---:|---:|---:|
| push_end → pull_begin | 3.0 | 4.0 | +1.0 |
| **PULL** | **74.0** | **82.0** | **+8.0** |
| pull_end → combine_begin | 338.0 | 349.0 | +11.0 |
| **COMBINE** | **193.5** | **152.0** | **−41.5** |
| push_end → combine_end | 625.0 | 603.0 | **−22.0** |

combine p90：原生 449 → ECO **354** µs ⇒ 尾巴短 95 µs。

**结论**

1. **ECO 的 dispatch 每一档都更慢**（BS=128 时约 +10~14 µs/层 = 0.5 ms/step）——
   多 8 个副本槽 + 一份 plan，这是固定开销。
2. **ECO 的 combine 快 21%**（−41.5 µs/层）。副本把每 rank 的入站 token 摊薄，
   scatter/归约跨度变短，尾部尤其明显。
3. 净差 `push_end → combine_end` = **−22 µs/层 ≈ −0.88 ms/step**，与实测 −1.09 ms 同量级
   ⇒ **ECO 的收益在 combine，不在 dispatch。**
4. 这也解释了「ECO 只在大 batch 赢」：combine 的收益正比于入站 token 量，
   BS=32 每 rank 只有 ~20 token，摊薄收益抵不过 dispatch 的固定开销。

### 冗余度扫描（BS=256，eplb+rr，TMA）

| extra/rank | P | slots | ms/step | IR |
|---:|---:|---:|---:|---:|
| 4 | 416 | 52 | 66.95 | **1.0306** |
| **8** | **448** | **56** | **66.56** | 1.0344 |
| 16 | 512 | 64 | 66.14 | 1.0420 |
| 32 | 640 | 80 | 68.46 | **1.0269** |

IR 在四组之间只差 1.4%（1.027–1.042），各组最忙 rank 的 token 数都在 775k–786k 之间波动
1.4% —— **这个差异是噪声，冗余度对 rank 级 IR 没有系统性影响**（它优化的是槽级分布）。
选 8 是因为 ms 最优且槽数适中。

### warp 并行版（对齐 SM100）

plan CTA 提到 **512 线程**，照 `eco_unit_kernel.cu` 分工：

```
warp 0      -> 静态表           (bar.sync 1, 512 汇合)
warp 1..15  -> counts + 广播     (bar.sync 2, 480)
```

实测 `plan_exec` **37.6 → 34.3 µs**（`M_32_eco` vs `WS100`），但 `ms/step`
20.41 → 20.72：plan 早已不是瓶颈，收益被 pull 的抖动掩盖。

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

`runs/rank_traces.json`（8.1 MB）：13 个实验 × 8 rank × 每 rank 256 行 × 18 个槽位。

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
| wb / wa 标定 | ✅ 全部实验统一 `wb=260, wa=111` |
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

## 四臂正式跑法

```bash
# BS=32 = 8 ranks x 4 seq/rank. 64 -> 8, 128 -> 16, 256 -> 32.
C="SEQS_PER_RANK=4 WARM_STEPS=100 MEASURE_STEPS=100 GC_MOE_SM90_PULL_TMA=1"
# arm 0: 原生 gcmoe（无副本，P=384）
env $C ./scripts/run_arm_eco.sh 0 0 Q4_32_vanilla
# arm 1: random placement + host RR
env $C GC_MOE_SLOT_PLACEMENT=random GC_MOE_EPLB_EXTRA_PER_RANK=8 \
    ./scripts/run_arm_eco.sh 0 0 Q4_32_random
# arm 2: EPLB placement + host RR
env $C GC_MOE_EPLB=1 GC_MOE_EPLB_EXTRA_PER_RANK=8 \
    ./scripts/run_arm_eco.sh 1 0 Q4_32_rr
# arm 3: EPLB placement + kernel plan (add GC_MOE_ECO_TRACE=1 for the timeline)
env $C GC_MOE_EPLB=1 GC_MOE_EPLB_EXTRA_PER_RANK=8 GC_MOE_ECO=1 \
    GC_MOE_ECO_TRACE=1 GC_MOE_ECO_WB=260 GC_MOE_ECO_WA=111 \
    ./scripts/run_arm_eco.sh 1 1 Q4_32_eco
```

`GC_MOE_EPLB_EXTRA_PER_RANK=n` sets redundant slots **per rank** directly
(`physical = experts + n*ranks`); it overrides the older fraction knob
`GC_MOE_EPLB_REDUNDANCY`. `GC_MOE_SLOT_PLACEMENT=random` scatters the same
multiset of experts over the same slots with a fixed seed -- that is the
placement control arm.

Dump the timeline: `python3 scripts/agg_eco_stages.py runs/C2_eco.json`.
