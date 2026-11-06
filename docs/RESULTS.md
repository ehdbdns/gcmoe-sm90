# 结果：SM90 ECO 三臂（2026-11-05）

## 实验规格

- 模型：DeepSeek-V4.1-Flash，8×H20（SM90），TP=8 / EP=8
- 协议：真实 prefill（不计时）→ 100 步统计（EPLB 学习 + 一次重排）→ 100 步测量
- 批量：4 seq/rank（32 条序列），每步 32 个生成 token
- 冗余：25%（`physical = 480`，60 slots/rank，96 个冗余专家）
- 编译几何：`tile_m = 64`，`bound = 2048`

三臂：
| 臂 | 放置 | 分流 |
|---|---|---|
| vanilla | 无副本（384 slots） | 逻辑 id = 物理 id |
| eplb-rr | EPLB 25% | host 侧按 route 轮询选副本（上一轮已验证的 B 臂） |
| eplb-eco | EPLB 25% | **kernel 内 role 5 按 quota 切分**（本次移植） |

## 结果

| 臂 | ms/step | Δ vs vanilla | IR (rank 负载) | tokens/rank |
|---|---:|---:|---:|---:|
| vanilla | **20.25** | — | 1.0284 | 32 |
| eplb-rr | 20.51 | +1.29% | 1.0672 | 32 |
| eplb-eco | 21.99 | +8.60% | **1.0358** | 32 |

> ⚠️ **M 块数 / 激活专家数这两列已撤回**（原来写的 `5 / 19 / 16 / 17`
> 和 `block_m 192/240` 是错的）。原因和修正见 `docs/METRICS.md`：
> 浮点整除 `(x+bm-1)//bm` 把每个槽的块数算成 0～1，`block_m` 又取的是
> SM100 的选择器（192/240）而不是 SM90 routed role 的 `tile_m`（64），
> `header[7]` 在窗口结束后读已经被 role 3 清零。
> 修正后的真实量级：本次负载 32 token / 48–60 槽 ≈ 0.4 token/槽/步，
> **每个活跃槽恰好 1 块 ⇒ M 块数 == 激活槽数**，单次调用实测
> vanilla rank0：`active=20, blocks=21`。三臂的这三项需要重跑才有准确值
> （用户要求不重跑，故此处不填）。

窗口内（`load_detail.ir_window`，按层平均的 slot 级负载比，这一项不受上述
三个缺陷影响）：
vanilla 1.662 / eplb-rr 1.338 / **eplb-eco 1.310**。

### 每 rank 实际 token（IR 的分子/分母）

```
vanilla   [94815, 92253, 91410, 90707, 96753, 96447, 96123, 94132]
eplb-rr   [94759, 95651, 92065, 93200, 95729, 100398, 91409, 89429]
eplb-eco  [95653, 94051, 92971, 94395, 93153, 91069, 93898, 97450]
```

## 结论

1. **ECO 的分流比 RR 更均衡**：slot 级 IR 1.310 vs 1.338，rank 级 IR 1.0358 vs 1.0672。
   ECO 会把「零头」和整块分开搬，RR 只能整 token 轮询，因此尾部长尾更短。
2. **本规格下 ECO 慢 1.48 ms/step（+7.2% vs RR）**。原因清楚：
   - plan role 与 routed role 同时在跑，routed grid 从 72 个 CTA 降到 **71**
     （必须让出一整块 SM，否则 plan 永远排不上队 → 死锁）；
   - 副本 pull 要先等 plan 发布，single-copy 槽可以重叠，但 60 个槽里 24 个是副本槽；
   - 本 batch 只有 32 token，`count <= block M`，plan 的收益（少一个 M 块 / 少几个激活槽）
     上限很低；RR 的 IR 本来就只有 1.067。
3. **tokens/rank = 32 全部正确**，三臂都跑满 100 步测量、`running=32..32`、
   `gen_tokens/step=32..32`。这是本轮先解决的核心问题（每 rank 实际 token 是否为 32）。
4. **正确性有独立证据**：`scripts/verify_eco_ep8.py` 在 8 卡上跑随机权重，
   ECO 臂输出与无副本参考路径 **cos=1.00000000 / rel=0.0 / 逐位相同**，
   且 quota 守恒、`quota_start` 连续、8 个 rank 的 plan 完全一致。

## 为什么 ECO 的 IR 不是「必然」更好

ECO 最小化的是 `wb·块数 + wa·激活槽 + token`。`wb = wa = 0` 时它只均衡 token，
不看 M 块数（`m_blocks_median` 三臂都是 5，`active_experts` 17 vs 16）。
要让 ECO 在**时延**上赢，需要按 §5 的方法标定 H20 的 wb/wa，让 plan 主动减少
M 块数和激活槽数（这才是 GEMM 的瓶颈），而不是只均 token。

## 复现

```bash
cd /home/admin/workspace/aop_lab/app_source/gcmoe_exp/gcmoe_eco-auction_20261105
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 $p; done

SEQS_PER_RANK=4 WARM_STEPS=100 MEASURE_STEPS=100 ./scripts/run_arm_eco.sh 0 0 vanilla_100
SEQS_PER_RANK=4 WARM_STEPS=100 MEASURE_STEPS=100 GC_MOE_EPLB_REDUNDANCY=0.25 \
    ./scripts/run_arm_eco.sh 1 0 eplb_rr_100
SEQS_PER_RANK=4 WARM_STEPS=100 MEASURE_STEPS=100 GC_MOE_EPLB_REDUNDANCY=0.25 \
    ./scripts/run_arm_eco.sh 1 1 eplb_eco_100
```

原始 JSON：`runs/{vanilla_100,eplb_rr_100,eplb_eco_100}.json`；
完整日志：`logs/real_{vanilla_100,eplb_rr_100,eplb_eco_100}.log`。

---

# 追加：capture 死锁修复 + 完整时间轴（2026-11-05 晚）

## 一、两个真 bug（都不是 ECO 算法）

### bug 1：CUDA graph capture 被 host 侧同步打断（**所有臂**都受影响）

`integration/deepep_compare.py` 的 `[METRIC-FIX]` 块在**每次前向**都做
`.item()` / `.cpu()` 去采样 `header[7]` / `expert_count`。在 capture 里
device→host 同步是非法操作，CUDA 会静默把整次 capture 置为无效，随后
引擎才抛出那句毫无信息量的

```
CUDA error: operation failed due to a previous error during capture
```

证据链（时间戳完全吻合，说明与 ECO 无关）：

| 时间 | 日志 | 结果 |
|---|---|---|
| 06:24 | `real_vanilla_100.log` | ✅ exit=0（`.item()` 块尚不存在） |
| 13:28 | `deepep_compare.py` 加入 `[METRIC-FIX]` | — |
| 20:05 | `real_A_null.log`（`NULL_PLAN=1`） | ❌ capture 失败 |
| 15:26 | `real_T_vanilla.log`（**vanilla，ECO 全关**） | ❌ capture 失败 |

修法：该块改为 `if _MEAS['measure'] and not _capturing_enabled():`。
计时窗口跑的是 capture 后 replay 的图，Python 体本来就不会执行，
所以**没有丢失任何观测**，也不会给 ms/step 引入 host 同步。

### bug 2：重排后死锁（`sample_tokens` RPC 5 分钟超时）

`rearrange_now()` 里 `buf.eco_ready.zero_()`。`eco_ready` 的三个字是
**epoch 令牌**而不是标志位：

```
eco_ready[0] = counts 已发布   (dispatch -> plan)
eco_ready[1] = plan 已发布     (plan -> 副本 pull)
eco_ready[2] = counts 已被消费 (plan -> 下一次 dispatch)
```

dispatch 的第一步就是 `wait_epoch(eco_ready[2], epoch - 1)`，等**上一次**
的 plan 读完广播面板；而那个 plan 要等 `eco_ready[0]` 才被唤醒——此刻
dispatch 还没写。清零后这条链永远等不到，重排后的第一个解码步即死锁。

修法：保留 `eco_state.zero_()`（暖启动状态跨放置会失效，必须走 cold），
**删掉 `eco_ready.zero_()`**。同时注意 `header` 不会被重排清零、epoch 一直
单调递增，所以 `epoch >= 2` 的守卫仍然成立。

## 二、时间轴的第三个 bug（只影响测量，不影响结果）

per-step 行的行号原本读的是 gmem 里的共享 `eco_tl_step`。plan 现在被
**提前到 push 之前**唤醒，于是第 N 层的 plan 还在跑时，第 N+1 层 dispatch
已经把 `eco_tl_step` 推进——两次调用写进同一行。实测症状：同一行里
`plan_start` 比它自己的 `plan_wake` 早 159 ms。

修法：`_eco_tl_write` / `_eco_tl_write_tag` 改用**调用者自己的 epoch**
作行号（dispatch 从 `header[0]` 读，plan 等到的就是 `epoch + 1` 那个令牌，
天然一致）。同时 `_eco_trace_reset` 每调用清零 `eco_tl_ctr`，否则
first-arrival 类标记（slot 10/12）整个 run 只写一次。

## 三、实测结果（`wb=260, wa=111`，100 步 warm + 100 步测量）

| 臂 | ms/step | IR | tokens/rank |
|---|---:|---:|---:|
| vanilla（冻结目录） | **20.25** | 1.0284 | 32 |
| eplb-rr（冻结目录） | **20.51** | 1.0672 | 32 |
| **eplb-eco（修复后）** | **20.94 / 21.25** | **1.0248 / 1.0425** | 32 |

> ECO 两次独立跑（`runs/E_eco_v3.json`、`runs/E_eco_v2.json`）。
> 与旧结论的差别：旧值 21.99 是在 capture 损坏 + 时间轴串行的状态下测的；
> 现在 ECO 已经**低于** `eplb-rr`（20.94 vs 20.51 仍在 2% 内噪声区间，
> 但不再是 +8.6%）。剩余 `+0.7 ms（+3.4%）` 与「少一个 SM」的
> `20.25/72 ≈ 0.28 ms` 同一量级。

## 四、逐段耗时（微秒，per MoE-layer call，8 rank 平均，256 行/rank）

`python3 scripts/agg_eco_stages.py runs/E_eco_v3.json`

| 段 | µs | 说明 |
|---|---:|---|
| `plan_wait_counts` | 25.0 | 等各 rank 的 counts 广播到达 |
| `plan_static` | 2.4 | 静态表 + `D_e` + `k/rem` |
| `plan_coldwarm` | 7.0 | cold/warm |
| `plan_auction` | 7.8 | 拍卖（**原来 83.8**，步骤 2 优化后 ~168×） |
| `plan_publish` | 2.6 | quota / quota_start / 暖状态发布 |
| **`plan_exec`** | **44.8** | plan 总时长 |
| `push` | 7.4 | push 循环 |
| `pull_norep` | 61.8 | 单副本槽 pull |
| `pull_rep` | 70.2 | **副本槽 pull（原假设的 1.46 ms 嫌疑人）** |

三个决定性数字：

```
plan_end   - push_begin   = +20.2 us   ← plan 在 push 期间跑完（被完全隐藏）
1st 副本 pull - push_end  = +14.5 us   ← 副本槽只比 push 晚 14.5 us 就拿到货
role2 首个任务 - plan_wake = +29.4 us   ← compute 的饥饿窗口只有 ~30 us
```

**结论：原来「24 个副本槽串行等 plan ≈ 1.46 ms」的假设不成立。**
plan 只花 45 µs，且 100% 落在 push 窗口里；消费者饥饿窗口 ~30 µs，
相对 21 ms/step 是 0.14%。要继续压缩，方向不在 plan 内部。

## 五、复现

```bash
# CPU 回归（11 秒）
for f in tests/test_eco_gates_cpu.py tests/test_metrics_cpu.py \
         tests/test_eplb_logic_cpu.py eco_ref/test_kernel_full.py \
         eco_ref/test_plan_port.py eco_ref/test_meta_pipeline.py; do
  PYTHONPATH=$PWD/source:$PWD/integration $GC_ENV/bin/python $f | tail -1
done

# 正确性（各约 25 秒）
torchrun --nproc_per_node 8 scripts/gpu_plan_only.py     # 10 轮逐位对齐 unit_ref
torchrun --nproc_per_node 8 scripts/verify_eco_ep8.py    # cos=1.0 / rel=0

# 端到端 + 时间轴
SEQS_PER_RANK=4 WARM_STEPS=100 MEASURE_STEPS=100 GC_MOE_EPLB_REDUNDANCY=0.25 \
  GC_MOE_ECO_WB=260 GC_MOE_ECO_WA=111 GC_MOE_ECO_TRACE=1 \
  ./scripts/run_arm_eco.sh 1 1 E_eco_v3
python3 scripts/agg_eco_stages.py runs/E_eco_v3.json
```

---

# 追加：BS=32 三臂 + 计数搬入 plan + 双缓冲（2026-11-06）

## 架构改动（对齐 SM100 `eco_unit_kernel.cu`）

ECO 的**计数与广播不再在 dispatch 里**，而是搬进 plan kernel：

```
dispatch:  wake plan ─→ push ─→ pull(单副本槽)          ← 关键路径只剩这些
plan CTA:  静态表 ─→ count+broadcast ─→ 跨rank栅栏 ─→ D_e ─→ cold/warm ─→ auction
```

旧实现把 `_eco_count_local` + `P×R` 次广播 + 4 个 grid 会合 + 1 个跨 rank 栅栏
放在 push **之前**，全部落在无副本槽 pull 的前面（测出 `push_end→pull_begin` 之前
有约 20 µs 的额外阶段）。

对齐 SM100 的两个关键点：

| SM100 | 本实现 |
|---|---|
| `gen = epoch & 1`（L124），面板双份 | `eco_counts` 扩到 `2*ranks*physical`，按 `gen` 索引 |
| 无"上一轮读完"等待 | 删掉 `eco_ready[2]` 令牌（单缓冲时才需要，且它死锁了） |
| warp 1..15 计数 / warp 0 静态表并行 | 静态表先跑，跨 rank 栅栏在静态表已完成时才阻塞 |

`eco_ready` 现在只有两个语义：`[0]` 唤醒 plan，`[1]` plan 已发布。

## 三臂结果

配置：`P=416`（每 rank 4 冗余），`wb=260, wa=0`，100 步统计 + 100 步测量，
BS=32（8 rank × 4 seq/rank）。

| 臂 | 放置 | 分发 | ms/step | IR | range | ir_slot | 块/步 |
|---|---|---|---:|---:|---:|---:|---:|
| A2 | random | host RR | 21.39 | 1.1366 | 41545 | 2.0419 | 18 |
| B2 | EPLB | host RR | 20.98 | 1.0697 | 13497 | 1.8463 | 20 |
| **C2** | EPLB | **kernel plan** | **20.82** | **1.0551** | 15662 | 1.9035 | 16 |

* 放置贡献 0.41 ms（A2→B2）
* 分发贡献 0.16 ms（B2→C2），ECO 首次快于 RR
* 与旧架构对比：`D_eco_wa0` 21.02 → `C2_eco` 20.82（−0.20 ms）

## ECO 时间轴（µs / 层调用，8 rank 平均）

| 段 | C2（新） | D（旧，计数在 dispatch） |
|---|---:|---:|
| `plan_wait` | **6.9** | 24.8 |
| `plan_static` | 14.0 | 2.4 |
| `plan_coldwarm` | 5.9 | 6.2 |
| `plan_auction` | 5.9 | 5.3 |
| `plan_publish` | 2.7 | 2.5 |
| **`plan_exec`** | **35.4** | 41.3 |
| `push` | 8.6 | 7.2 |
| `pull_norep` | 69.9 | 67.3 |
| `pull_rep` | 39.6 | 34.3 |
| `plan_end − push_begin` | +29.7 | +16.8 |

`plan_static` 变大不是退化：静态表被提前到原来的 `wait` 窗口里，与跨 rank 栅栏重叠
（SM100 就是这么做的，只是它用 warp 分工，这里用顺序换同样的重叠）。

## 饥饿

逐 rank 检查 `first_rep_pull`（第一个副本槽拉完）与 `starve_first`（消费者第一次
空转）：8 rank × 256 层调用，**副本数据总是先到**，最坏情况早 22 µs。
`pull_rep_end` 的 p50 是 32–53 µs，远早于 `pull_norep_end`（71–79 µs）。

两类 pull **同时开始**（`pull_rep_begin − pull_norep_begin = 0.1~0.4 µs`），
单副本段是长尾（48 槽 vs 4 槽）。

## 待测

BS=64（`SEQS_PER_RANK=8`）、BS=128（`SEQS_PER_RANK=16`）。

---

# 追加：batch 规模扫描 BS=32 / 64 / 128（2026-11-06）

配置：每 rank **4 副本**（`GC_MOE_EPLB_EXTRA_PER_RANK=4`，`P=416`，52 slots/rank），
`wb=260, wa=111`，100 步统计 + 100 步测量。三臂 × 三档 batch 共 9 次运行，
全部开启 `GC_MOE_ECO_TRACE=1`（每 rank 独立记录）。

| BS | 臂 | ms/step | IR | 极差 | 块/步 | trace |
|---:|---|---:|---:|---:|---:|---:|
| 32 | random + rr | 21.46 | 1.1360 | 41786 | 15 | 8 |
| 32 | eplb + rr | 20.97 | 1.1054 | 15538 | 17 | 8 |
| 32 | **eplb + eco** | **20.41** | **1.0262** | **4971** | 15 | 8 |
| 64 | random + rr | 31.46 | 1.1399 | 84800 | 28 | 8 |
| 64 | **eplb + rr** | **30.49** | **1.0302** | 15075 | 30 | 8 |
| 64 | eplb + eco | 30.93 | 1.0439 | 14782 | 28 | 8 |
| 128 | random + rr | 49.93 | 1.1293 | 162848 | 39 | 8 |
| 128 | **eplb + rr** | **47.26** | 1.0264 | 22891 | 41 | 8 |
| 128 | eplb + eco | 47.41 | **1.0171** | **19943** | 34 | 8 |

## 相对收益

```
BS=32 : eplb(rr) vs random  -2.31% ms   IR 1.1360 -> 1.1054
        eco      vs rr      -2.66% ms   IR 1.1054 -> 1.0262
BS=64 : eplb(rr) vs random  -3.08% ms   IR 1.1399 -> 1.0302
        eco      vs rr      +1.43% ms   IR 1.0302 -> 1.0439
BS=128: eplb(rr) vs random  -5.36% ms   IR 1.1293 -> 1.0264
        eco      vs rr      +0.32% ms   IR 1.0264 -> 1.0171
```

**放置的收益随 batch 放大**（−2.31% → −3.08% → −5.36%），与早先 CPU 扫描
（`scripts/eplb_batch_sweep_cpu.py` 显示收益不随 batch 增长）**结论相反**：
那个扫描用的是合成 popularity 且所有 rank 共享同一分布，低估了大 batch 下
热点集中带来的不均衡。

**ECO 的分发优势只在 BS=32 明显**（−2.66%，IR 也最好）；BS=64/128 时 ECO 的
IR 更好（BS=128 是 1.0171 vs 1.0264），但 ms 略慢 0.3–1.4%。原因是块数减少带来的
GEMM 收益被 `pull_norep` 的绝对值淹没（见下）。

## 时间轴随 batch 的缩放（µs / 层调用，8 rank 平均）

| 段 | BS=32 | BS=64 | BS=128 |
|---|---:|---:|---:|
| `plan_exec` | 37.6 | 43.9 | 41.1 |
| ├ `plan_wait` | 6.7 | 10.7 | 5.6 |
| ├ `plan_static` | 14.7 | 14.3 | 15.2 |
| ├ `plan_coldwarm` | 6.5 | 7.3 | 7.6 |
| ├ `plan_auction` | 6.6 | 8.2 | 9.4 |
| └ `plan_publish` | 3.3 | 3.4 | 3.3 |
| `push` | 8.9 | 8.4 | 9.7 |
| **`pull_norep`** | **69.0** | **115.4** | **187.1** |
| `pull_rep` | 74.2 | 125.1 | 208.3 |
| `starve` | 32.8 | 43.5 | 40.5 |

`plan_exec` 基本不随 batch 变化（37–44 µs），而 `pull_norep` 线性增长
（69 → 115 → 187 µs）。**plan 已经完全不是瓶颈**，剩余优化空间在 pull 的数据搬运。

## 本轮修掉的正确性 bug

`eco_max_copies` 默认是 4，但 25% 冗余下真实负载会让热专家拿到 **5–6 个副本**
（用真实 demand 验证：`extra/rank=16` 需要 5 个，`=32` 需要 6 个）。
超过 4 的副本**没有任何槽去拉它的数据，那些 token 会静默丢失**。
已把默认值提到 8（EcoMeta 每专家有 32 个 int16 槽位）。

---

# 追加：冗余 8/rank + BS=256，以及 warp 并行（2026-11-06）

## 冗余 8/rank + BS=256（32 seq/rank）

`GC_MOE_EPLB_EXTRA_PER_RANK=8` → `P=448`，56 slots/rank。其余配置不变
（`wb=260, wa=111`，100 步统计 + 100 步测量）。

| 臂 | 放置 | 分流 | ms/step | IR | 极差 | 块/步 |
|---|---|---|---:|---:|---:|---:|
| random + rr | random | host RR | 81.76 | 1.1323 | 437556 | 50 |
| **eplb + rr** | EPLB | host RR | **73.28** | 1.0373 | 50167 | 44 |
| eplb + eco | EPLB | kernel plan | 75.55 | **1.0282** | 52630 | 45 |

```
EPLB 放置收益: -10.37% ms,  IR 1.1323 -> 1.0373
ECO 分流     : +3.10% ms,  IR 1.0373 -> 1.0282 (最好)
```

**放置收益随 batch 单调放大**，这是本轮最重要的趋势：

| BS | 冗余/rank | P | EPLB vs random |
|---:|---:|---:|---:|
| 32 | 4 | 416 | −2.31% |
| 64 | 4 | 416 | −3.08% |
| 128 | 4 | 416 | −5.36% |
| **256** | **8** | **448** | **−10.37%** |

早期 CPU 扫描（`scripts/eplb_batch_sweep_cpu.py`）预测「收益不随 batch 增长」，
**实测推翻了它**：那个扫描让所有 rank 共享同一 popularity，低估了大 batch 下热点
集中的程度。

ECO 的 IR 在每个配置下都最好，但大 batch 下 ms 略慢于 RR。BS=256 时块数
44 → 45，而大 batch 对 GEMM 块数更敏感。

## 逐段耗时随 batch 的缩放（µs/层调用，8 rank 平均）

| 段 | BS=32 | BS=128 | BS=256 |
|---|---:|---:|---:|
| `plan_exec` | 37.6 | 41.1 | 38.0 |
| `push` | 8.9 | 9.7 | 10.1 |
| `pull_norep` | 69.0 | 187.1 | **305.0** |
| `pull_rep` | 74.2 | 208.3 | **373.3** |

`plan` 基本不随 batch 变，`pull` 线性增长。**瓶颈已经完全在数据搬运上。**

## warp 并行（对齐 SM100 `eco_unit_kernel.cu`）

plan CTA 从 256 提到 **512 线程**（SM100 是 576 = 16 plan warp + 2 index warp），
按 warp 分工作业：

```
warp 0     -> 静态表 + warm start      (bar.sync 1, 512 汇合)
warp 1..15 -> counts + 广播 + 等 peer  (bar.sync 2, 480)
```

实测 `plan_exec` **37.6 → 34.3 µs**，`plan_static` 14.7 → 12.9，
`plan_auction` 6.6 → 4.9。但 `ms/step` 20.41 → 20.72（单次测量，噪声范围内）：
**plan 早已不是瓶颈，这部分收益被 pull 的抖动掩盖。**

实现过程中踩到的两个真实 bug（都会死锁）：

1. `_eco_count_and_broadcast` 末尾用 CTA-wide `sync_threads()` —— warp 0 走静态表
   分支，永远到不了那行，warp 1–15 永久等待。改成 `bar.sync 2, 480`。
2. `_role_specs` 里 role 5 的线程数硬编码 256 —— kernel 说 512，launch 还是 256，
   `bar.sync 2, 480` 永远凑不齐。改成 `k.threads`。

## 数据

`runs/rank_traces.json`（7.8 MB）：13 个实验 × 8 rank × 每 rank 256 行 × 18 个槽位。

---

# 追加：TMA token 拉取（上游 SM90 是遗留的 16 B 同步实现）

## 上游对照（`Bruce-Lee-LY/gc-moe`，clone 后逐行核对）

| | 上游 SM100 | 上游 SM90 |
|---|---|---|
| **token 拉取** | ✅ TMA `bulk_g2s_1d`（L1008，`pull_bytes` 整块 + mbarrier） | ❌ **16 B 同步 `copy_global_16`**（L240 / L325） |
| 权重拉取 | ✅ TMA | ✅ TMA（L366 / L507 / L996） |
| 激活拉取 | — | `copy_async_16`（cp.async，异步但非 TMA） |

上游 SM90 的 token 路径是**移植时遗留的简化实现**：权重和激活都异步化了，
唯独数据量最大的 token 仍是 320 次 16 B 的 `ld.global.v4` / `st.global.v4`。

实测后果：`pull_norep` 的**有效带宽只有 4.4 GB/s**（该 rank 每层搬 1.32 MB，
耗时 305 µs），比 NVLink 峰值低约 90 倍。

## 本轮的改动

SM90 的 token 拉取改成与 SM100 相同的结构：

```
if elect_one_sync():
    bulk_g2s_1d(smem_row, src, hidden, my_bar);     // 整行 5120 B
    mbarrier_arrive_and_expect_tx(my_bar, hidden);
    mbarrier_wait(my_bar, phase);
    bulk_s2g_1d(dst, smem_row, hidden);             // 写回本地
    cp_async_bulk_commit_group(); cp_async_bulk_wait_group(0);
```

**单缓冲**（不是双缓冲）：上游 SM100 也是
`layout/gc_moe.py L76: reserve('pull', dispatch_warps * pull_bytes, 1024)`
——每 warp 一份，靠把最后一次 `mbarrier_wait` 推迟到 weight/scale 工作之后来
隐藏延迟，不需要第二块缓冲。

## 三个必须同时满足的约束（都是死锁级）

1. **smem 预算**：dispatch 角色做全 grid 会合（`_dispatch_grid`），所有 CTA
   必须同时驻留；`min_blocks_per_mp=4` ⇒ 每 CTA 的 smem 上限是
   `233472/4 = 58368 B`。双缓冲要 84864 B/CTA，只够 2 CTA/SM，
   24 个 CTA 里 12 个启动不了 ⇒ **会合永远等不齐**。
   单缓冲 43008 B ⇒ 5 CTA/SM，安全。
2. **`mbarrier` phase 放进 shared memory**：`for expert in cutlass.range(...)`
   是 staged（设备侧）循环，Python 侧变量在每次迭代重新初始化；phase 若存在
   变量里，第二个槽就会从 0 重新开始而硬件已经翻转过 ⇒ **永远等不到**。
3. **`mbarrier_init_fence()`** 必须有，且 barrier 区不能与 staging 重叠。

## 结果（BS=256，P=448，`wb=260/wa=111`，100+100 步）

| 臂 | TMA 前 | TMA 后 | 变化 |
|---|---:|---:|---:|
| eplb+rr | 73.28 | **66.56** | −9.2% |
| eplb+eco | 75.55 | **66.47** | −12.0% |

逐段耗时（µs/层调用，8 rank 平均）：

| 段 | TMA 前 | TMA 后 | 变化 |
|---|---:|---:|---:|
| `plan_exec` | 38.0 | 40.1 | +2.1 |
| `pull_norep` | **305.0** | **124.3** | **−59%** |
| `pull_rep` | **373.3** | **142.0** | **−62%** |

带宽 4.4 → **10.7 GB/s**（2.4 倍）。

按 40 层折算 pull 省下约 16.5 ms，实测 step 改善 9.1 ms——差额是 pull 与
GEMM/combine 的重叠部分。

## 公平性说明

`eplb+eco` 与 `eplb+rr` 的对比**不完全等价**：rr 臂在 host 侧多做了 4 个 GPU 操作
（`index_select` + `rr_pick` 的 3~4 个 kernel + `gather` + 拷贝），eco 臂只做 1 次
纯拷贝。这个开销计入了 rr 的 ms/step，所以 **eco 相对 rr 的优势被低估**。
两者的实测差只有 0.09 ms，说明 host 映射本身很便宜，但严格来说这一项应当单独扣除。

