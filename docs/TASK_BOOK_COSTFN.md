# 任务书：用实测耗时函数替换 ECO 的 cost（t19_costfn）

> 压缩上下文后必读。工作目录 `app_source/gcmoe_exp/t19_costfn`。
> 基线 = **mb-claim 已开启**（按 M 块领取任务），本任务只换耗时函数。

## 1. 用户的三条明确要求（2026-11-17）

1. **gap 用 B 口径**：按实测，每块付 **38 次** 交接空闲
   （L1 18 个 ntile + L2 20 个 ntile），即 `38 × 1.821 = 69.2 µs/块`。
   *注：上一轮把 69 µs 塞进 wb 的等价做法在 A/B 里 8/8 IR 更差，
   但用户明确要求按实测，故本轮照做，用实验检验。*
2. **push 不重复计**：push 已在 kernel 记录的 task span 内（960/960 验证），
   所以用 `span − push` 得纯 GEMM，再把 push 按 valid 加回。
3. **只换耗时函数**：这个函数没有物理单位，只是"量化某 rank 的耗时"。

## 2. 新耗时函数（全部系数由 mb-claim trace 拟合）

来源：`t18_abstrace/docs/abstrace/AT256_eco_mbclaim_rank0_full.abstrace_rank0.csv`

```
nq(v) = 8/16/32/40/48/64 的量化档（GEMM 按档收费）

纯 GEMM（已 ×ntile，span − push）:
  L1: 14.33 + 0.5042·nq   [×18]
  L2:  7.02 + 0.2122·nq   [×20]
push（valid 线性，已 ×ntile）:
  0.765 + 0.0303·v        [×38]

=> C_block(v) = 398.4 + 13.32·nq(v) + 1.151·v     [µs]
   T_gap      = 1.821 µs / 次

cost(slot) = Σ_blocks C_block(valid_i) + (#blocks) × 38 × 1.821
```

`_eco_costC(slot, y, hp)` 的结构刚好匹配：
* `y`   = 整块数（每块 valid = block_m = 64）
* `hp`  = 是否有零头块（valid = rem）

```
cost = y·C_block(64) + (hp ? C_block(rem) : 0) + (y + hp)·GAP_TOTAL
```

## 3. 要改的 4 个同构函数（缺一个就自相矛盾）

| 函数 | 作用 | 改动 |
|---|---|---|
| `_eco_rank_time` | T[r] = Σ_slots cost | 换成新式（去掉 tok 项） |
| `_eco_costC` | 单个 unit 的 cost | 同上（核心） |
| `_eco_out` / `_eco_in` | 差分 Δ | 必须与新 cost 一致，否则 auction 走错方向 |
| `_eco_do_warm` 里 `unit = wb + bm` | warm start 步长 | 换成新式差分步长 |

**关键**：这些函数原来都含 `+ q`（token 项）。新函数**没有 token 项**，
所以所有 `tok`/`q` 参与的地方都要重算，不能只改一处。

## 4. 系数怎么进内核

用 `cutlass.const_expr` 常量（编译期定值），单位统一为 **1/1024 µs 的整数**
（避免浮点），即：
```
CB_GEMM_FIX  = 398.4 * 1024  ≈ 408000
CB_GEMM_NQ   = 13.32 * 1024  ≈  13640
CB_PUSH_V    = 1.151 * 1024  ≈   1179
CB_PUSH_FIX  = 29.07 * 1024  ≈  29770
GAP_TOTAL    = 69.2  * 1024  ≈  70860
```
新增 env `GC_MOE_ECO_COSTFN=1` 启用（默认 0 = 旧 token 式），便于 A/B。

`nq(v)` 用与 GEMM 完全相同的档位表：
```python
for lim in (8,16,32,40,48):
    if v <= lim: return lim
return 64
```

## 5. CPU 验证（GPU 之前必须做）

1. 单调性：`C_block` 对 v 单调递增。
2. 与实测对齐：v=8/16/32/48/64 的 C_block 应与
   545/661/892/1124/1355 µs 吻合（±3%）。
3. 差分一致性：`_eco_out/_eco_in` 必须等于 `costC(y±1) - costC(y)`。
4. 与旧函数的排序对比：给出一个典型槽位分布，两种 cost 下的排序差异。

## 6. 要跑的实验（8 个）

4 档 BS × {eco, eco_smr}，全部 `GC_MOE_MB_CLAIM=1` + `GC_MOE_ECO_COSTFN=1`：

| 标签 | BS | spr | smr |
|---|---|---|---|
| C_32_eco | 32 | 4 | 0 |
| C_32_eco_smr | 32 | 4 | 1 |
| C_64_eco | 64 | 8 | 0 |
| C_64_eco_smr | 64 | 8 | 1 |
| C_128_eco | 128 | 16 | 0 |
| C_128_eco_smr | 128 | 16 | 1 |
| C_256_eco | 256 | 32 | 0 |
| C_256_eco_smr | 256 | 32 | 1 |

配置同主表：`WARM=1000 MEASURE=1000`、`EXTRA_PER_RANK=8`、`PULL_TMA=1`、
`GC_IRSTEP_STEPS=1024`、开 trace、模型走本机副本（加载 ~20s）。

**对照**：主表已跑好的 `M_{BS}_eco` / `M_{BS}_eco_smr`（mb-claim + 旧 token 式）。

## 7. 执行纪律

* 每 20 s 轮询，禁长 sleep。
* 出错立即停、定位、修复、继续。
* 每个 run 完成立即打印 ms/IR 并追加 `/tmp/costfn_progress.log`。
