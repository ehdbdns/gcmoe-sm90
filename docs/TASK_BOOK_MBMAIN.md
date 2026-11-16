# 任务书（2026-11-16 夜）：权重 A/B → 主表补跑（mb-claim）

> 压缩上下文后必读本文件，按「执行顺序」继续，不要重新决策。

## 0. 用户的两条硬要求（不可违反）

1. **权重对比的 8 个实验不能开 mb-claim**，因为历史没开。
   否则"新权重更好"会与被混入的"并行度修复"混淆，结论失效。
2. **主表要开到 mb-claim**，这是这一轮的改进点，所有臂都必须开，
   这样臂之间才公平（不是只给某一臂开）。

⇒ 结论：mb-claim 必须做成**运行期开关**，同一份源码跑两种配置。

## 1. 阶段 0：把 mb-claim 变成开关（必须先做）

* 新增 env `GC_MOE_MB_CLAIM`，**默认 0**（= 历史行为，tile 粒度领取）。
* 三处改动都用 `cutlass.const_expr` 编译期分叉，两条路径都保留：
  | 位置 | 开 (=1) | 关 (=0，历史) |
  |---|---|---|
  | dispatch 发布 | 每 (expert,ntile,mblock) 一张票 | 每 (expert,ntile) 一张票，`active_mb=0` |
  | `_claim_command` | `mblk = active_mb[slot]` | `mblk = 0xFFFFFFFF`（哨兵） |
  | `_produce_tasks` | 只 load 该 mblock | `for start in range(0,count,max_m)` |
* CPU 验证：`GC_MOE_MB_CLAIM=0` 与 `=1` 都能编译通过（smoke_compile_eco）。
* 这样历史配置可复现，且两条路径共用同一份源码。

## 1.5 权重口径修正（用户 2026-11-16 追加）

用户指出：新权重还须计入 **combine push（随 valid）** 与 **每个 M 块任务之间的空白**。
已量化，见 `docs/WB_FIT_V2.md`：

```
push  = 0.765 + 0.0303*valid   (µs)        # 已在 task span 内，v1 没漏，只是斜率没分离
gap   = 1.82 µs / 任务                      # 两版实现都稳定（旧 1.69 / 新 1.82）
每块  = 496.7 + 13.32*nq + 1.150*valid      # 已×38 个 ntile
```

⇒ **wb = 34.3**（v1 的 29.5 只含 GEMM；push 只加 0.04，空白加 4.8）

**因此本轮 A/B 改为 26 vs 34**（原定 26 vs 29 已废弃）。
34 这个值还要在 mb-claim=1 上再验一次（空白在两种粒度下都在）。

## 2. 阶段 1：权重 A/B（mb-claim=0，共 16 个实验）

为什么是 16 而不是 8：**本机没有旧权重的可比数据**。
`runs/` 只有 `AT256_eco.json`（旧权重/BS256/eco，可复用）
和 `AT256_eco_mbclaim.json`（那是 mb-claim 开的，不能用来比权重）。
20 臂主表是在**改动前的源码**上跑的（EPLB 还是旧放置），
与本机当前源码不可直接比。所以要得到干净的 A/B，必须**两个权重都现在跑**。

| 组 | 权重 | 臂 | BS |
|---|---|---|---|
| A | wb=26 wa=0（旧） | eco, eco_smr | 32/64/128/256 |
| B | **wb=34 wa=0（新，含 push+gap）** | eco, eco_smr | 32/64/128/256 |

全部 `GC_MOE_MB_CLAIM=0`，其余与历史一致：
`WARM=1000 MEASURE=1000`、`EXTRA_PER_RANK=8`、`PULL_TMA=1`、
`GC_IRSTEP_STEPS=1024`、开 trace。

**判定规则**（先写死，避免事后解释）：
对 eco 臂比较 ms/step，新权重在 **≥3/4 个 BS 上 ms 更低** 且
IR 不劣于旧权重（差值 < 0.005）⇒ 采用新权重；否则用旧权重。

## 3. 阶段 2：主表 20 臂（mb-claim=1 + 冻结的权重）

4 BS × 5 臂 = 20：
`vanilla / random / rr / eco / eco_smr`
（random 用 `GC_MOE_SLOT_PLACEMENT=random` + `GC_MOE_EPLB_REARRANGE=0`）

全部 `GC_MOE_MB_CLAIM=1`，权重用阶段 1 定下的值。

## 4. 每个 run 的固定配置

```bash
bash scripts/warm_page_cache.sh          # 权重加载优化：必须先跑
WARM_STEPS=1000 MEASURE_STEPS=1000
GC_MOE_SM90_PULL_TMA=1
GC_MOE_EPLB_EXTRA_PER_RANK=8             # 每 rank 4 个冗余
GC_MOE_ECO_MAX_COPIES=8
GC_MEASURE_IR_STEP=1 GC_MOE_IR_STEP=1 GC_IRSTEP_STEPS=1024 GC_MOE_LAYERS=40
GC_MOE_ECO_TRACE=1 GC_MOE_PULL_TRACE=1
GC_MOE_FINE_TRACE=1 GC_MOE_FINE_FLOOR=1 GC_MOE_FINE_EPOCHS=1
```

## 5. 执行纪律

* 每 20s 轮询一次，**禁止 **长 sleep。
* 出错立即停轮询、定位、修复、继续；不要跳过实验。
* 每个 run 完成立即把 ms/IR 打印出来并追加到 `/tmp/task_progress.log`。
* 一个 run 约 19 分钟（加载 15 + 解码 4），16+20=36 个 run。

## 6. 进度

| 阶段 | 状态 |
|---|---|
| 0 开关 | 进行中 |
| 1 权重 A/B | 待跑 |
| 2 主表 | 待跑 |

---

## 7. 夜间执行记录（追加，压缩上下文后必读）

### 7.1 已完成：mb-claim 开关化
`GC_MOE_MB_CLAIM`，默认 0。三处分叉都已实现并有 `cutlass.const_expr` 守卫：
dispatch 两处发布 / `_claim_command` / `_produce_tasks`。
`=0` 与 `=1` 均通过 `smoke_compile_eco.py`。

### 7.2 已修的 bug（关键）
关闭路径最初复用 `mblk` 哨兵，导致 `command[0]=0`，
producer 跳过整轮 L1 → consumer 在 `wait_count(l1_done)` 永久自旋。
**症状**：8 rank 卡在 39.9/52.5 GB、GPU 0%、日志停在 `[eplb-dbg] hook reached`。
**修法**：`=0` 分支必须提交真实命令（`command[0]=1`），只有 `=1` 分支才用哨兵判空。

### 7.3 已查明：加载慢不是 bug
| run | 权重加载 |
|---|---|
| `AT256_eco`（成功） | **26.85 s** |
| `AT256_eco_mbclaim`（成功） | **317.75 s** |

差别只在**当时页缓存是否命中**，与 mb-claim 无关。8 个 worker 同时从 NAS
读 510 GB，冷缓存约 317 s。
**对策（进行中）**：把模型复制到本机 `/home/admin/local_models/DeepSeek-V4.1-Flash`，
之后 `MODEL=` 指向本地，加载稳定在 ~30 s。

### 7.4 待办
- [ ] 等本地副本复制完（约 510 GB）
- [ ] 用 `MODEL=/home/admin/local_models/DeepSeek-V4.1-Flash` 重跑阶段 1（26 vs 34）
- [ ] 阶段 2 主表 20 臂
