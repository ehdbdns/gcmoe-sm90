# 任务规格备份：把 SM100 的 ECO plan 逐行抄到 SM90（CuTe DSL）

> 备份自用户消息（2026-11-05）。每次压缩上下文后请重读本文件。

## 第 0 步：先读这 6 个文件

路径前缀：`app_source/gc_moe_compare_20261104/gc_moe_eco/gc_moe/`（原始写的是 `sources/gc-moe-eco/gc_moe/`）

| 文件 | 行数 | 用途 | 重要度 |
|---|---:|---|---|
| `eco/unit_ref.py` | 306 | ★★★★★ **唯一算法规格**：bit-exact numpy 参考 | 必读全文 |
| `eco/eco_unit_kernel.cu` | 689 | SM100 plan kernel（CUDA C++，nvcc 独立编译） | 必读全文 |
| `eco/eco_unit_auction.cuh` | 407 | 拍卖实现（被 .cu 在 L643 include） | 必读全文 |
| `eco/plan.py` | 121 | 编译配置 + `ECO_TIME_WEIGHTS` | 读 L1-50 |
| `eco/include/eco_meta.cuh` | 70 | EcoMeta 内存布局 | 必读全文 |
| `impls/sm100_fp8_fp4_gc_moe.py` | — | SM100 dispatch 侧 ECO 改造 | **只读思路，不要照搬** |

阅读顺序：`unit_ref.py`（算法）→ `.cu` + `.cuh`（实现结构）→ `eco_meta.cuh`（数据布局）。

## 第 1 步：函数级映射表

### 1.1 算法层（unit_ref.py）

| unit_ref.py | 行 | SM90 函数 | 说明 |
|---|---|---|---|
| `rank_time(q)` | L49-51 | `_eco_rank_time` | `T[r]=Σ_slots[wb·⌈q/bm⌉+wa·(q>0)+q]` |
| `_cost(s,y,hp)` | L53-55 | `_eco_costC` | `wb·(y+hp)+wa·[y+hp>0]+y·bm+hp·rem[phy[s]]` |
| `_out(s)` | L60-64 | `_eco_out` | 拿走一个单位的时间变化 → (Δ, ispart) |
| `_in(s,ispart)` | L66-68 | `_eco_in` | 放入一个单位的时间变化 |
| `_move(a,b,ispart)` | L70-78 | `_eco_move` | 零头转移（改 part）/ 整块转移（y[a]--, y[b]++） |
| `quota()` | L80-84 | `_eco_emit_quota` | `quota[s]=y[s]·bm+(rem[e] if part[e]==s)` |
| **`plan(n,prev)`** | L87-105 | `_eco_plan_assign` | n→k/rem → 无副本专家定死 → cold/warm → rounds |
| **`_cold(n)`** | L107-129 | `_eco_do_cold` | 均分 + 按 `t0·bm−(x%bm)·(wb+bm)` 排序取整 + jp |
| **`_warm(n,qp,npv)`** | L131-173 | `_eco_do_warm` | ★ congruent 反解 + δ 增减 |
| **`_auction_round(hop2)`** | L182-231 | `_eco_do_auction_round` | ★ 含 two-hop 的 best_out 预计算 |
| **`_accept(offers)`** | L233-265 | `_eco_do_accept` | ★ win2/drop 冲突消解 + 前缀接受 |

### 1.2 结构层（.cu）→ CuTe DSL
`.cu` 的函数划分、循环结构、smem 布局可直接照搬，只换语法。
`__shared__` 数组 → `__init__` 字节偏移 + `self._eco_s/_eco_set`；`bar.sync` → `cute.arch.sync_threads()`；
576 线程（16 plan warp + 2 index warp）→ **256 线程，一个线程管 1 专家或 1 槽**。

## 第 2 步：五处必须逐字照搬（验收重点）

### 2.1 warm start 的 congruent 规则（L142）
```python
own = next((j for j in range(len(cs)) if qs[j] >= remp and (qs[j] - remp) % bm == 0), -1)
u[j] = (qs[j] - (remp if j == own else 0)) // bm + (j == own)
```
★ own == -1（上轮 plan 与本轮 counts 不一致）⇒ 该专家整块丢弃（u 保持 0）。

### 2.2 δ 增减（L158-173）
```python
delta = k[e] - sum(u)
while delta > 0:   # 加给 (t0[rank]+add[j]*unit, j) 最小者
while delta < 0:   # 从 u[j] > 0 中 (t0[rank]+add[j]*unit, -j) 最大者减
if own < 0 or u[own] == 0:   # 零头换到 (t0[rank], -j) 最大者
```
`unit = wb + bm`。

### 2.3 two-hop（L186-231）
```python
best_out[r1] = (key, d1, in2, ip, s1b, s2)   # 每卡只留 key=t[rank[s2]]+in2 最小者
if rank[s2] != r0 and phy[s1b] != e and int(t[r1]) + in1 + d1 < lim and k2 < lim:
```

### 2.4 `_accept` 的 win2 / drop（L236-244）
win2[transit] = 该中转槽上所有两跳 offer 里 ck=(-t_src, s0) 最小的那条；
某槽既出让(own)又当中转(transit) ⇒ 保留 ck 更小的，另一条进 drop。

### 2.5 前缀接受的两条 break（L257-259）
```python
if cur + c[4] >= o[3]: break
if (cur + c[4] - self.slack) * R > tot: break
```

## 第 3 步：CUDA C++ → CuTe DSL 对照

### 3.1 基础读写
| CUDA C++ | CuTe DSL |
|---|---|
| `p[i]` / `p[i]=v` | `cell(base, Uint64(off)+i*Uint64(4), Uint32)[0]` |
| `__shared__ int y[P]; y[s]` | `shared_cell(slab, Uint32(off)+s*Uint32(4), Uint32)[0]` |
| `atomicAdd(&T[r], c)` | `atom_shared_add_u32(slab + off + r*4, c)` |
| `__syncthreads()` / `bar.sync` | `cute.arch.sync_threads()` |
| `threadIdx.x` | `tid, _, _ = cute.arch.thread_idx()` |
| `blockIdx.x` | `block, _, _ = cute.arch.block_idx()` |
| `for (i=0;i<n;i++)` | `for i in cutlass.range(n)` |
| 编译期 N | `for i in cutlass.range_constexpr(N)` |
| grid-stride | `for i in cutlass.range(tid, n, self.threads)` |
| `(uint32_t)x` / `(uint64_t)x` | `Uint32(x)` / `Uint64(x)` |
| `x >> 16` / `x & 0x7FFF` | `x >> Uint32(16)` / `x & Uint32(0x7FFF)` |

### 3.2 ★ 三个不能照抄的点
1. **没有 Uint16**（SM90 只导入 Boolean/Float32/Int64/Uint8/Uint32/Uint64）：把两个 int16 当 u32 读，按 k 奇偶取半字，`k` 必须 `range_constexpr`。
```python
word = cell(base, Uint64(64) + Uint64(e * Uint32(32) + Uint32(k)) * Uint64(2), Uint32)[0]
raw = (word & Uint32(0xFFFF)) if (k % 2) == 0 else (word >> Uint32(16))
```
2. `__shared__` 必须手算字节偏移。
3. 成本是整数运算，全程 Uint32。
★ copies 步长固定 32（`copies[E][32]`），循环上界才是 `eco_max_copies`（建议 4）。

### 3.3 线程数 576 → 256
★ `_accept` 的接受段必须单线程串行（`if tid == 0:`）。

## 第 4 步：SM90 适配清单
### 4.1 role 编号：plan 用 **role 5**（role 4 是 shared expert）。
### 4.2 `__init__` 里 smem 偏移
```python
_p = self.ranks * self.local          # = P
_q = max(self.eco_logical, 1)         # = E
_r = self.ranks
self.eco_y_off      = 0
self.eco_part_off   = self.eco_y_off      + _p * 4     # E
self.eco_rem_off    = self.eco_part_off   + _q * 4     # E
self.eco_k_off      = self.eco_rem_off    + _q * 4     # E
self.eco_n_off      = self.eco_k_off      + _q * 4     # E
self.eco_pn_off     = self.eco_n_off      + _q * 4     # E
self.eco_x_off      = self.eco_pn_off     + _q * 4     # P
self.eco_g_off      = self.eco_x_off      + _p * 4     # P
self.eco_u_off      = self.eco_g_off      + _p * 4     # P
self.eco_add_off    = self.eco_u_off      + _p * 4     # P
self.eco_pq_off     = self.eco_add_off    + _p * 4     # P
self.eco_phy_off    = self.eco_pq_off     + _p * 4     # P
self.eco_act_off    = self.eco_phy_off    + _p * 4     # P
self.eco_T_off      = self.eco_act_off    + _p * 4     # R
self.eco_T0_off     = self.eco_T_off      + _r * 4     # R
self.eco_bo_off     = self.eco_T0_off     + _r * 4     # R*7
self.eco_wn_off     = self.eco_bo_off     + _r * 7 * 4 # P
self.eco_dp_off     = self.eco_wn_off     + _p * 4     # P
self.eco_ac_off     = self.eco_dp_off     + _p * 4     # P
self.eco_nof_off    = self.eco_ac_off     + _p * 4     # 1
self.eco_nacc_off   = self.eco_nof_off    + 4          # 1
self.eco_of_off     = self.eco_nacc_off   + 4          # P*13
self.eco_cum_off    = self.eco_of_off     + _p * 13 * 4  # E*(R+1)
self.eco_smem = (self.eco_cum_off + _q * (_r + 1) * 4 + 1023) // 1024 * 1024
self.eco_slack = self.eco_wb + self.max_m + self.eco_wa
self.eco_cold_rounds = 6
```
★ P=320,E=256,R=8 时 eco_smem ≈ 46 KB；assert <= 200 KB。**不要引入 Uint16。**

### 4.3 layout 6 个 view（在 `expert_arrive` 之后 / 本仓库实际位置见实现）
```python
reserve('eco_ready', (2,), 'int32', 4)
reserve('eco_quota', (experts,), 'int32', 4)
reserve('eco_quota_start', (experts,), 'int32', 4)
reserve('eco_counts', (ranks * experts,), 'int32', 4)
reserve('eco_meta', (16 + experts * 16 + experts + ranks,), 'int32', 4)
reserve('eco_state', (2 * experts,), 'int32', 4)
```

### 4.4 config 7 字段
`eco_plan / eco_logical_experts / eco_max_copies(4) / eco_wb / eco_wa / eco_rounds(1) / eco_cold_rounds(6)`

### 4.5 role 5 分支
```python
elif cutlass.const_expr(self.role == 5):
    alloc = cutlass.utils.SmemAllocator()
    slab = cute.recast_ptr(alloc.allocate(self.smem, 1024), dtype=Uint8)
    self._eco_plan(base, peers, rank, tokens, slab)
```
★ `__call__` 里 role 5 用 `self.eco_smem` 与 `grid=(1,1,1)`。

### 4.6 `_combine` 里清 ECO 标志（`expert == 0` 时清 `eco_ready[0]`、`eco_ready[1]`）

## 第 5 步：dispatch 侧必须新写
### 5.1 SM100 是 token cursor；SM90 是 1 CTA/专家 ⇒ 用 CTA 编号分段。
### 5.2 判据
```
num_fixed = EcoMeta.num_fixed[rank]
slot <  num_fixed ⇒ 单副本：lo=0, hi=total        【不等 plan】
slot >= num_fixed ⇒ 副本槽：lo=quota_start[slot], hi=lo+quota[slot]  【先等 plan】
```
### 5.3 CTA 编号分段
```python
if block >= 1 and block <= local:
    slot_idx = block - 1
    replicated = slot_idx >= num_fixed
    if replicated:
        if tid == 0: wait_epoch(eco_ready + 1, epoch, False)
        cute.arch.sync_threads()
    self._pull_eco_slot(..., slot_idx, replicated)
```
★ `__init__` 必须校验 `grid >= local + 1`。

### 5.4 ★★★★ 唤醒链
```
role 1 (dispatch)                                   role 5 (plan)
① 清 send_count，会合 #1（本地 11/12）
② push 全部 route 到该逻辑专家的【每个活跃副本】的 rank
③ 会合 #2（本地 13/14，fence_sys，发布 header[1]）
④ 等所有 peer 的 header[1]
⑤ 写 eco_counts[rank][e] = send_count[e]；fence_sys
⑥ 本地会合 #3（22/23）
⑦ ★ st_release(header[24], epoch, sys) + wait_epoch(peer.header[24], sys)
⑧ ★ st_release(eco_ready[0]) → plan: wait → 静态表 → D_e/cumsum(读所有 peer) → cold/warm → quota → st_release(eco_ready[1])
⑨ 副本段 CTA：wait_epoch(eco_ready[1])
```
少 ⑦ ⇒ 各 rank D_e 不同 ⇒ 错数不报错；少 ⑧ ⇒ 死锁。

### 5.5 plan 必须读【所有 rank】的 counts 行（否则 D_e 只有 1/R）。
### 5.6 push 必须写进每个活跃副本（单副本专家 copies 只有一项，自动一份）。

## 第 6 步：EcoMeta 布局
```
int32 [0,16)                     header（不用）
int32 [16, 16+E*16)              copies[E][32] ← int16，0x7FFF=空，bit15=passive
int32 [16+E*16, +P)              phy2log[P]
int32 [16+E*16+P, +R)            num_fixed[R]
```

## 第 7 步：wb/wa 必须重新标定（B300 的 {64:(260,111)} 不能用于 H20）
先跑起来：`eco_wb = bm 或 8`、`eco_wa = 0`、`eco_rounds = 1`、`eco_cold_rounds = 6`、`eco_max_copies = 4`。

## 第 8 步：9 个已知坑
1. eco_counts_ready 从未 release ⇒ plan 死等。
2. 视图名不一致（统一 `eco_ready + 0/+1`）。
3. `_eco_ncopies` 缺 slab 参数。
4. `_eco_do_accept` 的 tot 传 0 ⇒ 条件失效。
5. 倒序 range 不编译/死循环 ⇒ 选择排序。
6. warm 的 pq[]/pn[] 没从 eco_state 填 ⇒ 退化成 cold。
7. 某专家无活跃副本时 break ⇒ token 消失。
8. copies 用 Uint16 读 ⇒ 不存在。
9. rem 必须按专家索引（`rem[phy[s]]`）。

## 第 9 步：实现顺序
静态表 + D_e → cold → quota + 守恒 → one-hop auction → accept → warm → two-hop（EP8 可省）。
每一步都和 unit_ref 的同名函数对拍。

## 第 10 步：验收
- 阶段 1：role 5 cubin 能编译启动；无 wait_epoch 超时；守恒 `Σ_k quota[copies[e][k]] == n[e]`。
- 阶段 2：V0 eco_plan=False 与 baseline bit-identical；V1 E==P 仍 bit-identical；V2 双副本 quota 和 = token 数、quota_start 连续；V3 8 rank 的 eco_quota 一致；**V4 unit_ref 与 kernel 逐位相同（最强）**。
- 阶段 3：plan 耗时、warm vs cold、wb/wa 标定。

## 第 11 步：不要做的事
- 不要用 SM100 的 wb/wa；不要照搬 token cursor；不要把 _warm 换成按比例缩放；
- 不要去掉 _accept 的两条 break；不要把 rem 按槽索引；不要假设副本数相同；不要一次写完再调。
