# 源码来源与改动清单

## 目录结构

```
source/        gc_moe 包（两臂共用同一份）
integration/   vLLM 后端集成（deepep_compare.py）
patches/       vLLM 兼容补丁
scripts/       驱动与启动脚本
config/        experiment.env
docs/          SPEC / SOURCE / RESULTS / REPRODUCE
runs/          结果 JSON
logs/          运行日志
```

## source/ 的基线

`source/` = `gc_moe_compare_20261104/gc_moe_eco_repo`，其上游是
`Bruce-Lee-LY/gc-moe` 的 `3d520f2`，再叠加：

1. **eco 分支**（`Natureal/gc-moe:eco`）：sm100/Blackwell 的 ECO，sm90 路径未使用
2. **EPLB（B 臂方案）**：从 `eplb_deepinfer_isolated_20261022/source/rtp_llm/eplb/deepinfer_periodic/` 移植

## 与原始 GC MoE 的 kernel 差异（完整清单）

`gc_moe/impls/sm90_fp8_fp4_gc_moe.py` —— **只有一处实质改动**：

```python
# 原版
self.ranks, self.local = config.ranks, config.experts // config.ranks   # 48
# 现在
self.logical_local = config.experts // config.ranks                     # 48
self.physical = int(getattr(config, 'physical', 0) or config.experts)   # 424
self.local = self.physical // config.ranks                              # 53
```

其余 diff 全部是 import 换行格式。**dispatch / compute / combine 三个 role
的循环体一行未改**，只是 `local` 从 48 变成 53。

`gc_moe/layout/sm90_gc_moe.py`：
```python
local = experts // ranks            ->  local = physical // ranks
reserve('send_count', (experts,))   ->  (physical,)
```
缓冲区随槽位数扩容，并新增 `slot_logical` / `expert_slots`。

`gc_moe/apis/sm90_gc_moe.py`：权重形状校验 `local`(48) → `phys_local`(53)。

## 从 DeepInfer/RTP-LLM 移植的文件

| 本目录文件 | 原始文件 | 用途 |
|---|---|---|
| `source/gc_moe/eplb/policy.py` | `deepinfer_periodic/policy.py` | `rebalance_experts_global`（复制专家 + 贪心装箱） |
| `source/gc_moe/eplb/deepinfer_periodic/shuffle.py` | 同名 | `WeightShuffler`（symm_mem IPC / NCCL P2P） |
| `source/gc_moe/eplb/deepinfer_periodic/config.py` | 同名 | `initial_layout` / `resolve_num_extra_experts` |

`runtime.py` **未采用**：它绑定了 RTP-LLM 的 `DeepGemmMaskedExecutor` 和
`DeepEpLowLatencyRouter`，无法用于 GC MoE。周期驱动逻辑在本目录
`integration/deepep_compare.py::rearrange_now` 里重写。

## 集成层改动（integration/deepep_compare.py）

| 位置 | 改动 |
|---|---|
| `_install_eplb` | 用 B 臂的 `initial_layout` + `install_eplb_plan` + `gather_slot_weights` 建立 53 行权重视图，并构造 `WeightShuffler` |
| `apply`（路由） | 逻辑 id → 物理槽位，`pick = arange(n) % count[id]`（RR） |
| `rearrange_now` | `policy.py` 重算 → rank0 广播 → `WeightShuffler.move` → 重指路由 |
| `_launch` | 与 vanilla 臂形状相同（**不再传 `replica_plan`**） |
| `ir_begin/ir_end` | 从 kernel 的 `expert_count` 读每 rank 负载，算 IR |

## vLLM 集成层的重排路径

`integration/deepep_compare.py::rearrange_now`，与 B 臂
`deepinfer_periodic/runtime.py::_rearrange` 同构：

1. 各层 `stats`（`[layers, per_rank]`）→ 全局物理槽位 → 逻辑专家 demand；
2. `all_reduce` 后 rank0 跑 `rebalance_experts_global`，广播新 layout；
3. `WeightShuffler.move()` 跨 GPU 搬 6 个张量/层；
4. `install_eplb_plan` 重指路由。

### 本轮修掉的两个真 bug

| # | 位置 | 症状 |
|---|---|---|
| 1 | `rearrange_now` | `stats` 是 `[53]` 却用 `[424]` 索引 → `index [424] vs self [384]` 崩溃。改为 `[layers, per_rank]` + 先按全局槽位展开 |
| 2 | `shuffle.py::_ipc_layer` | **同 rank 源槽位会被后续写入覆盖**，读到脏数据。改为同 rank 从快照 `self.buffers` 读 |
| 3 | `_LAYER_OF` 的键 | 用 `data_ptr` / 无强引用的 `id()` 做键时会**碰撞**（allocator 复用地址、对象 id 回收）→ 多个层写同一行 stats，测出 tokens/rank 虚高 17 倍。改为 `id(payload)` + `_LAYER_REFS` 持强引用 |
| 4 | `meas_step` | 读 counters 前未 `torch.cuda.synchronize()`，`step()` 是异步提交 → 读到上一/半写状态。已加同步 |
| 5 | `meas_begin` | 未清零 counters，warm 期的累计值污染第一个测量步。已加清零 |

### 验证

**CPU 回归测试**（`tests/test_eplb_logic_cpu.py`，2 秒，无需 GPU）：

```
T1 layer index: 40 unique for 40 layers
T2 demand: sum=240 (expect 240)
T3 shuffle reference: bad rows=0/424
T5 distinct layer rows: 40/40
CPU_TEST PASS
```

**GPU 单测**（`torchrun -n8`，约 1 分钟，不加载模型）：

```
PRE_REARRANGE  rel=0.000e+00                 重排前 EPLB == vanilla
LAYER0_CHECK   bad rows=0/53                 重排后每行权重正确
SLOTMAP        bad entries=0                 slots_of 指向正确
POST_REARRANGE cos=1.00000012 rel=0.000e+00  重排后数值完全一致
```

## 曾经走错的路（不要重复）

1. **`gc_moe_src` 的 `replica_plan` 方案**：kernel 内 water-filling 切 token，
   在 vLLM 的 Green Context（6 dispatch SM + 72 compute SM）下
   `_plan_replica` / `_pull_range` 的 rendezvous 错位，**永久死锁**。
   裸 torchrun 能过，vLLM 里不行。
2. `build_eplb_layout`（另一套放置算法）：不是 B 臂的算法。
3. 自己写的 `_reshuffle`：B 臂用的是 `shuffle.py::WeightShuffler`。
