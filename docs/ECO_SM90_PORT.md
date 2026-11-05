# ECO → SM90：移植记录（2026-11-05）

把 SM100 的 ECO plan（`eco/unit_ref.py`，唯一算法权威）完整移植到 SM90 CuTe DSL 路径。
不是重新设计：`eco_ref/plan_port.py` 是逐行 Python 移植，`eco_ref/test_kernel_full.py`
是 kernel 控制流的 CPU 仿真，两者都与 `unit_ref.py` 逐位对拍；kernel 再与它逐位对拍。

## 1. 已落地的文件（全部带 `[ECO-SM90]` 注释标记）

| 文件 | 改动 |
|---|---|
| `source/gc_moe/impls/sm90_fp8_fp4_gc_moe.py` | role 5：`_eco_plan` / `_eco_do_cold` / `_eco_do_warm` / `_eco_do_auction_round` / `_eco_publish_*`；dispatch 侧逻辑 id 路由 + 每个活跃副本 push + 唤醒链；pull 侧按 quota 切分 |
| `source/gc_moe/layout/sm90_gc_moe.py` | `eco_ready / eco_quota / eco_quota_start / eco_counts / eco_meta / eco_state` 六个 view（`eco=False` 时字节布局不变） |
| `source/gc_moe/apis/sm90_gc_moe.py` | 7 个 config 字段（`GC_MOE_ECO_*` 环境变量） |
| `source/gc_moe/jit_kernels/impls/sm90_fp8_fp4_gc_moe.py` | 编译 role 5（256 线程、`eco_smem`、`grid=1`） |
| `source/gc_moe/jit_kernels/heuristics/sm90_gc_moe.py` | ECO 下 routed grid 让出一个 SM 给 plan |
| `source/gc_moe/runtime/sm90_gc_moe.py` | role 图允许 role 5 |
| `source/gc_moe/eco/meta_sm90.py` | host 侧 EcoMeta（`copies[E][32]` int16 / `phy2log[P]` / `num_fixed[R]`） |
| `integration/deepep_compare.py` | `GC_MOE_ECO=1`：buffer 开 ECO 区、装 EcoMeta、`topk_idx` 保持逻辑 id、重排后重装 meta |
| `scripts/run_arm_eco.sh` | 三臂启动脚本（vanilla / eplb-rr / eplb-eco） |

## 2. 验收结果

### V4：kernel vs `unit_ref.py`（GPU，逐位）
`scripts/gpu_plan_only.py`：8 rank 同时跑 role 5，读真实 `eco_counts`，与 `UnitPlanner.plan()` 比较。

| 配置 | cold | warm |
|---|---|---|
| 3 seeds × {(rounds=1,wb=64,wa=0), (2,64,32), (3,0,0)} | 全部 `equal=True` | 全部 `equal=True` |

9 组 × 2 轮全部逐位一致，且 `Σ quota[copies[e]] == n[e]`、`quota_start` 连续。

### 端到端（vLLM，8×H20）

| 臂 | ms/step | IR | tokens/rank |
|---|---:|---:|---:|
| vanilla（无副本，无 plan） | 20.25 | 1.0284 | 32 |
| eplb + host RR | 20.51 | 1.0672 | 32 |
| eplb + ECO plan | 21.99 | **1.0358** | 32 |

（同目录 `gcmoe_eplb_ep8_20261104` 的对照：vanilla 80.45 ms / EPLB-RR 73.77 ms，
那是 32 seq/rank 的规模，与本轮 4 seq/rank 不可直接比较。）

### 正确性（`scripts/verify_eco_ep8.py`，无模型，8 GPU）
```
ECO_EP8 finite=True cos=1.00000000 rel=0.000e+00 conserved=True
        quota_start_contiguous=True plan_identical=True hot_copies_used=3/3 OK
```
即：ECO 臂的输出与「无副本」参考路径**逐位相同**，热专家确实被拆到 3 个副本上，
8 个 rank 各自算出的 plan 完全一致。

## 3. 移植中修掉的真实 bug（都是静默错数，不报错）

| # | 问题 | 症状 |
|---|---|---|
| 1 | `copies` 的 passive 位没过滤 | `_eco_ncopies` 把 passive 副本算成活跃副本，cold 把 expert 切成 3 份而不是 2 份 |
| 2 | `P=480 > 256` 线程 | `if tid < P:` 只覆盖前 256 个槽，其余槽的 phy/y 全是 0 |
| 3 | 冷启动键用 Uint32 | `t0*bm - (x%bm)*(wb+bm)` 常为负，回绕成巨大值，排序取整选错副本 |
| 4 | `jp = argmax(g*bm - x)` 用 Uint32 | 同上，零头归属错 |
| 5 | offer 存储下标 `r*S+j` 但分组扫描 `0..nof` | 读到别的槽的 offer，接受数恒为 0 |
| 6 | `gof` 用「源 rank 的 offer 数」当「接收方段基址」 | 多个接收方的段重叠，offer 串台 |
| 7 | `_accept` 用 `break` | DSL 不允许 staged break，改成 `stopped` 闸门 |
| 8 | `head + 24 * 4` | `head` 是 int32 指针，元素算术；写成字节偏移会越界覆盖 `peer_table` |
| 9 | plan CTA 与 persistent routed role 抢 SM | 整个调用死锁（8 卡 100% 不退出） |
| 10 | pull 侧 `total` 与 `lo/hi` 不是同一轮读 | 副本 pull 区间错，token 丢失 |
| 11 | passive 副本也参与 pull | 2 个专家的 token 被拉两次（多 16 token，cos 0.99） |

## 4. 与 SM100 的差异（必要，不是简化）

1. **role 5**（SM90 的 role 4 是 shared expert）。
2. **256 线程**，一个线程管 1 个专家或 1 个槽；所有 `P=480` 的循环都是 grid-stride。
3. **plan 的 SM**：routed role 是 persistent 的，每个 CTA 占满一个 SM。
   plan 必须与它同时在跑，所以 routed grid = `compute_sms - 1`（SM100 同样
   「compute 少起一个 cluster，把 SM 让给 plan」）。实测：留半个 SM 都还会死锁。
4. **two-hop = 0**：EP8 是完全图，一跳必然可达；SM100 的 `eco_unit_config`
   在 ≤32 rank 时也是 `hop2 = 0`，因此这里没有省略任何有效路径。
5. **`eco_state` 用 int32**（SM100 用 uint16，SM90 DSL 没导入 Uint16）。

## 5. wb / wa

SM100 的 `ECO_TIME_WEIGHTS` 是 B300 拟合值，不能用于 H20。本轮默认
`wb = 0, wa = 0`（token 主导目标），这也是 SM100 的 `GC_MOE_ECO_OBJECTIVE=token`
分支。标定方法（未做，时间紧）：

1. 固定 token，扫块数 → 斜率 = wb
2. 固定 token 与块数，扫活跃槽数（`{65}` vs `{64,1}`）→ 斜率 = wa
3. 归一化，使 token 系数为 1

## 6. 复现命令

```bash
# CPU（秒级）
for f in eco_ref/test_*.py tests/test_eplb_logic_cpu.py; do \
  <env>/bin/python $f; done

# GPU：只跑 plan（~20s，无模型）
source scripts/env.sh
export GC_MOE_SRC=$PWD/source PYTHONPATH=$GC_MOE_SRC:$GC_SHIM:$PWD/integration
torchrun --nproc_per_node 8 scripts/gpu_plan_only.py

# GPU：端到端正确性（无模型，~30s）
torchrun --nproc_per_node 8 scripts/verify_eco_ep8.py

# vLLM 端到端
SEQS_PER_RANK=4 WARM_STEPS=100 MEASURE_STEPS=100 ./scripts/run_arm_eco.sh 1 1 eplb_eco_100
SEQS_PER_RANK=4 WARM_STEPS=100 MEASURE_STEPS=100 ./scripts/run_arm_eco.sh 0 0 vanilla_100
```

## 7. 代码位置索引

| 功能 | 位置 |
|---|---|
| plan 主编排（D_e → cold/warm → rounds → 发布） | `impls/sm90_fp8_fp4_gc_moe.py::_eco_plan` |
| cold（均分 / 排序取整 / jp） | `_eco_do_cold`（+ `_eco_cold_key`） |
| warm（congruent 反解 + δ 增减） | `_eco_do_warm` |
| auction（候选 / 报价 / 分组 / 前缀接受） | `_eco_do_auction_round` |
| 成本函数 | `_eco_costC` / `_eco_out` / `_eco_in` / `_eco_move` |
| EcoMeta 解码 | `_eco_copy` / `_eco_slot_of` / `_eco_ncopies` / `_eco_phy` |
| dispatch 侧（逻辑 id → 每个活跃副本） | `_dispatch` 里 `cutlass.const_expr(self.eco)` 分支 |
| 唤醒链（header[24] 跨 rank + `eco_ready[0/1]`） | `_dispatch` 尾部 + `_eco_plan` 开头/结尾 |
| pull 侧切分（含 passive 跳过） | `_dispatch` 的 `lo/hi` 段 + `_eco_is_active` |
| role 5 资源与放置 | `jit_kernels/impls/sm90_fp8_fp4_gc_moe.py::_role_specs`、heuristics、runtime |

## 8. 已知限制

- `eco_max_copies = 4`：EPLB 5% 冗余时热点专家最多 4 副本，够用；
  若把热点需求拉到 10×，一个专家会有 12 个副本，必须同步提高这个上限
  （kernel 的 per-expert 寄存器数组长度同样是 4）。
- EPLB 布局用均匀需求 → 副本分布比较均匀，ECO 的可优化空间有限；
  IR 1.0358 说明这个 batch 下负载本来就接近均衡。
- 未标定 wb/wa，plan 目前优化的是「最忙 rank 的 token 数」。
