# 交接：GC-MoE 单层 trace 时间线（第二轮）

> 上一份交接文档：`HANDOVER_TRACE_TASK.md`（任务书）+ `MY_FAILURE_AUDIT_AND_HOW_TO_CORRECT.md`
> （前任自述的 12 个失误）。本文只写**我在那份基础上做了什么、发现了什么、留下什么**。
>
> **读者的分工**：本文之后，代码不再由我修改。我只按任务书执行八卡实验。

---

## 0. 一句话总结

前任的两份文档把我引到了正确的**方法**（按 trace 行号取同一次调用、别信相关性命中），
但也留下了一个**没被验证的假设**：`pull_tl` 的打点与 torch trace 的调用可以按位置直接配对。
我按行号重做后发现这个假设在 `FINAL_V2` 上**根本不成立**（两者的 epoch 空间、窗口长度都不同），
于是给出三条可量化的结论 + 一套能自证的对齐协议，并把对齐所需的 epoch 记录补进了内核。

---

## 1. 我沿用（确认有效）的部分

| 前任的结论 | 我的验证 |
|---|---|
| 一次 forward = 5 个连续 gc_moe kernel 行，`call = row // 5` | 8 rank × 39680 call，period-5 角色模式 **0 mismatch** |
| 按 role 数组下标取第 k 项是错的 | 复现了：`/tmp/k11` 的 5 个数组同下标相差可达 53 s |
| 角色几何：24×256 / 71×256 / 71×384 / 1×512 / 288×384 | 逐 rank 断言通过 |
| 缺 `wait token` 段 | 确认必须单列；但**起点定义前任也弄错了**（见 §3） |
| 别用相关性猜 step/layer | 我用了 3 次相关性，3 次都被判据否掉（见 §4） |

---

## 2. 我在前任基础上修正的第一个错误：`ts` 和 `dur` 的单位

**前任的图/表把 `dur` 当成 µs、`ts` 当成 ns 混用**，导致物理约束全部算错（我一开始也踩了）。

**判定方法（30 秒，可复用）**：同一个 call 内
```
routed.ts − shared.ts   ÷   shared.dur   == 1.005   （8 个 call 全部如此）
```
两者比值恒为 1.005 ⇒ **`ts` 与 `dur` 同单位，都是 µs**。

改用一致单位后，交接文档给的 5 条硬约束全部落进预期区间：

| 约束 | 预期 | 实测（8 rank） |
|---|---|---|
| shared − dispatch | −2…+3 µs | **−1.06…+1.79** |
| routed − shared_end | >0，+0.9…+1.2 | **+0.86…+1.28**（8/8 为正） |
| combine − routed_end | +2.6…+4.5 | **+2.24…+3.52** |
| count_exchange 末 == pull 起 | 无缝 | **0.000000 µs** |
| wait+GEMM == routed 总时长 | 恒等 | **0.000000 µs** |

---

## 3. 修正的第二个错误（**用户指出**）：`GEMM` 的起点

前任和我最初都把 `wait token` 的终点定在 `pull_tl[5]` = **最后一个本地槽拉完**，
于是"GEMM 要等 pull 结束才开始"，出现"某些 rank 在 pull 结束后很久才开始算"的荒谬现象。

**正确语义（源码为证）**：
```python
# pull_tl[15] —— _pull_tl_min_first 的 docstring
"""[PHASES] "first compute CTA got a task" stamp, min-reduced."""
```
- `[15]` = **第一个 compute CTA 真正领到任务** → **GEMM 起点**
- `[5]`  = 最后一个本地槽拉完 → dispatch 结束（不是 GEMM 起点）

**实测差多少**（同一 call，8 rank）：

| | 数值 |
|---|---|
| `[15] − [10]` 首个任务 | **196–204 µs** |
| `[5] − [10]` 全槽拉完 | **254–597 µs** |
| 差 | **51–400 µs（均值 190）** |

⇒ **GEMM 与 pull 并行 51–400 µs**，`wait token` 只剩 **−0.1…+12.3 µs**
（负值 = 首个任务在 routed kernel 启动时已就绪，即无等待）。

修正后的段耗时（8 rank 均值，µs）：

```
count exchange 19.9 | pull token 369.5 | shared 194.0 | ECO plan 108.7
wait token  5.1 | GEMM 659.8 | combine 292.2
```

> `[15]` 与 routed 起点来自两个时钟域（kernel globaltimer vs CUPTI），
> 相减在 256 个 call 上散布 **−14…+16 µs，中位 +3.8**。
> 因此单 call 的 GEMM 起点可能微负；**原始值保留在 `checks`，只有画图时 clamp 到 0**。

---

## 4. 第三个（也是最关键的）发现：三种数据源的窗口**互不重叠**

这是 `FINAL_V2` 这份归档数据的硬事实，不是脚本 bug：

| 数据源 | 环深度 | 覆盖 |
|---|---|---|
| torch trace | 无环 | 39680 call = **992 步** |
| `pull_tl` | **256** | 最后 256 call = **6.4 步** |
| `ir_step.tables` | 1024 步 | 前 **1000 步** |
| `fine_rows` | 524288 记录 | 82 个 epoch = 测量窗口**开头** |

`FINAL_V2` 实测：
```
ir_step window : epoch 47081 .. 87080
pull_tl window : epoch 87385 .. 87640      ← 差 304 epoch
overlap        : NONE
```

**后果**：任何一个 call，最多只能拿到其中两项。
- 有 `pull_tl` 的 call（最后 256 个）→ 7 段齐全，但 **`ir_step` 表里没有它的 token/M 块数**
- 有 `ir_step` 的 call（前 1000 步）→ 有 token/M 块，但 `pull_tl` 已被覆盖

⇒ **这正是"能不能同时拿到 7 段 + token/M 块"这个问题的答案：在 `FINAL_V2` 上不能。**

**我也验证了前任 `ONE_CALL.md` 的 call（行 39605–39609）落在这个区间之外**
（call 7921，epoch 不在 87385–87640 内），所以它表里的 `pull`/`waittok` 数值
**不可能来自 `pull_tl` 打点**，只能是插值。本文档的图改用 call 39679（环内）。

### 我试过、但被自己的判据否掉的匹配方法（记录以免重复）

| 方法 | 结果 | 判据 |
|---|---|---|
| role 数组同下标 | 相差 53 s | 直接否 |
| `pull` 窗口 vs `dispatch dur` 相关 | 最佳 0.758 / 次佳 0.756 | 差距 0.0024 << 5%，**无效** |
| rank 画像 L1 匹配 | 最佳 0.129 / 次佳 0.148 | 差距 13%，**无效** |
| 80-call 整段指纹 | 最佳 18.1 µs / 次佳 18.8 µs | 差距 0.7 µs，**无效** |
| M 块数向量 vs routed 时长 | 相关 0.23 / 0.015 | 太弱，**无效** |

> 结论：**窗口不重叠时，任何统计匹配都不该被采信。** 必须让窗口本身重叠。

---

## 5. 我为解决对齐而做的三个代码改动（已落盘、已编译通过）

### 5.1 `ir_step_epoch`：让内核自己记录 epoch（核心）

**问题**：`irstep_begin()` 在 host 端写 `epoch0 = buf.header[0]`，
而 CUDA-graph 重放下 host 读到的是**冻结值**。实测偏差 225 epoch（=5.6 步）。

**改法**（3 处，最小侵入）：

| 文件 | 改动 |
|---|---|
| `source/gc_moe/layout/sm90_gc_moe.py` | `reserve('ir_step_epoch', (40*1024,), 'int32', 4)` |
| `source/gc_moe/impls/sm90_fp8_fp4_gc_moe.py` | `_ir_step_write` 里 `ir_step_epoch[idx] = epoch`（普通 store，非 atom） |
| `integration/deepep_compare.py` | `irstep_end()` dump `rows_epoch` |
| `scripts/real_decode_100_100.py` | `tables['epochs'] = rows_epoch` |

这样 `ir_step` 的每个 (layer, step) 单元格都带着**内核写入的 epoch**，
与 `pull_tl` 的 `epoch % 256` 行号可以**精确 join**，不再依赖 host 读。

### 5.2 `irstep_begin()` 的锚点改读 `eco_tl_step`

原代码：`mark[0] = buf.header[0]`（stale），而紧接着又算出 `_epoch_now = eco_tl_step`
却只用于 fine_floor。已改为 `mark[0] = _epoch_now`。
（**注意**：实测这个改动**没有解决**根因——`eco_tl_step` 在 host 侧同样可能 stale，
所以真正可靠的是 5.1 的内核自记。）

### 5.3 抽取脚本按 4-role / 5-role 自适应

`GC_MOE_SHARED_IN_GC=0` 时 shared 走 vLLM（marlin），gc_moe 只有 **4 个 kernel**，
不是 5 个。抽取脚本已按 `(grid, block)` 自适应，两种构建都能解析。

---

## 6. 交付物（已推送）

仓库 `github.com/ehdbdns/gcmoe-sm90`，目录 `docs/final_orig_layer/`：

| 文件 | 说明 |
|---|---|
| `one_layer_call39679.md` | 完整说明：段的精确定义、8 rank 校验、plan 预测、窗口冲突 |
| `one_layer_call39679.perfetto.json` | 8 rank × 7 段 = **56 段**，可直接拖进 ui.perfetto.dev |
| `one_layer_call39679.png` | 全层时间线（3 条并发泳道 × 8 rank） |
| `one_layer_call39679_zoom.png` | 0–260 µs 放大（看清 pull 与 GEMM 的重叠） |
| `one_layer_call39679.segments.json` | 全部原始数值 + 逐 rank 校验 + clamp 量 |
| `scripts/{extract_tail,build_layer,plot_layer,extract_one_call}.py` | 可复现全流程 |

**该层的 plan 预测**（同 epoch）：预测跨度 5.7% vs 实测 58.5%，L1=1.1867。
原因可读出：rank 4 的 GEMM 942 µs / combine 仅 15 µs，rank 5 相反（547/400），
而 plan 的 cost 只含 GEMM+push+gap，**不含 combine**。

---

## 7. 未完成 / 需要下一轮做的（按优先级）

### 7.1 【必须】跑一次"窗口真正重叠"的采样（我做到一半）

已经确认可行的做法（`GC_MEASURE_FLUSH=0` + 少步数）：

```bash
export GC_MOE_MEASURE_FLUSH=0     # 重排后不再 flush 5 步
--warm-steps 200 --measure-steps 4
```

但**仍有三个坑**（我踩过，下一个执行者不要再踩）：

1. **profiler 只捕获 2 步，不是 4 步**。实测 80 个 call（中间 1 个步边界），
   而 `ir_step` 记了 8 步、`pull_tl` 覆盖 6.4 步。**原因未查清**。
   → 建议先确认 `profiler` 与 `engine.step()` 的计数关系，或改用
   `prof_step` 明确控制。
2. **host 侧 `epoch0` 不可信**（见 5.2）。已用 5.1 的内核自记绕过。
3. **单步耗时会偏大**（实测 423 ms/step vs 正常 86 ms），
   因为 profiler 开销被摊到极少步数上。**不要用这种 run 的 ms/step 做性能结论**，
   只用来取 token/M块/时间线。

### 7.2 待验证：`ir_step_epoch` 是否真的可用

后台最后一次 run（`ALIGNEDE`）就是为了验证这一点：
```bash
python3 /tmp/rt/join_exact.py runs/ALIGNEDE.json prof/alignede
```
**期望输出**：`ring epochs present in ir_step table: 256 / 256`。
若命中数 >0，就可以用 epoch 精确 join，把 7 段 + token/M块 画在同一张 perfetto 里。

若命中数为 0，说明还有一层偏移，**需要再查 `_ir_step_write` 的 `idx` 与
`n_layers` 的 −1 偏移**（源码里 `layer = (epoch + n_layers - 1) % n_layers`）。

### 7.3 采样"重拍后某几层"的正式脚本还没写

需求（用户原话）：*"就采样重拍后的某几层记录 m 块 token 数块，并把这几层画 perfetto 即可"*。
前置条件就是 7.2 通过。之后：
1. 在 `ir_step_epoch` 可 join 的前提下，挑 2–3 个 epoch；
2. 每个 epoch 出 1 张 perfetto（7 段）+ 1 张 PNG；
3. token/M块 从 `ir_step.tables` 按 `epochs` 反查，**不再靠猜**。

---

## 8. 环境与文件位置（下一轮直接用）

```
工作目录  /home/admin/workspace/aop_lab/app_source/gcmoe_exp/t19_costfn
模型      /home/admin/local_models/DeepSeek-V4.1-Flash   (~476 GB, 加载 ~20-25 s)
数据集    /mnt/nas2/nanjun.cp/MoE-Bench-Experiments/datasets/dsv4-flash_online_input_ids_26-10-08_1.jsonl
```

启动模板（已验证可跑）：

```bash
P=/home/admin/workspace/aop_lab/app_source/gc_moe_rtp_ep8_20261031/env3/bin/python
cd /home/admin/workspace/aop_lab/app_source/gcmoe_exp/t19_costfn
source config/experiment.env && source scripts/env.sh
export PYTHONPATH=$PWD/source:$PWD/patches:$PWD/integration
export GC_MOE_EPLB=1 GC_MOE_ECO=1 GC_MOE_EPLB_EXTRA_PER_RANK=8
export GC_MOE_ECO_WB=26 GC_MOE_ECO_WA=0 GC_MOE_MB_CLAIM=1
export GC_MOE_PULL_TRACE=1 GC_MOE_ECO_TRACE=1 GC_MOE_IR_STEP=1 GC_MEASURE_IR_STEP=1
export GC_MOE_MEASURE_FLUSH=0
export GPU_UTIL=0.72
$P -u scripts/real_decode_100_100.py --label X --output runs/X.json \
   --seqs-per-rank 32 --warm-steps 200 --measure-steps 4 \
   --profile-out prof/x
```

一次性约 **9–10 分钟**（prefill ~60 s + warm 200 步 ~15 s + profiler 段）。

### 运行前检查（每次都要）

```bash
nvidia-smi --query-gpu=memory.used --format=csv,noheader   # 必须全 4 MiB
```

**已知环境坑**（沿用前任记录，仍有效）：
- NFS 偶发抖动：`cd` 失败就重试，不要当成 bug；
- `GC_MOE_NO_PREFETCH=1`（否则权重加载 657 s）；
- 杀进程要连 `VLLM::EngineCore` 父进程一起杀；
- 不要用 `rm -f`（本环境策略会拒绝），需要清理时改用覆盖写。

### 后台运行（本环境必须）

```bash
setsid nohup /tmp/rt/run_xxx.sh > /tmp/rt/outer.log 2>&1 < /dev/null &
```
直接 `nohup ... &` 会被回收。

---

## 9. 现存临时文件（可复用，也可无视）

| 路径 | 内容 |
|---|---|
| `/tmp/rt/tail{0..7}.json` | FINAL_V2 尾部 3000 行（600 call）快照 |
| `/tmp/rt5/tail{0..7}.json` | ALIGNED5 全量 gc_moe 行（80 call） |
| `/tmp/rt/FINAL_V2_min.json` | FINAL_V2 的 pull_rows + ir_step 精简版 |
| `/tmp/rt/layout.bak` `/tmp/rt/impl.bak` `/tmp/rt/deepep_compare.py.bak` | §5 改动前的备份 |
| `/tmp/rt/join_exact.py` | §7.2 的对齐验证脚本 |
| `/tmp/rt/run_aligned*.sh` | 三个对齐实验的启动脚本 |

---

## 10. 如果只记三件事

1. **`ts` 和 `dur` 都是 µs**（判定：`routed.ts−shared.ts` ÷ `shared.dur` = 1.005）。
2. **`GEMM` 起点是 `pull_tl[15]`（首个 CTA 领到任务），不是 `[5]`（全槽拉完）**；
   两者差 51–400 µs，GEMM 与 pull 是**并行**的。
3. **`FINAL_V2` 的 trace / `pull_tl` / `ir_step` 窗口互不重叠**，
   所以那份归档数据里"7 段 + token/M 块"不可能同时拿到；
   必须重跑，且要用内核写入的 `ir_step_epoch` 做 join，不要用 host 读的 `epoch0`。
