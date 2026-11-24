# 【SM90】REPLY_SM90_H20_SHARED_PAIR_CPU — fused 慢 6.34% 全在「GC graph 起点→首个 routed CTA」

> 本回执由 **SM90 执行者**（8×H20）提交。SM100 任务书不在本执行者范围内。
> 纯 CPU 只读诊断，**未占用 GPU、未新增实验**，正在运行的八臂未被打断。

## 0. 结论

| 指标 | fused（arm2）| unfused（arm3）| 差值（fused−unfused）| 百分比（分母 unfused）|
|---|---:|---:|---:|---:|
| **整步 decode（host）** | 92.909 ms/步 | 89.681 ms/步 | **+3.228 ms** | **+3.60%** |
| **完整 FFN（设备）** | 47.430 ms/步 | 44.601 ms/步 | **+2.829 ms** | **+6.34%** |

**FFN 差距的构成（六段，ms/步，五项之和 = FFN 差值）**：

| 段 | fused | unfused | 差值 | 占 FFN 差值的比例 |
|---|---:|---:|---:|---:|
| FFN 起点 → GC graph 起点 | 5.335 | 7.127 | **−1.792** | −63.3% |
| **GC graph 起点 → 首个 routed CTA** | 6.223 | 1.163 | **+5.060** | **+178.9%** |
| 首个 routed CTA → 末个 routed CTA | 27.143 | 26.914 | +0.229 | +8.1% |
| 末个 CTA → combine 结束 | 6.969 | 7.660 | −0.690 | −24.4% |
| combine 结束 → FFN 结束 | 1.760 | 1.737 | +0.022 | +0.8% |
| **合计** | **47.430** | **44.601** | **+2.829** | **100%** |

**即：fused 的全部劣势（以及更多）集中在 `GC graph 起点 → 首个 routed CTA` 这一段，+5.060 ms/步。**
其余各段 fused 反而更快或基本持平（前段 −1.79、combine 段 −0.69）。

## 1. 逐对一致性（320 个 layer×rank 配对，非均值掩盖）

| 段 | 均值 | 中位数 | 最小 | 最大 | 符号一致 |
|---|---:|---:|---:|---:|---|
| FFN 总跨度 | +70.73 µs/层 | +70.85 | +65.49 | +75.95 | ✅ **全为正** |
| before_gc | −44.81 | −44.81 | −46.55 | −42.99 | ✅ **全为负** |
| **gc_to_first** | **+126.69** | +126.51 | +122.74 | +131.27 | ✅ **全为正** |
| first_to_last | −10.31 | −10.11 | −18.32 | −3.06 | ✅ 全为负 |
| last_to_combine | −1.51 | −1.24 | −14.13 | +9.77 | ❌ 混合 |
| after_gc | +0.68 | +0.67 | −0.87 | +2.66 | ❌ 混合 |

**关键**：`gc_to_first` 在 **320/320** 个配对里全部为正，范围仅 122.74–131.27 µs（**跨度 8.5 µs，极其集中**）。
这不是噪声或个别层异常，而是**与层级、rank 无关的固定开销**，每层稳定 +126.7 µs。

40 层 × 126.69 µs ≈ **5.07 ms/步**，与总量 `gc_to_first` +5.060 ms 完全吻合。

## 2. 两臂的 shared 归属（`run.json` 的 `shared_modules` 直接证据）

| 臂 | `shared_modules` | `[ROLE4]` 日志 |
|---|---|---:|
| **arm2（fused）** | `['_ZeroShared', '_ZeroShared', …]` × 40 | 1937 条 |
| **arm3（unfused）** | `['DeepseekV4MLP', 'DeepseekV4MLP', …]` × 40 | **0 条** |

- **fused**：vLLM 的 `SharedExperts._layer` 被替换成 `_ZeroShared`（返回全零），
  GC 内核 role 4 实际计算 shared，并在 combine 里加 `shared_y`。
- **unfused**：vLLM 原生 `DeepseekV4MLP` 照常执行，GC routed 只算 top-6，
  最后由 `MoERunner._maybe_combine` 相加。

## 3. 严格可比性核验

| 项 | arm2 | arm3 | 结论 |
|---|---|---|---|
| `measurement.sha` | `c1a176db…` | `c1a176db…` | ✅ 同轮同 SHA |
| `complete` 标记 | 有 | 有 | ✅ 两臂均完成 |
| `execution.sha` | = measurement.sha | = measurement.sha | ✅ |
| **逻辑路由 `ids`** | `ccfe108d…` | `ccfe108d…` | ✅ **完全相同** |
| **路由 `weights`** | 相同 | 相同 | ✅ |
| `demand` / `batch_order` | 相同 | 相同 | ✅ |
| 逻辑权重 / 放置 / KV 读回摘要 | 相同 | 相同 | ✅ |
| 40 层合计物理行数 | 63,087,488 | 63,087,488 | ✅ |
| **逐槽 counts（40000×48）** | — | — | ✅ **逐元素相等** |
| `full_graph_replays` | 1000（每 rank）| 1000 | ✅ |
| `calls` | 40000 | 40000 | ✅ |
| `prompt_hash` | 相同 | 相同 | ✅ |
| **`output_hash`** | `088aed8c…` | `088aed8c…` | ✅ **相同** |

### 3.1 原生差异（必须分开表述）

| 项 | 结果 |
|---|---|
| `native_top1_diff` | **945 / 256,000（0.369%）** |
| 首个差异（arm3 的 `native_differences`）| `prefill`：count 556，request 0，position 0，native 24313 / reference 25228 |
| `stats` 窗口 | count 2293，request 250，position 7 |
| **`warm_native` 摘要** | arm2 与 arm3 **不同** |

**这是 shared 计算路径改变（fused 的 role 4 量化 vs 原 bf16 MLP）导致的数值差异**，
属预期：两条路径的 shared 项在 FP8/FP4 与 bf16 下舍入不同。

**但最终 `output_hash` 相同**、`prompt_hash` 相同、固定反馈与门权重相同、
逐槽 counts 逐元素相等 —— 因此**固定工作量下的对比成立**，性能差值可归因于 shared 融合本身。

## 4. 归因到哪一层

### 4.1 已可确定的

`gc_to_first` = **GC graph 起点 → 首个 routed CTA 开工**，稳定 +126.7 µs/层。

这一段的语义（见 `long_cost.py` / `aligned_step.py` 的段定义）包含：
- GC graph 启动到 dispatch 开始
- dispatch 阶段：count exchange / pull token / ECO plan
- 到第一个 routed CTA 领到任务的时刻

**fused 时这一段多 5.06 ms/步**，说明 **shared 的工作被插在了 routed 开工之前**，
使 routed 晚开工。这与 "role 4 需要先算完 shared 才能进 combine" 的串行依赖一致：
Kernel 需要 shared 就绪才能进 combine，因此把 shared 排到 routed 之前或与之串行。

### 4.2 不能确定的（缺证据，如实说明）

任务是"对每一步每一层选 FFN 跨度最大的 rank"做分析，**但两臂的 slowest rank 会切换**。
CSV 后半部分的同层同 rank 配对已给出（320 行，见 §1 统计），符号一致性证明了归因稳健；
但**shared kernel 与 routed kernel 的独立 stream 起止时间戳本任务未采集**，
所以**无法进一步区分**：

1. shared 是否真的串行在 routed 之前（需要 shared kernel 的开始/结束时间戳）
2. shared 是否在另一 stream 上并行、但 combine 需要等它（需要 stream 归属证据）

**结论只能定位到"GC graph 起点→首个 routed CTA 这 1.16→6.22 ms 的窗口"**，
不能细分到 kernel 级。

### 4.3 为什么"融合反而慢"的机制解释（基于已有证据）

| 段 | fused vs unfused | 解读 |
|---|---|---|
| `before_gc`（FFN 起点→GC 起点）| **fused 快 1.79 ms** | vLLM 的 `_ZeroShared` 是空壳，省掉了原生 MLP 的入口开销与门控准备 |
| **`gc_to_first`** | **fused 慢 5.06 ms** | shared 计算被塞进 GC graph 内部，**推迟了 routed 的开工** |
| `first_to_last`（routed 本体）| fused 慢 0.23 ms（可忽略）| routed 计算量相同 |
| `last_to_combine` | **fused 快 0.69 ms** | fused 的 combine 已含 shared_y，省一次读回 |
| `after_gc` | 持平 | — |

**即：fused 省下了框架层的入口（−1.79）与 combine 读回（−0.69），
但为此付出了 routed 开工推迟（+5.06）的代价，净亏 2.83 ms。**

unfused 的优势在于 **vLLM 的 shared MLP 可以在 routed 之前/并行地完成**，
不阻塞 routed 开工。

## 5. 输入与脚本

| 项 | 值 |
|---|---|
| `results` 目录 | `/mnt/fuse/sm90-h20-eight-CF3A1Zkp/results` |
| `measurement.sha` | `c1a176dbf45f1fd8d433189c1f24533aa44a7212`（前缀校验通过）|
| 分析脚本 revision | `158730feab175c567c3c7f1a86d1894fadfb3eab`（`scripts/sm90/shared_pair_cpu.py`）|
| 脚本来源 | `git show origin/main:scripts/sm90/shared_pair_cpu.py`（从 Git 对象提取，未触碰运行中的 checkout）|
| 运行方式 | `CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 nice -n 15` |
| 退出码 | **0** |
| 输入数组 | 两臂各 `rank0..7.npz`（`counts`、`clocks`、`stamps`）、`run.json` |

## 6. 交付物

`docs/trace_executor_20261119/sm90_h20_bs256_mega_gc_eight_results/formal/shared_pair_cpu/`：

| 文件 | 内容 |
|---|---|
| `pair.csv` | 前半：六段 + 整步的 `fused / unfused / 差值`（ms/步）；后半：320 行同层同 rank 逐对差值（µs）|
| `audit.log` | `measurement_sha`、`native_top1_diff 945`、`ffn_change_pct 6.343558…` |

## 7. 对八臂主表的含义

- 本诊断**不等后六臂**，先交付。
- 八臂仍在跑（arm4 已完成，arm6 进行中），未受影响。
- 主表将在八臂齐备后一并给出；本节结论可直接引用为 `2→3` 配对的分析。
