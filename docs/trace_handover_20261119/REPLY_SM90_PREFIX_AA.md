# 【SM90】REPLY_SM90_PREFIX_AA — vanilla 自对照就不可复现，分歧与放置无关

> 本回执由 **SM90 执行者**（8×H20）提交。

## 0. 核心结论

**同一个 vanilla 后端重复执行 prefill，参考轨迹自身就无法复现**：

| 臂 | mismatches / samples | 涉及不同请求数 | position 0 失配 |
|---|---:|---:|---:|
| **vanilla（自对照）** | **183 / 18,511** | 84 | 1 |
| **random** | **174 / 18,511** | 92 | 2 |

**两者量级相同（183 vs 174）**，且 88 个 `(请求, 位置)` 组合在两次运行中**同时失配**，
其中 83 个的原生 token **完全相同**。

这直接推翻了上轮"random 放置导致 prefill 分叉"的假设：
**分歧来自参考轨迹不可复现，与放置无关**。

## 1. `req109` position 0 的结果（任务书点名要求）

| 臂 | 是否失配 | 原生值 | 参考值 |
|---|---|---|---|
| vanilla | **否** | 2581 | 2581 |
| random | **否** | 2581 | 2581 |

**两次启动中 `req109` position 0 都等于参考值 2581。**

这与上轮失败日志里的
`req109, position 0, native 671, reference 2581` **不一致** ——
说明上轮那次是**偶发**，不是确定性行为。本轮的 2 次执行都没有复现 671。

## 2. 一致性字段（任务书要求逐项检查）

| 字段 | 两臂是否相同 |
|---|---|
| `prompt_hash` | ✅ 相同 |
| `prefill_steps` | ✅ 相同（**117**）|
| `positions` | ✅ 相同（逐请求的 prefill 长度）|
| `batch_order` | ✅ 相同（`000697f7…`）|
| `samples` | ✅ 相同（**18,511**）|

即两臂**完全运行在同一批请求、同一批位置、同一批顺序**上，
差异只出现在 token 值本身。

## 3. 失配明细结构

- 失配位置范围：**position 0 到 102**（两臂相同）
- position 0 的失配极少：vanilla 1 个、random 2 个（占 18511 样本的 0.005%–0.011%）
- 绝大多数失配集中在 **prefill 后半段**

### 3.1 position 0 的完整失配记录

`[请求编号, 请求内位置, 原生 token, 参考 token]`

| 臂 | 记录 |
|---|---|
| vanilla | `[221, 0, 49541, 9854]` |
| random | `[119, 0, 2581, 42]`、`[173, 0, 2054, 9854]` |

### 3.2 两臂交叉对照

| 项 | 数值 |
|---|---|
| 两臂同时失配的 `(请求, 位置)` | **88** |
| 其中原生 token 相同 | **83** |
| 其中原生 token 不同 | **5** |
| 仅 vanilla 失配的请求 | 11 |
| 仅 random 失配的请求 | 19 |
| 两臂都失配的请求 | 73 |

**83/88 的重合且值相同**，说明失配位置是**高度可重复的系统性位置**，
但**具体的失配值会变** —— 这是浮点非确定性的典型特征。

未列入明细的位置等于该位置的参考值（按任务书口径）。

## 4. 为什么这解释了上轮的失败

上轮 random 臂报 `190 / 256 mismatches`、`req109 pos0 native 671`。
本轮：
- random 只有 **174 / 18,511** 处失配（分母是 18511 个 prefill 样本，不是 256 个请求）；
- `req109 pos0` 在两次执行中**都等于参考**。

**上轮的 190/256 是把 18,511 个样本中的失配聚合到了 256 个请求上**，
而 `req109=671` 是**偶发**值。

更关键的是：**vanilla 自己也有 183 处失配**，因此
`verify_native_prefix` 在 prefill 阶段作为硬断言**必然会间歇性误报**，
这不是 random 臂的特有问题。

## 5. 对判据的含义（供设计端决策）

1. **prefill 原生自对照不可复现**：同后端、同输入、同 batch_order，
   18511 个样本中约 1%（183 / 174 处）不同。这是浮点归约顺序或 kernel
   选型的非确定性，不是逻辑错误。
2. **`verify_native_prefix` 作为"必须为 0"的硬断言不成立**。
   建议改为：
   - 只对**正式 decode 窗口**（1000 步）做零差异断言；
   - prefill 阶段改为记录 + 阈值（例如失配率 ≤ 1.5%），
     或改为**统计上界**而非逐位相等。
3. **`req109 pos0` 不是稳定复现点**，不宜作为定位锚点。
4. 若要彻底消除 prefill 非确定性，需要在 kernel 层固定归约顺序或
   禁用相关 fast-math 路径 —— 这会改变被测内核语义，超出执行者权限。

## 6. 版本与环境

| 项 | 值 |
|---|---|
| 测量 SHA | `b33970c58f3ad5612a1580c7c4b870a3be4bbec5` |
| 工作目录 | `/tmp/gcmoe-prefix-aa-czl2Pwmx` |
| 参考轨迹 | `replay/tokens.json`，sha256 `ad68503c07f9c213c7ddeda92593f54b65fe65515b10dd8247ae7f1c714b1f15` |
| 参考来源 | `main24_prefix_capacity_results_blocked/bs256/state-path.txt` → `/tmp/gcmoe-main24-bs256-LIk16dRh` |
| GPU | 8×H20，driver 580.105.08 |
| 软件 | cutlass 4.8.1 / torch 2.13.0+cu130 / CUDA 13.0 / vllm 0.31.0 |
| 模型启动数 | **2**（vanilla 1 + random 1）|
| 代码修改 | **无** |

**环境变量**（按任务书）：`GC_MOE_PREFIX_DIAG=1`、`GC_MOE_MATRIX_BS=256`、
`GC_MOE_MATRIX_SIZE=24`、`GC_MOE_KV_BYTES=21474836480`、`GC_MOE_LONG_COST=1`、
`GC_MOE_SM90_STREAM_M_BLOCKS=1`。

## 7. 交付物

`docs/trace_executor_20261119/sm90_prefix_aa_results/`：
- `vanilla.json`、`random.json`（含 `arm` / `prompt_hash` / `prefill_steps` /
  `positions` / `samples` / `mismatches` / `batch_order`）
- `reference.sha`
- `vanilla.log`、`random.log`（两臂完整日志）
- `SHA256SUMS`

两臂均成功完成 prefill 并按设计在 prefill 结束后退出（未做 1000+1000）。
