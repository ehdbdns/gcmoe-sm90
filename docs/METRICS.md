# 指标定义与测量方式（含 2026-11-05 的修正）

三个指标都来自 kernel 自己的计数器，不引入额外 RPC（每步 RPC 会让
80 ms/step 变成 142 ms/step，上一轮已经踩过）。

## 1. 每 rank 实际 token（IR 的分子分母）

来源：`expert_count[]`，role 1 在**每次调用**把该槽拉到的行数写进去。

- 窗口级：`stats[layer][slot]` 表在窗口内累加，除以 `steps` 和 `world`（kernel
  的 `total` 累加了所有源 rank 的 `send_count`，而 `send_count` 是全局限量、
  每个 rank 内容相同 ⇒ 值含 8 倍放大，必须除 `world`）。
- `IR = max_rank(per-step tokens) / mean_rank(per-step tokens)`，
  用整窗口平均，不用单次采样（单层采样有 ~7% 的 Poisson 噪声，和要测的不均衡
  同量级）。

**这一项一直是对的**，三臂都是 32 tokens/rank。

## 2. 每层激活专家数（每步）

来源：`header[7]`，role 1 在**本次调用**里每激活一个槽 +1，role 3 在每次调用
开头清零。

### ❌ 之前是错的

`meas_end()` 在窗口**结束后**才读 `header[7]`。那时下一次调用可能已经跑过
role 3，值被清零 ⇒ 三个臂全都报告 `active_experts_kernel_header = 0`。

### ✅ 现在

`_run_into()` 在每次 launch 返回后立刻快照 `header[7]`，窗口内取中位数
（`active_experts_sampled_median`），并保留最后一次的值
（`active_experts_kernel_header`）。

实测单次调用（32 token、rank 0、vanilla、48 槽）：**20 个槽激活**。

## 3. 每层 M 块数

真实定义：routed role 对每个槽按 `for start in range(0, count, tile_m)` 分块，
`tile_m` 来自编译期 schedule。所以

```
blocks = Σ_slots ceil(tokens[slot] / tile_m)
```

### ❌ 之前是两个错误叠加

1. **浮点整除**：代码写 `(per_step + bm - 1) // bm`。对 float 这是
   `floor_divide`，`(0.5 + 15) // 16 = 0` ⇒ 每步不足 1 个 token 的槽贡献 **0 块**。
   32 token / 60 槽 ≈ 0.4 token/槽/步，于是每个活跃槽都算成 0～1 块。
2. **用错了 `block_m`**：`_block_m()` 调的是 SM100 的
   `get_block_m_for_gc_moe`，返回 192（vanilla）/240（EPLB），
   而 SM90 routed role 实际的 `tile_m` 是 **64**（`capacity=2048`）
   或 16（小 batch 几何）。

### ✅ 现在

- `blk = torch.ceil(per_step / tile_m).sum(dim=1)`
- `_block_m()` 直接取 `get_sm90_role_config(...).tile_m`（与编译用的是同一个值）。

### 修正前后的量级

本次实验的负载很低（32 token / 48–60 槽 ≈ 0.4 token/槽/步），此时
**每个活跃槽恰好 1 块**，所以

```
blocks == active slots
```

实测单次调用（vanilla, rank 0, 48 槽, tile_m=16）：
`active=20, blocks=21`（有一个槽跨了 2 块）。

之前上报的 `m_blocks_median = 5.0` 是因为大部分槽被浮点整除算成 0；
`tile_m` 用错在**低负载下不影响**结果（0.4/64 和 0.4/240 都是 1 块），
只在一个槽超过一个块时才有差别（`tests/test_metrics_cpu.py` T2 用
100 token/槽的热点验证了这一点）。

## 4. 结论：这个实验里 M 块数 = 激活槽数

`wb`（块权重）和 `wa`（激活槽权重）在当前负载下**共线**，这正是
`docs/ECO_SM90_PORT.md` §5 说的「`wa = 0` 在 `count ≤ max_m` 时不失最优」。
所以 ECO 现在的优化目标（`wb = wa = 0`，纯 token）和「按块优化」在
这个 batch 下是同一个问题 —— 想看到 ECO 的时延收益，需要更大的 batch
（让一个槽跨多块）或真正的 wb/wa 标定。

## 5. 原始字段对照

| 字段 | 含义 | 是否可信 |
|---|---|---|
| `tokens_per_rank_median/min/max` | 窗口平均，每 rank token | ✅ |
| `per_layer_median_tokens` | 每层每步平均 token | ✅ |
| `ir` / `ir_detail` | rank 级 IR | ✅ |
| `ir_window` | slot 级 IR（层平均） | ✅ |
| `sample_tokens_per_rank` | 单次调用该 rank 拉的 token | ✅ |
| `active_experts_sampled_median` | **新增**：每次 launch 后采样 `header[7]` 的中位数 | ✅ |
| `active_step` | 单次调用 `expert_count>0` 的槽数 | ✅ |
| `m_blocks_step` | 单次调用的块数（已用 ceil + tile_m） | ✅ |
| `m_blocks_median/min/max` | 窗口平均块数（已修正） | ✅（修正后） |
| `block_m` | 现在 = routed role 的 `tile_m` | ✅（修正后） |
| `active_experts_kernel_header` | 最后一次 launch 的 `header[7]` | ✅（修正后，之前恒为 0） |
| `active_ever_median` | 窗口内「至少收到过一个 token」的槽数 | ⚠️ 不是占用率，别当激活数用 |

## 6. 回归测试

`tests/test_metrics_cpu.py`（`METRICS_CPU PASS`）逐条钉住这三个缺陷：

- T1：`x = 0.5` 的槽必须算 1 块（旧公式算 0）
- T2：热点槽超过一个块时，`tile_m` 的选择会改变结果（并记录「低负载下
  块数 == 激活槽数」这一事实）
- T3：快照得到 20 个激活槽，晚读 `header[7]` 得到 0
- T4：窗口总量守恒
