# 实验规格：GC MoE EP8 + EPLB(10%)

## 目的

在真实 DeepSeek-V4.1-Flash decode 上，比较 **GC MoE 原始内核** 与
**GC MoE + EPLB 冗余专家(10%) + 实时 RR 分发** 两臂。

## 硬件 / 软件

| 项 | 值 |
|---|---|
| GPU | 8 × H20 95.07 GiB |
| 并行 | TP=8, EP=8, DP=1（TP 不切 MoE token） |
| torch / vLLM | 2.13 / 0.31.0 |
| GC MoE 源码 | `source/`（本目录内，见 SOURCE.md） |

## 两臂差异

| 项 | vanilla | EPLB |
|---|---|---|
| `GC_MOE_EPLB` | 0 | 1 |
| 物理槽位 | 384（48/rank） | **424（53/rank）**，+40 = 10% |
| 放置 | 无 | `policy.py::rebalance_experts_global`（DeepInfer） |
| 权重 | 48 行/rank | `gather_slot_weights` 扩到 53 行/rank |
| 路由 | 逻辑 id 直接入 kernel | 逻辑 id → **物理槽位（RR）** |
| kernel | 原样 | 仅 `local` 48→53 |

## 测量协议

```
add_request(256 或 32 条真实 NAS prompt)
  → engine.step() 直到每条序列吐出第一个 token（prefill，不计时）
  → warm 窗口（EPLB 学习，不计时）
  → [可选] rearrange（本实验默认关闭，见 GC_MOE_EPLB_NO_REARRANGE）
  → flush 5 步（丢弃）
  → measure 窗口 N 步，device-synced 整窗计时
```

计时口径：`time_window()` 在窗口前后各做一次 `torch.cuda.synchronize()`
（通过 worker RPC），窗口内 `step()` 连续提交。**不能逐步计时** ——
vLLM 0.31 默认 `async_scheduling=True`，单步 `step()` 只测到入队延迟。

## 自证字段（每次运行必须检查）

| 字段 | 期望 |
|---|---|
| `running_min/max` | 等于序列总数（batch 未衰减） |
| `gen_tokens_per_step` | 等于序列总数（decode 每步每序列 1 token） |
| `ctx_tokens_per_step` | 0（测量窗口纯 decode） |
| `tokens_per_rank` | 等于序列总数（TP 不切） |
| `eplb` | 两臂分别为 false / true |
| 日志 | `[eplb] installed: physical=424 slots/rank=53 ...`（仅 EPLB 臂） |

## 已知约束

1. **显存**：EPLB 多占 0.59 GiB（GC buffer）+ 3.5 GiB（模型），KV cache
   少 6.04 GiB，256 序列时是硬上限。
2. **`GC_MOE_EPLB_NO_REARRANGE=1`**：本实验只验证**静态布局**。B 臂原实验做过
   一次重排，本轮的 vLLM 适配层把重排默认关掉（未验证）。
3. **`VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD=0`**：attention 的 multi-stream
   GEMM 会和 GC MoE 的 Green Context 抢 SM，2048-token profile batch 下
   默认流会永久等一个 aux event。
