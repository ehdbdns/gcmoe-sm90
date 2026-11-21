# 【SM90】REPLY_SM90_MAIN24_STABLE_DECODE_BS1024 — 六臂全部通过；价格对照在 BS1024 **首次反向**

> 本回执由 **SM90 执行者**（8×H20）提交。SM100 任务书不在本执行者范围内。

## 0. 结论

| 项 | 结果 |
|---|---|
| 实测 SHA | `782ea2983b0aff2f7d5998c716a0b898e869a978` |
| 实际模型启动数 | **6**（每臂一次，顺序 vanilla → random → rr → **eco** → **eco_linear** → eco_smr）|
| `exit-code.txt` | **0** |
| `pairs.csv` | **15 对**，`strict_pair_comparable` **15/15 True** |
| `warm_native_equal` | **15/15 True** |
| `native_diff` | **15/15 = 0** |
| 每对 `measured_samples` | **1,024,000**（1000 步 × 1024 token）|
| 每步 1024 token | 六臂各 **2000 步** |
| scheduler 抢占 | 六臂日志 **0 条** |
| 代码修改 | **无**（预检一次通过）|

任务书要求 `GC_STEP_BOUNDARY=1` 的打点：六臂 `run.json` 均含 `measured.host_boundary_ns`，
八份 `rank*.npz` 均含 `step_boundary`（1000×4）。**旧 SHA 未启动过模型**，
因此本轮直接使用带打点的新 SHA，无"旧 SHA 粗分段"的限制。

## 1. 主表（BS1024，两窗口各 1000 步）

| arm | E2E ms/步 | vs vanilla | 完整 MoE ms/步 | vs vanilla | 重排 | routed MAE |
|---|---:|---:|---:|---:|---|---:|
| vanilla | **319.857** | — | 136.670 | — | 否 | 0.0409 |
| random | **328.608** | **+2.74%** | 144.479 | +5.71% | 否 | 0.0478 |
| rr | **296.879** | **−7.18%** | 110.331 | −19.27% | 是 | 0.0352 |
| eco_linear | **293.857** | **−8.13%** | 108.723 | −20.45% | 是 | 0.0527 |
| eco | **300.970** | **−5.91%** | 108.722 | −20.45% | 是 | 0.0378 |
| **eco_smr** | **284.554** | **−11.04%** | **102.094** | **−25.30%** | 是 | 0.1017 |

统计窗口（重排前）：vanilla 309.702 / random 319.546 / rr 323.515 /
eco 330.741 / eco_linear 325.547 / eco_smr 316.132 ms/步。

**eco_smr 依然最快**（−11.04%），收益随 BS 单调放大：
BS256 −6.17% → BS512 −10.17% → **BS1024 −11.04%**。

**注意顺序带来的差异**：本档按任务书要求把 `eco` 排在 `eco_linear` 之前。
两臂的 MoE 完全相同（−0.001%），但 E2E 是 eco 比 eco_linear 慢 2.42%，
因此表内 eco 的 −5.91% 不能与 BS256/512 的 eco（−4.31% / −7.56%）直接连成趋势线。

## 2. 价格主对照 `eco_linear → eco`：**BS1024 首次反向**

| BS | E2E Δ（fine vs linear）| MoE Δ | routed MAE Δ | 结论 |
|---:|---:|---:|---:|---|
| 256 | **+0.51%** | −0.12% | −0.0538 pp | fine 更准但 E2E 更慢 |
| 512 | **+0.56%** | −0.14% | −0.0329 pp | 同上，几乎逐位重合 |
| **1024** | **+2.42%** | **−0.001%** | −0.0149 pp | **MoE 打平，E2E 却慢 2.42%** |

**这是三档里第一次出现"MoE 层完全打平而 E2E 显著变慢"**。
因为两臂的 MoE 时间只差 0.001 ms（108.723 vs 108.722），
增加的 7.1 ms 全部落在 MoE 之外 —— 与 §3 的边界打点互相印证：

## 3. 步边界诊断（`cost_boundary_report.py` 的 stdout，见 `cost_boundary_report.txt`）

单位 ms，`linear = eco_linear`，`fine = eco`，`delta = fine − linear`。

### 3.1 GPU 区间（每 rank 8 卡平均；括号内为配对中位数与"fine 更慢"的样本数）

| 区间 | n | linear | fine | Δ 均值 | Δ 中位 | fine 更慢 |
|---|---:|---:|---:|---:|---:|---:|
| `between_steps`（全窗）| 999 | 86.433 | 93.539 | **+7.106** | +5.425 | 970/999 |
| `within_moe` | 1000 | 108.439 | 108.467 | **+0.028** | +0.024 | 547/1000 |
| `between_layers` | 1000 | 98.985 | 98.968 | −0.017 | +0.075 | 729/1000 |
| `combine_to_forward_return` | 20 | 0.1433 | 0.1426 | −0.0007 | −0.0005 | 7/20 |
| `forward_to_sampler` | 20 | 0.00246 | 0.00265 | +0.0002 | +0.0003 | 14/20 |
| `sample_tokens` | 20 | 3.8955 | 4.0340 | **+0.1385** | +0.1121 | 18/20 |
| `sampler_to_next_execute` | 20 | 79.283 | 86.582 | **+7.299** | +5.064 | **20/20** |
| `execute_to_first_moe` | 20 | 6.7426 | 7.2986 | +0.5559 | +0.3956 | 16/20 |
| `marked_gaps` | 20 | 90.067 | 98.059 | **+7.993** | +5.547 | **20/20** |
| `unmarked_gaps` | 939 | 86.428 | 93.522 | +7.094 | +5.394 | 910/939 |

**五段之和核对**：`combine_to_forward_return + forward_to_sampler + sample_tokens
+ sampler_to_next_execute + execute_to_first_moe` 与 `marked_gaps` 同量级（90.07 / 98.06），
差值由未打点的步内片段构成，方向与 `unmarked_gaps`（+7.09）一致。

**结论**：**几乎全部 +7.1 ms 落在 `sampler_to_next_execute` 段**（+7.30 ms，20/20 全部更慢）。
该段是"sample_tokens 返回到下一次 execute_model 入口"的流上经过时间，
**可以包含排队与等待**。`within_moe` 只是 +0.028 ms，`sample_tokens` 本身 +0.14 ms，
`combine→forward_return` 甚至略快 —— 即 **MoE 算子与 sampler 计算都不是原因**。

### 3.2 主机侧（逐 step，1000 样本）

| 区间 | linear | fine | Δ 均值 | Δ 中位 | fine 更慢 |
|---|---:|---:|---:|---:|---:|
| `host_10step`（十个 ten-step 块）| 293.857 | 300.970 | **+7.113** | +6.838 | 100/100 |
| `schedule` | 40.515 | 42.323 | **+1.809** | +1.652 | 970/1000 |
| `update_from_output` | 5.939 | 6.255 | +0.316 | +0.306 | 944/1000 |
| `other_get_output_including_worker_wait` | 221.39 | 222.62 | +1.234 | +1.097 | 889/1000 |
| `other_engine_step` | 25.306 | 28.828 | **+3.523** | +2.075 | 981/1000 |
| `collect` | 0.703 | 0.935 | +0.232 | +0.232 | 974/1000 |

**方向一致**：主机 10 步块 +7.11 ms 与 GPU `between_steps` 的 +7.11 ms **几乎完全相等**，
说明两套打点互相独立地指向同一差额。

**但主机侧无法把它归到单一环节**：`schedule` +1.81、`other_engine_step` +3.52、
`other_get_output` +1.23、`collect` +0.23、`update_from_output` +0.32，
合计约 +7.1 ms，分散在四个段里，且 `other_get_output` **包含等待 worker**。
**因此不能把差额归因为 CPU 调度或某个 GPU 算子**——证据只支持"fine 在
sample_tokens 之后到下一次 execute_model 之间多等了约 7 ms，且这段时间里
调度与 engine.step 其余部分各占一部分"。

## 4. SM-reuse 对照 `eco → eco_smr`

| 项 | eco | eco_smr | Δ |
|---|---:|---:|---:|
| E2E ms/步 | 300.970 | **284.554** | **−5.45%** |
| 完整 MoE ms/步 | 108.722 | **102.094** | **−6.10%** |

BS256 −1.94%/−4.26% → BS512 −2.82%/−5.25% → **BS1024 −5.45%/−6.10%**，
SM-reuse 收益随 BS 单调增长。`auction.csv` 中 eco 与 eco_smr 的
offers/accepted 完全相同（433,250 / 293,457），确认是 role 6 复用而非配额。

## 5. 资源（任务书 §验收）

### 5.1 显存（`run.json` 的 `stats/measured.initial.device_memory_bytes`）

| arm | 最低剩余（stats）| 最低剩余（measured）| 最高 reserved | 总显存 |
|---|---:|---:|---:|---:|
| vanilla | 21.32 GiB | 21.07 GiB | 65.64 GiB | 95.07 GiB |
| random | 13.38 GiB | 13.13 GiB | **72.97 GiB** | 95.07 GiB |
| rr | 13.38 GiB | 12.14 GiB | 72.97 GiB | 95.07 GiB |
| eco | 13.32 GiB | 12.10 GiB | 72.96 GiB | 95.07 GiB |
| eco_linear | 13.34 GiB | 12.10 GiB | 72.96 GiB | 95.07 GiB |
| eco_smr | 13.33 GiB | **12.08 GiB** | 72.96 GiB | 95.07 GiB |

> `free` 为设备剩余显存读数，**包含 CUDA allocator 之外的占用**（NCCL/NVSHMEM 对称堆、
> CUDA context、驱动预留）。与 `95.07 − 72.96 = 22.11 GiB` 的差额约 10.0 GiB 即来自这些部分。
>
> 余量随 BS 收窄：BS256 13.03 → BS512 13.97 → **BS1024 12.08 GiB**。
> BS2048 若沿用 20 GiB KV，reserved 侧预留只剩约 12 GiB 的缓冲，风险显著上升。

### 5.2 KV 容量与利用率

| 项 | 六臂共同值 |
|---|---|
| `GPU KV cache size` | **1,724,380 tokens**（BS256 1,530,922 / BS512 1,597,922）|
| `GC_MOE_KV_BYTES` | 21,474,836,480 B = **20.00 GiB** |
| **峰值 KV 利用率** | **48.9%**（六臂相同；BS512 23.5%）|

KV token 容量随 `max_model_len` 增大而继续变化（1.53M → 1.60M → 1.72M），
**不能把上一档的 token 数当作下一档上限**。

### 5.3 工作量

| 项 | 值 |
|---|---|
| 基础批次 | 512 条（与 BS512 完全同一批，SHA 校验通过 `e7c4aee9…`）|
| 实际请求 | **1024**（512 × 2，每条作为独立请求）|
| prefill token | **927,462**（463,731 × 2）|
| `decode_budget` | 2,971（`1+1000+1000+970slack`）|
| `GC_KV_BUDGET` | 5,000,000（需求约 4.1M，余量充足）|
| prefill 引擎步 | 645 |

## 6. 拍卖统计（BS1024）

| arm | rounds | offers | accepted | 接受率 | 每调用成交 | 无出价轮 | 有出价全拒 |
|---|---:|---:|---:|---:|---:|---:|---:|
| eco | 40,165 | 433,250 | 293,457 | 67.73% | 7.336 | 33 | **0** |
| eco_linear | 40,166 | 432,968 | 293,329 | 67.75% | 7.333 | 35 | **0** |
| eco_smr | 40,165 | 433,250 | 293,457 | 67.73% | 7.336 | 33 | **0** |

**接受率继续上升**：BS256 64.3% → BS512 65.7% → **BS1024 67.7%**。
`eligible_donors` 1.75M、`feasible_donors` 1.39M、`blocked_before_offer` 352–353k
（BS512 为 380k），**目的地可行性瓶颈略有缓解**。

## 7. 环境与预检

| 项 | 值 |
|---|---|
| 测量 SHA | `782ea2983b0aff2f7d5998c716a0b898e869a978`（worktree `/tmp/gcmoe-prefix-BzC7FVRH/code`，`git status` 干净）|
| 工作目录 | `/tmp/gcmoe-prefix-BzC7FVRH` |
| 现场 state | `/tmp/gcmoe-main24-bs1024-fhSPzc57`（**44 GB**，KV 二进制已按成功路径释放，清单保留）|
| GPU | 8×H20，driver 580.105.08 |
| 软件 | cutlass 4.8.1 / torch 2.13.0+cu130 / CUDA 13.0 / vllm 0.31.0 |
| 环境 | `GC_MOE_ORDER_PREFILL=1`、`GC_MOE_STABLE_PREFILL_TOPK=1`、`GC_MOE_STABLE_DECODE_TOPK=1`、**`GC_STEP_BOUNDARY=1`**、`GC_REPEAT_BS512_PROMPT_SHA256=e7c4aee9…`、`GC_KV_BUDGET=5000000`、`GC_MAIN24_SINGLE_BS=1024`、`GC_MOE_KV_BYTES=21474836480` |

**两个 top-k 开关实读**：六臂 `run.json` 的 `controls` 中两者**全部为 1**。

**预检**：26 项 CPU PASS；六臂编译全过（`[1,4,2,3]` 11.3–11.4 s、
`[1,4,2,5,3]` 16.8–17.0 s、`[1,4,2,5,6,3]` 26.5 s）；
八卡 SM-reuse 门禁 `reuse-0.sha` 与 `reuse-1.sha` **逐 rank 相同**。

## 8. 交付物

`docs/trace_executor_20261119/sm90_main24_stable_decode_bs1024_results/`（159 个文件，269 MB）：

- `delivery/bs1024/`：`pairs.csv`（15 对）、`main_table.csv`、`layers.csv`、`cost_pairs.csv`、
  **`cost_boundary.csv`**、`auction.csv`、各臂 `run.json`/`runner.log`/`status.txt`/8 份 `rank*.npz`
- `cost_boundary_report.txt`（`cost_boundary_report.py` 的 stdout）
- `dataset.sha256`（两份源文件 SHA256）、`preflight/`、`preflight.log`、`progress.log`、
  `exit-code.txt`（=0）、`state-path.txt`、`delivery/SHA256SUMS`

## 9. 需要设计端注意

1. **`eco_linear → eco` 在 BS1024 反向放大到 +2.42%**，且 MoE 完全打平。
   若继续用这一对做"价格主对照"，BS1024 的数据会与 256/512 的趋势断裂；
   建议把 `sampler_to_next_execute` 段（+7.30 ms，20/20）作为后续排查的明确入口。
2. 主机侧拆解显示 +7.1 ms 分散在 `schedule`/`other_engine_step`/`other_get_output` 各段，
   **不足以归因到 CPU 调度或单一 GPU 算子**；如需进一步定位，需要更细的引擎内打点。
3. **BS2048 内存预判**：最低剩余显存已降到 12.08 GiB、峰值 KV 48.9%。
   若 2048 档 KV 需求翻倍（约 98%），20 GiB/卡将不够，需要设计端先决定 KV 字节数与
   prompt 预算（当前 512 条基础 × 4 = 2048 请求，prefill 约 1.85M token）。
