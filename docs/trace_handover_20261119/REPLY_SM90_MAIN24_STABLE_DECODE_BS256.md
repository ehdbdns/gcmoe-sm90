# 【SM90】REPLY_SM90_MAIN24_STABLE_DECODE_BS256 — 稳定 prefill+decode top-k 后六臂全部通过，15 对严格判定全 True

> 本回执由 **SM90 执行者**（8×H20）提交。SM100 任务书不在本执行者范围内。

## 0. 结论：**六臂全部零差异通过，15 对全部可严格比较**

| 项 | 结果 |
|---|---|
| 实际模型启动数 | **6**（预算 6，每臂一次）|
| `exit-code.txt` | **0** |
| `pairs.csv` | **15 对** |
| `native_diff == 0` | **15 / 15** |
| `warm_native_equal == True` | **15 / 15** |
| `strict_pair_comparable == True` | **15 / 15** |
| `measured_samples` | 每对 **256,000**（1000 步 × 256 token）|
| 每步 256 token | **六臂 × 2000 步 = 12,000 步全部满足** |

**这是本系列第一次拿到全部 15 对都可严格归因的 BS256 六臂主表。**
此前 `REPLY_MAIN24_COST` 的 20 个非 vanilla 臂全部因 `warm_native_equal=false` 被判不可比；
`REPLY_SM90_MAIN24_STABLE_BS256` 的 random 臂在 stats 窗口 4 处失配即停。
本轮加入 `GC_MOE_STABLE_DECODE_TOPK=1` 后该缺口消失 —— 与
`REPLY_SM90_STABLE_DECODE_AA` 的推断完全一致。

## 1. 主表（BS256，两窗口各 1000 步）

| arm | E2E ms/步（正式）| vs vanilla | 完整 MoE ms/步 | vs vanilla | 重排 | token IR | M 块 IR |
|---|---:|---:|---:|---:|---|---:|---:|
| vanilla | **94.588** | — | 40.415 | — | 否 | 1.407 | 1.219 |
| random | **98.231** | **+3.85%** | 43.335 | +7.23% | 否 | 1.489 | 1.222 |
| rr | **90.998** | **−3.80%** | 35.303 | −12.65% | 是 | 1.083 | 1.075 |
| eco_linear | **90.052** | **−4.80%** | 35.643 | −11.81% | 是 | 1.057 | 1.056 |
| eco | **90.512** | **−4.31%** | 35.600 | −11.91% | 是 | 1.060 | 1.056 |
| **eco_smr** | **88.757** | **−6.17%** | **34.083** | **−15.67%** | 是 | 1.060 | 1.056 |

统计窗口（未重排）的 E2E：vanilla 93.545 / random 96.851 / rr 100.004 /
eco_linear 99.048 / eco 99.906 / eco_smr 98.188 ms/步
（重排前的四条 arm 明显更慢，重排后回落到 88.8–91.0，见 §4）。

**排序与方向**：`eco_smr` 最快 → `eco_linear` → `eco` → `rr` → `vanilla` → `random`，
与 `REPLY_MAIN24_COST` 的旧口径排序**完全一致**（random 最差、eco_smr 最快）。

## 2. 价格主对照 `eco_linear → eco`

| 项 | eco_linear | eco | Δ |
|---|---:|---:|---:|
| E2E ms/步 | 90.052 | 90.512 | **+0.51%（fine 更慢）** |
| 完整 MoE ms/步 | 35.643 | 35.600 | **−0.12%（fine 略快）** |
| routed MAE | 0.1276 pp | **0.0738 pp** | **−0.0538 pp（fine 更准）** |
| graph MAE | 0.2127 pp | 0.2181 pp | +0.0054 pp |
| `primary_price` | linear（slack 90）| fine（slack 11469）| — |

与旧口径一样：**fine 价格预测更准（routed MAE 优 42%），但 E2E 反而慢 0.51%**，
MoE 层内则略快 0.12%。这说明"价格更准"没有转化为端到端收益，
差异主要在 MoE 之外（调度/开销）—— 与历史观察一致。

## 3. SM-reuse 对照 `eco → eco_smr`

| 项 | eco | eco_smr | Δ |
|---|---:|---:|---:|
| E2E ms/步 | 90.512 | **88.757** | **−1.94%** |
| 完整 MoE ms/步 | 35.600 | **34.083** | **−4.26%** |

SM-reuse 依旧是**最稳定的单项优化**（MoE −4.26%、E2E −1.94%），
与旧口径（MoE −4.36%、E2E −1.86%）几乎重合。
`auction.csv` 中 eco 与 eco_smr 的 offers/accepted 完全相同（291,116 / 187,082），
确认收益来自 role 6 复用 dispatch 的 6 个 SM，而非配额差异。

## 4. 资源（任务书 §验收 1 要求）

### 4.1 显存（每臂 8 rank 取值，`run.json` 的 `stats/measured.initial.device_memory_bytes`）

| arm | 最低剩余（stats）| 最低剩余（measured）| 最高 reserved | 总显存 |
|---|---:|---:|---:|---:|
| vanilla | 22.87 GiB | 22.64 GiB | 61.78 GiB | 95.07 GiB |
| random | 14.72 GiB | 14.47 GiB | **69.20 GiB** | 95.07 GiB |
| rr | 14.72 GiB | 13.48 GiB | 69.20 GiB | 95.07 GiB |
| eco_linear | 14.28 GiB | **13.04 GiB** | 69.19 GiB | 95.07 GiB |
| eco | 14.28 GiB | **13.04 GiB** | 69.19 GiB | 95.07 GiB |
| eco_smr | 14.27 GiB | **13.03 GiB** | 69.19 GiB | 95.07 GiB |

> **说明**：`free` 是任务书要求的设备剩余显存读数，**包含 CUDA allocator 之外的占用**
> （NCCL/NVSHMEM 对称堆、CUDA context、驱动预留）。因此它与
> `95.07 − 61.78 = 33.29 GiB` 的差额约 10.4 GiB 不能当作 allocator 可用量。
> ECO 四臂的 reserved 比 vanilla 高 7.4 GiB（69.19 vs 61.78），
> 对应 ECO 的额外副本/统计缓冲，剩余显存因此从 22.87 降到 14.27 GiB。
> **最低点 13.03 GiB（eco_smr 正式窗口）**，为后续 512 档的余量基线。

### 4.2 vLLM 公布的 KV 容量与利用率

| 项 | 六臂共同值 |
|---|---|
| `GPU KV cache size` | **1,530,922 tokens**（Maximum concurrency 329.73x @4,643）|
| `GC_MOE_KV_BYTES` | 21,474,836,480 B = **20.00 GiB** |
| **峰值 KV 利用率** | **11.4%**（六臂完全相同）|
| `kv cache group sizes` | `[32,32,32,32,32,32,32,64,8]`，lcm block 64 |

1,530,922 与任务书 §512/1024/2048 容量依据引用的数值完全一致。

### 4.3 每步 256 token

六臂 `runner.log` 中各 **2000 条** `256 generation tokens` 记录（stats + measured 各 1000），
`long_window()` 的 `rec['gen_tokens'] != [nseq]*1000` 断言未触发。
random 臂**本轮也完整跑通 2000 步**（上一轮在 stats 第 620 步即失败）。

## 5. 两个 top-k 开关的实读确认（任务书要求单列）

逐臂从 `run.json` 的 `controls` 读取：

| arm | `GC_MOE_STABLE_PREFILL_TOPK` | `GC_MOE_STABLE_DECODE_TOPK` |
|---|:--:|:--:|
| vanilla | **1** | **1** |
| random | **1** | **1** |
| rr | **1** | **1** |
| eco_linear | **1** | **1** |
| eco | **1** | **1** |
| eco_smr | **1** | **1** |

**所有六臂都开启了两个稳定开关**，因此计时包含这一公共实现，
按任务书要求只与本轮相同设置的数据比较。

## 6. 与旧口径的计时差异（必须注意）

本轮 vanilla 正式窗口 **94.588 ms/步**，旧口径（`REPLY_MAIN24_COST`）为 **90.231 ms**，
差异 **+4.8%**；稳定 prefill-only 的上一轮为 91.143 ms（+3.8%）。原因：
`GC_MOE_STABLE_DECODE_TOPK=1` 用 `argsort(stable=True)` 替换了原生 top-k 内核，
**改变了 decode 的实际执行路径**，这部分开销对六臂相同。

**因此本表的绝对值不可与旧表并列**；跨轮可比的只有**相对 vanilla 的百分比**：

| arm | 本轮 vs vanilla | 旧口径 vs vanilla | 偏移 |
|---|---:|---:|---:|
| random | +3.85% | +4.91% | −1.06 pp |
| rr | −3.80% | −3.58% | −0.22 pp |
| eco_linear | −4.80% | −4.37% | −0.43 pp |
| eco | −4.31% | −5.03% | +0.72 pp |
| eco_smr | −6.17% | −6.80% | +0.63 pp |

**五个臂的相对收益方向与量级全部稳定**（偏差 ≤1.1 pp），说明
6.17% 的 eco_smr 收益不是稳定 top-k 开销带来的假象。

## 7. 拍卖统计（BS256，两个 ECO 臂）

| arm | rounds | offers | accepted | 接受率 | 每调用成交 | 无出价轮 | 有出价全拒 |
|---|---:|---:|---:|---:|---:|---:|---:|
| eco_linear | 40,151 | 289,930 | 186,670 | 64.38% | 4.667 | 34 | **0** |
| eco | 40,169 | 291,116 | 187,082 | 64.26% | 4.677 | 34 | **0** |

与旧口径 BS256（eco_linear 64.59% / eco 64.54%）相差 <0.3 pp。
`eligible_donors` 1.42M、`feasible_donors` 1.04M、`blocked_before_offer` 375–378k，
与历史"瓶颈在目的地可行性判定"的结论一致。
**eco 与 eco_smr 的 offers/accepted 逐值相同**（291,116 / 187,082），确认 SM-reuse 不改变配额。

## 8. 环境、预检与版本

| 项 | 值 |
|---|---|
| 测量 SHA | `956bb3fa16deb9472a1493c265e69fbba7ff0b55`（worktree `/tmp/gcmoe-prefix-RITlft3l/code`，`git status` 干净）|
| 工作目录 | `/tmp/gcmoe-prefix-RITlft3l` |
| 现场 state | **`/tmp/gcmoe-main24-bs256-oOB492Em`**（12 GB，`replay/` 为 route 回放；`kv/` 下的二进制 KV 已在成功路径按 `run_main20.sh` 释放）|
| GPU | 8×H20，driver 580.105.08 |
| 软件 | cutlass 4.8.1 / torch 2.13.0+cu130 / CUDA 13.0 / vllm 0.31.0 |
| 环境 | `GC_MOE_ORDER_PREFILL=1`、`GC_MOE_STABLE_PREFILL_TOPK=1`、`GC_MOE_STABLE_DECODE_TOPK=1`、`GC_MAIN24_SINGLE_BS=256`、`GC_MOE_KV_BYTES=21474836480`、`GC_MOE_MATRIX_SIZE=24`；显式 `unset GC_MOE_DECODE_STATS_AA GC_MOE_VANILLA_STATS_REPLAY` |
| 代码修改 | **无**（预检一次通过，未做编译修复）|

**预检**：`ALL CPU CHECKS PASSED`（26 项 PASS，含 `LONG_COST`、`MAIN20_CPU`、
`stable prefill top-k` 与新增 decode 选择器用例）；
六种角色组合编译全部通过（`[1,4,2,3]` 12.0–12.1 s、`[1,4,2,5,3]` 17.9–18.0 s、
`[1,4,2,5,6,3]` 27.9 s）；
八卡 SM-reuse 门禁 `reuse-0.sha` 与 `reuse-1.sha` **逐 rank 完全相同**。

## 9. 交付物

`docs/trace_executor_20261119/sm90_main24_stable_decode_bs256_results/`（155 个文件，156 MB）：

- `delivery/bs256/pairs.csv`（15 对）、`main_table.csv`、`layers.csv`、`cost_pairs.csv`、`auction.csv`
- `delivery/bs256/<arm>/`：`run.json`、`runner.log`、`status.txt`（均 `exit_code=0`）、
  `installed-layout.json`、`hardware.csv`、8 份 `rank*.npz`
- `delivery/bs256/checkpoint/`（prefill 与 stats 快照清单）、`stats-demand.npz`、`placement.json`、
  `workload.sha256`、`state-path.txt`（含现场 state 路径）
- `delivery/SHA256SUMS`、`preflight/`、`preflight.log`、`progress.log`、`exit-code.txt`（=0）

**未提交**：route 回放与 KV 清单留在现场 `/tmp/gcmoe-main24-bs256-oOB492Em`（12 GB）。
成功路径下脚本按设计释放了 `kv/rank*.bin`（每卡 20 GiB × 8），因此现场仅 12 GB；
清单（`rank*.json`、`*rank*.pt`）完整保留，可复核 SHA。

## 10. 后续（512/1024/2048）容量提示

本轮 BS256 的关键余量数据：

- **最低剩余显存 13.03 GiB**（eco_smr 正式窗口），发生在 KV 利用率仅 11.4% 时。
- 六臂 reserved 峰值为 69.20 GiB；vanilla 只有 61.78 GiB，ECO 多出 7.4 GiB。
- KV 每卡 20 GiB 对应 1,530,922 token；正式窗口峰值利用率 11.4%，
  即实际每卡用到约 2.3 GiB KV。

因此 512 档若维持 20 GiB/卡 KV，**KV 侧余量充足**（利用率约 23%），
但 **reserved 侧只剩约 13 GiB 的池外余量**，需要按 ECO 臂（而非 vanilla）来规划，
并优先采集 eco_smr 的调度与空闲显存水位。
