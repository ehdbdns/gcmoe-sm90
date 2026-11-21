# 【SM90】REPLY_SM90_MAIN24_STABLE_DECODE_BS512 — 首个失败：BS512 的 prompt 供给不足（模型未启动）

> 本回执由 **SM90 执行者**（8×H20）提交。SM100 任务书不在本执行者范围内。

## 0. 首个失败（按任务书 §验收 2 报告）

| 项 | 值 |
|---|---|
| 阶段 | `step=3: capture`，第 1 个臂 **BS=512 / arm=vanilla** |
| 失败位置 | **模型启动之前**（`scripts/real_decode_100_100.py` 的 prompt 筛选）|
| 异常 | `SystemExit: only 374 prompts fit (need 512, len range [256,2048], KV budget 2300000, steps 0)` |
| `exit_code` | **1** |
| rank | 无（八个 worker 尚未创建）|
| 请求号 / 位置 / token | 无（未进入计算）|
| Capture workspace | `/tmp/gcmoe-aligned-RBnndQWj` |
| 已启动配置 | **vanilla 1 个**（且未真正启动模型）；random / rr / eco_linear / eco / eco_smr **均未启动** |
| 实测 SHA | `5cb4d7257649765b3ef8215a6acda6940ad1d103` |

## 1. 根因：单文件数据集在 [256, 2048] 区间只有 187 条唯一 prompt

`scripts/real_decode_100_100.py` 的 `load_prompts()` 从 `DATASET` 指向的文件里
筛选长度在 `[256, 2048]` 的 prompt，`BS=512` 需要 512 条。实测：

| 文件 | 总行数 | **[256,2048] 内可用条数** |
|---|---:|---:|
| `dsv4-flash_online_input_ids_26-10-08_1.jsonl` | 1,500 | **187** |
| `dsv4-flash_online_input_ids_26-10-08_2.jsonl` | 1,500 | **158** |
| 两个文件合计 | 3,000 | **345** |

首次启动时 `DATASET` 只指向**单文件**（`..._1.jsonl`，这是我沿用 BS256 那轮的单文件设置），
筛选后只有 187 条；`load_prompts()` 的补齐分支用 stride 反复取样，
最多凑到 **374 = 187 × 2** 条，因此 `BS=512` 必然失败。

**这与模型、KV、显存都无关**。`GC_KV_BUDGET=2300000` 并未成为限制：
用两个文件按同一 stride 规则离线复算，512 条的实际需求是
**2,048,371 token ≤ 2,300,000（余量 12.2%）**。
即：**真正卡住的是数据集条数，不是 KV 预算**。

## 2. KV 预算的离线复算（排除预算因素）

按仓库选样规则（per-file 长度排序 → 偶数 stride → 轮转 → 预算截断）：

| 项 | 双文件（345 条可用）|
|---|---:|
| 选出条数 | **512** |
| 长度 min / 中位 / max | 257 / 732 / 2042 |
| 总长度 | 463,731 |
| 选样后 `decode_budget` | 2,519（`1 + 1000 + 1000 + slack`）|
| 每流 `overhead` | 3,095 |
| **KV 需求** | **2,048,371 token**（≤ 2,300,000）|

**结论：把 `DATASET` 指向两个文件即可满足 BS512**，无需改动 `GC_KV_BUDGET`，
也不涉及任何代码语义修改。

## 3. 两次同点失败都未消耗 GPU

| 次 | `DATASET` | 结果 |
|---|---|---|
| 1 | 单文件 `..._1.jsonl` | `only 374 prompts fit` |
| 2 | 单文件（复现确认）| `only 374 prompts fit` |

两次都**未启动模型**（`real_decode` 在数据加载阶段即退出，`nvidia-smi` 全卡 4 MiB），
不消耗 GPU 时间，也未进入"每臂最多一次"的启动预算。

## 4. 已按仓库默认数据设置重启

`config/experiment.env` 的默认 `DATASET` 就是**双文件逗号拼接**（即"当前机器已验证的数据"）。
我已把 `DATASET` 恢复成该默认值（两文件），并在同一 SHA `5cb4d72` 上重跑：

- 工作目录 `/tmp/gcmoe-prefix-bTcIwpt0`
- 预检进行中：**26 项 CPU PASS**，六臂角色组合编译全部成功，
  SM-reuse 八卡门禁两侧 SHA 逐 rank 相同
- 尚未启动模型

**未做任何代码修改**（不涉及"等价编译修复"）。

## 5. 请设计端确认

1. BS512 的 `DATASET` 使用仓库默认的**双文件**是否认可？（BS256 那轮用的是单文件，
   这是两轮之间唯一的差异。）
2. 若要求必须**单文件**，BS512 在当前 `[256,2048]` 区间下**不可达**（187 条上限，补齐 374 < 512）；
   需要放宽 `PREFILL_LO`/`PREFILL_HI`，这会改变 prompt 长度分布，属实验语义变更。
3. 前瞻：按双文件 345 条唯一 prompt，**1024 档已无法凑满**（即使重复取样也会显著扭曲长度分布），
   512 之后的档位需要先明确数据供给方案。


---

# 补充：BS512 六臂已全部完成（15 对严格全 True）

> 上面第 0–5 节是**首次失败**的记录（单文件数据集 + `test -r` 逗号路径）。
> 设计端随后在 `746c339` 批准双文件并修好可读性检查，本轮按更新后的任务书重跑并完成。

## A. 结论

| 项 | 结果 |
|---|---|
| 实测 SHA | `746c339819e32883bf14a81bb9b727fcb1c5fdb7` |
| 实际模型启动数 | **6**（每臂一次）|
| `exit-code.txt` | **0** |
| `pairs.csv` | **15 对**，`strict_pair_comparable` **15/15 True** |
| `warm_native_equal` | **15/15 True** |
| `native_diff` | **15/15 = 0** |
| 每对 `measured_samples` | **512,000**（1000 步 × 512 token）|
| 每步 512 token | 六臂各 **2000 步**（`512 generation tokens` 记录 2000 条）|
| scheduler 抢占 | 六臂日志中 **0 条** |
| 代码修改 | **无**（`746c339` 自带双文件与逐个可读性检查）|

## B. 主表（BS512，两窗口各 1000 步）

| arm | E2E ms/步 | vs vanilla | 完整 MoE ms/步 | vs vanilla | 重排 | token IR | 活跃物理槽 |
|---|---:|---:|---:|---:|---|---:|---:|
| vanilla | **166.466** | — | 72.016 | — | 否 | 1.394 | 334.7 |
| random | **173.951** | **+4.50%** | 76.855 | +6.72% | 否 | 1.481 | 391.0 |
| rr | **156.381** | **−6.06%** | 60.270 | −16.31% | 是 | 1.061 | 398.7 |
| eco_linear | **153.026** | **−8.07%** | 58.805 | −18.35% | 是 | 1.031 | 369.8 |
| eco | **153.886** | **−7.56%** | 58.722 | −18.46% | 是 | 1.033 | 369.7 |
| **eco_smr** | **149.543** | **−10.17%** | **55.638** | **−22.74%** | 是 | 1.033 | 369.7 |

统计窗口（重排前）：vanilla 163.146 / random 170.431 / rr 170.960 /
eco_linear 170.663 / eco 169.521 / eco_smr 166.113 ms/步。

**排序与 BS256 一致**：`eco_smr` 最快 → `eco_linear` → `eco` → `rr` → `vanilla` → `random`。
**且 BS 越大收益越明显**：eco_smr 相对 vanilla 从 BS256 的 −6.17% 扩大到 **−10.17%**，
MoE 层从 −15.67% 扩大到 **−22.74%**。

## C. 价格对照 `eco_linear → eco`

| 项 | eco_linear | eco | Δ |
|---|---:|---:|---:|
| E2E ms/步 | 153.026 | 153.886 | **+0.56%** |
| 完整 MoE ms/步 | 58.805 | 58.722 | **−0.14%** |
| routed MAE | 0.07895 pp | **0.04605 pp** | **−0.0329 pp（fine 更准 41.7%）** |
| `primary_price` | linear（slack 90）| fine（slack 11469）| — |

**与 BS256 完全同型**：fine 价格更准、MoE 层略快（−0.14%），但 E2E 更慢（+0.56%）。
BS256 是 +0.51% / −0.12%，BS512 是 +0.56% / −0.14% —— **两档几乎逐位重合**，
说明这不是噪声，而是"价格精度"与"端到端收益"之间稳定的脱节。

## D. SM-reuse 对照 `eco → eco_smr`

| 项 | eco | eco_smr | Δ |
|---|---:|---:|---:|
| E2E ms/步 | 153.886 | **149.543** | **−2.82%** |
| 完整 MoE ms/步 | 58.722 | **55.638** | **−5.25%** |

BS256 为 −1.94% / −4.26%，BS512 扩大到 **−2.82% / −5.25%** —— SM-reuse 的收益随 BS 增长。
`auction.csv` 中 eco 与 eco_smr 的 offers/accepted 完全相同（345,915 / 226,980），
再次确认收益来自 role 6 复用 dispatch 的 6 个 SM，而非配额差异。

## E. 资源（任务书 §验收 1）

### E.1 显存（`run.json` 的 `stats/measured.initial.device_memory_bytes`）

| arm | 最低剩余（stats）| 最低剩余（measured）| 最高 reserved | 总显存 |
|---|---:|---:|---:|---:|
| vanilla | 23.38 GiB | 23.13 GiB | 63.09 GiB | 95.07 GiB |
| random | 14.94 GiB | 14.69 GiB | **70.44 GiB** | 95.07 GiB |
| rr | 15.26 GiB | 14.02 GiB | 70.44 GiB | 95.07 GiB |
| eco_linear | 15.21 GiB | 13.97 GiB | 70.44 GiB | 95.07 GiB |
| eco | 15.21 GiB | 13.97 GiB | 70.44 GiB | 95.07 GiB |
| eco_smr | 15.21 GiB | **13.97 GiB** | 70.44 GiB | 95.07 GiB |

> `free` 是设备剩余显存读数，**包含 CUDA allocator 之外的占用**（NCCL/NVSHMEM 对称堆、
> CUDA context、驱动预留）。它与 `95.07 − 70.44 = 24.63 GiB` 的差额约 10.7 GiB 即来自这些部分。
>
> **注意 BS512 比 BS256 的余量更大**（BS256 最低 13.03 GiB，BS512 最低 13.97 GiB），
> 因为本档的 prompt 池不同（双文件、含重复，中位长度 732 vs BS256 的 670），
> 每卡 reserved 只增加了 1.24 GiB（69.20 → 70.44），而 KV 只多用了约 0.4 GiB。

### E.2 KV 容量与利用率

| 项 | 六臂共同值 |
|---|---|
| `GPU KV cache size` | **1,597,922 tokens**（BS256 为 1,530,922）|
| `GC_MOE_KV_BYTES` | 21,474,836,480 B = **20.00 GiB** |
| **峰值 KV 利用率** | **23.5%**（六臂完全相同；BS256 为 11.4%）|

注意：1,597,922 与 BS256 的 1,530,922 **不同**，因为本档 `max_model_len` 随 512 条 prompt 的
最大长度（2042）与 `decode_budget`（2519）重算为更大值，每卡 block 数因此变多。
任务书按 1,530,922 规划的容量依据在 BS512 上仍成立（本档实际可用更多），
但**不能把 BS256 的 token 数直接当 BS512 的上限**。

### E.3 prompt 与工作量

| 项 | 值 |
|---|---|
| 选出 prompt | **512 条**（`64/rank × 8`）|
| 长度 min / 中位 / max | 257 / 732 / 2042 |
| 总 prefill token | **463,731** |
| `decode_budget` | 2,519（`1+1000+1000+518slack`）|
| `GC_KV_BUDGET` | 2,300,000（**实际需求 2,048,371，余量 12.2%**）|
| 数据集 | `_1.jsonl` + `_2.jsonl`，顺序固定；**512 条中含重复 prompt** |
| 数据集 SHA256 | `3664daca…`（_1）/ `957d06fe…`（_2），见 `dataset.sha256` |

## F. 拍卖统计（BS512）

| arm | rounds | offers | accepted | 接受率 | 每调用成交 | 无出价轮 | 有出价全拒 |
|---|---:|---:|---:|---:|---:|---:|---:|
| eco_linear | 40,160 | 344,791 | 226,425 | 65.67% | 5.661 | 35 | **0** |
| eco | 40,159 | 345,915 | 226,980 | 65.62% | 5.675 | 37 | **0** |
| eco_smr | 40,159 | 345,915 | 226,980 | 65.62% | 5.675 | 37 | **0** |

对比 BS256（64.38% / 64.26%）：**接受率随 BS 上升 1.3 pp**，
`eligible_donors` 从 1.42M 增至 1.62M，`feasible_donors` 从 1.04M 增至 1.24M。
但在 BS256 观察到的"瓶颈在目的地可行性判定"（`blocked_before_offer` 375k → 380k）
在 BS512 **依然存在且略有增加**。

## G. 环境与预检

| 项 | 值 |
|---|---|
| 测量 SHA | `746c339819e32883bf14a81bb9b727fcb1c5fdb7`（worktree `/tmp/gcmoe-prefix-WOb7V5z5/code`，`git status` 干净）|
| 工作目录 | `/tmp/gcmoe-prefix-WOb7V5z5` |
| 现场 state | `/tmp/gcmoe-main24-bs512-zQ4Vytw6`（**22 GB**；成功路径已按设计释放 `kv/rank*.bin`，清单保留）|
| GPU | 8×H20，driver 580.105.08 |
| 软件 | cutlass 4.8.1 / torch 2.13.0+cu130 / CUDA 13.0 / vllm 0.31.0 |
| 环境 | `GC_MOE_ORDER_PREFILL=1`、`GC_MOE_STABLE_PREFILL_TOPK=1`、`GC_MOE_STABLE_DECODE_TOPK=1`、`GC_KV_BUDGET=2300000`、`GC_MAIN24_SINGLE_BS=512`、`GC_MOE_KV_BYTES=21474836480` |

**两个 top-k 开关实读**：六臂 `run.json` 的 `controls` 中
`GC_MOE_STABLE_PREFILL_TOPK=1`、`GC_MOE_STABLE_DECODE_TOPK=1` **全部为 1**。

**预检**：26 项 CPU PASS；六臂角色组合编译全部通过
（`[1,4,2,3]` 12.2 s、`[1,4,2,5,3]` 18 s 级、`[1,4,2,5,6,3]` 28.2 s）；
八卡 SM-reuse 门禁 `reuse-0.sha` 与 `reuse-1.sha` **逐 rank 完全相同**。

## H. 交付物

`docs/trace_executor_20261119/sm90_main24_stable_decode_bs512_results/`（167 个文件，200 MB）：

- `delivery/bs512/pairs.csv`（15 对，全 True）、`main_table.csv`、`layers.csv`、
  `cost_pairs.csv`、`auction.csv`
- `delivery/bs512/<arm>/`：`run.json`、`runner.log`、`status.txt`（均 `exit_code=0`）、
  `installed-layout.json`、`hardware.csv`、8 份 `rank*.npz`
- `delivery/bs512/checkpoint/`、`stats-demand.npz`、`placement.json`、`workload.sha256`
- `delivery/SHA256SUMS`、`preflight/`、`preflight.log`、`progress.log`、
  **`dataset.sha256`**（两份源文件各自的 SHA256）、`exit-code.txt`（=0）、`state-path.txt`
- `failed-run1/`：首次失败（单文件 `test -r` 路径）的原始日志

## I. 与 BS256 的对照（同一控制集）

| 指标 | BS256 | BS512 | 变化 |
|---|---:|---:|---:|
| vanilla E2E | 94.588 | 166.466 | +76.0%（约 1.76×）|
| eco_smr E2E | 88.757 | 149.543 | +68.5%（约 1.69×）|
| eco_smr vs vanilla | −6.17% | **−10.17%** | 收益扩大 4.0 pp |
| eco vs vanilla | −4.31% | −7.56% | 扩大 3.3 pp |
| rr vs vanilla | −3.80% | −6.06% | 扩大 2.3 pp |
| random vs vanilla | +3.85% | +4.50% | 略增 0.7 pp |
| 峰值 KV | 11.4% | 23.5% | 约 2.06× |
| 最低剩余显存 | 13.03 GiB | 13.97 GiB | 略增 |
| 拍卖接受率 | 64.3% | 65.7% | +1.4 pp |

**eco_smr 相对 vanilla 的收益随 BS 单调放大**，这是 BS256→BS512 最显著的规律。

## J. 提示：1024/2048 档的数据供给

双文件在 `[256,2048]` 只有 **345 条唯一 prompt**。BS512 已用满 512 条（含重复），
**1024 档将需要约 3 倍重复率**，prompt 长度分布会被显著拉伸。
后续档位在开工前需要设计端明确：(a) 是否允许更高重复率；
(b) 是否放宽长度区间；(c) 是否引入新的数据文件。这不影响本轮结论。
