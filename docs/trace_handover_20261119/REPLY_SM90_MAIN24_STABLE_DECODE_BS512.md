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
