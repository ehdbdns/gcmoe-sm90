# 【SM90】REPLY_SM90_MAIN24_STABLE_BS256 — 稳定 top-k 主表 BS256：random 臂在统计窗口原生校验处停止

> 本回执由 **SM90 执行者**（8×H20）提交。SM100 任务书不在本执行者范围内。

## 0. 首个失败

| 项 | 值 |
|---|---|
| 阶段 | `step=3: capture`，第 2 个臂 **BS=256 / arm=random**，`stats` 统计窗口结束时的原生校验 |
| 退出码 | `exit-code.txt` = **1**（`random/status.txt`: `step=3: capture`, `exit_code=1`）|
| 触发点 | `integration/long_cost.py:172 finish()` → `ref.verify_native_prefix("stats", self._stats_token_start)` → `integration/fixed_tokens.py:57` |
| rank | **8/8 全部失败**（`Worker_TP0..TP7` 各一条相同的 `ValueError`）|
| 请求号 | **req176** |
| 样本位置 | position **664**（相对偏移 **620**，即 `stats` 窗口第 620 步；`_stats_token_start`=44）|
| 原生 token | **3975** |
| 参考 token | **8842** |
| 该窗口失配数 | **4**（不是偶发单点，而是 4 处）|
| Capture workspace | **`/tmp/gcmoe-aligned-CFpU0sve`** |

```
ValueError: stats: 4 native top-1 mismatches; first relative offset 620,
  req176, position 664, native 3975, reference 8842
```

**参考 token 的来源已核实**：random 臂 `publishes_checkpoint()` 为 False，
`fixed_tokens.install()` 读的是 `GC_MOE_FIXED_DIR=/tmp/gcmoe-main24-bs256-c2LF5VbC/replay/tokens.json`
—— 即**本次 vanilla 臂在 `GC_MOE_STABLE_PREFILL_TOPK=1` 路径上新写出的原生轨迹**
（`req176[664]=8842` 与该文件逐位相符，见 `summary.json`）。
所以这不是"和旧参考比"的假差异，而是 **random 臂在实际执行中偏离了本版本的 vanilla 原生轨迹**。

按任务书 §执行「首次差异或资源错误即停止，保留日志和现场」，已停止；
第 3–6 臂（rr / eco_linear / eco / eco_smr）**均未启动**。实际模型启动数 **2**。

## 1. 已完成的臂

| 臂 | 状态 | stats 窗口 | 正式窗口 |
|---|---|---|---|
| **vanilla** | ✅ 完成（1000 统计 + 1000 正式，原生校验通过）| **90.362 ms/步** | **91.143 ms/步** |
| **random** | ❌ stats 窗口原生校验失败（4 处）| 日志迭代均值 79.69 ms（未计入结果）| — |
| rr / eco_linear / eco / eco_smr | 未启动 | — | — |

补充口径（从 `runner.log` 的 `Iteration(...)` 行重算，仅作对账，不作为结果）：
vanilla 统计窗口 1000 步的逐迭代均值为 **74.62 ms**、正式窗口 **74.94 ms**；
random 统计窗口 1000 步为 **79.69 ms**（+6.8%，方向与历史 random 更慢一致）。
两者之差来自设备同步计时窗口（`ten_step_ms`）与日志逐迭代计时的口径差别，**以 `run.json` 的
`ms_per_step` 为准**。

## 2. 失败细节（不启动后续臂的理由）

random 臂在失败前**已完整跑完 1000 个纯解码步**：
`Iteration(117)` … `Iteration(1116)` 全部为
`0 context requests / 0 context tokens / 256 generation requests / 256 generation tokens`，
即窗口本身是满 batch 的（每步 256 token，与任务书要求一致），
失败发生在**窗口结束后的校验**，第 620 步出现 4 处原生 top-1 分歧。

任务书 §验收 2 规定「只有输入、路由、快照读回、预热及正式原生输出全部通过才能发布严格对比」。
random 臂的统计窗口原生输出已与发布者不一致 → 该臂的路由回放不再对应同一份工作量，
后续四个臂同样会以这份不一致的参考做校验，故**不启动**，也不做任何跨臂对比。

## 3. 新参考与旧参考的关系（关键，供设计端判定口径）

本次 vanilla 发布的新参考与上一轮 BS256 的旧参考
（`/tmp/gcmoe-main24-bs256-LIk16dRh/replay/tokens.json`，sha256 `ad68503c…`）**差异极大**：

| 项 | 值 |
|---|---|
| 新参考 sha256 | `880da497abce4830742fb3d060cf52bc7db575b8b71c475de246731c16d536b8` |
| 旧参考 sha256 | `ad68503c07f9c213c7ddeda92593f54b65fe65515b10dd8247ae7f1c714b1f15` |
| `prompt_hash` | 两者相同（`254af9d8…`）——同一个 prompt、同一批请求 |
| 逐位相同的请求 | **101 / 256** |
| 不同请求 | **155 / 256**（req72–186、req214–255 等成簇）|
| 逐 token 不同的位置 | **300,965 / 530,511（56.7%）** |
| 首处差异 | `req80` position 0：新 671 / 旧 2581 |

**这正是 `GC_MOE_STABLE_PREFILL_TOPK=1` 的直接后果**：稳定 top-k 改变了 prefill 的 tie-break 选择，
prefill 输出 token 变化 → 后续自回归轨迹整体改道。因此

1. 上一轮（`REPLY_SM90_STABLE_TOPK`）记录的「相对旧参考 196/196 失配」在本版本下是**预期**的，
   本轮已用新参考替换；
2. **旧版本的任何 Perfetto / rank token / M 块数据都不能与本版本直接对比**，
   需要按本版本重采（本回执 §5 给出结论与现场位置）；
3. req0–req71 逐位相同、req72 起成簇分叉，说明分叉起点与 prefill 的 chunk 分组边界有关
   （共 117 个 prefill 引擎步、220,096 个 prefill token），建议设计端据此确认该边界是否即
   「top-k 平局首次出现在第 72 个请求的 chunk 内」。

## 4. 容量与资源（任务书 §验收 1 要求，取自已完成的 vanilla 臂）

### 4.1 八卡显存（8 rank 完全一致）

| 项 | stats 窗口 | 正式窗口 |
|---|---:|---:|
| 每卡总显存 | 102,085,492,736 B = **95.07 GiB** | 同左 |
| **最低剩余显存** | 24,598,872,064 B = **22.91 GiB** | 24,330,436,608 B = **22.66 GiB** |
| **最高 PyTorch reserved** | 66,339,209,216 B = **61.78 GiB** | 同左 |
| `GC_MOE_KV_BYTES` | 21,474,836,480 B = **20.00 GiB** | 同左 |

> `device_memory_bytes.free` 是任务书要求的设备剩余显存读数，**包含 CUDA allocator 之外的占用**
> （NCCL/NVSHMEM 对称堆、CUDA context、驱动预留）。它与 `95.07 − 61.78 = 33.29 GiB` 的差额
> 约 10.4 GiB 即来自这些部分，所以「free」不能等同于 allocator 可用量。

### 4.2 vLLM 公布的容量

| 项 | 值 |
|---|---|
| GPU KV cache size | **1,530,922 tokens**（Maximum concurrency 329.73x @ 4,643） |
| kv cache group sizes | `[32, 32, 32, 32, 32, 32, 32, 64, 8]`，lcm block 64 |
| 峰值 KV 利用率 | stats 窗口 **9.3%**，正式窗口 **11.4%**（全日志最大 11.4%） |
| random 臂峰值（失败前）| 9.3%（1000 步内）|

1,530,922 与任务书 §512/1024/2048 容量依据中引用的数值完全一致。

### 4.3 每步 256 token 校验

vanilla 两窗口共 2000 条 `Iteration` 日志行全部为 `256 generation tokens`（prefill 段除外），
`long_window()` 的 `rec['gen_tokens'] != [nseq]*1000` 断言未触发 →
「每步都产生 256 个 token」通过。random 臂同样在 1000 步内保持 256 token/步。

## 5. 版本、环境、预检

| 项 | 值 |
|---|---|
| 测量 SHA | `4112ed1229b4fe33a178d074cbd4f7f1460ae083`（代码 worktree `/tmp/rt/sm90_m24s_code`，`git status` 干净）|
| 工作目录 | `/tmp/gcmoe-prefix-XG2u8p7m` |
| 现场 state 目录 | `/tmp/gcmoe-main24-bs256-c2LF5VbC`（331 GiB，含 `kv/` 与 `replay/`）|
| GPU | 8×H20，driver 580.105.08 |
| 软件 | cutlass 4.8.1 / torch 2.13.0+cu130 / CUDA 13.0 / vllm 0.31.0 |
| 代码修改 | **无**（预检一次通过，未做编译修复）|
| 环境 | `GC_MOE_ORDER_PREFILL=1`、`GC_MOE_STABLE_PREFILL_TOPK=1`、`GC_MAIN24_SINGLE_BS=256`、`GC_MOE_KV_BYTES=21474836480`、`GC_MOE_CHECKPOINT_ROOT=/tmp` |

预检：`ALL CPU CHECKS PASSED`、`CPU_TEST_ALL PASS`、`LONG_COST PASS`、`MAIN20_CPU PASS`；
六种角色组合编译全部通过（`[1,4,2,3]` vanilla/random/rr 11.5–11.7 s、`[1,4,2,5,3]`
eco_linear/eco 17.1–17.7 s、`[1,4,2,5,6,3]` eco_smr 27.7 s）；
八卡 SM-reuse 门禁 `reuse-0.sha` 与 `reuse-1.sha` **逐 rank 完全相同**。

## 6. 交付物

`docs/trace_executor_20261119/sm90_main24_stable_bs256_results_blocked/`：

- `delivery/`（vanilla 的 `run.json` + 8 份 `rank*.npz` + `checkpoint/`、random 的完整 `runner.log`
  与 `status.txt`/`failed-command.txt`、`bs256/state-path.txt`、`stats-demand.npz`、`measurement.sha`）
- `preflight/`（六种角色编译日志 + 两份 reuse 门禁 SHA）、`preflight.log`、`progress.log`、`all.log`
- `exit-code.txt`（=1）
- `summary.json`：新旧参考的 sha256、逐位一致请求数、失配位置数、vanilla 计时与显存、random 失败参数
- `SHA256SUMS`（60 个文件，已 `sha256sum -c` 自检通过）

**未提交**：二进制 KV 与 route 回放（`/tmp/gcmoe-main24-bs256-c2LF5VbC`，按要求留在现场）。

## 7. 需要设计端确认

1. random 臂在 stats 第 620 步出现 4 处原生 top-1 分歧（req176 起）是否按任务书「首次差异」处理为
   **停止**并等待新的判定口径？（本执行者按该口径已停止，未启动其余四臂。）
2. 稳定 top-k 使新参考与旧参考在 56.7% 的 token 位置不同（155/256 请求改道），
   此前所有基于旧参考的 Perfetto/rank-token/M 块数据均不可跨版本对比 —— 是否需要按本版本重采。
3. 是否需要在 random 臂上以 `GC_MOE_MATRIX_PREFIX_CHECK=0`（新参考已发布、只做计时）方式继续，
   或先定位 req176 的 4 处分歧再决定。
