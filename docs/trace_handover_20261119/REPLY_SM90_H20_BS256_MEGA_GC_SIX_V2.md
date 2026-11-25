# 回执 · SM90 · BS256 Mega/GC 六臂（第二版，按用户指示提前停止）

执行者：SM90 八卡执行者。任务书 `docs/trace_executor_20261119/SM90_H20_BS256_MEGA_GC_SIX.md`（第二次执行），测量 SHA `3c2d8e8556fa7ffddf26a1f51e979dec34d7f61e`。

## 摘要

用户在 arm1 完成后指示「跑完当前臂先停」。因此本轮回执只覆盖 **arm0（发布者）、arm1（Mega 原生）、arm5（Mega EPLB）、arm2（GC 原生 fused）** 四个臂，arm3/arm4/arm6 未启动。**主表无法补齐六行**，本回执按任务书「失败或缺失行写明原因，禁止填 0 替代」如实列出。

## 上一轮失败点已解决

`d80a616` 回执指出 arm0 在选提示阶段失败（256 条需求 961,728 token vs 预算 740,000）。本轮 `3c2d8e8` 修复后在模型加载前打印：

```
[data] picked 256 prompts, len 256/256/256 (min/med/max), KV need 65536 / budget 740000
[six-budget] prefixes=256 tokens=65536 per_stream_overhead=2449 required=692480 limit=740000
[data] 256 prompts (32/rank x 8), len 256..256
```

- 256-token 截断已真正生效（长度恒为 256）
- `required=692480 < limit=740000`，余量 47,520
- 与任务书给定的预期行 **逐字段吻合**

Mega JIT 预检同样通过（上一轮的另一个阻塞点）：
```
Mega JIT preflight passed: nvcc with .../toolchain/bin/x86_64-conda-linux-gnu-g++
```

## 四臂状态

| 臂 | 配置 | exit | complete | 产物 | 说明 |
|---|---|---|---|---|---|
| arm0 | 发布者（vanilla，仅生成参考）| **0** | ✅ | 8×NPZ + run.json | 不进入主表 |
| arm1 | **Mega · vLLM shared · 原生 48 槽** | **0** | ✅ | 8×NPZ + run.json | **成功** |
| arm5 | **Mega · vLLM shared · EPLB 56 槽** | **1** | ❌ | 无 | **OOM（见下）** |
| arm2 | GC · fused split shared · 原生 48 槽 | 无（被外部终止）| ❌ | 无 | 跑满 1000 步，收尾阶段被 `pkill` 波及 |
| arm3 / arm4 / arm6 | GC 其余三臂 | — | — | — | **未启动**（用户要求提前停止）|

## arm1（Mega 原生）成功：本轮唯一完整的新测量

| 指标 | 值 |
|---|---|
| `[six-memory] post-warmup free` | **10,715,725,824 B ≈ 9.98 GiB** |
| `[warm] 1000 decode steps` | **92.95 ms/step** |
| `measured.ms_per_step` | **93.162 ms/步** |
| prefill 步数 | 177 |
| batch / layers | 256 / 40 |
| 层序对齐 | `stats first_epoch=7481`，`(7481−1) % 40 = 0` ✅ |
| KV 预算 | `required=692480 / limit=740000` |

**这是本项目第一次拿到 Mega 后端的有效整步耗时**（上一轮两条 Mega 臂都在图捕获前 OOM，无任何有效整步或 FFN 耗时）。

## arm5（Mega EPLB）OOM 详情

发生在 **kernel warmup 阶段**（不是 KV 分配阶段）：

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 128.00 MiB.
GPU 7 has a total capacity of 95.07 GiB of which 35.31 MiB is free.
Including non-PyTorch memory, this process has 94.97 GiB memory in use.
Of the allocated memory 89.63 GiB is allocated by PyTorch, and 2.56 GiB is reserved but unallocated.
```

| 指标 | arm5（EPLB 56 槽） | arm1（原生 48 槽） |
|---|---|---|
| `[six-memory] post-warmup free` | 4,730,454,016 B ≈ **4.41 GiB** | 10,715,725,824 B ≈ **9.98 GiB** |
| 峰值显存（nvidia-smi） | 96,997 MiB | 85,863 MiB |
| 结果 | **OOM** | 通过 |

差距约 5.6 GiB。相比上一轮（EPLB 臂余 3.00 GiB、原生余 15.96 GiB），10 GiB KV 已把两者都推高，但 **EPLB 的 8 个额外物理槽（56 vs 48）仍然超出**。任务书禁止单臂改 KV/槽数，故未做任何调整。

## arm2（GC 原生 fused）：数据未落盘，原因是我终止 launcher

- 全程无 OOM，`[six] GC M=64 roles=(1, 4, 8, 2, 3)` 已打印（任务书要求的 fused 证据），显存 47.7 GiB
- `[warm] 1000 decode steps: 88.73 ms/step`
- 正式窗口跑到 Iteration 2176（与 arm0/arm1 相同终点），**1000 步已全部执行**
- 但随后我按用户指示 `pkill -TERM -f 'run_eight.sh --six-bs256'` 停止 launcher，该信号波及 arm2 的进程树，导致它在收尾阶段被 `SIGKILL`，`run.json`/NPZ 未写出
- 这是我的操作造成的，不是脚本或用例缺陷

因此 arm2 的 `88.73 ms/step` 只是 `[warm]` 统计窗口读数，**不能当作正式测量值使用**。

## 已产出的数据（可用）

| BS | 臂 | 后端 | shared | 放置 | 整步 ms/步（正式窗口）|
|---|---|---|---|---|---|
| 256 | arm0 | 发布者 | — | vanilla | 89.738（参考）|
| 256 | **arm1** | **Mega (DeepGEMM FP8×FP8)** | vLLM | 原生 48 槽 | **93.162** |

`[warm]` 统计窗口读数（非正式）：arm0 87.77、arm1 92.95、arm2 88.73 ms/step。

对比：同一 BS256 配置下，此前 GC 后端在 replay 口径的整步约 87–89 ms/步。Mega 原生整步 93.16 ms 比 GC 原生（arm2 的 warm 读数 88.73）慢约 **+5.0%**，但 arm2 无正式测量值，**此对比不构成结论**。

## 未完成项

- arm3（GC · vLLM shared · 原生）未启动
- arm4（GC · fused · EPLB）未启动
- arm6（GC · vLLM shared · EPLB）未启动
- 因此主表缺 4 行、`pairs.csv` 缺全部 12 对、`verification.json` 未生成、summary 未生成

## 若需续跑

任务书支持在同一 SHA、同一现场续跑：

```bash
bash scripts/sm90/run_eight.sh --six-bs256 --resume "$job/results"
```

已完成臂（arm0、arm1）会按校验和跳过。若要先修 arm5 的 OOM，需要设计端在下列之一做出裁定（均超出执行者权限）：

1. 把 `GC_MOE_KV_BYTES` 从 10 GiB 降到 6–8 GiB（arm5 余量 4.41 GiB，还需约 2 GiB 缓冲）
2. 单独为 EPLB 臂降低 KV，接受与原生臂不同的 KV 容量（会破坏公平性口径）
3. 放弃 arm5/arm4/arm6 三条 EPLB 臂，只交付原生三臂（1、2、3）

## 现场与退卡

- 现场：`/mnt/fuse/sm90-bs256-six-v2-Y9tGZbel`
  - `results/arm0/`（complete，8×NPZ + run.json + runner.log）
  - `results/arm1/`（complete，8×NPZ + run.json + runner.log）
  - `results/arm5/runner.log`（OOM 完整栈，第 22057 行起）
  - `results/arm2/runner.log`（64 MB，含 `[six] GC M=64 roles=(1,4,8,2,3)` 与 1000 步完整日志）
  - `results/{gpus.csv,processes.csv,measurement.sha,six-mode,input.sha256*,model-mount.txt,mega-import.log,mega-jit-*/}`
- 八卡已释放：各 4 MiB、无 compute app、无残留 worker
- 磁盘剩余 3,613 GB
- 代码修改：**无**（diff 为空）

## 归档内容说明

`docs/trace_executor_20261119/sm90_h20_bs256_mega_gc_six_v2_results/`（230 MB）：

| 内容 | 说明 |
|---|---|
| `launcher.log` | 含 wheel 校验与 Mega JIT 预检通过 |
| `results/arm0/` | 发布者完整：`complete`、`artifacts.sha256`、8×NPZ、`run.json`、`installed-layout.json`、`runner.log` |
| `results/arm1/` | **Mega 原生完整**：同上全套 |
| `results/arm2/runner.log` | 前 300 KB（保留 `[six] GC M=64 roles=(1,4,8,2,3)` 与 1000 步记录），64 MB Kineto 转储已截断 |
| `results/arm5/` | OOM 首错：`runner.log`、`status.txt`、`failed-command.txt` |
| `results/replay/` | 仅布局 JSON 与 `tokens.json`；12 GB 的 `stats-rank*.pt` route bank 未提交 |
| `results/` 根 | `gpus.csv`、`processes.csv`、`measurement.sha`、`six-mode`、`input.sha256*`、`model-mount.txt`、`mega-import.log`、`mega-jit-tir8FRmC/` |

大型 KV（172 GB `kv/*.bin`）与 route bank 保留在现场 `/mnt/fuse/sm90-bs256-six-v2-Y9tGZbel/results/`，未进入 git。
