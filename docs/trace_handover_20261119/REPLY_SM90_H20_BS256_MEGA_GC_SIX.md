# 回执 · SM90 · 8×H20 · BS256 · Mega 与 GC 六臂

执行者：SM90 八卡执行者。任务书 `docs/trace_executor_20261119/SM90_H20_BS256_MEGA_GC_SIX.md`（commit `74a6127`）。

## 结论：publisher（arm0）在选提示阶段即失败，六臂一行未产出

`exit-code.txt = 1`、`results/arm0/exit = 1`、`results/arm0/status.txt = step=3: capture / exit_code=1`。
arm1–arm6 从未启动（只有 `results/arm0/` 存在）。

## Mega JIT 预检通过（上一轮的阻塞点已解除）

```
Mega JIT preflight passed: /usr/local/cuda/bin/nvcc with
/home/admin/workspace/aop_lab/app_source/deep_ep25_ar_probe_20261025/toolchain/bin/x86_64-conda-linux-gnu-g++
```

wheel 哈希校验 OK（`deep_gemm-2.8.2+7292e7b`）。上一轮的 JIT 模板歧义在本轮未复现。

## 失败点：提示选择与 KV 预算自相矛盾

runner.log 连续两行：

```
[data] picked 256 prompts, len 257/670/2042 (min/med/max), KV need 220096 / budget 740000
only 198 prompts fit (need 256, len range [256,2048], KV budget 740000, steps 2897)
```

### 量化

| 项 | 值 |
|---|---|
| 选中提示的 token 总和 | 220,096 |
| 每流 KV overhead（`steps`） | 2,897 |
| 256 流总需求 | 220,096 + 256×2,897 = **961,728** |
| `GC_KV_BUDGET` | 740,000 |
| **缺口** | **−221,728（需求超预算 30.0%）** |
| 每流预算 / 每流需求 | 2,891 / 3,757 token |

不到 256 条能塞进 740,000 的原因就是这个：**平均每流需要 3,757 token，而预算只够 2,891 token/流**，于是贪心选择只能装下 198 条。

### 两个数字为何矛盾

`scripts/real_decode_100_100.py` 的 `select_decode_prompts` 在 `SM90_SIX_ARM=1` 时分两处被调用：

1. 第 715 行：走 `SM90_SIX_ARM` 分支 → `[p[:256] for p in load_prompts(256, 256, 2048)]`，日志打印 **`[data] picked 256 prompts`**。
2. 第 727 行的 KV 适配循环（最多 3 次）发现 `sum(len(p) + overhead) > kv_pool`，于是**再次**调用 `select_decode_prompts`；但 `SM90_SIX_ARM` 分支**不读预算、原样返回同一批 256 条**，循环空转。
3. 循环 3 次未收敛后，`prompts` 仍是那 256 条；最终由 `load_prompts` 内部按预算过滤时报出 **`only 198 prompts fit`**。

也就是说：`SM90_SIX_ARM` 的「固定取 256 条真实前缀」语义和 `GC_KV_BUDGET=740000` 的容量上限**在数学上不相容**，代码只是把矛盾暴露成了这条错误。

### 顺带发现：截断分支未生效

任务书要求「各取真实前 256 个 token」，但日志里 `len 257/670/2042` 说明**截断没有出现在实际路径上**——若真按 `p[:256]` 截断，长度应恒为 256、总需求约 65,536 + 256×2,897 = 807,168，仍超 740,000，但缺口会小得多。这说明两条路径的提示集也不一致，需要设计端一并确认。

## 已启动次数与状态

| 项 | 值 |
|---|---|
| 语言模型启动 | **1 次**（arm0 publisher，失败） |
| 六臂消费者 | **0 次** |
| 主表 / pairs / verification | **无法产出** |
| 结果盘可用 | 3,613 GB（需 250 GB，充足） |
| 八卡 | 已释放，各 4 MiB、无 compute app、无残留 worker |
| 代码修改 | **无**（diff 为空）|

## 为什么没有自行修复

任务书明确：

> 执行者只可修复不改变语义的编译环境问题，附 diff、工具版本与复验；运行期 OOM、数据不符或环境不满足时归档首错。**不得单臂改 KV、BS、图、精度、放置、步数、前缀**。

本故障需要改动三者之一才能继续，全部落在禁止范围内：

- 提高 `GC_KV_BUDGET` / `GC_MOE_KV_BYTES`（改 KV 配置）
- 降低 `steps`（改步数）
- 真正启用 256-token 截断（改前缀）

因此我只固化首错与现场。

## 建议的修复方向（供设计端裁定，未执行）

1. **让 `SM90_SIX_ARM` 分支真正截断**：返回 `[p[:256] ...]` 并断言 `len(p)==256`，此时总需求约 807,168，仍略超 740,000 → 还需配合第 2 或第 3 项。
2. **放宽预算**：把 `GC_KV_BUDGET` 提到 ≥ 1,010,000（约 13.7 GiB/rank KV），或把 `GC_MOE_KV_BYTES` 提到 14 GiB。
3. **缩短测量步数**（会改变实验语义，最不推荐）。

我需要明确指令指定其中一项，才可附 diff 继续。

## 现场与归档

- 现场目录：`/mnt/fuse/sm90-bs256-six-T4ktsVpa`
  - `results/arm0/runner.log`（第 15–16 行为矛盾对）
  - `results/arm0/status.txt`（`step=3: capture` / `exit_code=1`）
  - `results/arm0/{exit,execution.sha,raw-location,runtime.log,failed-command.txt,wall-seconds}`
  - `results/{gpus.csv,processes.csv,measurement.sha,input.sha256,input.sha256.log,model-mount.txt,mega-import.log,attention-import.log,conversion.jsonl,six-mode}`
  - `launcher.log`（含 Mega JIT 预检通过与 wheel 校验）
- 未生成任何 KV、route bank 或 NPZ（失败发生在 prefill 之前）。
- 已归档到 `docs/trace_executor_20261119/sm90_h20_bs256_mega_gc_six_results/`（仅首错与启动证据，无测量数据）。
