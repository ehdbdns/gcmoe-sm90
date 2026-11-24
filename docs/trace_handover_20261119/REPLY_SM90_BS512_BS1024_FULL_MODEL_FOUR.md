# 回执 · SM90 · BS512 / BS1024 整模四臂复测

执行者：SM90 八卡执行者。任务书 `docs/trace_executor_20261119/SM90_BS512_BS1024_FULL_MODEL_FOUR.md`（commit `74a5395`）。

## 结论：BS512 vanilla 第一臂即失败，无法产出任何主表行

`bs512.exit = 1`。失败发生在统计窗口收尾的 `_gc_moe_long_finish('stats')`，八个 rank 同型报错：

```
[route-diag] pos=40000 n=40000 epochs_eq=True layers_eq=False layer0=0 counts_min=0 ord_ok=True
ValueError: incomplete 1000-step route, layer, epoch or ordinal history
```

## 诊断：只有 layer 序列判据失败，其余全部通过

`integration/long_cost.py::finish_phase` 的六个判据：

| 判据 | 观测 | 结论 |
|---|---|---|
| `position == len(epochs)` | 40000 == 40000 | ✅ 1000 步 × 40 层，调用数完全正确 |
| `epochs == arange(first, first+n)` | True | ✅ epoch 连续，窗口边界干净 |
| `layers[0] == 0` | 0 | ✅ 首项层号正确 |
| `counts >= 0` | min = 0 | ✅ 物理槽计数合法 |
| `ordinals` 一致 | True | ✅ 路由 ordinal 全对 |
| **`layers == (epochs - 1) % 40`** | **False** | ❌ **唯一失败项** |

也就是说：**观测到的数据量完整、epoch 连续、ordinal 正确，只是整体层序相对 `(epoch-1) % 40` 这个基准存在固定偏移。** `layer0=0` 而 `layers_eq=False` 同时成立，说明第 0 项层号为 0，但后续层序与该公式不吻合 —— 典型的**窗口起点未对齐到 40 层边界**。

## 根因：统计窗口起点没有对齐模型层周期，而对齐守卫在本任务里被关掉了

`long_cost.py::begin()` 里本身有对齐守卫：

```python
if os.getenv("SM90_EIGHT_ARM"):
    next_epoch = int(buf.header[0].item())
    if (next_epoch - 1) % tape.layers:
        raise ValueError(f"{phase}: rank {rank} layer 0 would use epoch {next_epoch}")
```

但 `scripts/sm90/run_full_model_four.sh` 第 40 行显式清除了它：

```
unset SM90_COLLECT_SE_ROUTES SM90_EIGHT_ARM SM90_EIGHT_BACKEND SM90_PAIRED_SMALL GC_BS256_TRACE_DIR
```

于是 `begin()` 不会在窗口开始前检查 `(next_epoch - 1) % 40`，也不会主动对齐。graph 捕获与 warmup 期间的 FFN 调用会推进设备侧 epoch 计数器，等 prefill 结束、统计窗口真正开始时，起点已经偏离 40 的整数倍，整个 1000 步窗口因此相对 `(epoch-1) % 40` 系统性错位。

时间线佐证（BS512 vanilla）：

| 时刻 | 事件 |
|---|---|
| 00:42:44 | flashinfer sparse MLA warmup |
| 00:42:47 | `Capturing CUDA graphs (FULL)` 完成 |
| 00:42:55 | `Iteration(0)`，prefill 开始 |
| 00:48:07 | `Iteration(267)`，首个满批 512 |
| 00:50:46 | `Iteration(1266)`，统计窗口收尾 → `route-diag` → 失败 |

从 267 到 1266 正好 999 步，decode 满批无 context 混入（统计窗口内 `context requests` 计数为 0），说明**窗口本身是干净的**，问题只在层相位。

## 已启动次数与当前状态

| 项 | 状态 |
|---|---|
| BS512 vanilla | ❌ 失败（`exit=1`，`step=3: capture`）|
| BS512 random / rr / eco | 未启动（`run_main20.sh` 在首臂失败即 `exit`）|
| BS1024 vanilla | ⏳ 已启动，同位置（正在 prefill / 即将进入统计窗口）|
| BS1024 其余三臂 | 未启动 |
| 语言模型启动 | **2 次**（BS512 vanilla、BS1024 vanilla）|
| 主表 / pairs / layers / auction | **无法产出任何行** |

BS1024 走完全相同的脚本路径与相同的 `unset`，预期会在同一判据上失败。

## 按任务书要求的处置

任务书规定：

> 执行者只可处理保持语义的编译故障并附 diff。其他故障交付具体阶段、原始报错、已启动次数和已完成结果。

本故障**不是编译故障**，而是窗口对齐判据与脚本 `unset` 组合导致的运行期失败，属语义/校验范畴，因此本执行者**未做任何代码修改**（diff 为空），只保留现场与首错。

## 建议的修复方向（供设计端决定，未执行）

三个候选中任一即可，但都会改变窗口起点或校验口径，需设计端裁定：

1. **恢复守卫**：在 `run_full_model_four.sh` 中不再 unset `SM90_EIGHT_ARM`（或在 `four==1` 时单独设一个对齐开关），让 `begin()` 提前拒绝或对齐。
2. **主动对齐**：在 `begin('stats')` 前把设备 epoch 推进到下一个 `40k+1`（丢弃不足一层的残余调用），代价是丢弃若干 warmup 调用，不影响 1000 步窗口本身。
3. **放宽判据**：把 `layers == (epochs-1) % 40` 改成"层序是一个 40 周期的循环移位"，即允许固定相位偏移。这会改变校验语义，风险最高。

## 现场与归档

- 现场目录：`/mnt/fuse/sm90-four-full-JblIP9a`
  - `delivery-512/bs512/vanilla/runner.log`（失败首错完整栈，5120–5150 行附近）
  - `delivery-512/bs512/vanilla/status.txt`（`step=3: capture`、`exit_code=1`）
  - `delivery-512/bs512/vanilla/{hardware.csv,preexisting-processes.csv,gpu.log,runtime.log,execution.sha}`
  - `delivery-1024/`（BS1024 进行中）
  - `launcher.log`、`inputs.sha256`、`input-check.log`、`model-mount.txt`、`measurement.sha`
- 未产生任何 KV / route bank 成品（失败发生在 `finish_phase` 落盘之前）。
- 权重加载证据：48/48 用时 3 秒（14.68 it/s），本地 ext4 副本 `/mnt/fuse/models/DeepSeek-V4.1-Flash`，`GC_MOE_NO_PREFETCH=1`，`warm_page_cache.sh` 已预热。**加载优化全部生效**。
- 八卡在失败臂退出后由 launcher 自动交给 BS1024 使用；当前仍在 BS1024 上运行。

## 请求

请确认修复方向后重新下发任务书。若允许我按方案 1（恢复对齐守卫，属于"保持语义"的健康检查）或方案 2（对齐 epoch）修改，请明确指定其中一个，我会附 diff 执行。
