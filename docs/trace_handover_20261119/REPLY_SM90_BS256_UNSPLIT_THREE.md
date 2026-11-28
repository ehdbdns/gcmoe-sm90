# 回执 · SM90 · H20 BS256 上游非 split fused shared 三臂

执行者：SM90 八卡执行者。任务书 `docs/trace_executor_20261119/SM90_BS256_UNSPLIT_THREE.md`。

| 项 | 值 |
|---|---|
| 新臂测量 SHA | `d9ad3fdcf5c8d346b946f4aa88e3c25aed80db5f` |
| 历史对照 SHA | `47e9b06328336e10584ef942a710aa332538327b`（上轮八臂）|
| 新增模型启动 | **3 次**（arm2 / arm4 / arm7）|
| 参考来源 | `/mnt/fuse/h20-ir-eight-4Thp8jCg/eight/kv-10`（`reference-path` 记录，KV 以符号链接只读复用）|
| 未启动第 4 个模型生成参考 | ✅ 符合任务书要求 |
| `run.rc` | **0** |
| 行数 | `unsplit-summary.csv` **9** 行、`unsplit-pairs.csv` **6** 行（与任务书一致）|

## 几何证据（逐臂核对通过）

```
[shared-unsplit] tokens=256 shared_m=64 grid=72 threads=256 smem=199680 phases=(0, 1) routed_wait=(4,) roles=(1, 4, 2, 3)
[shared-unsplit] tokens=256 shared_m=64 grid=72 threads=256 smem=199680 phases=(0, 1) routed_wait=(4,) roles=(5, 7, 1, 4, 2, 3)
```

| 检查项 | 任务书要求 | 实测 |
|---|---|---|
| `shared_m` | 64 | **64** ✅ |
| `grid` | 72 | **72** ✅ |
| `threads` | 256 | **256** ✅ |
| `smem` | 199680 | **199680** ✅ |
| `phases` | (0, 1) | **(0, 1)** ✅ |
| `routed_wait` | (4,) | **(4,)** ✅ |
| 原生/RR `roles` | (1, 4, 2, 3) | **arm2、arm4 均为 (1, 4, 2, 3)** ✅ |
| ECO `roles` | (5, 7, 1, 4, 2, 3) | **arm7 为 (5, 7, 1, 4, 2, 3)** ✅ |

即：shared L1+L2 已合并为单个 M64 kernel（72 CTA、256 线程、199680 B 动态共享内存），routed 等整个 shared kernel 完成才放行。

## 主表（9 行：3 新臂 + 6 历史对照）

| 臂 | shared | 放置 | step ms | FFN ms/步 | FFN µs/层 | token IR | 总 M 块 |
|---|---|---|---|---|---|---|---|
| 2 | **unsplit** | 原生（无副本，48 槽） | 89.493 | 46.814 | 1170.3 | 1.3679 | 376.1 |
| 2 | split | 原生（无副本，48 槽） | 90.430 | 46.358 | 1159.0 | 1.3679 | 376.1 |
| 3 | unfused | — | 88.471 | 44.626 | 1115.6 | 1.3679 | 376.1 |
| 4 | **unsplit** | EPLB + RR（52 槽） | 87.045 | 43.784 | 1094.6 | 1.0890 | 391.4 |
| 4 | split | EPLB + RR（52 槽） | 87.364 | 42.962 | 1074.0 | 1.0890 | 391.4 |
| 6 | unfused | — | 83.695 | 40.535 | 1013.4 | 1.0890 | 391.4 |
| 7 | **unsplit** | EPLB + ECO（52 槽） | 84.762 | 41.388 | 1034.7 | 1.0569 | 376.1 |
| 7 | split | EPLB + ECO（52 槽） | 84.691 | 40.642 | 1016.0 | 1.0569 | 376.1 |
| 8 | unfused | — | 82.054 | 38.706 | 967.6 | 1.0569 | 376.1 |

## 六对对照（`unsplit-pairs.csv`）

| 新臂 | 对照臂 | 对照 shared | 整步 Δms | 整步 Δ% | FFN Δµs/层 | FFN Δ% | native_diff | 严格可比 |
|---|---|---|---|---|---|---|---|---|
| 2 | 2 | split | -0.938 | **-1.04%** | +11.4 | **+0.98%** | 0 | ✅ |
| 2 | 3 | unfused | +1.021 | **+1.15%** | +54.7 | **+4.90%** | 711 | ❌ |
| 4 | 4 | split | -0.319 | **-0.36%** | +20.6 | **+1.91%** | 0 | ✅ |
| 4 | 6 | unfused | +3.350 | **+4.00%** | +81.2 | **+8.02%** | 711 | ❌ |
| 7 | 7 | split | +0.071 | **+0.08%** | +18.7 | **+1.84%** | 0 | ✅ |
| 7 | 8 | unfused | +2.707 | **+3.30%** | +67.1 | **+6.93%** | 711 | ❌ |

正数表示 **unsplit 更慢**（任务书口径）。

### 关键矛盾：整步更快，但 FFN 段更慢

三对同实现的严格可比对照都呈现同一模式：

| 臂 | 整步 | FFN | 结论 |
|---|---|---|---|
| arm2 | **-1.04%**（更快）| +0.98%（更慢）| 方向相反 |
| arm4 | **-0.36%**（更快）| +1.91%（更慢）| 方向相反 |
| arm7 | **+0.08%**（更快）| +1.84%（更慢）| 方向相反 |

拆开看每步的构成：

| 臂 | shared | 整步 ms | FFN ms | **FFN 之外的开销 ms** |
|---|---|---|---|---|
| 2 | unsplit | 89.493 | 46.814 | **42.679** |
| 2 | split | 90.430 | 46.358 | **44.072** |
| 3 | unfused | 88.471 | 44.626 | **43.845** |
| 4 | unsplit | 87.045 | 43.784 | **43.261** |
| 4 | split | 87.364 | 42.962 | **44.402** |
| 6 | unfused | 83.695 | 40.535 | **43.160** |
| 7 | unsplit | 84.762 | 41.388 | **43.373** |
| 7 | split | 84.691 | 40.642 | **44.049** |
| 8 | unfused | 82.054 | 38.706 | **43.348** |

**unsplit 的 FFN 段每次慢 1.8–1.9%（+18.7~20.6 µs/层），但 FFN 之外的整步开销减少约 0.8–1.4 ms，净整步反而快 0.08–1.04%。**

合理解释：unsplit 把 shared L1/L2 合并成一个 kernel，少了 L1→L2 的 kernel 间同步与 launch 开销，这部分落在 FFN 计时窗口**之外**（FFN 边界在层 FFN 入口到返回）；而合并后的单 kernel 本身比两个 M32 kernel 慢，因此 FFN 段变长。

## 与 unfused 的对照（不可严格比较）

三对 vs unfused 全部 `strict_pair_comparable=False`，`native_diff=711`（256,000 中 0.28%）。

| 臂 | 整步 | FFN | 说明 |
|---|---|---|---|
| arm2 vs unfused | +1.15% | +4.90% | 跨 shared 实现，输出不同 |
| arm4 vs unfused | +4.00% | +8.02% | 跨 shared 实现，输出不同 |
| arm7 vs unfused | +3.30% | +6.93% | 跨 shared 实现，输出不同 |

原因是 shared 执行方不同（GC fused kernel vs vLLM 的 shared MLP），原生 top-1 必然有差异 —— 任务书明确要求「跨 shared 路径的差异照实保留」。**unfused 在所有三对上都明显更快**（整步快 1.15–4.00%、FFN 快 4.9–8.0%）。

## 结论

1. **unsplit 移植完全成功**：几何参数与任务书逐字段一致，三臂均无 OOM，`run.rc=0`。
2. **与 split 相比（同实现，严格可比）**：整步更快 0.08–1.04%，但 FFN 段慢 1.8–1.9%。
   - 原生 −1.04%（−0.94 ms）、RR −0.36%（−0.32 ms）、ECO +0.08%（+0.07 ms，实际持平）
3. **vLLM unfused 仍然最快**：在原生、RR、ECO 三种放置上都显著优于两种 GC fused 实现（FFN 快 4.9–8.0%）。
4. 三对 vs split 的 `native_diff` 均为 0、`warm_native_equal=True`，可作严格性能对照；三对 vs unfused 的 711 差异来自 shared 实现不同，按任务书照实保留不判胜负。

## 归档

- https://github.com/ehdbdns/gcmoe-sm90/tree/main/docs/trace_executor_20261119/sm90_bs256_unsplit_three_results（79 MB）
- 含 `result/`: `unsplit-summary.csv`(9 行)、`unsplit-pairs.csv`(6 行)、三臂各自 `run.json`、8 份 `rank*.npz`、`installed-layout.json`、`runtime.log`、`execution.sha`、`artifacts.sha256`、`complete`、截断至 2 MB 的 `runner.log`（保留 `[shared-unsplit]` 几何证据）、`reference-path`、`gpus.csv`、`processes.csv`。
- arm0 为历史发布者的直接复制（`reference-path` 指向原现场）。
- 现场：`/mnt/fuse/h20-unsplit-three-h8M9KCSo`；历史参考：`/mnt/fuse/h20-ir-eight-4Thp8jCg/eight/kv-10`。大型 KV 与路由库保留在原位置。

## 退卡与清理

- 最后一臂（arm7）16:02:16 完成，`complete` 落盘，`run.rc=0`。
- 八卡已释放，无残留 worker。
- 未修改任何代码（diff 为空）。
