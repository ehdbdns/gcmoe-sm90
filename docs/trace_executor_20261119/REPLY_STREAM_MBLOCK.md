# REPLY_STREAM_MBLOCK — 失败：baseline（stream=0）仍无法编译

## 结论

**停在 A 轮（`GC_MOE_SM90_STREAM_M_BLOCKS=0`）的步骤 2，未进入采集。**
`5fabf5b` 已经按建议把 baseline 的整槽 pull 恢复为内联实现，但**编译仍然失败**，
错误位置与上一轮完全相同（`sm90_fp8_fp4_gc_moe.py:2517`，外层
`for expert in cutlass.range(...)`）。

## 版本

| 项 | 值 |
|---|---|
| 起始 SHA | `5fabf5bf380468bae03b4f6075f7e3727e5eeb8a` |
| 执行 SHA | 同起始（**本轮我没有改任何代码**）|
| 工作区 | 干净（`git diff` 为空）|

> 说明：我上轮提交的 `f78fd98`（4 处 DSL 兼容修复）**已被 `5fabf5b` 完整包含**，
> 现在 `5fabf5b` 的树与我的修复版逐字节一致，`git diff` 为空。

## 编译矩阵（本轮实测）

| 配置 | 结果 |
|---|---|
| `STREAM=0` + `GC_MOE_STEP_PROFILE=0` | **1 error** ❌ |
| `STREAM=0` + `GC_MOE_STEP_PROFILE=1` | **1 error** ❌ |
| `STREAM=1` + `GC_MOE_STEP_PROFILE=0` | 0 error ✅ `[1,2,5,3]` |
| `STREAM=1` + `GC_MOE_STEP_PROFILE=1` | 0 error ✅ `[1,4,2,5,3]` |

`tests/test_aligned_step_cpu.py` → **ALIGNED_STEP_CPU PASS**

错误原文：

```
error[TYPE_DYNAMIC_EXPR_UNSUPPORTED]: A value carried through this `for` is a
plain Python value (Meta value) (a `SM90GCMoERoleKernel`) that cannot be turned
into a Runtime value (Staged value), so the `for` cannot run at run time.
    --> source/gc_moe/impls/sm90_fp8_fp4_gc_moe.py:2517
 2515 |             pull_phase_smem[tid] = Uint32(0)
 2516 |         cute.arch.sync_threads()
 2517 |         for expert in cutlass.range(block, self.local, self.grid):
```

## 本轮排除的可能（都已实测）

| 假设 | 实验 | 结果 |
|---|---|---|
| 我的 4 处修复是问题 | 用 `5fabf5b` **原版**（不含我的改动）编译 | **仍 1 error**（同位置）|
| stream 分支代码导致 | 把 `if const_expr(self.stream_blocks)` 改成 `and False` 屏蔽 | 仍 1 error |
| `_publish_l1_batch` 的 `cell()` 写法 | 改回 `self._field(...)[idx]` | 仍 1 error |
| baseline 调用 `_pull_slot_rows` 传别名 | `5fabf5b` 已改为内联 | 仍 1 error |
| 显式 `Uint32()` 包装调用参数 | 试过 | 仍 1 error |
| `if const_expr(not stream)` 代替 `else` | 试过 | 仍 1 error |
| 字面量步长 `cutlass.range(lo, hi, 64)` | 试过 | 仍 1 error |
| 单迭代循环包裹 baseline | 试过 | 仍 1 error |

所有试验均已回滚，工作区干净。

## 我能确定的边界

1. **`2d126f9`（8c4f6f6 的父提交）在 `STREAM=0` 下编译通过** —— 上轮已实测。
2. **`5fabf5b`（含内联恢复）在 `STREAM=0` 下仍失败** —— 本轮实测。
3. 因此 `8c4f6f6` 引入的**某个非 pull 路径的改动**破坏了 baseline 的 CuTe lowering，
   而它不在我已排查的范围内（pull 循环、stream 分支、`_publish_l1_batch`、
   `command` 布局、`_release_stream_l2`、`block_ready`）。

**下一步建议**：用 `git bisect` 式的方法，在 `2d126f9..5fabf5b` 之间把
`8c4f6f6` 的 419 行改动按函数分成几段逐个加回，定位触发点。这需要改代码结构，
超出「最小编译兼容修复」授权，故未实施。

## 保留的数据

采集目录（本轮，未产生任何采样）：
`/tmp/gcmoe-aligned-<本轮未创建>` —— 因为入口在步骤 2 就退出，
`run_aligned_ep8.sh` 已创建目录并写入：

```
start.sha  gpu.log  runtime.log  cpu.log  compile-base.log  failed-command.txt  status.txt
```

上轮目录 `/tmp/gcmoe-aligned-G6ifvad3` 同样保留。

**原始 8 份 torch trace / capture.json / fine.json：不存在**（未进入步骤 3）。

## 未执行

- A 轮步骤 3–4：未执行
- B 轮（`STREAM=1` 采集）：未执行（任务书要求 A→B 顺序，且 A/B 必须可比）
- `stream_results/`：空
- 未提交任何 counts / perfetto / detail

## 备注

`STREAM=1` 的编译是**干净的**（两种 profile 模式都通过）。如果设计端确认
可以先用 `STREAM=1` 单独采集（放弃本轮 A/B 对照），我可以立即执行——
但需要明确指示，因为这与任务书「按顺序跑旧方案和新方案」冲突。
