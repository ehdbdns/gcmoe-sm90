# REPLY_STREAM_MBLOCK — 失败：baseline 编译不过

## 结论

**停在 A 轮（`GC_MOE_SM90_STREAM_M_BLOCKS=0`）的步骤 2：常规四角色编译失败。**
B 轮（`STREAM=1`）编译正常，但我没往下跑——任务书要求按顺序 A 然后 B，
且 A/B 必须是同一配置下的对照。

## 版本

| 项 | 值 |
|---|---|
| 起始 SHA | `8c4f6f63e9130911a6f8eb816245b049bf8a3d8e` |
| 执行 SHA | 起始 SHA + 我的 4 处编译兼容修复（单独提交）|
| 是否改语义 | **否**（只改 DSL 不接受的下标/布局写法）|

## 机器

8 × H20 / 580.105.08 / 97871 MiB，起始全部 4 MiB 空闲。
torch 2.13.0+cu130 / CUDA 13.0 / vLLM 0.31.0。

## 步骤 1：通过

工作区干净；`git merge --ff-only` 快进到 `8c4f6f6`；8 卡空闲。

## 步骤 2：失败

```bash
GC_MOE_SM90_STREAM_M_BLOCKS=0 bash scripts/trace_layer/run_aligned_ep8.sh "$rev" --fine-trace
```

- `tests/test_aligned_step_cpu.py` → **ALIGNED_STEP_CPU PASS**
- `smoke_compile_eco.py`（STREAM=0）→ **失败**

```
error[TYPE_DYNAMIC_EXPR_UNSUPPORTED]: A value carried through this `for` is a
plain Python value (Meta value) (a `SM90GCMoERoleKernel`) that cannot be turned
into a Runtime value (Staged value), so the `for` cannot run at run time.
    --> source/gc_moe/impls/sm90_fp8_fp4_gc_moe.py:2517
 2517 |         for expert in cutlass.range(block, self.local, self.grid):
```

对照：

| 配置 | 结果 |
|---|---|
| `STREAM=0` + `GC_MOE_STEP_PROFILE=0` | **1 error** |
| `STREAM=0` + `GC_MOE_STEP_PROFILE=1` | **1 error** |
| `STREAM=1` + `GC_MOE_STEP_PROFILE=0` | 0 error（`[1,2,5,3]`）|
| `STREAM=1` + `GC_MOE_STEP_PROFILE=1` | 0 error（`[1,4,2,5,3]`）|

用 `2d126f9`（8c4f6f6 的父提交）的 impl 在 STREAM=0 下编译**通过**，
确认是本轮提交引入的回归。

## 我做的 4 处编译兼容修复

全部属于 `TYPE_DYNAMIC_EXPR_UNSUPPORTED` 这一类，仓库此前也遇到过（见
`pull_gemm_detail.py` 那轮的 `rec[4]→rec[3]`、`ir_step_epoch` 的 `st_global_u32`）：

| # | 位置 | 原写法 | 改为 |
|---|---|---|---|
| 1 | `_publish_l1_batch` | `self._field(base,'active_expert',Uint32)[batch] = expert` 等 3 条 | `cell(base, off + batch*4, Uint32)[0] = ...` |
| 2 | `_release_stream_l2` | `self._field(base,'stream_l2_batch',Uint32)[l2_index] = batch` | 同上 |
| 3 | stream 分支 | `self._field(base,'expert_count',Uint32)[expert] = hi-lo`（及 `fine_epx`）| 同上 |
| 4 | `_consume_tasks` | `command_words = 5 if self.stream_blocks else 4` | 恒为 5 |

第 4 点是必要的：stream 分支会索引 `command[4]`，而 STREAM=0 时布局只有 4 个字，
DSL 仍会在编译期构建该下标并使整个 kernel 折叠失败。

**修复后**：`STREAM=1` 两种 profile 模式都编译通过；`STREAM=0` **仍然 1 error**。

## 剩余错误的根因（供设计端）

新抽出的 `_pull_slot_rows` 有两个调用形态：

```python
# streaming
for row_lo in cutlass.range(lo, hi, self.max_m):
    self._pull_slot_rows(..., lo, Uint32(row_lo), row_hi)

# baseline
self._pull_slot_rows(..., lo, lo, hi)
```

被调函数内部是 `for row in cutlass.range(row_lo + tid, row_hi, self.threads)`。
传**循环变量**（streaming）时能折叠；传**三个 staged 别名**（baseline）时不能，
DSL 把错误报在外层 `for expert` 上。

试过但**无效**的做法（均已回滚，未留在树里）：

1. 调用点显式 `Uint32(lo)` 包装
2. `if cut_expr(not stream)` 代替 `else`
3. `cutlass.range(lo, hi, 64)` 字面量步长
4. 让 baseline 也走一次单迭代 `cutlass.range` 循环

**建议的修法（二选一，都会改变新代码结构，故未擅自实施）**：

- baseline 路径保留 `8c4f6f6` 之前的**内联 pull 循环体**（不调用新函数）；
- 或让 `_pull_slot_rows` 只接收一个 `[row_lo, row_hi)` 切片（不再接收 `lo` 与
  `row_lo` 两个参数），由调用方保证 `row_lo == lo`。

## 保留的数据

采集目录 `/tmp/gcmoe-aligned-G6ifvad3`（本次失败轮，**未产生任何采样**）：

```
start.sha  gpu.log  runtime.log  cpu.log  compile-base.log  failed-command.txt  status.txt
```

`status.txt`：`step=2: CPU regression and CUDA compile / exit_code=1`

原始 8 份 torch trace / capture.json / fine.json：**不存在**（未进入步骤 3）。

## 未执行

- A 轮步骤 3–4（采集、解析）：未执行
- B 轮（`STREAM=1`）：未执行
- `stream_results/`：空
- 未提交任何 counts / perfetto / detail
