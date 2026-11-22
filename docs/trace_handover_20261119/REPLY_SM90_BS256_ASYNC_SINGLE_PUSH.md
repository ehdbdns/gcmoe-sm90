# 【SM90】REPLY_SM90_BS256_ASYNC_SINGLE_PUSH — single_push 在第一个统计步即大面积原生失配

> 本回执由 **SM90 执行者**（8×H20）提交。SM100 任务书不在本执行者范围内。

## 0. 结论：**输出审计变化，按失败回执**

| 臂 | 状态 | 结果 |
|---|---|---|
| **baseline** | ✅ `exit_code=0` | 完整 1000+1000 步，历史校验全部通过 |
| **single_push** | ❌ `exit_code=1` | **stats 窗口第 1 个被接受的 token 即失配，152,669 / 256,000 处** |

任务书 §执行 明确：

> 保留原固定输入、路由、源内 ordinal、EPLB、ECO plan、成本模式、stream、输出和逐 rank 审计。
> **若输出或审计变化，按失败回执，保留首个分歧。**

本轮 single_push 的输出与原 ECO 路径**大面积不同**，属该条款。已停止，未启动第三臂（本身只有两臂）。

## 1. 首次失败（任务书 §交付 3 要求）

| 项 | 值 |
|---|---|
| 阶段 | `step=3: capture`，`arm=single_push` 的 **stats 窗口校验** |
| 退出码 | **1** |
| 触发点 | `integration/long_cost.py` 的 `finish('stats')` → `fixed_tokens.verify_native_prefix("stats", …)` |
| rank | **8/8 全部**（`Worker_TP0..TP7`，日志中 17 处该断言）|
| 首个失配 | **`req0`，position 117，relative offset 0** |
| native | **80729** |
| reference | **294** |
| 失配总数 | **152,669 / 256,000（59.6%）** |
| Capture workspace | **`/tmp/gcmoe-aligned-Xv03HEYY`** |

```
ValueError: stats: 152669 native top-1 mismatches;
  first relative offset 0, req0, position 117, native 80729, reference 294
```

**prefill 与模型加载本身正常**：117 个 prefill 引擎步完成，`Iteration(117)` 起进入 decode，
1000 个 decode 步全部 256 token/步（`long_window` 的 `gen_tokens` 断言未触发）。
失败发生在**窗口结束后的原生 token 比对**。

## 2. 数量的含义：不是数值抖动

| 对照 | 失配数 / 总数 | 比例 |
|---|---:|---:|
| 本臂 single_push（本次）| **152,669 / 256,000** | **59.6%** |
| `REPLY_SM90_DECODE_VANILLA_REPLAY`（旧缺陷）| 7 / 256,000 | 0.003% |
| `REPLY_SM90_MAIN24_STABLE_BS256`（旧缺陷）| 4 / 256,000 | 0.002% |

**59.6% 与 0.002–0.003% 差四个数量级**。这不是 top-k 平局或归约顺序的微小差异，
而是**单副本 token 的物理行序/发布时机改变了计算结果**：
首个失配出现在 `req0` 的 **position 117（relative offset 0）**，即 decode 的第一步就不同。
从第一步起就不同，说明它不是逐步累积的漂移。

## 3. 已排除的因素

| 项 | 证据 |
|---|---|
| 数据/输入 | `DATASET` 与 baseline **完全相同**（同一单文件），`prompt_hash` 一致 |
| 历史 KV 与回放 | 两臂共用保留现场 `/tmp/sm90-bs256-trace-pbpMbH4u`；启动前逐 rank 逐阶段 `cmp` KV manifest 通过 |
| 硬件差异 | `hardware.csv` 与历史 `ref/eco/hardware.csv` `diff -u` 无差异（driver 580.105.08、500 W、1980 MHz）|
| P2P/拓扑 | `nvidia-smi topo -m` 八卡全 **NV18**；`topo -p2p a` 八卡两两 **OK**（双向 P2P 原子支持满足）|
| CPU 门禁 | 任务书指定的 `tests/test_async_single_push_cpu.py` **`ASYNC_SINGLE_PUSH_CPU PASS`** |
| 编译 | 一次通过，**执行者未做任何修复** |
| baseline 自身 | 同一 checkout、同一命令、同一现场，baseline **完整通过** |

**因此问题定位在 `a4e8854` 的 single_push 算法本身**，不是环境、不是数据、也不是执行者操作。

## 4. baseline 的完整数据（本轮分母，同命令同现场）

`moe.csv`：

| arm | steps | graph→first ms | first→last ms | last→combine ms | **moe_ms** | 变化 |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 999 | 8.0270 | 25.7711 | 2.2840 | **36.0821** | — |
| baseline | 1000 | 8.0270 | 25.7715 | 2.2841 | 36.0826 | — |

`run.json`：stats **99.808 ms/步**、measured **97.238 ms/步**。
`controls` 中 `GC_MOE_SM90_ASYNC_SINGLE_PUSH` **未设置**（baseline 用默认 ECO 路径），符合任务书。

baseline 的 `verification.txt`：
`eco historical workload verified for both windows and all eight ranks`。

baseline 五层最慢 rank（step 2，供 new arm 对照时使用）：

| layer | rank | tokens | M 块 | 活跃槽 | 首块可领 µs | pull 完成 µs | routed 退出 µs |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2 | 194.0 | 50 | 49 | 196 | 163 | 879.4 |
| 8 | 0 | 202.0 | 47 | 43 | 200 | 208 | 885.8 |
| 20 | 3 | 180.0 | 49 | 44 | 204 | 182 | 890.3 |
| 32 | 1 | 186.0 | 48 | 40 | 202 | 249 | 886.6 |
| 39 | 3 | 208.0 | 47 | 43 | 195 | 200 | 878.5 |

## 5. 无法交付的项（任务书 §交付 2，因单臂失败）

以下要求**只对成功臂有效**，single_push 因失败无 Perfetto/counts/detail，故无法给出：

- single_push 的粗粒度与 absolute/detail Perfetto、counts.json、detail.json；
- 每 rank 的 token 与 M 块数；
- 逐层的**满块总数**、`declared_sources < 8` 的满块数、`finished_sources < 8` 的满块数、尾块数；
- `finalize` 与 `L1 publish` 的分离解读。

**没有用 baseline 的数据冒充**，也没有为了拿到这些字段而放宽校验或改分流。

## 6. 交付物

`docs/trace_executor_20261119/sm90_bs256_async_push_results/`（43 个文件，82 MB）：

| 路径 | 内容 |
|---|---|
| `baseline/` | `run.json`、8 份 `rank*.npz`、`runner.log`、`verification.txt`、`hardware.csv`、`measurement.path`、`status.txt`、`runtime.log`、`perfetto/`（5 层 × 4 文件）|
| `single_push/` | `runner.log`（5,132 行，含八 worker 完整栈）、`hardware.csv`、`run.log`（driver 日志，含全部 17 处断言）、`status.txt`（`step=3: capture`，`exit_code=1`）、`failed-command.txt` |
| `moe.csv` | baseline 的 999/1000 步主指标 |
| `topo_m.txt`、`topo_p2p.txt` | 八卡 NV18 拓扑与双向 P2P 结果 |
| `state-path.txt` | 试次工作目录 |

**未提交**：raw Kineto、KV、replay（留在 `/tmp/sm90-bs256-eco-p2dQrgMq` 与保留现场）。

## 7. 清理

| 项 | 结果 |
|---|---|
| 作业退出 | 两臂均结束，`nvidia-smi --query-compute-apps` **0 条** |
| 残留 PID | **0**（`run_aligned_ep8.sh` / `real_decode_100_100.py` 进程均无）|
| 临时 checkout | `/tmp/sm90-async-push-run1` 已 `git worktree remove` 并 `prune` |
| 临时输出 | `/tmp/sm90-bs256-eco-p2dQrgMq` 保留（含 single_push 的 raw 目录，虽为空，作为失败现场）|
| 历史现场 | `/tmp/sm90-bs256-trace-pbpMbH4u`（KV + replay）**未触碰** |

## 8. 需要设计端确认

1. **single_push 的输出与原路径不等价**（59.6% 的 token 不同，第一步即分叉）。
   任务书把"源 rank 间预留的先后决定物理行序"归为"传输算法的一部分"，
   但本结果表明**行序改变已经改变了数值结果**，不是纯调度变化。
   在这条算法能保持输出等价之前，其 MoE 计时没有意义。
2. 建议的定位入口：`a4e8854` 的 `sm90_fp8_fp4_gc_moe.py` 中
   "一次原子预留当前源的全部行，然后直推"与"满块收到 M 行即发布"两处。
   首个失配在 `req0` position 117（decode 第一步），
   说明问题不在尾部块，而在**首个满块的发布条件**。
3. 若需要继续，建议先用**单步 A/A**（同一输入、同一步、仅切 async 开关）
   对比 `req0` 在 position 117 的 logits 与所选专家，再决定是否重跑 1000 步。
