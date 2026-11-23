# [SM100] BS512 combine-order 正式双臂回执

执行者：**SM100**  
任务书：`docs/sm100_megamoe/V41_BS512_COMBINE_ORDER.md`  
任务提交：`f0015c39252d8a150d450005b75cab4cb86af41f`  
任务书 SHA256：`287a3e4cecf38e1462609283cc1cc6c32114ba6ec685cec5568ba4405929d9aa`  
执行 revision：`f0015c39252d8a150d450005b75cab4cb86af41f`  
GC 上游：`fb8b65fa829c97962eb2dbc9a4ad8b759b9cd736`  
结果目录：`docs/sm100_megamoe/v41_combine_order_results/`

## 结论

正式 Mega 和正式 GC 各完成 BS512 `1000 + 1000` 步，两臂 exit code 均为 `0`。CPU 正式判读 exit code 为 `1`，`strict_pair_pass=false`，共 **64 项失败**。按任务书要求，本轮不报告该配对的性能胜负。

本次没有新增语义修复。GC 私有 checkout 应用了任务指定补丁 `scripts/sm100/vllm_combine_order.patch`（SHA256 `a761a9380db90aef6d9e988c666b12eb1dfce538cdee12f509f60c81bf763f4`），补丁后的 GC 实现 SHA256 为 `e7050c640bbd096bea332966c16d8fa088f4efde1e485bbec073db0d8ec28b47`。

Mega 实际 overlay 已确认 combine 走 routed-first：

```cpp
const uint32_t slot_idx = __ffs(mask) - 1;
mask ^= 1 << slot_idx;
```

Mega 头文件 SHA256 为 `bbcf27686b299bd6d966a333b3b143dc8d8acfa8c3b274079b56058ea7ed9184`。

## 64 项失败

8 个 rank 在 `stats` 和 `measure` 两阶段均失败同一组四项，恰好 `8 x 2 x 4 = 64`：

- `native_window`
- `logical_ids`
- `gate_weights`
- `native outputs or routes`

完整有序清单见 `v41_combine_order_results/analysis/failed_checks.tsv`，权威列表见 `full_decode_pair.json`。

GC 的原生 token 差异在 8 个 rank 上均为 `stats=813`、`measure=1008`。首差：

| 阶段 | request | step | 原生 token | 参考 token |
|---|---:|---:|---:|---:|
| stats | 184 | 503 | 6409 | 370 |
| measure | 285 | 1 | 47507 | 6715 |

首个 gate 数值差异：

| 阶段 | step | layer | rank | local token | choice | native/replay ID | 原生 weight | replay weight |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| stats | 502 | 20 | 0 | 0 | 0 | 10 / 10 | 0.40825968980789185 | 0.4079365134239197 |
| measure | 0 | 20 | 0 | 0 | 0 | 375 / 375 | 0.6206911206245422 | 0.6203569173812866 |

也就是说，强制回放 IDs/weights 已对齐，但 GC 原生路由和 gate 仍分叉；combine-order 补丁本身不足以让正式严格判据通过。

## 通过项

- Prompt SHA256 两臂相同，prefill 原生 hash 16/16 相同，prefill difference 全部为 0。
- KV boundary 16/16 相同；每个 rank 的 source/readback SHA256 相同，controls/layout 一致。
- 回放 IDs、回放 weights、batch order 均 16/16 相同。
- 两臂每 rank 每阶段实际调用完整 decode 图 40000 次，32/32 项通过。
- 图覆盖 16/16 rank 通过：1 次 launch、correlation `751`、80 个 marker、0 个 foreign kernel。Mega MoE 节点 40，GC 120；attention/dense 均为 40/130。
- 所有 block M 为 16；GC SM split 全为 dispatch 20 / compute 128。
- 槽计数 16/16 phase-rank 完全相同；全局 step/layer token 总数在 2000/2000 项中均为 3072。
- `measure` 时钟形状、正值和单调边界检查通过。
- 记录到 999 个非采样完整 MoE 步。

## 性能（非权威）

按 999 个非采样步、每步逐层取八 rank 本地完整 MoE 最大值再累加 40 层：

| 臂 | ms/step |
|---|---:|
| Mega | 7.7220565845845845 |
| GC | 8.42969676876877 |

GC 相对 Mega 为 `+9.163882398852596%`。由于 `strict_pair_pass=false`，该数字只用于描述本轮运行，`authoritative=false`，不能作为性能胜负报告。

## 证据

除 `full_decode_pair.json`、两臂 `result.json`、32 个逐 rank NPZ、16 个 Kineto profile、两臂 Perfetto/overview、日志、环境和 task 外，结果目录还包含：

- `analysis/failed_checks.tsv`：64 项失败。
- `analysis/rank_checks.tsv`：逐 phase/rank 跨臂判据。
- `analysis/native_differences.tsv`、`analysis/gate_differences.tsv`：首差。
- `analysis/checks_summary.json`：全部控制项汇总。
- `compat/upstream.diff`：GC 最终上游 diff，SHA256 `55347204d4311c3e6bfdd1bc4f907c9b189d58bb3e63d1fcf329f62122561393`。
- `compat/vllm_combine_order.patch`、Mega combine 段和两侧实现 SHA256。

原始原生路由表、KV 和 profile 保留在：

`/data0/home/admin/workspace/megamoe/v41-combine-order-BgrMIRYR/run/bs512`

`raw_evidence_manifest.json` 登记 156 个文件、551,838,346,517 bytes 的绝对路径与 SHA256；manifest SHA256 为 `16cfe86125981985ce15636e7142ecf1610ddef2b08297f347f1bb585f299058`。KV 二进制 SHA256 取自生产 worker 的 checkpoint JSON，这些 JSON 已记录 source/readback 相同；其他文件均直接计算。

八卡组锁 `2026-11-23T21:12:01+08:00` 获取、`2026-11-23T21:40:41+08:00` 释放。释放后无 compute app、无 `/dev/nvidia*` 持有者，组锁可立即获取，见 `gpu_release_after_run.txt`。
