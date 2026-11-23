# [SM100] V41 BS512 vLLM combine-order 正式双臂

任务书：`docs/sm100_megamoe/V41_BS512_COMBINE_ORDER.md`  
任务提交：`f0015c39252d8a150d450005b75cab4cb86af41f`  
任务书 SHA256：`287a3e4cecf38e1462609283cc1cc6c32114ba6ec685cec5568ba4405929d9aa`  
执行 revision：`f0015c39252d8a150d450005b75cab4cb86af41f`  
GC 上游：`fb8b65fa829c97962eb2dbc9a4ad8b759b9cd736`  
新增语义修复：无。只应用任务指定的 `scripts/sm100/vllm_combine_order.patch` 到 GC 私有 checkout。

## 结论

一个正式 Mega 和一个正式 GC 均完成 BS512 `1000 + 1000` 步，两臂进程 exit code 均为 `0`。CPU 正式判读 exit code 为 `1`，`strict_pair_pass=false`，共 64 项失败；因此本回执不报告该配对的性能胜负。

combine 顺序修改本身已实际进入 GC：补丁 SHA256 为 `a761a9380db90aef6d9e988c6661b12eb1dfce538cdee12f509f60c81bf763f4`，GC 实现文件 SHA256 为 `e7050c640bbd096bea332966c16d8fa088f4efde1e485bbec073db0d8ec28b47`。Mega 实际 overlay 的 combine 段确认使用 routed-first 的 `__ffs(mask) - 1`；头文件 SHA256 为 `bbcf27686b299bd6d966a333b3b143dc8d8acfa8c3b274079b56058ea7ed9184`。

但补丁没有消除原生路由/gate 分叉。8 个 rank 在 `stats` 和 `measure` 两阶段都失败同一组四项检查，故恰好为 `8 x 2 x 4 = 64` 项。

## 正式判据

| 项目 | 结果 |
|---|---|
| Mega BS512 `stats` | 1000 步，exit `0` |
| Mega BS512 `measure` | 1000 步，exit `0` |
| GC BS512 `stats` | 1000 步，exit `0` |
| GC BS512 `measure` | 1000 步，exit `0` |
| CPU 正式判读 | exit `1` |
| `strict_pair_pass` | **false** |
| 失败项 | **64** |
| 非采样完整 MoE 步 | 999 |
| 性能结论 | **非权威，不报告胜负** |

每阶段每 rank 失败项固定为：`native_window`、`logical_ids`、`gate_weights`、`native outputs or routes`。完整 64 项见 `analysis/failed_checks.tsv` 和 `full_decode_pair.json`。

## 原生差异首例

GC 的原生 token 差异在 8 个 rank 上均为：`stats` 813 个、`measure` 1008 个。首差如下：

| 阶段 | request | step | 原生 token | 参考 token |
|---|---:|---:|---:|---:|
| `stats` | 184 | 503 | 6409 | 370 |
| `measure` | 285 | 1 | 47507 | 6715 |

首个 gate 数值差异如下；两侧 top-k ID 仍相同，但权重不同：

| 阶段 | step | layer | rank | local token | choice | native/replay ID | 原生 weight | replay weight |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `stats` | 502 | 20 | 0 | 0 | 0 | 10 / 10 | 0.40825968980789185 | 0.4079365134239197 |
| `measure` | 0 | 20 | 0 | 0 | 0 | 375 / 375 | 0.6206911206245422 | 0.6203569173812866 |

逐 rank 的 `native_diff` 与首差见 `analysis/native_differences.tsv`，逐 rank 首个 gate 差异见 `analysis/gate_differences.tsv`。这说明即使强制回放的 IDs/weights 与 Mega 相同，GC 的原生路由和 gate 仍未对齐。

## 已通过的控制项

- Prompt SHA256 两臂相同：`b4fee0121ad43c17fdd5fccf5c4809602468725776257facc16f867e4c25cd86`。
- prefill 原生结果 16/16 phase-rank 相同，prefill difference 全部为 0。
- KV boundary 16/16 phase-rank 相同；每个 rank 的 `source_sha256 == readback_sha256`，两臂 controls/layout 一致。
- 强制回放 IDs、回放 weights、batch order 均为 16/16 相同。
- 两臂每 rank 都使用完整 decode 图；每 rank 每阶段有 40000 次完整图 MoE 调用，32/32 项通过。
- 所有实际 block M 均为 16；GC 每 rank 的 SM split 均为 dispatch 20 / compute 128。
- 图覆盖 16/16 rank 通过：每个 profile 均匹配 1 次 launch、correlation `751`、80 个 marker、0 个 foreign kernel。Mega 的 MoE 节点为 40，GC 为 120；两臂 attention/dense 节点均为 40/130。
- 槽计数 16/16 phase-rank 完全相同，形状均为 `(1000, 40, 48)`；全局每 step/layer token load 在 2000/2000 项中均为 3072。
- `measure` 时钟形状、正值和单调边界检查通过。

完整机器判读见 `analysis/checks_summary.json`，逐 rank 表见 `analysis/rank_checks.tsv`。

## 采样层边界

采样点为 `measure` 的 step 2 / L20；两臂每 rank token 数和 M 块数完全相同，总计 3072 tokens、433 M blocks。

| 指标 | Mega | GC |
|---|---:|---:|
| 最早进入 MoE 的 rank | 7 | 2 |
| 最晚进入 MoE 的 rank | 5 | 5 |
| 八卡入口时差 | 2.642578125 us | 5.8623046875 us |
| 完整层跨度 | 227.541015625 us | 277.041015625 us |
| 最长 count exchange | 22.752 us (rank 0) | 17.312 us (rank 2) |
| 最大对齐半宽 | 0.784 us | 0.832 us |

明细在 `analysis/l20_ranks.csv`、`analysis/l20_summary.json` 和两臂 `perfetto/`。

## 性能数字（非权威）

排除采样步 2，对剩余 999 步逐层取八 rank 本地完整 MoE 最大值，累加 40 层后求均值：

| 臂 | ms/step |
|---|---:|
| Mega | 7.7220565845845845 |
| GC | 8.42969676876877 |

GC 相对 Mega 为 `+9.163882398852596%`（有符号降幅 `-9.163882398852596%`）。由于 `strict_pair_pass=false`，`analysis/performance_summary.json` 中 `authoritative=false`；该数字只描述本轮执行，不能作为性能胜负结论。

## 证据与释放

交付目录包含两臂 `result.json`、32 个逐 rank NPZ、16 个 anchor、16 个 Kineto profile、两臂各 5 层 Perfetto/overview、完整日志、环境、task、combine 补丁与上游最终 diff。

大体积原生路由、KV 和 profile 保留在执行机：

- 原始根：`/data0/home/admin/workspace/megamoe/v41-combine-order-BgrMIRYR/run/bs512`
- `raw_evidence_manifest.json`：156 个文件，551,838,346,517 bytes（约 513.94 GiB）
- manifest SHA256：`16cfe86125981985ce15636e7142ecf1610ddef2b08297f347f1bb585f299058`
- 完整绝对路径清单：`raw_evidence_paths.txt`

KV 二进制 SHA256 取自生产 worker 写出的 checkpoint JSON；这些 JSON 已记录 source/readback 相同。其余文件在生成 manifest 时直接计算 SHA256。

八卡组锁于 `2026-11-23T21:12:01+08:00` 获取、`2026-11-23T21:40:41+08:00` 释放。释放后核验无 compute app、无 `/dev/nvidia*` 持有者、组锁可立即获取；见 `gpu_release_after_run.txt`。
