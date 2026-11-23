# [SM100] V41 BS512 首次 decode 分叉结果

任务书：`docs/sm100_megamoe/V41_BS512_FIRST_DIFF.md`  
执行 SHA：`f96d5b47994d10ead36bb84e55039032376dc80a`  
任务书 SHA256：`d5719b6378ac16ebfcfeacde9d623a4c3cb3c5b828bc869206cb02b45fd04a04`  
GC 上游：`fb8b65fa829c97962eb2dbc9a4ad8b759b9cd736`  
旧锚点归档：`37bc326`（原运行 revision `ecf5c52332c4bed43786f96118a6e3f5d863458a`）

## 结论

- Mega 与 GC 正式运行 exit code 均为 `0`。每臂执行 1 个 stats decode step，每 rank 实际记录 40/40 个有序 MoE 层调用，实际 block M 全部为 16。
- 两臂均使用 BS512 整步 decode CUDA Graph，`full_decode=true`。Mega 无 SM split；GC 使用上游默认分区，实际为 dispatch 20 / compute 128。
- 两臂 prefill token 与请求顺序完全一致：512 个请求，`prefill.json` 与 `stats-tokens.npy` 均逐字节相同，prompt SHA256 均为 `b4fee0121ad43c17fdd5fccf5c4809602468725776257facc16f867e4c25cd86`。
- L0、L1、L2 的所有 rank，以及 L3 的 rank 0--6，全部采集张量无差异。差异只在 **L3 / rank 7 / row 33**。
- 最早可观测差异是 **MoE 输入 `x`**，不是输出 `y`：`x[33,2]` 为 Mega `209`、GC `210`。同行的 `weights` 和 `y` 已不同，因此不能把输出差异单独归因给 GC MoE 内核；本采样点之前的 prefill KV/attention 与上游路径仍待区分。

## 首差

| tensor | 不同元素 / 总数 | 首个索引 | Mega | GC | 行/列范围 |
|---|---:|---|---:|---:|---|
| `x`（FP8 字节） | 1263 / 327680 | `[33,2]` | 209 | 210 | 仅 row 33；列 2--5115 |
| `weights` | 6 / 384 | `[33,0]` | 0.2723998427391052 | 0.27266591787338257 | 仅 row 33；六个 gate 全不同 |
| `y`（bf16） | 4863 / 327680 | `[33,0]` | -0.049072265625 | -0.0517578125 | 仅 row 33；列 0--5119 |

`sf` 与 `ids` 在所有 4x8 个采集点均完全一致。`y` 在磁盘上是 int16 的 bf16 位型，表中数值已按 bf16 解码。

## L3 / rank 7 / row 33 六门

两臂 IDs 均为 `[169, 41, 369, 224, 240, 328]`。六个 gate 权重均不同：

| gate | expert | Mega | GC | GC - Mega |
|---:|---:|---:|---:|---:|
| 0 | 169 | 0.2723998427391052 | 0.27266591787338257 | +0.00026607513427734375 |
| 1 | 41 | 0.33976319432258606 | 0.33931583166122437 | -0.00044736266136169434 |
| 2 | 369 | 0.27418071031570435 | 0.2732819616794586 | -0.0008987486362457275 |
| 3 | 224 | 0.20908476412296295 | 0.20913782715797424 | +0.000053063035011291504 |
| 4 | 240 | 0.21272066235542297 | 0.21361152827739716 | +0.0008908659219741821 |
| 5 | 328 | 0.19185076653957367 | 0.19198694825172424 | +0.00013618171215057373 |

按任务要求，在 `stats step 0 / rank 7 / L3 / row 33` 与 `37bc326` 的旧两臂结果逐值比对：IDs 完全相同；Mega 和 GC 的六门权重旧新最大绝对差均为 `0.0`。旧分叉已精确复现，见 `analysis/six_gate_comparison.csv` 与 `analysis/old_anchor_reproduction.json`。

## 证据

- 原始图层快照共 64 个：`mega|gc/layer{0..3}-rank{0..7}.npz`；每个快照为 `x=(64,5120) uint8`、`sf=(64,40) int32`、`ids=(64,6) int64`、`weights=(64,6) float32`、`y=(64,5120) int16`。
- 原始快照与 stats 数组保留在执行机 `/data0/home/admin/workspace/megamoe/v41-first-diff-9QujkJ0g/run/bs512`。`raw_snapshot_manifest.json` 登记 114 个文件、65,986,474 bytes，逐文件 SHA256；清单自身 SHA256 为 `61cc4a7b47028c04418f39861740ade13648a4c163a518c3a01c034fdd07d301`。
- 全量逐层、逐 rank、逐张量比较在 `analysis/layer_rank_tensor_diff.csv`；紧凑结论在 `first_diff.json` 与 `analysis/summary.json`。
- 本轮未新增编译或语义修复。兼容层沿用既有审计/时钟插桩和 GC capture 修复；Mega 入口 SHA256 为 `9b391d7a93e4bbfc8b8de4041e9628272e9d3bd531e385a4c6318b700fd692c1`，GC 入口为 `c4f01e5ee8b0fd564a4548b16857aaca013a119f0c901c22798c39404c9a38c6`，最终 tracked diff SHA256 为 `487fdc6c60fcee3ad313e9ae61e33396b5d2bda082fc2c9d6e3f415a50f879a3`。

## 环境与释放

八卡均为 NVIDIA L20D；`nvidia-smi` 报告 compute cap 8.9、275040 MiB，Torch 报告 capability `(10,3)`、267.69 GiB。驱动 `580.105.08`，torch `2.13.0+cu130`，vLLM `0.31.0`，cutlass-dsl `4.8.1`，Triton `3.7.1`，ptxas `13.5.37`。`GC_MOE_DISPATCH_SMS` 与 `GC_MOE_BALANCED_WORKQUEUES` 未设置，实际分区由驱动报告为 20/128。

八卡组锁于 `2026-11-23T19:23:21+08:00` 获取、`2026-11-23T19:32:24+08:00` 释放。运行后无 compute app、无 `/dev/nvidia*` 持有者，组锁可立即获取，见 `gpu_release_after_run.txt`。本轮性能计时不作结论。

## 文件索引

| 文件 | 内容 |
|---|---|
| `first_diff.json` | 任务生成的紧凑首差 |
| `analysis/first_diff_details.json` | 首差、全部差异与 row 33 细节 |
| `analysis/layer_rank_tensor_diff.csv` | 4 层 x 8 rank x 5 张量全量比较 |
| `analysis/six_gate_comparison.csv` | 新旧两臂六门逐值对照 |
| `mega/`、`gc/` | 两臂 result、prefill、model log、起止与 exit code |
| `logs/` | launcher、组锁、GPU、版本、ptxas、权重与 revision |
| `compat/` | 上游最终 tracked diff 与 untracked 源码归档 |
| `raw_snapshot_manifest.json` | 原始数组路径、字节数与 SHA256 |

