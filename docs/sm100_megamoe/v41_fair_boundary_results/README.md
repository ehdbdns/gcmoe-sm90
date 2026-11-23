# [SM100] V41 BS512 公平输入边界结果

任务书：`docs/sm100_megamoe/V41_BS512_FAIR_BOUNDARY.md`  
任务提交：`f91491499e122ac51b3c27eb799667f4a92e6e27`  
任务书 SHA256：`a0d59361aa4d85e8e462eb01f6ba53fbf6b8241f3335a33b639dd0962dc4c8d2`  
执行 revision：`f91491499e122ac51b3c27eb799667f4a92e6e27`  
本地保持语义的插桩修复：`6196c0253258af115f2e68e3875d720537727897`  
GC 上游：`fb8b65fa829c97962eb2dbc9a4ad8b759b9cd736`

## 结论

两臂短跑 exit code 均为 `0`。每臂执行 3 个 BS512 原生 greedy decode step；每 rank 记录 `120 = 3 x 40` 次整图 MoE 调用，实际 block M 全为 16。Mega 无 SM split；GC 实际为 dispatch 20 / compute 128。CPU 判读退出码为 `3`，`short_strict_pass=false`，因此没有启动正式 1000+1000 复测，也没有性能结论。

共同 prefill 生效：两臂 `prefill.json` 和 `stats-tokens.npy` 的起点相同，prompt SHA256 均为 `b4fee0121ad43c17fdd5fccf5c4809602468725776257facc16f867e4c25cd86`。8 个 rank 均写到同一份 32 GiB KV checkpoint，GC 写回后的 source/readback SHA256 与 Mega 完全一致，controls 也一致。旧锚点 **step 0 / L3 / rank 7** 的六个 gate 已完全相同，原先在该锚点的分叉消失。

新的最早自然轨迹差异在 **step 0 / L7 / rank 6**，阶段是该层专家 **BF16 输出 `y`**，不是输入：`y[24,4991]` 为 Mega `-6.705522537231445e-07`、GC `-6.742775440216064e-07`。该调用点的 attention 摘要、专家 hidden、top-k IDs 和 gate 全部相同。

## 首步差异

step 0 只有 5 个差异，全部是专家 `y`，输入、IDs、gate 和 attention 摘要均相同：

| layer | rank | 首个 `y` 索引 | Mega | GC |
|---:|---:|---|---:|---:|
| 7 | 6 | `[24,4991]` | -6.705522537231445e-07 | -6.742775440216064e-07 |
| 16 | 3 | `[19,404]` | -0.061767578125 | -0.0615234375 |
| 33 | 0 | `[25,1812]` | 225 | 224 |
| 37 | 4 | `[25,405]` | -88 | -87.5 |
| 39 | 5 | `[40,521]` | -0.007080078125 | -0.007049560546875 |

step 1 开始传播到 attention：最早 `attn_in` 差异是 step 1 / L35 / 全部 8 rank；最早 `hidden_states` 与 `topk_weights` 差异是 step 1 / L34 / rank 4。`topk_ids` 首次差异出现在 step 2 / L36 / rank 7（`[60,5]`，Mega `271`、GC `64`）。全部原生差异为 82 个定位点：step 0/1/2 分别为 5/42/35 个，覆盖全部 8 rank。

输出 token 只有 1/1536 个不同：request `281`、step `2`，Mega `22`、GC `23`。槽计数的全局 step/layer 负载仍全为 3072，但 rank 1、4、5 在 step 2 分别有 1、1、2 个槽计数差。

## 同输入专家重放

每 rank 120 个重放点，共 960 个。15 个点被标记，step 0/1/2 各 5 个。所有 15 个点都有：

- 实际量化输入、hidden、top-k IDs、gate、`x`、`sf` 和 `shared_sf` 两臂完全相同；
- 7 个 routed/shared slot 的 partial hash 两臂完全相同；
- Mega 自回放与自然图 `y` **逐元素相同**（`different=0`）；
- GC 在相同输入和相同 partial hash 下，最终 `y` 仍有 1--9 个元素不同。

首例仍为 step 0 / L7 / rank 6：1 个元素不同，最大绝对差 `3.725290298461914e-09`。因此本轮可把已定位的剩余差异收窄到“相同 slot partial 之后的最终 BF16 归约/写回路径”，而不是输入量化或路由。

判读同时发现：15 个标记点的 Mega `partials_verified=false`，GC 为 `true`；即 Mega 的 slot 求和未能由该诊断公式逐位复原实际输出。这是本次 strict 判据失败的另一项证据/限制，不能解释为 partial hash 不同。

## 判据

37 项检查中 33 项通过。失败项为 `tokens`、`rank1_counts`、`rank4_counts`、`rank5_counts`，另有 82 个原生差异和 15 个同输入重放差异，故 `short_strict_pass=false`。逐项结果见 `checks.tsv`，完整机器判读见 `results/fair_boundary.json`。

## 证据与原始数据

原始 `native-*.pt`、`replay-*.json`、stats 数组和共同 KV 保留在 `/data0/home/admin/workspace/megamoe/v41-fair-boundary-nmRZZ9Q9/run/bs512`。`raw-manifest.tsv` 登记 130 个文件、277,414,112,145 bytes 的绝对路径、字节数和 SHA256；其中 KV 使用 checkpoint manifest 中经双端读回验证的 SHA256。清单 SHA256 为 `3fec6e013cc97969dd61c3a7f54e1717ad536d5ecd897c141cca880d748c9fa8`。

`codefix.diff` 是使 V4.1 模型枚举到 `DeepseekV4DecoderLayer` 并按 `layer_id` 排序的保持语义修复。上游最终 diff 在 `upstream.diff`（SHA256 `689fda072ab75f8366f56549de2e84e02f475bf34c9e1b0ed9aad44be9273904`）。

八卡组锁于 `2026-11-23T20:12:07+08:00` 获取、`2026-11-23T20:37:46+08:00` 释放。释放后无 compute app、无 `/dev/nvidia*` 持有者，组锁可立即获取，见 `gpu_release_after_run.txt`。短程插桩不计入性能。
