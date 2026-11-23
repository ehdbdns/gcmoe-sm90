# [SM100] BS512 原生 greedy Mega/GC 双臂回执

执行者：SM100。未执行或混写 SM90 任务。  
任务书：`docs/sm100_megamoe/V41_BS512_UPSTREAM_GREEDY_PAIR.md`  
执行 SHA：`ecf5c52332c4bed43786f96118a6e3f5d863458a`  
任务书 SHA256：`ab8b68706f58a66076d02ba035af30f0053294754a6c5bcac6445b9034b629ea`  
结果目录：`docs/sm100_megamoe/v41_upstream_greedy_pair_results/`

## 结果

Mega 和 GC 两臂均完成 1000 步统计和 1000 步测量，正式运行 exit code 均为 0。两臂使用 BS512 完整 decode graph，各自生成并保留自己的原生 greedy token 和 KV；采样记录只读。八卡组锁于 `18:30:43` 获取，`18:52:20` 释放。

1. 图覆盖 16/16 rank 通过：Mega 与 GC 每 rank 均匹配同一次 `cudaGraphLaunch`，关联号 `751`；每 rank 80 个 marker、40 attention、130 dense、40 Mega MoE 或 120 GC MoE 节点，foreign kernel 为 0。
2. 每 rank 每阶段完整图调用 40,000/40,000，32/32 arm-phase-rank 通过。token 数组均为 1000 steps × 512 requests；各 arm-phase 的逐步逐层八卡 token 合计均为 3072；prefill 的 512 个请求两臂完全一致。
3. 两臂两阶段 rank 0--7 的实际 block M 全部为 16。GC 使用锁定上游默认值，`GC_MOE_DISPATCH_SMS` 和 `GC_MOE_BALANCED_WORKQUEUES` 均未设置；驱动实际分区为 dispatch 20 / compute 128。
4. Mega backend SHA256 `9b391d7a93e4bbfc8b8de4041e9628272e9d3bd531e385a4c6318b700fd692c1`；GC 上游 `fb8b65fa829c97962eb2dbc9a4ad8b759b9cd736`，入口 SHA256 `c4f01e5ee8b0fd564a4548b16857aaca013a119f0c901c22798c39404c9a38c6`，最终 tracked diff SHA256 `487fdc6c60fcee3ad313e9ae61e33396b5d2bda082fc2c9d6e3f415a50f879a3`。

## 独立生成差异

prefill 相同，首个 decode gate 差异在 `stats step 0, L10, local row 59, top-k 0`，权重 `0.6256834268569946` 对 `0.6255426406860352`；首个 rank 0 route id 差异在 `stats step 0, L13, local row 59, top-k 1`，`62` 对 `373`。首个 token 差异为 `stats step 2, request 252`，Mega `1`、GC `201`。

原生 token 差异数：`stats 109950/512000`，`measure 198198/512000`。`measure step 0, request 0` 已分别生成 `223` 和 `15135`，说明进入测量阶段时轨迹已分叉。每 rank 的 route id、gate weight、物理槽计数差异及首次位置已列入 `native_greedy_pair.json`、`analysis/native_differences.csv` 和 `analysis/native_differences.json`。这是独立生成轨迹的正常放大，不按跨臂 bit-exact 验收。

## L20

采样点 `measure step 2, L20`。Mega tokens `[582,216,311,158,362,178,286,979]`、M 块 `[68,45,51,41,52,43,47,86]`；GC tokens `[597,229,298,150,354,176,310,958]`、M 块 `[68,46,50,40,53,41,50,87]`。两臂 token 合计均 3072；M 块合计 433 / 435。

| 指标 | Mega | GC |
|---|---:|---:|
| 最早/最晚进入 rank | 4 / 3 | 3 / 6 |
| 入口时差 | 1.1484 us | 2.4805 us |
| 完整层跨度 | 221.8857 us | 271.6025 us |
| 最长 count exchange | rank 4，21.280 us | rank 2，15.424 us |
| 最大对齐半宽 | 0.8315 us | 0.8160 us |
| 最大绝对对齐漂移 | 1.1128 us | 1.3208 us |

GitHub L20：

- https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/sm100_megamoe/v41_upstream_greedy_pair_results/analysis/l20_summary.json
- https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/sm100_megamoe/v41_upstream_greedy_pair_results/mega/perfetto/step002_layer20.perfetto.json
- https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/sm100_megamoe/v41_upstream_greedy_pair_results/gc/perfetto/step002_layer20.perfetto.json

## 性能

排除测量 step 2，对其余 999 步逐层取八 rank 本地完整 MoE 时长最大值，每步累加 40 层后对步求均值：

| 臂 | ms/step |
|---|---:|
| Mega | 7.7219034715 |
| GC | 8.4396620300 |

以同轮 Mega 为基准，GC 有符号耗时变化为 **+9.2951%**。这是两臂各自独立生成时的负载描述；首处分叉后 token、KV 与路由不同，不能解释为同一路由上纯内核成本增加。

## 证据与释放

最终上游 diff 在 `compat/upstream-final.diff`，并复制到执行机 `run/archive/upstream-final.diff`；本轮沿用已验证的 DSL 兼容和 GC 外层 capture 修复，无新增语义修改。大路由 NPY 与原始 Kineto profile 留在执行机 `/data0/home/admin/workspace/megamoe/v41-native-greedy-VPCWiSrW/run/bs512`。`raw_evidence_manifest.json` 共 107 条、3,980,659,606 bytes（约 3.71 GiB），SHA256 `c33496a7693e5fe5b414d38a021a5473a871c3f122b2410c8233412d9a5cc92c`。

环境按实测保留：`nvidia-smi` 显示 NVIDIA L20D / compute_cap 8.9 / 275040 MiB，Torch 报告 capability `(10,3)` / 约 267.69 GiB；torch `2.13.0+cu130`、vLLM `0.31.0`、cutlass-dsl `4.8.1`、Triton `3.7.1`、驱动 `580.105.08`。运行后无 compute app、无设备持有者，组锁可获取，见 `gpu_release_after_run.txt` 与 `lock-release.txt`。
