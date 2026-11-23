# [SM100] V41 BS512 原生 greedy Mega/GC 双臂结果

任务书：`docs/sm100_megamoe/V41_BS512_UPSTREAM_GREEDY_PAIR.md`  
执行 SHA：`ecf5c52332c4bed43786f96118a6e3f5d863458a`  
任务书 SHA256：`ab8b68706f58a66076d02ba035af30f0053294754a6c5bcac6445b9034b629ea`  
GC 上游：`fb8b65fa829c97962eb2dbc9a4ad8b759b9cd736`

## 结论

- Mega 和 GC 两臂均完成 1000 步统计和随后 1000 步测量，正式运行 exit code 均为 0；八卡组锁于 `18:30:43` 获取，`18:52:20` 释放。
- 两臂均使用 BS512 完整 decode graph。16/16 rank 找到同一次匹配的 `cudaGraphLaunch`，关联号 `751`；每 rank 40,000/40,000 次完整图调用，32/32 arm-phase-rank 通过。
- 两臂使用各自独立的原生 greedy 轨迹和 KV。prefill 的 512 个请求完全一致；进入 decode 后先出现极小 gate/route 差异，最终在 `stats step 2, request 252` 首次产生不同 token，之后轨迹分叉。
- 原生 token 差异：`stats` 109,950 / 512,000，`measure` 198,198 / 512,000。该数量是独立生成后的轨迹分叉结果，不是回放或配对执行失败。
- L20 入口时差：Mega `1.1484 us`，GC `2.4805 us`；跨 rank 完整层跨度：Mega `221.8857 us`，GC `271.6025 us`。
- 正式 MoE 指标：Mega `7.7219034715 ms/step`，GC `8.4396620300 ms/step`；以同轮 Mega 为基准，GC 有符号耗时变化为 **+9.2951%**。

## 运行配置

- 模型：DeepSeek V4.1 Flash，TP8/EP8，融合 shared，40 个专家层，top-6，每 rank 48 个本地槽。
- 推理：BS512，native greedy，固定 prompt，vLLM 0.31.0 V2 `FULL_DECODE_ONLY`，capture size 512，每个 rank 每步一次模型图提交；采样和调度在模型图外。
- 两臂独立生成并保留自己的 KV；没有跨臂共享 token、route 或 KV。
- Mega backend：`run/overlay/deep_gemm/mega/__init__.py`，SHA256 `9b391d7a93e4bbfc8b8de4041e9628272e9d3bd531e385a4c6318b700fd692c1`。
- GC backend：固定上游 `fb8b65fa829c97962eb2dbc9a4ad8b759b9cd736`，入口 `gc_moe/gc/__init__.py` SHA256 `c4f01e5ee8b0fd564a4548b16857aaca013a119f0c901c22798c39404c9a38c6`；最终 tracked diff SHA256 `487fdc6c60fcee3ad313e9ae61e33396b5d2bda082fc2c9d6e3f415a50f879a3`。
- 两臂 prompt SHA256 均为 `b4fee0121ad43c17fdd5fccf5c4809602468725776257facc16f867e4c25cd86`。

GC 使用锁定上游的 Green Context 默认分区和自动 M 块大小。`GC_MOE_DISPATCH_SMS` 与 `GC_MOE_BALANCED_WORKQUEUES` 在 native-greedy 分支明确 unset；驱动实际报告每个 rank 都是 **dispatch 20 / compute 128**。证据见 `sm_env_check.txt`。

## 结构与 block M

| 检查项 | 结果 |
|---|---|
| Mega exit / GC exit | 0 / 0 |
| 图覆盖 | 16/16 rank |
| 每 rank 每阶段完整图调用 | 40,000/40,000，32/32 arm-phase-rank |
| token 数组 | 两臂两阶段均为 1000 steps × 512 requests |
| 逐步逐层八卡 token 合计 | 3072，4/4 arm-phase 全部通过 |
| prefill 不一致请求 | 0/512 |

两臂、两阶段的 rank 0--7 实际 block M 全部为 16：

| rank | Mega stats/measure block M | GC stats/measure block M | GC dispatch/compute SM |
|---:|---:|---:|---:|
| 0 | 16 / 16 | 16 / 16 | 20 / 128 |
| 1 | 16 / 16 | 16 / 16 | 20 / 128 |
| 2 | 16 / 16 | 16 / 16 | 20 / 128 |
| 3 | 16 / 16 | 16 / 16 | 20 / 128 |
| 4 | 16 / 16 | 16 / 16 | 20 / 128 |
| 5 | 16 / 16 | 16 / 16 | 20 / 128 |
| 6 | 16 / 16 | 16 / 16 | 20 / 128 |
| 7 | 16 / 16 | 16 / 16 | 20 / 128 |

完整记录见 `analysis/block_m_sm_split.csv`。该任务的完整图证据为以下 16 份；每份都是同一 rank 的 step 2 profile：

| arm | rank | correlation | markers | attention | dense | MoE | foreign | pass |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| mega | 0 | 751 | 80 | 40 | 130 | 40 | 0 | yes |
| mega | 1 | 751 | 80 | 40 | 130 | 40 | 0 | yes |
| mega | 2 | 751 | 80 | 40 | 130 | 40 | 0 | yes |
| mega | 3 | 751 | 80 | 40 | 130 | 40 | 0 | yes |
| mega | 4 | 751 | 80 | 40 | 130 | 40 | 0 | yes |
| mega | 5 | 751 | 80 | 40 | 130 | 40 | 0 | yes |
| mega | 6 | 751 | 80 | 40 | 130 | 40 | 0 | yes |
| mega | 7 | 751 | 80 | 40 | 130 | 40 | 0 | yes |
| gc | 0 | 751 | 80 | 40 | 130 | 120 | 0 | yes |
| gc | 1 | 751 | 80 | 40 | 130 | 120 | 0 | yes |
| gc | 2 | 751 | 80 | 40 | 130 | 120 | 0 | yes |
| gc | 3 | 751 | 80 | 40 | 130 | 120 | 0 | yes |
| gc | 4 | 751 | 80 | 40 | 130 | 120 | 0 | yes |
| gc | 5 | 751 | 80 | 40 | 130 | 120 | 0 | yes |
| gc | 6 | 751 | 80 | 40 | 130 | 120 | 0 | yes |
| gc | 7 | 751 | 80 | 40 | 130 | 120 | 0 | yes |

原始 profile 留在执行机，路径、字节数和 SHA256 见 `raw_evidence_manifest.json`；仓库内只保留紧凑证据 `analysis/graph_launch.json` / `.csv`。

## 独立生成轨迹差异

两臂 prefill 相同。第一个 decode 路由差异出现在 Mega/GC 都尚未产生不同 token 的早期层，随后差异沿各自 KV 和上下文累积：

- `stats step 0, L10, local row 59, top-k 0`：第一次 gate 权重不同，Mega `0.6256834268569946`，GC `0.6255426406860352`。
- `stats step 0, L13, local row 59, top-k 1`：rank 0 第一次 route id 不同，Mega `62`，GC `373`。
- `stats step 0, L11, local slot 13`：第一次物理槽计数不同，Mega `116`，GC `117`。
- `stats step 2, request 252`：第一次原生 token 不同，Mega `1`，GC `201`。
- `measure step 0, request 0`：measure 阶段开始时轨迹已经分叉，Mega token `223`，GC token `15135`。

差异数量：

| phase | rank | route ids | gate weights | physical slot counts |
|---|---:|---:|---:|---:|
| stats | 0 | 4,477,918 | 7,044,418 | 1,214,310 |
| stats | 1 | 4,565,495 | 6,658,871 | 1,199,172 |
| stats | 2 | 4,632,803 | 6,981,562 | 1,205,254 |
| stats | 3 | 4,737,227 | 7,007,605 | 1,208,205 |
| stats | 4 | 3,830,542 | 6,678,050 | 1,207,633 |
| stats | 5 | 4,637,523 | 7,066,447 | 1,215,700 |
| stats | 6 | 4,353,881 | 6,808,097 | 1,203,311 |
| stats | 7 | 4,434,323 | 6,952,834 | 1,210,909 |
| measure | 0 | 9,793,030 | 14,705,474 | 1,491,743 |
| measure | 1 | 10,430,927 | 14,823,228 | 1,476,583 |
| measure | 2 | 10,331,169 | 14,785,965 | 1,485,123 |
| measure | 3 | 10,333,767 | 14,774,104 | 1,483,189 |
| measure | 4 | 9,254,091 | 14,660,621 | 1,482,167 |
| measure | 5 | 10,359,640 | 14,776,250 | 1,489,208 |
| measure | 6 | 10,082,760 | 14,744,666 | 1,479,559 |
| measure | 7 | 9,862,626 | 14,707,728 | 1,488,998 |

每 rank 的精确首次位置和首值见 `analysis/native_differences.csv`。这是“两臂各自原生 greedy 生成”的结果，不能用跨臂 bit-exact 相等来验收；差异从一条极小的 gate 数值开始，经 route 和 KV 分叉放大。性能段的后续步因此也不代表同一 token 路由上的纯内核差异。

## L20

采样点为 `measure step 2, L20`。两臂此时已经独立生成，token 与 M 块不再相同：

| rank | Mega tokens | Mega M blocks | GC tokens | GC M blocks |
|---:|---:|---:|---:|---:|
| 0 | 582 | 68 | 597 | 68 |
| 1 | 216 | 45 | 229 | 46 |
| 2 | 311 | 51 | 298 | 50 |
| 3 | 158 | 41 | 150 | 40 |
| 4 | 362 | 52 | 354 | 53 |
| 5 | 178 | 43 | 176 | 41 |
| 6 | 286 | 47 | 310 | 50 |
| 7 | 979 | 86 | 958 | 87 |
| 合计 | 3072 | 433 | 3072 | 435 |

| 指标 | Mega | GC |
|---|---:|---:|
| 最早进入 MoE 的 rank | 4 | 3 |
| 最晚进入 MoE 的 rank | 3 | 6 |
| 八卡入口时差 | 1.1484 us | 2.4805 us |
| 完整层跨度（最早入口到最晚出口） | 221.8857 us | 271.6025 us |
| 出口时差 | 0.6318 us | 1.1855 us |
| 最长 count exchange | rank 4，21.280 us | rank 2，15.424 us |
| 最大对齐半宽 | 0.8315 us | 0.8160 us |
| 最大绝对对齐漂移 | 1.1128 us | 1.3208 us |

GitHub L20：

- `analysis/l20_summary.json`、`analysis/l20_ranks.csv`
- `mega/perfetto/step002_layer20.perfetto.json`
- `gc/perfetto/step002_layer20.perfetto.json`

step 2 的 L0、8、20、32、39 overview、详细 Perfetto 和 counts 均在两臂目录下。

## 性能

方法：排除测量 step 2，对其余 999 步逐层取八 rank 本地完整 MoE 时长最大值，每步累加 40 层，再对步求均值。

| 臂 | ms/step |
|---|---:|
| Mega | 7.7219034715 |
| GC | 8.4396620300 |

以同轮 Mega 为基准，GC 有符号耗时变化为 **+9.2951%**（GC 更慢）。这是任务要求的独立 native-greedy 模式；首处分叉之后两臂各自的路由和后续 KV 不同，因此该值描述本轮独立生成负载，不应解释成同一 token/route 轨迹上、纯内核成本增加 9.2951%。

## 修复、环境与释放

沿用上一轮已验证的私有 DSL 兼容和 GC 外层 capture 修复；本轮没有新增语义修改。最终上游 tracked diff 在 `compat/upstream-final.diff`，状态在 `compat/upstream-status.txt`。DSL wheelhouse 使用本机 `dsl-wheelhouse-4.8.1`，五个 wheel 均通过 `dsl-wheelhouse.sha256` 校验。

环境标识按实测保留：`nvidia-smi -L` 显示 NVIDIA L20D、compute_cap 8.9、每卡 275040 MiB；Torch 报告 capability `(10, 3)`、每卡约 267.69 GiB、8 卡。版本为 torch `2.13.0+cu130`、vLLM `0.31.0`、cutlass-dsl `4.8.1`、Triton `3.7.1`、驱动 `580.105.08`、ptxas CUDA `13.5.37`。

正式运行结束后 `nvidia-smi --query-compute-apps=gpu_uuid,pid` 为空，`fuser` 无设备持有者，组锁可立即获取。证据见 `gpu_release_after_run.txt`、`lock-release.txt` 和 `logs/{acquired,released}.log`。

大型路由 NPY 和原始 Kineto profile 留在执行机 `/data0/home/admin/workspace/megamoe/v41-native-greedy-VPCWiSrW/run/bs512`。`raw_evidence_manifest.json` 共 107 条、3,980,659,606 bytes（约 3.71 GiB），SHA256 `ca691fcddee984e0a6feaa3e5b1a5ff4a8ba3413ea92cdff000f94fec0150dd9`；路径清单在 `raw_evidence_paths.txt`。
