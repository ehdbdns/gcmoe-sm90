# 【SM100】V4.1 Flash BS512 直跑五臂与旧矩阵 CPU 相位合并回执

执行者：**SM100**。本轮只执行下列两份 SM100 任务书，未执行任何 SM90 任务书。
本回执合并 GPU 直跑与 CPU-only 归档复核结果。

## 结论

- BS512 直跑五臂全部完成：`mega`、`gc`、`mega_rr`、`gc_rr`、`gc_eco` 的退出码均为
  **0**，`summary.json` 的 `failures` 为空，80/80 个 rank NPZ 哈希校验通过。
- 正式测量窗口的完整 MoE 时间为：Mega **17.955101 ms/步**、GC **18.578540
  (+3.47%)**、Mega-RR **21.021418 (+17.08%)**、GC-RR **65.681585
  (+265.81%)**、GC-ECO **18.825545 (+4.85%)**。这是各臂自然 greedy 解码结果，
  不是跨臂固定 route 的受控比较。
- CPU 相位复核覆盖旧矩阵的四个 BS。GC 相对 Mega 的 MoE 差距几乎全部落在
  `start → first routed`：BS512 该区间为 **63.398273 ms/步**，占 **63.912034
  ms/步** 总差距的 **99.20%**；`routed` 和 `combine` 两段合计仅约 0.514 ms/步。
- 旧矩阵使用受控 replay/固定 route；本轮 BS512 是自然生成。两者的绝对时间和
  GC-Mega 差值不能当作同一 workload 直接相减。

## 任务、源码与环境

| 项目 | 值 |
|---|---|
| GPU 任务 | `docs/sm100_megamoe/V41_DIRECT_BS512.md` |
| GPU 任务 SHA256 | `1d25419614efb385ee10bc00bf4f32ae34cbe8575039d3cbf2ee920d0820f024` |
| GPU 执行源码 | `f6808405a920faedc9d781d56dbe0fcc22267c48` |
| CPU 任务 | `docs/sm100_megamoe/V41_CPU_PHASES.md` |
| CPU 任务 SHA256 | `09ec0eef7738f8ecbd7a531ca6b7aebdeb8e144a688c13c4d1da17a128b3bc07` |
| CPU 分析源码 | `44c63e8ccbc1298a35e83d942b50484345fcde6e` |
| 回执打包基线 | `fe58cba2948a85a7e57cda4820a1f9622db304e7` |
| GCMoE 上游 | `fb8b65fa829c97962eb2dbc9a4ad8b759b9cd736` |
| GPU | 8 × NVIDIA L20D，275040 MiB/卡，compute capability `(10, 3)` |
| Python / Torch | 3.12 / 2.13.0 |
| vLLM | wheel SHA256 `39132b433967f0962d03d8a5f633b9bee6b32708f566c7d5637b79f8d0ef3433` |
| CuTe DSL | 4.8.1 |
| ptxas | CUDA 13.5.37 |

八卡 UUID、完整包清单和 ptxas 输出分别见
`v41_direct_bs512_results/gpus.txt`、`versions.txt`、`ptxas.txt`。

## GPU：BS512 直跑五臂

运行自 `2026-11-22T18:37:45+08:00` 开始。每臂启动独立 vLLM 进程，BS512、
TP8/EP8、40 层融合 shared expert、32 GiB KV/rank；每臂先跑 1000 步统计，再跑
1000 步正式测量。mega 的统计计数生成一份 40 层 DeepInfer 放置，三条 EPLB 臂使用
同一份。所有臂使用同一提示 SHA256
`b4fee0121ad43c17fdd5fccf5c4809602468725776257facc16f867e4c25cd86`。

### 五臂结果

`MoE Δ` 是同一轮相对 Mega 的有符号变化，正数表示更慢。`routed` 与 `graph`
误差为 MAE/MAX，单位百分点。

| Arm | 墙钟 ms/步 | 完整 MoE ms/步 | MoE Δ | token IR | 总 M 块 | 峰值 M 块 | 活跃槽 | M-block IR | routed MAE/MAX | graph MAE/MAX | ECO 成交 | block M | dispatch/compute |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `mega` | 96.152 | 17.955101 | +0.00% | 1.493821 | 423.491 | 64.848 | 338.594 | 1.224381 | 0.306/1.559 | 2.178/13.834 | — | 16 | Mega |
| `gc` | 97.927 | 18.578540 | +3.47% | 1.497782 | 422.898 | 64.854 | 337.446 | 1.226282 | 0.136/0.878 | 2.242/13.600 | — | 16 | 12/136 |
| `mega_rr` | 97.733 | 21.021418 | +17.08% | 1.069246 | 451.322 | 59.782 | 398.797 | 1.059726 | 0.109/0.686 | 2.199/13.135 | — | 16 | Mega |
| `gc_rr` | 175.877 | 65.681585 | +265.81% | 1.069172 | 451.612 | 59.797 | 399.281 | 1.059333 | 1.248/61.297 | 1.939/55.847 | — | 16 | 12/136 |
| `gc_eco` | 98.051 | 18.825545 | +4.85% | 1.140472 | 423.784 | 54.527 | 354.904 | 1.029357 | 0.117/0.804 | 2.124/12.449 | 238659 | 16 | 12/136 |

原始五行表在 `v41_direct_bs512_results/bs512/metrics.csv`。完整 MoE 口径为每步逐层
取八卡最大设备跨度后累加，再对 1000 步取均值。

### 放置、RR 和 ECO

- `plan.npz` SHA256：
  `ddcb82d5151cb353b62095a935a0141d5bd3f59ad095e2418b5d88e09b2a0ecf`。
- `mega_rr`、`gc_rr`、`gc_eco` 均记录同一 plan SHA；40 层逐层 layout SHA 在
  三臂间逐层一致，运行期抽样槽位不匹配数为 0。每层 changed slots 为 445–448。
- RR 调用数：`mega_rr` 与 `gc_rr` 在 stats/measure 均为每 rank 40000；
  `gc_eco` 为 stats 40000、measure 0；Mega/GC 为 0。与任务预期一致。
- 实际 GC 绿上下文为 `dispatch=12, compute=136`；启动器请求的
  `GC_MOE_DISPATCH_SMS=8` 由上游运行时取整为实际 12。
- GC-ECO 使用上游默认时间目标：`(block M, wb, wa, warm, cold, two-hop) =
  (16, 249, 140, 1, 6, 0)`，plan role 为 576 threads；未覆盖
  `GC_MOE_ECO_UNIT_CFG` 或 `GC_MOE_ECO_OBJECTIVE`。ECO 路径内部按上游实现用
  `compute_sms - 2`（即 134 个 compute SM）编译。

### 自检与时间线

每臂正式窗口均通过每步满批 512 token、40 层 eager/graph 边界、有限输出、设备
时间戳、放置安装和 RR 调用数检查；`measure.ranks[*].validated_layer_graphs` 均为
40。该模式下没有 KV checkpoint、token 表、route 档案、跨臂 top-1 或十组配对，
因此没有伪造受控比较结论。

| Arm | 启动 | 结束 |
|---|---|---|
| `mega` | 2026-11-22T18:38:00+08:00 | 2026-11-22T18:44:39+08:00 |
| `gc` | 2026-11-22T18:44:40+08:00 | 2026-11-22T18:52:18+08:00 |
| `mega_rr` | 2026-11-22T18:52:19+08:00 | 2026-11-22T18:59:15+08:00 |
| `gc_rr` | 2026-11-22T18:59:17+08:00 | 2026-11-22T19:10:38+08:00 |
| `gc_eco` | 2026-11-22T19:15:55+08:00 | 2026-11-22T19:25:01+08:00 |

`gc_rr` 的 routed/graph 最大误差明显高于其余臂，且 MoE 慢 265.81%；进程与自检
均未失败，本回执按实测保留该结果，不作未经验证的根因归因。

## CPU：旧矩阵关键 rank 相位

任务书脚本以每步每层本臂八卡完整设备跨度选择关键 rank，三个区间都取同一 rank。
它复核旧矩阵中物理需求相同的原生 Mega/GC 两臂，覆盖 BS256/512/1024/2048；每条
数据来自 1000 步 × 40 层。`CUDA_VISIBLE_DEVICES=""`，未启动模型。

| BS | Arm | start→first routed | first→last routed | last routed→combine end | MoE 总时 |
|---:|---|---:|---:|---:|---:|
| 256 | `mega` | 15.249182 | 4.306677 | 0.798949 | 20.354808 |
| 256 | `gc` | 78.866674 | 4.373523 | 0.972600 | 84.212797 |
| 256 | `gc − mega` | 63.617491 | 0.066847 | 0.173650 | 63.857989 |
| 512 | `mega` | 15.253257 | 5.345018 | 1.119758 | 21.718032 |
| 512 | `gc` | 78.651530 | 5.603934 | 1.374603 | 85.630066 |
| 512 | `gc − mega` | 63.398273 | 0.258916 | 0.254845 | 63.912034 |
| 1024 | `mega` | 14.305372 | 5.845911 | 1.148435 | 21.299718 |
| 1024 | `gc` | 73.797807 | 6.111558 | 1.379219 | 81.288584 |
| 1024 | `gc − mega` | 59.492436 | 0.265646 | 0.230784 | 59.988866 |
| 2048 | `mega` | 13.673697 | 6.266095 | 1.277592 | 21.217384 |
| 2048 | `gc` | 79.271124 | 6.589163 | 1.495574 | 87.355861 |
| 2048 | `gc − mega` | 65.597427 | 0.323068 | 0.217981 | 66.138476 |

单位为 **ms/步**。四个 BS 的第一段占 GC-Mega 总差距分别为 99.63%、99.20%、
99.17% 和 99.18%。结论是旧受控矩阵的主要差距来自首次 routed 前的等待/启动路径，
不是 routed 主体或 combine；本轮自然生成直跑则把 GC 对 Mega 的完整 MoE 差缩到
0.623 ms/步（+3.47%）。

脚本对归档执行了形状、全局收包 token 数、关键边单调性以及 Mega/GC 物理计数一致性
检查。输出中八个 `moe_ms` 再与原矩阵各自 `metrics.csv` 的 `moe_graph_ms`
逐项比较，最大浮点差为 `3.553e-15 ms`（全部为同一数值的舍入误差）。任务输出：
`docs/sm100_megamoe/v41_old_pair_phases.csv`，SHA256
`1ec2016ddc1a953eb3330a85aebb3ab56e8d963d17b766ae16871bb4a79cf1ce`。

## 归档、释放与清理

- 轻量证据目录：`docs/sm100_megamoe/v41_direct_bs512_results/`。
- 80 个 stats/measure rank NPZ 保留在执行机
  `/data0/home/admin/workspace/megamoe/v41-direct-bs512-TQgZTf46/bs512/`；相对
  哈希清单为结果目录的 `delivery/rank-files.sha256`。清理后再次验证为
  **SHA_OK (80/80)**；大 NPZ 未提交 Git。
- 私有 upstream、dsl-env、overlay、DG/ECO 编译缓存已清理，运行目录由 711 MiB
  降至 155 MiB；保留 rank NPZ、逐臂 result/log/start/end/exit-code 和交付目录。
- `run_five.sh` 于 `2026-11-22T19:25:04+08:00` 写入释放记录。
  `2026-11-22T19:26:02+08:00` 检查：`nvidia-smi --query-compute-apps` 为空，
  `/dev/nvidia0..7` 无持有者，八卡利用率 0%。运行锁与组锁均返回
  `*_RELEASED`。
- 本轮没有真实失败；未修改模型、输入、计时语义或上游实现。唯一新增的是仓库既有
  `natureal_matrix_audit.patch` 插桩。

证据入口：`v41_direct_bs512_results/`、`v41_old_pair_phases.csv`。
