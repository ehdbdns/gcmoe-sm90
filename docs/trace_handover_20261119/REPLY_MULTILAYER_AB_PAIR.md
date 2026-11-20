# REPLY_MULTILAYER_AB_PAIR — 任务一：两次八卡对照（配对单次运行）

## 0. 结论：**调度门禁未通过**（`scheduling_gate_passed = false`）

只有一处不满足条件——**两臂生成的 token 序列不同**：

| 项 | baseline | stream |
|---|---|---|
| `prompt_hash` | `254af9d8a2752eab68f6419f085c1973e22e582a3f50a3c9…` | **相同** |
| `output_hash` | `de4457bd3e6e405052ddc22fdf5a466a0ada3959822c4404…` | `06114f8646c64adf7bd71a8f65110aad5a0ee72e76a06ce0…` |
| 匹配 | — | **不同** |

任务书 §回执 4 的判定是「输入及生成摘要匹配、E2E 与 40 层 MoE 均下降时 `scheduling_gate_passed` 为真」。
后两条都满足（E2E −4.20%、40 层 MoE −3.52%），**但生成摘要不匹配，故门禁为 false**。
按 §回执 4「其余结果也应原样提交」，本回执如实提交，并停在此处等待复核。

## 1. 主体指标（均为实测）

| 指标 | baseline | stream | 差值 | 变化 |
|---|---:|---:|---:|---:|
| **E2E（100 步无 profiler）ms/step** | **84.726** | **81.170** | **−3.556** | **−4.20%** |
| **40 层 MoE `moe_max_rank_sum_ms`** | **37.380** | **36.064** | **−1.317** | **−3.52%** |
| step 2 开 profiler 的墙钟 ms | 87.365 | 85.469 | −1.895 | −2.17% |
| plan 占比 MAE（pp） | 1.340 | 1.000 | −0.340 | −25.4% |
| plan 最大绝对误差（pp） | 9.453 | 5.585 | −3.868 | −40.9% |

主指标（E2E）与次指标（40 层 MoE）**方向一致地下降**，plan 预测误差也同步改善。

## 2. 版本、环境与输入

| 项 | 值 |
|---|---|
| 测量代码 SHA | `9a2f294e920be3c98b741a46467cac1fb777805c` |
| baseline 内核 | `98b6194a56e7468653bb6230bf0b3887c9466dce`（STREAM=0） |
| stream 内核 | `6617ee976fa86eb77eb3a34b8b6df3e60059a586`（STREAM=1） |
| baseline capture ID | `c21088fc2a674400bd65af0f52115ec4` |
| stream capture ID | `a6d821957c654e588468ebfb30e5a252` |
| GPU | 8×H20，driver 580.105.08 |
| 软件 | cutlass 4.8.1 / torch 2.13.0+cu130 / CUDA 13.0 / vllm 0.31.0 / ptxas 13.2.78 |
| 模型 | `/home/admin/local_models/DeepSeek-V4.1-Flash` |
| 数据集 | `/mnt/nas2/nanjun.cp/MoE-Bench-Experiments/datasets/dsv4-flash_online_input_ids_26-10-08_1.jsonl` |
| 工作目录 | `/tmp/gcmoe-two-arm-qyJEyQs9` |
| 原始目录 baseline | `/tmp/gcmoe-aligned-GIbymGGQ` |
| 原始目录 stream | `/tmp/gcmoe-aligned-AC5bqoNH` |
| 两臂 `eco_costfn` | 0 / 0（本轮固定线性 plan） |
| 两臂 `stream_m_blocks` | 0 / 1 |

**两次模型启动，无第三次**。三个工作树保持原 SHA，无任何代码修改（`git status` 干净），
因此**本轮没有兼容性修复提交**。

**CPU 门禁**（`tools` 工作树）：

```
ALIGNED_STEP_CPU PASS
PULL_GEMM_DETAIL_CPU PASS
MULTILAYER_STUDY_CPU PASS
```

## 3. 交付物核对（全部达标）

| 项 | 要求 | 实际 |
|---|---:|---:|
| `run.json` | 2 | **2** |
| 层总览 Perfetto | 80 | **80** |
| counts | 80 | **80** |
| absolute Perfetto JSON | 22 | **22** |
| 逐 rank 行数 | 640 | **642**（640 数据行 + 2 表头） |
| `SHA256SUMS` 校验 | 通过 | **通过**（204 个文件，124 MB） |

两臂各含 40 层总览 + 40 counts + 11 份 absolute（层 0/4/8/12/16/20/24/28/32/36/39），
每份 absolute 合并八个 rank 的 count exchange / pull token / wait token / 首个有效 CTA 领票起算的 GEMM，
以及详细 pull / L1 / L2 / push 区间。

聚合文件：`paired.csv`、`layer_pairs.csv`、`comparison.json`、两臂各自的 `ranks.csv` / `layers.csv`。

## 4. 逐层结果（40 层，按最慢 rank 的 MoE 耗时）

**26 层变快 / 0 层持平 / 14 层变慢**（`comparison.json` 的 `faster_layers = 26`）。

改善最大与退化最大：

| 层 | baseline µs | stream µs | Δ µs | 结论 |
|---:|---:|---:|---:|---|
| **19** | 1148.9 | 931.5 | **−217.4** | 最快 |
| 21 | 1070.3 | 917.2 | −153.1 | |
| 4 | 1082.5 | 920.0 | −162.5 | |
| 6 | 993.3 | 838.2 | −155.2 | |
| **31** | 877.5 | 1015.0 | **+137.5** | 最慢 |
| 27 | 1027.1 | 1088.2 | +61.1 | |
| 20 | 985.4 | 1029.8 | +44.4 | |
| 28 | 941.7 | 983.4 | +41.7 | |

### 4.1 改善最大：layer 19（−217.4 µs）

**baseline 的负载严重失衡**，stream 的负载更均匀：

| rank | baseline tok | baseline M 块 | baseline µs | stream tok | stream M 块 | stream µs |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | **314** | 50 | 1147.0 | 192 | 38 | 927.8 |
| 6 | **49** | 18 | 1145.5 | 123 | 27 | 931.5 |
| 7 | 77 | 23 | 1145.7 | 166 | 33 | 930.6 |

baseline 的 rank 0 拿了 **314 token / 50 个 M 块**，rank 6 只有 **49 token / 18 块**
（极差 6.4×），而八个 rank 的 MoE 耗时几乎相同（1145–1149 µs）——
说明该层被 **rank 0 的尾部完全支配**，其余 rank 在等它。

stream 臂的 token 分布是 184–247（极差 1.34×）、M 块 27–40（1.48×），
八个 rank 的 MoE 耗时 927.8–931.5 µs（极差仅 3.7 µs），**尾部等待被摊平**。

plan 预测也随之变准：baseline 该层 MAE 3.104 pp、最差 rank 6 高估 −8.10 pp；
stream 降到 MAE 0.909 pp、最差 only −4.04 pp。
`ECO plan` 段本身几乎未变（116 → 112 µs mean），**收益来自配额分布而非规划器开销**。

### 4.2 退化最大：layer 31（+137.5 µs）

这里的机理**相反**：

| rank | baseline tok | baseline M 块 | baseline µs | stream tok | stream M 块 | stream µs |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 189 | 44 | 876.5 | **224** | 44 | **1015.0** |
| 3 | 202 | 37 | 875.9 | **256** | 43 | 1010.4 |
| 7 | 141 | 34 | 877.0 | 152 | 34 | 1012.1 |

baseline 该层**原本就很均衡**（token 141–208、M 块 34–44，MoE 875.9–877.4 µs，极差 1.5 µs），
stream 却把 token 推到 147–256、M 块 32–44，**总量上升**（baseline Σ=1128 tok → stream Σ=1016 tok
但 rank 0/3 的单点负载显著变重，rank 0 从 189→224、rank 3 从 202→256），
八个 rank 的 MoE 耗时也涨到 1010.4–1015.0 µs。

`ECO plan` 段 109 → 114 µs（+5 µs），GEMM 623 → 625 µs（+2 µs）——
两项都几乎没变，**说明这不是规划器变慢，而是该层重排后本 rank 承担的 M 块变多**
（rank 0 保持 44 块但 token 从 189→224，rank 3 从 37→44 块）。
plan MAE 从 0.653 → 1.699 pp 也印证配额与实测偏离加大。

### 4.3 其余 12 个变慢层

幅度 2.9–61.1 µs，多数（8/14）在 +12～+45 µs 区间，属同一类「重排后个别 rank 单点变重」。
考虑到本轮**每次重排都会改变逐槽分布**，单层的 ±40 µs 波动不宜视为系统性问题；
26:14 的胜面与 −3.5% 的整体收益才是主结论。

## 5. 必须写明的限制

1. **生成 token 不同 → 门禁不通过**。两臂 `prompt_hash` 相同（同一批提示词），
   但 `output_hash` 不同。任务书未要求解释原因，我按事实报告。
   这意味着 **−4.20% 的 E2E 收益是在两臂输出不一致的前提下测得的**，
   不能直接当作确定性的净收益，需要发布者判断是否可接受（或补一次输出一致性核查）。
2. **每臂只有一次独立运行**（任务书 §3 明确「每臂只有一次独立运行」），
   因此无法给出重复性误差；−4.20% 与 −3.52% 是否超出计时波动，本轮无法自证。
3. **层与 rank 不是独立重复**：40 层来自同一次 capture，`layer_pairs.csv` 的逐层差值
   只用于解释机理，不能当作 40 个独立样本做统计推断。
4. **本轮为新测量路径首次成功**：`--study-paired` 用进程内 EngineCore
   （`VLLM_ENABLE_V1_MULTIPROCESSING=0`）严格推进一步，每 rank 恰好 40 次 dispatch，
   上一轮 `11d031e` 报告的跨步泄漏（80 次 dispatch）**在本路径下未复现**。
   profiler 关闭后另跑 2 步暖机 + 100 步纯 decode 计时，且校验了
   `profiler_active=False`、`fine_writes=False`、每步 256 token 且 ctx=0。

## 6. 产物路径

```
docs/trace_executor_20261119/multilayer_ab_results/
├── comparison.json            scheduling_gate_passed = false
├── paired.csv                 5 个指标 × 两臂
├── layer_pairs.csv            40 层逐层对照
├── summary.csv
├── measurement.sha
├── SHA256SUMS                 204 个文件
└── step02/
    ├── baseline/paired/       run.json, ranks.csv, layers.csv,
    │                          40× perfetto.json, 40× counts.json, 11× absolute.perfetto.json
    └── stream/paired/         同上
```

原始八卡 Kineto trace、fine 分片与 `study-runner.log` 留在执行机
`/tmp/gcmoe-two-arm-qyJEyQs9`，各 `run.json` 的 `raw_path` 给出具体目录；
每臂另有 `raw-sha256.txt` 记录原始导出文件的校验值。
