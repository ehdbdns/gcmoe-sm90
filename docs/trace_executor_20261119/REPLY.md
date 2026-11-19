# REPLY — 八卡执行成功（step 2，layer 8 / 20）

## 结论

**任务书步骤 1–4 全部通过，两层均已解析并通过全部验收判据。**
单一入口 `scripts/trace_layer/run_aligned_ep8.sh` 一次跑通，退出码 0。

## 版本

| 项 | 值 |
|---|---|
| 起始 SHA | `dc15ee38591add1de44b81c11d02d06d9387af9a` |
| 执行 SHA | `dc15ee38591add1de44b81c11d02d06d9387af9a` |
| 是否有编译修复 | **无**（`d92e714` 已由设计端修好，我未改任何代码）|
| 工作区 | 干净（入口脚本 `test -z "$(git status --porcelain)"` 通过）|

## 机器与运行配置

```
8 × NVIDIA H20, driver 580.105.08, 97871 MiB/卡, 起始全部 4 MiB
torch 2.13.0+cu130 / CUDA 13.0 / vllm 0.31.0 / capability 9.x × 8
MODEL=/home/admin/local_models/DeepSeek-V4.1-Flash
DATASET=.../dsv4-flash_online_input_ids_26-10-08_1.jsonl
```

配置由入口脚本固定（与任务书一致）：
EPLB=1、逐层 EPLB、EXTRA_PER_RANK=8、ECO=1、WB=26/WA=0/COSTFN=0、
MAX_COPIES=8、MB_CLAIM=1、PULL_TMA=1、SHARED_IN_GC=1、SM_REUSE=0、
DISPATCH_SMS=6、CAPACITY=BOUND=2048、BLOCK_M=64、
PULL_TRACE=1、ECO_TRACE=1、FINE_TRACE=0、IR_STEP=0、LAYERS=40、
MEASURE_FLUSH=2、ENFORCE_EAGER=0、CUDAGRAPH_MODE=FULL_DECODE_ONLY。

## 步骤 2：两套编译均通过

```
compiled roles [1, 2, 5, 3]      (GC_MOE_STEP_PROFILE=0)   9.8s
compiled roles [1, 4, 2, 5, 3]   (GC_MOE_STEP_PROFILE=1)  11.5s
ALIGNED_STEP_CPU PASS
```

## 步骤 3：采集

- prefill 256 prompts → 200 步 warmup → 逐层 EPLB 重排 → 2 个未采样步 → 1 个采样步
- `step = 2`（重排后零基编号，符合任务书预期）
- 8/8 worker 导出成功；256 条序列各生成 1 个 token（`decode_tokens = 256`）

## capture ID 与 epoch 解析（§4.1）

| 项 | 值 |
|---|---|
| capture ID | **`e19037d2c1c24ea081a8fba6308b3753`**（8 rank 共享）|
| 设备快照候选区间 | first=14681 → next=14801（各 rank）|
| 候选匹配结果 | 每 rank `[14721]` — **8 卡唯一一组** |
| 最终 40 个连续 epoch | **14721 – 14760** |
| 每 rank 调用数 | 40（8 rank 全部相同）|

八个 rank 的 epoch 序列逐一比对：全部 `first=14721, last=14760, n=40`。
设备快照推进量为 120（=3×40），新解析器在候选范围内用「整步层间 dispatch 间隔 +
逐层设备打点」严格配对，唯一定位到 14721–14760 —— 与 torch trace 的 40 个调用一一对应。

## §4.2：每层 56 段、段和恒等

| 检查 | layer 8 | layer 20 |
|---|---|---|
| 区段总数 | **56**（8 rank × 7 段）✅ | **56** ✅ |
| 每个段名出现次数 | 各 8 ✅ | 各 8 ✅ |
| `count exchange + pull token == dispatch dur` | max 偏差 **2.23e-04 µs** ✅ | **2.23e-04 µs** ✅ |
| `wait token + GEMM == routed dur` | max 偏差 **1.72e-04 µs** ✅ | **2.11e-04 µs** ✅ |

（偏差为浮点尾数，非逻辑不符。）

## §4.3：pull 与 GEMM 重叠

**layer 8**（满足 `pull_after_first_claim_us>0` 且 `overlap>0` 的 rank：**全部 8 个**）

| rank | tokens | routed_rows | m_blocks | block_m | pull_after_claim | overlap_us | wait | GEMM |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 206 | 1648 | 35 | 64 | 76 | 77.52 | 2.0 | 603.8 |
| 1 | 186 | 1488 | 32 | 64 | 96 | 97.29 | 3.0 | 540.6 |
| 2 | 192 | 1536 | 33 | 64 | 107 | 108.23 | 3.0 | 567.8 |
| 3 | 192 | 1536 | 31 | 64 | 177 | 178.03 | 2.0 | 583.4 |
| 4 | 190 | 1520 | 35 | 64 | 141 | 143.06 | 2.0 | 577.4 |
| 5 | 194 | 1552 | 29 | 64 | 97 | 98.93 | 2.0 | 550.2 |
| 6 | 189 | 1512 | 31 | 64 | 400 | 401.42 | 3.0 | 816.4 |
| 7 | 187 | 1496 | 31 | 64 | 71 | 72.69 | 2.0 | 556.5 |

**layer 20**（满足条件的 rank：**7 个**，rank 0–6；rank 7 为 −61，因其仅 24 token 提前算完）

| rank | tokens | routed_rows | m_blocks | block_m | pull_after_claim | overlap_us | wait | GEMM |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 229 | 1832 | 37 | 64 | 60 | 62.06 | 2.0 | 647.0 |
| 1 | 226 | 1808 | 33 | 64 | 114 | 115.14 | 3.0 | 623.1 |
| 2 | 219 | 1752 | 31 | 64 | 183 | 183.53 | 3.0 | 642.3 |
| 3 | 218 | 1744 | 30 | 64 | 181 | 182.39 | 2.0 | 625.5 |
| 4 | 215 | 1720 | 33 | 64 | 96 | 97.64 | 3.0 | 623.0 |
| 5 | 194 | 1552 | 30 | 64 | 267 | 267.95 | 3.0 | 647.3 |
| 6 | 211 | 1688 | 34 | 64 | 89 | 89.71 | 3.0 | 647.3 |
| 7 | 24 | 192 | 9 | 64 | -61 | 0.00 | 2.0 | 123.4 |

任务书要求「每个交付层至少有一个 rank 同时满足两者」，两层分别有 8 个和 7 个 rank 满足。
最大重叠 layer 8 rank 6 = **401.42 µs**；layer 20 rank 5 = **267.95 µs**。
在 Perfetto 中可核对这些 rank 的 `pull token` 与 `GEMM` 区段确实重叠。

## §4.4：八卡负载记录

每层 0–7 共八条记录，见上两表的 `tokens` / `routed_rows` / `m_blocks` / `block_m`。

**守恒校验**：

```
layer 8  : SUM(routed_rows) = 12288 = 256 × 6 × 8  ✅
layer 20 : SUM(routed_rows) = 12288 = 256 × 6 × 8  ✅
```

`block_m = 64`（全部 rank），`m_blocks` 为逐槽 `ceil(owned / 64)` 之和。

**plan 代价与 L1 差（解析器给出的值）**：

| 层 | step | layer | epoch | plan_vs_measured_ratio_l1 |
|---:|---:|---:|---:|---:|
| 8 | 2 | 8 | 14729 | **0.594525** |
| 20 | 2 | 20 | 14741 | **0.647502** |

## §4.5：PNG（可选）

未导出。`plot_layer.py` 读取的是 `segments.json` 结构，与本任务的 `counts.json`
不同，直接调用会 `KeyError: 'segments'`。任务书标注该项为可选，故跳过，
未为此改动任何解析器或绘图脚本。

## 产物与校验

交付到 `docs/trace_executor_20261119/results/`：

```
capture.json                                    （另附）
step002_layer08_epoch14729.counts.json
step002_layer08_epoch14729.perfetto.json
step002_layer20_epoch14741.counts.json
step002_layer20_epoch14741.perfetto.json
```

原始数据保留在采集目录 `/tmp/gcmoe-aligned-XY0xN5ot`（含 8 份 torch trace 与全部日志，未删改）。

校验值：

```
97c03db48ba8bb2e89d1143516653de906c1ec15c1eb3706c312c65355e7ecee  deliver/step002_layer08_epoch14729.counts.json
4c772bee56fb53dc0fe9887faf4f514c0b877eb641d58cbbe8a8e0de9e74132f  deliver/step002_layer08_epoch14729.perfetto.json
76370fcc5e24d47f8ca4f605e32de5a02a4e8fe46bc361b04e015be047506858  deliver/step002_layer20_epoch14741.counts.json
6c4e2d3fbfefd3665bc3797aa1177ef93f719d869521f2e94eef06d987b64c35  deliver/step002_layer20_epoch14741.perfetto.json
1c051b109a7cefada8254cfc86df5763fa35483be2cfe0db7f3889b5c6f9b6df  raw/capture.json
a12c0590be03a5cc3e19ff45c97208281542c38a010405781c762a40de1de87c  raw/rank0.json
3b4b84c78b4b6f1ed3fa815d345462325bf403d3bae12fc3a30466aa21bbc4d2  raw/rank1.json
b3f63d1f7dba0a87f31881432679a410b1ff4910b5d8870bce36aa572f980be3  raw/rank2.json
492eec6025332b7063a4892d92d1f2679c5cb3007b0839925c0374fe8b337e7a  raw/rank3.json
a10faa742079d5cd018da0f7758e8a6357dcdb9f630537af29e6b20458b93904  raw/rank4.json
f91867ea6b23a58a74069249ee08c77ab6baa13dd240baa14c78836c5b1767e4  raw/rank5.json
02afc07b5edd01d4c116f15d6b2cc59915337e36b77b871cedfbfc2722def048  raw/rank6.json
0d6f92b830d8f0fb46d42dd06d2da89f3159721cb1a13df4fbe7ad82bdc041d1  raw/rank7.json
```

## 未改动清单

- 代码：未改（起始 SHA = 执行 SHA）
- 实验配置、内核行为、打点、调度、数据、解析器、测试、验收阈值、层选择：全部按任务书冻结
