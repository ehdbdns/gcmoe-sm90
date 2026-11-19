# 1 号执行者对接文档

本目录是「用 torch profile 做 GC-MoE 单层时间线」这件事的完整交接。

| 文件 | 内容 |
|---|---|
| `01_任务书_单层torch-profile时间线.md` | 任务目标、环境路径、实验启动方式、正确读 trace 的方法、交付物清单 |
| `02_1号执行者失误总账与质询清单.md` | 1 号执行者犯的 12 个错误、每个错误对应的「30 秒内识破」方法、可信输出的验收标准 |

## 接手前必读的三句话

1. **不要相信 1 号执行者产出的任何 trace 结果文件。** 他的方法是错的：
   把 5 个 role 各自排序后用同一个数组下标取，取到的不是同一次调用
   （实测相差 53 秒）。因此 `one_layer.png`、`GROUPED.md` 的数值、
   `perfetto_from_torch.json`、`one_call_from_torch.perfetto.json`、
   `/tmp/k11`、`gcmoe_t20_aligned2/` 全部作废。

2. **唯一可信的起点**是 `gcmoe_t20_aligned2` 之外的两处原始产物：
   - `gcmoe_exp/t19_costfn/prof/final_v2/traces/rank{0..7}.json`（torch profile 原始 trace）
   - `gcmoe_exp/t19_costfn/runs/FINAL_V2.json`（与上面**同一次运行**的结果）
   请从这两处重新做，不要复用中间的派生文件。

3. **两条硬校验**，不满足就是又取错了调用：
   - 同一次调用 5 个 role 的**原始 ts 绝对值**必须落在同一个几十微秒窗口内
   - 各分段时长**求和**必须等于对应 kernel 的总时长

## 任务要求（摘要）

- 用 **torch profile** 追踪**重排之后**的某几个层（不需要全部 40 层，1~2 层足够）
- 分段必须包含：`count exchange` / `pull token` / `shared L1+L2` / `ECO plan` /
  **`wait token`** / **`GEMM`** / `combine`
- 物理约束：
  - `shared` 与 `dispatch` **几乎同时启动**（−3 ~ +3 µs）
  - `routed` **必须**在 `shared` 结束之后启动（+1 µs 量级）
  - `combine` 在 `routed` 之后
  - `wait token + GEMM == routed kernel 总时长`
- 同一层给出 8 个 rank 的 **token 数、M 块数、plan 预测耗时与实测耗时**，并判断 plan 准不准
- 产出 perfetto 文件 + PNG 时间线图 + 说明文档，推送到本仓库

## 工作方式要求

先读源码再动手；用最小规模验证（只抽几行 trace，不要全量扫 2.7 GB × 8）；
不要长时间 sleep（20 s 轮询并汇报）；启动前核对脚本；报错立即停；
CPU 短测通过再上真实推理；权重加载要开 `warm_page_cache.sh`；
不要陷入完美主义；检查数据物理合理性后再交付。
