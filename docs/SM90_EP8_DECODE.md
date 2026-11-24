# 8×H20 上的 MoE 解码性能

<!-- ep8-results -->

- **BS256**　EPLB 上从 RR 切到 ECO，MoE 从 891.15 到 842.43 µs/层，缩短 48.72 µs/层（-5.47%）。拍卖接受 187,082 / 291,116 次报价（64.26%），零成交 0 次层调用。[原始 CSV](trace_executor_20261119/sm90_bs256_main_five_plan_comm_results/delivery/bs256/main_table.csv)
- **BS512**　等待第二版整模结果。
- **BS1024**　等待第二版整模结果。

## 整模主表

| BS | 臂 | MoE（µs/层） | 相对原生 | 相对 RR | token IR | 总 M 块 | 峰值 rank M 块 | 活跃物理槽 | M 块 IR | routed MAE（百分点） | graph MAE（百分点） | 成交/层调用 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | 原生 | 1010.22 | 0.00% | +13.36% | 1.407 | 371.66 | 56.58 | 292.12 | 1.219 | 0.0943 | 1.9273 | — |
| 256 | 随机 + RR | 1084.74 | +7.38% | +21.72% | 1.489 | 415.42 | 63.41 | 341.34 | 1.222 | 0.1105 | 1.9999 | — |
| 256 | EPLB + RR | 891.15 | -11.79% | 0.00% | 1.083 | 404.55 | 54.36 | 356.06 | 1.075 | 0.0616 | 0.4843 | — |
| 256 | EPLB + ECO | 842.43 | -16.61% | -5.47% | 1.060 | 371.66 | 49.05 | 315.34 | 1.056 | 0.0834 | 0.2185 | 4.68 |

每次层调用先取八个 rank 从 graph 起点至 combine 结束的最大跨度，再对 1000 步 × 40 层等权平均。变化为本臂耗时除以同档参考臂耗时再减一，负值表示更快。总 M 块与活跃槽按八卡求和，峰值 M 块按 rank 取最大；IR 为每次层调用的八卡最大值除以均值。MAE 比较各 rank 的预测与实测耗时份额。表中每档四臂的六对对照均满足预热输出一致、正式 top‑1 零差异。

<!-- /ep8-results -->

## 实验设定

DeepSeek‑V4.1‑Flash，TP8/EP8。每臂先用 1000 个满批 decode 步统计需求，随后在固定输入、路由和 KV 边界下测量 1000 步。每步覆盖 40 个 MoE 层。公共配置为 FULL_DECODE_ONLY CUDA Graph、M64、按 M 块就绪发号、fused split shared expert，关闭 SM-reuse。每个 BS 内依次比较原生无副本、随机副本放置 + RR、逐层 EPLB + RR 和逐层 EPLB + ECO。后两臂的专家副本所在 rank 相同，RR 轮转分发，ECO 按预测负载拍卖分流。[四臂定义与验收条件](trace_executor_20261119/SM90_BS512_BS1024_FULL_MODEL_FOUR.md)

| 整模全局 BS（请求数） | 256 | 512 | 1024 |
|---|---:|---:|---:|
| 按 EP8 均摊的请求数 / rank | 32 | 64 | 128 |

每个请求在一个 decode 步生成一个新 token。整模使用 TP8，路由审计对每个源 rank 各记录一份形状为 `[BS, 6]` 的 top-6 路由；主表的接收 token 和 M 块指标来自实际物理槽。下方独立算子 Perfetto 使用每个源 rank 各自输入的 128 个 token，总计 1024 个输入、6144 个路由项。[整模路由口径](trace_executor_20261119/SM90_BS512_BS1024_HISTORICAL_ROUTES.md) · [算子输入口径](trace_executor_20261119/SM90_BS1024_SE_BENCH_PERFETTO.md)

### Plan 的代价和 SM 放置

线性代价以 token 为单位。若 rank 上的物理专家槽分别接收 $q_s$ 个路由项，则

$$
C_r = 26 M_r + T_r,\qquad M_r=\sum_s \lceil q_s/64\rceil,\quad T_r=\sum_s q_s.
$$

线性版最终保留 **wb = 26、wa = 0**。一个新增 M 块按 26 个 token 的代价计，尾块在各槽内分别向上取整。本次主表的在线 ECO 实测使用 **fine cost（mode 2，slack 11469）**。下表中的预测误差也按该次报告的主价格计算。[线性权重 A/B](WB_AB_RESULT.md) · [BS256 实测配置](trace_executor_20261119/SM90_BS256_MAIN_FIVE_PLAN_COMM.md)

ECO plan 独占通信分区 6 个 SM 中的 1 个，剩余 5 个承载 20 个 dispatch CTA，计算分区保有 72 个 SM。BS256 的独立对照中，plan 从 compute 挪到 dispatch 后，每步 40 层合计的 MoE 耗时从 **35.7547 降至 33.8395 ms（−5.36%）**。其中 graph 起点至首个真实 routed CTA 的时间从 8.0254 降至 6.3890 ms，贡献约 85% 的节省。该对照取 999 个未采样步。[两臂回执和实际 SM 落点](trace_handover_20261119/REPLY_SM90_BS256_PLAN_ON_COMM.md)

## BS1024 单层 Perfetto

两份文件使用历史第 20 层的 EPLB 放置与相同的 benchmark 输入。下表取第 2 次 replay（sample 1、epoch 10）的 rank 0，以各自文件的时间轴为准，单位 µs。八 rank 的整次 replay 跨度为 RR **683.08 µs**、ECO **627.55 µs**；这组单层读数与上面的整模均值分别统计。[RR Perfetto JSON](trace_executor_20261119/sm90_bs1024_se_bench_results/rr.replay2.perfetto.json) · [ECO Perfetto JSON](trace_executor_20261119/sm90_bs1024_se_bench_results/eco.replay2.perfetto.json)

| rank 0 阶段 | EPLB + RR 起止 | EPLB + ECO 起止 |
|---|---:|---:|
| count exchange → pull token | 2.3–11.3 → 11.3–59.3 | 3.4–25.4 → 25.4–88.4 |
| ECO plan | — | 1.5–67.9 |
| shared expert L1 | 1.5–79.9 | 3.9–84.4 |
| shared expert L2 | 5.2–186.9 | 9.0–202.7 |
| wait token | 80.2–84.1 | 84.7–88.5 |
| routed GEMM | 84.1–655.3 | 88.5–595.7 |
| combine | 658.8–682.1 | 600.1–626.5 |

1. **交换与拉取。** 同一 dispatch 轨先显示从启动到首次 pull 的 count exchange 区间，再显示 pull token。此时 shared L1 已在运行。ECO 的 plan 也与 count exchange、pull 的前段及 shared 计算重叠。
2. **shared 与 routed。** split 将 shared L1、L2 放在独立 kernel 中；轨道显示的是 kernel 的驻留区间。L1 的就绪结果供 L2 消费。两臂 rank 0 都在 L1 结束后进入 wait token，随后第一个 CTA 真正开始 routed GEMM，此时 shared L2 仍在运行。L2 与 routed 的重叠时间分别为 **102.8 µs** 和 **114.2 µs**。该样本 RR 的八个 rank 都在本地 pull 结束后开始 routed；ECO 在 rank 3 上两者还重叠 **8.2 µs**。
3. **计算至收尾。** routed GEMM 从首个 CTA 实际领到计算任务开始，覆盖最后一个计算 CTA 完成及其末尾 push。随后 combine 完成本地汇总。rank 0 的接收路由项从 763 增至 777，M 块从 55 降至 47，routed 跨度从 **571.1 降至 507.2 µs**。两条时间线中的并行阶段不能直接相加。[阶段边界与设备时钟](../scripts/sm90/se_perfetto.py)

> **截图位置**　BS1024，第 20 层放置，replay 2。可在此并排放置 RR 和 ECO 的 Perfetto 截图。
