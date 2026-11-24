# 回执 · SM90 · BS512 / BS1024 五臂三层 replay2

执行者：SM90 八卡执行者。任务书 `docs/trace_executor_20261119/SM90_BS512_BS1024_FIVE_REPLAY2.md`，代码与测量同为 `80fed59aed018d77b71370dcb37cb8b8a7cc6d49`。
未改动任何实验参数、脚本或 kernel；无编译失败，无保持语义的修改（diff 为空）。

## 运行口径

- 一次启动内完成 L8 / L20 / L32（层号从零起），复用逻辑权重、shared 权重与通信缓冲区；未启动 vLLM、未加载 checkpoint、未构建 KV。
- EP8、M64、stream、TMA pull、6 个 dispatch SM、H5120 / I2304、1 个 fused shared expert、top-6 沿用上一轮算子 benchmark。
- BS512 每源 rank 64 token、八卡合计 3072 路由项；BS1024 每源 rank 128 token、合计 6144。三层各用固定 benchmark 输入；L20 沿用上一轮种子；五臂同层输入及逻辑权重相同。
- 每层安装仓库内该 BS、该臂的真实布局（`delivery/bs{512,1024}/<arm>/installed-layout.json`）；ECO 逐层校核 `prepared_matrix_price == {'mode': 2, 'slack': 11469}`。
- 每层预热 3 次 + 捕获 + 5 次 replay，再以 torch.profiler 记录 3 次 graph replay；**展示与三层均值固定取 sample 1（第二次 replay）**。三份 replay 全部存档。
- 两侧各十次八卡启动全部在八张 H20 同时空闲时开始（`gpus.csv` 为 8×NVIDIA H20，`processes.csv` 为空）。

## 主表（µs/层，三层等权平均 L8/L20/L32，sample 1）

| BS | 臂 | MoE µs/层 | 相对 vanilla | token IR | 总 M 块 | 峰值 rank M 块 | 活跃物理槽 | M 块 IR | routed MAE (pp) | graph MAE (pp) | 拍卖成交(均值/合计) | 输出一致 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 512 | vanilla | 566.592 | +0.00% | 1.0755 | 384 | 48 | 384 | 1.0000 | 0.0371 | 0.0513 | — | 是 |
| 512 | random | 634.261 | +11.94% | 1.1372 | 446 | 56 | 446 | 1.0052 | 0.0235 | 0.1374 | — | 是 |
| 512 | rr | 631.093 | +11.38% | 1.0807 | 440 | 56 | 440 | 1.0190 | 0.0202 | 0.1656 | — | 是 |
| 512 | eco | 576.256 | +1.71% | 1.0764 | 384 | 48 | 384 | 1.0000 | 0.0213 | 0.0412 | 0 / 0 | 是 |
| 512 | eco_smr | 538.464 | -4.96% | 1.0764 | 384 | 48 | 384 | 1.0000 | 0.0569 | 0.0393 | 0 / 0 | 是 |
| 1024 | vanilla | 624.779 | +0.00% | 1.0425 | 384 | 48 | 384 | 1.0000 | 0.0352 | 0.0823 | — | 是 |
| 1024 | random | 696.992 | +11.56% | 1.1337 | 448 | 56 | 448 | 1.0000 | 0.0535 | 0.2431 | — | 是 |
| 1024 | rr | 690.987 | +10.60% | 1.0681 | 446 | 56 | 446 | 1.0037 | 0.0338 | 0.1185 | — | 是 |
| 1024 | eco | 627.403 | +0.42% | 1.0543 | 384 | 49 | 384 | 1.0208 | 0.0433 | 0.0767 | 0 / 0 | 是 |
| 1024 | eco_smr | 593.120 | -5.07% | 1.0543 | 384 | 49 | 384 | 1.0208 | 0.0475 | 0.0826 | 0 / 0 | 是 |

相对变化分母为同 BS 的 vanilla 三层均值；负值表示更快。完整 MoE 取同层八 rank 中最大的「本地 graph 起点 → combine 结束」跨度。

## 比较图

- BS512：[moe.svg](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/trace_executor_20261119/sm90_bs512_bs1024_five_replay2_results/bs512/moe.svg)
- BS1024：[moe.svg](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/trace_executor_20261119/sm90_bs512_bs1024_five_replay2_results/bs1024/moe.svg)

## 逐层明细（µs/层）

### BS512

| 层 | 臂 | MoE µs | 总 M 块 | 峰值 M 块 | 活跃槽 | 拍卖成交 | token IR | M 块 IR |
|---|---|---|---|---|---|---|---|---|
| L8 | vanilla | 565.536 | 384 | 48 | 384 | — | 1.0547 | 1.0000 |
| L20 | vanilla | 566.240 | 384 | 48 | 384 | — | 1.0833 | 1.0000 |
| L32 | vanilla | 568.000 | 384 | 48 | 384 | — | 1.0885 | 1.0000 |
| L8 | random | 634.432 | 446 | 56 | 446 | — | 1.1771 | 1.0045 |
| L20 | random | 634.848 | 446 | 56 | 446 | — | 1.1120 | 1.0045 |
| L32 | random | 633.504 | 445 | 56 | 445 | — | 1.1224 | 1.0067 |
| L8 | rr | 628.960 | 439 | 56 | 439 | — | 1.0755 | 1.0205 |
| L20 | rr | 634.240 | 439 | 56 | 439 | — | 1.0885 | 1.0205 |
| L32 | rr | 630.080 | 441 | 56 | 441 | — | 1.0781 | 1.0159 |
| L8 | eco | 574.272 | 384 | 48 | 384 | 0 | 1.0599 | 1.0000 |
| L20 | eco | 578.368 | 384 | 48 | 384 | 0 | 1.0859 | 1.0000 |
| L32 | eco | 576.128 | 384 | 48 | 384 | 0 | 1.0833 | 1.0000 |
| L8 | eco_smr | 536.608 | 384 | 48 | 384 | 0 | 1.0599 | 1.0000 |
| L20 | eco_smr | 538.048 | 384 | 48 | 384 | 0 | 1.0859 | 1.0000 |
| L32 | eco_smr | 540.736 | 384 | 48 | 384 | 0 | 1.0833 | 1.0000 |

### BS1024

| 层 | 臂 | MoE µs | 总 M 块 | 峰值 M 块 | 活跃槽 | 拍卖成交 | token IR | M 块 IR |
|---|---|---|---|---|---|---|---|---|
| L8 | vanilla | 625.056 | 384 | 48 | 384 | — | 1.0430 | 1.0000 |
| L20 | vanilla | 624.512 | 384 | 48 | 384 | — | 1.0273 | 1.0000 |
| L32 | vanilla | 624.768 | 384 | 48 | 384 | — | 1.0573 | 1.0000 |
| L8 | random | 702.720 | 448 | 56 | 448 | — | 1.1758 | 1.0000 |
| L20 | random | 692.832 | 448 | 56 | 448 | — | 1.0977 | 1.0000 |
| L32 | random | 695.424 | 448 | 56 | 448 | — | 1.1276 | 1.0000 |
| L8 | rr | 689.856 | 446 | 56 | 446 | — | 1.0768 | 1.0045 |
| L20 | rr | 690.912 | 447 | 56 | 447 | — | 1.0781 | 1.0022 |
| L32 | rr | 692.192 | 446 | 56 | 446 | — | 1.0495 | 1.0045 |
| L8 | eco | 629.984 | 384 | 49 | 384 | 0 | 1.0430 | 1.0208 |
| L20 | eco | 626.528 | 384 | 49 | 384 | 0 | 1.0443 | 1.0208 |
| L32 | eco | 625.696 | 384 | 49 | 384 | 0 | 1.0755 | 1.0208 |
| L8 | eco_smr | 592.352 | 384 | 49 | 384 | 0 | 1.0430 | 1.0208 |
| L20 | eco_smr | 591.712 | 384 | 49 | 384 | 0 | 1.0443 | 1.0208 |
| L32 | eco_smr | 595.296 | 384 | 49 | 384 | 0 | 1.0755 | 1.0208 |

## 30 个 Perfetto（每层每臂一份，8 rank，仅第二次 replay）

### BS512

| 层 | vanilla | random | rr | eco | eco_smr |
|---|---|---|---|---|---|
| L8 | [vanilla](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/trace_executor_20261119/sm90_bs512_bs1024_five_replay2_results/bs512/layer8/vanilla.replay2.perfetto.json) | [random](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/trace_executor_20261119/sm90_bs512_bs1024_five_replay2_results/bs512/layer8/random.replay2.perfetto.json) | [rr](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/trace_executor_20261119/sm90_bs512_bs1024_five_replay2_results/bs512/layer8/rr.replay2.perfetto.json) | [eco](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/trace_executor_20261119/sm90_bs512_bs1024_five_replay2_results/bs512/layer8/eco.replay2.perfetto.json) | [eco_smr](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/trace_executor_20261119/sm90_bs512_bs1024_five_replay2_results/bs512/layer8/eco_smr.replay2.perfetto.json) |
| L20 | [vanilla](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/trace_executor_20261119/sm90_bs512_bs1024_five_replay2_results/bs512/layer20/vanilla.replay2.perfetto.json) | [random](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/trace_executor_20261119/sm90_bs512_bs1024_five_replay2_results/bs512/layer20/random.replay2.perfetto.json) | [rr](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/trace_executor_20261119/sm90_bs512_bs1024_five_replay2_results/bs512/layer20/rr.replay2.perfetto.json) | [eco](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/trace_executor_20261119/sm90_bs512_bs1024_five_replay2_results/bs512/layer20/eco.replay2.perfetto.json) | [eco_smr](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/trace_executor_20261119/sm90_bs512_bs1024_five_replay2_results/bs512/layer20/eco_smr.replay2.perfetto.json) |
| L32 | [vanilla](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/trace_executor_20261119/sm90_bs512_bs1024_five_replay2_results/bs512/layer32/vanilla.replay2.perfetto.json) | [random](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/trace_executor_20261119/sm90_bs512_bs1024_five_replay2_results/bs512/layer32/random.replay2.perfetto.json) | [rr](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/trace_executor_20261119/sm90_bs512_bs1024_five_replay2_results/bs512/layer32/rr.replay2.perfetto.json) | [eco](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/trace_executor_20261119/sm90_bs512_bs1024_five_replay2_results/bs512/layer32/eco.replay2.perfetto.json) | [eco_smr](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/trace_executor_20261119/sm90_bs512_bs1024_five_replay2_results/bs512/layer32/eco_smr.replay2.perfetto.json) |

### BS1024

| 层 | vanilla | random | rr | eco | eco_smr |
|---|---|---|---|---|---|
| L8 | [vanilla](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/trace_executor_20261119/sm90_bs512_bs1024_five_replay2_results/bs1024/layer8/vanilla.replay2.perfetto.json) | [random](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/trace_executor_20261119/sm90_bs512_bs1024_five_replay2_results/bs1024/layer8/random.replay2.perfetto.json) | [rr](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/trace_executor_20261119/sm90_bs512_bs1024_five_replay2_results/bs1024/layer8/rr.replay2.perfetto.json) | [eco](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/trace_executor_20261119/sm90_bs512_bs1024_five_replay2_results/bs1024/layer8/eco.replay2.perfetto.json) | [eco_smr](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/trace_executor_20261119/sm90_bs512_bs1024_five_replay2_results/bs1024/layer8/eco_smr.replay2.perfetto.json) |
| L20 | [vanilla](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/trace_executor_20261119/sm90_bs512_bs1024_five_replay2_results/bs1024/layer20/vanilla.replay2.perfetto.json) | [random](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/trace_executor_20261119/sm90_bs512_bs1024_five_replay2_results/bs1024/layer20/random.replay2.perfetto.json) | [rr](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/trace_executor_20261119/sm90_bs512_bs1024_five_replay2_results/bs1024/layer20/rr.replay2.perfetto.json) | [eco](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/trace_executor_20261119/sm90_bs512_bs1024_five_replay2_results/bs1024/layer20/eco.replay2.perfetto.json) | [eco_smr](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/trace_executor_20261119/sm90_bs512_bs1024_five_replay2_results/bs1024/layer20/eco_smr.replay2.perfetto.json) |
| L32 | [vanilla](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/trace_executor_20261119/sm90_bs512_bs1024_five_replay2_results/bs1024/layer32/vanilla.replay2.perfetto.json) | [random](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/trace_executor_20261119/sm90_bs512_bs1024_five_replay2_results/bs1024/layer32/random.replay2.perfetto.json) | [rr](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/trace_executor_20261119/sm90_bs512_bs1024_five_replay2_results/bs1024/layer32/rr.replay2.perfetto.json) | [eco](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/trace_executor_20261119/sm90_bs512_bs1024_five_replay2_results/bs1024/layer32/eco.replay2.perfetto.json) | [eco_smr](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/trace_executor_20261119/sm90_bs512_bs1024_five_replay2_results/bs1024/layer32/eco_smr.replay2.perfetto.json) |

每份含 8 个 rank；轨道为 combine、shared expert L1、shared expert L2、`count exchange + pull token`（同轨）、wait token、routed GEMM；两个 ECO 臂另有 ECO plan。routed GEMM 起止取实际领到任务的首个 CTA 到完成的末个 CTA（eco_smr 含 role 6 参与的计算）。每段带 `sample`、`epoch`、`layer`、`arm`、`bs`、`rank_tokens`、`m_blocks`。

## 校验结果

- **输出一致性**：BS512 与 BS1024 各层五臂逐 rank 输入 SHA256 相同、同一层第二次 replay 输出 SHA256 相同；2 BS × 3 层 × 5 臂共 30 组全部 `output_equal=True`，无「输出差异」组。
- **逐 rank 逻辑路由**：每层八 rank 总路由项 = bs×6（3072 / 6144）；逐 arm 逐逻辑专家核对物理槽计数与请求路由项一致（`sm90_se_profile.py` 的 `physical counts differ from requested logical routes` 断言全部通过）。
- **replay 完整性**：`epoch + 3`、输出有限；八 rank 取同一 replay 的 epoch 一致。
- **ECO 一致性**：eco 与 eco_smr 的逐槽分配、拍卖计数完全一致（`SM reuse changed ECO allocation` 未触发）；在线 role-5 fine cost 与离线 `slot_cost` 逐槽求和相等；八 rank 拍卖记录一致，成交数在三层与两臂均为 0（本轮 448 槽 / 单副本前置未触发迁移拍卖）。
- **物理槽数**：vanilla 48/rank（384 逻辑=384 物理），random/rr/eco/eco_smr 56/rank（448 物理）。

## 结论摘要

- BS512：vanilla 566.6 → random +13.74%、rr +11.38%、eco +1.71%、eco_smr **−4.96%**。
- BS1024：vanilla 624.8 → random +11.56%、rr +10.60%、eco +0.42%、eco_smr **−5.07%**。
- EPLB 的 rr 两档均比 vanilla 慢约 10–11%（本口径的完整 MoE 含 plan 与通信；448 槽使 M 块 384→446 增加约 16%），random 与 rr 同向且更慢；eco 把 M 块压回 384 后基本回到 vanilla 水平，再叠加 role 6 复用 dispatch SM（eco_smr）在 64 token（BS512）与 128 token（BS1024）两种稠密度下都稳定获得约 5% 的完整 MoE 收益。
- 两档 BS 的 ECO 预测误差都在 0.02–0.08 pp（routed）与 0.02–0.10 pp（graph）量级，除 BS1024 random 的 graph MAE 0.24 pp 外均 < 0.15 pp。

## 归档与退卡

- 全部产物（两档 BS 原样目录）：[https://github.com/ehdbdns/gcmoe-sm90/tree/main/docs/trace_executor_20261119/sm90_bs512_bs1024_five_replay2_results](https://github.com/ehdbdns/gcmoe-sm90/tree/main/docs/trace_executor_20261119/sm90_bs512_bs1024_five_replay2_results)，含 `execution.sha`、`gpus.csv`、`processes.csv`、五臂日志、三层 × 五臂的原始三次 Kineto trace（`rank*.json`）、设备戳（`rank*-device.json`，含物理槽计数与拍卖计数）、`layers.csv`、`ranks.csv`、`main_table.csv`、`moe.svg`。
- 原始数据在本机的完整位置：`/mnt/fuse/sm90-five-replay2-2UJy19Zb/bs512/`、`/mnt/fuse/sm90-five-replay2-2UJy19Zb/bs1024/`（八个 rank 的 trace、device JSON 与 CSV/SVG 与提交版本一致）。
- 最后一臂退出后八卡已释放：`nvidia-smi` 八卡各 4 MiB、无 compute app；本任务 worker（torchrun / benchmark_gc_moe_se.py）全部退出；调度器设备掩码未改动。
- 清理：本任务未产生私有编译产物（未编译内核、未建 JIT 私有缓存）。
