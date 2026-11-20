# 固定 stream 的 fine cost 两臂复核

回执提交 `b009512`，测量提交 `6ed27c2`。同一台 8×H20 共启动两次模型，两臂均在回放与测量期间使用逐 M 块 stream。线性价格和 fine cost 各自参与 ECO 拍卖，本轮沿用已下发的 valid 定价，含 L2 末尾的 push。以下数据来自已交付的 step 2 采样和同进程的 100 步计时。[原始交付](../trace_executor_20261119/valid_cost_results) · [逐 rank 复算数据](VALID_COST_ROUTED_RANKS.csv)

## Routed 负载预测

主指标取每个 rank 的最后一个 tile 结束时刻减去第一个 tile 开始时刻。分别从同一次 absolute Perfetto 的 6/7 型打点还原 11 层 × 8 个 rank 的跨度，核对每个 L2 的 push 区间均位于对应 tile 内。对每层分别将八卡的预测分与实测跨度归一化，误差等于两者占比之差乘 100。表中 MAE 与最大绝对误差的单位均是百分点。每一行固定该臂已经观察到的物理配额和时间，对同一组结果用两种模型打分。

| 实际拍卖使用的价格 | 线性模型 MAE | fine 模型 MAE | 线性模型最大绝对误差 | fine 模型最大绝对误差 |
|---|---:|---:|---:|---:|
| 线性 | 0.1450 | 0.1114 | 0.4356 | 0.3557 |
| fine | 0.1369 | 0.0996 | 0.5681 | 0.4903 |

两臂各用自己上线的模型时，MAE 从 0.1450 降至 0.0996，降低 31.3%。最大绝对误差从 0.4356 升至 0.4903，后者位于第 20 层 rank 6。固定任一臂的分配再打分，fine 模型的 MAE 和最大绝对误差均较低。第 36 层 rank 4 在 fine 臂的**整层 graph** 误差为 +1.2601 个百分点，同一个 rank 的**精确 routed** 误差为 +0.2326 个百分点。[第 20 层细采](../trace_executor_20261119/valid_cost_results/step02/stream/paired/step002_layer20_epoch14501.absolute.perfetto.json) · [第 36 层细采](../trace_executor_20261119/valid_cost_results/step02/stream/paired/step002_layer36_epoch14517.absolute.perfetto.json)

全 40 层都有从首次有效领票到 routed kernel 退出的普通 Perfetto 区间。用这个近似区间，两臂上线模型的 MAE 分别为 0.1316 和 0.1130。精确首末 tile 只在 11 层有打点。CSV 中其余层的精确跨度及对应误差留空，`valid_rows_to_block_count` 给出每个有效行数对应的 M 块个数。[线性臂各层](../trace_executor_20261119/valid_cost_results/step02/baseline/paired) · [fine 臂各层](../trace_executor_20261119/valid_cost_results/step02/stream/paired)

## 实测耗时

| 指标 | 线性 | fine | 变化 |
|---|---:|---:|---:|
| 11 个细采层，每层最慢 rank 的精确 routed 跨度，再取层均值 | 618.18 µs | 616.00 µs | −0.35%，8 层快、3 层慢 |
| step 2 每个 rank 的 40 层 MoE graph 跨度之和，再取八卡最大值 | 33.9774 ms | 33.9610 ms | −0.048% |
| 100 个无 profiler 步的 E2E 均值 | 80.9502 ms/步 | 80.8545 ms/步 | −0.118% |

10 个连续的十步计时段中 fine 臂 6 段快、4 段慢。两臂各运行一次，这轮的耗时差尚无独立重复测量验证。[配对耗时](../trace_executor_20261119/valid_cost_results/paired.csv)

## 回执核对

- 已核验交付包校验和，并从两臂审计记录重新生成比较结果。`cost_inputs_match`、`same_run_controls`、`same_logical_routes` 与 `strict_pair_comparable` 均为 true。105 步的 26,880 个原生 top-1 全部相同。
- 回执正文的物理路由栏有误。`same_physical_routes` 实际为 false。第 2 步的 320 个层与 rank 配对中，296 个的逐槽分配改变，265 个的 routed 行数改变，186 个的 M 块数改变。这些是拍卖的输出。每臂 320 条 rank 数据，两份 CSV 共 640 条数据及两个表头。
- `scheduling_gate_passed` 仅供 stream 开关对照使用。比较器对 cost 实验将其设为 false。本轮的 `cost_pair_faster` 为 true，表示对照有效且两项主耗时的观测均值均较低。比较器内的 `prediction_mae_lower` 和 `prediction_max_error_lower` 使用整层 graph 占比，两项依次为 true、false。[原始比较 JSON](../trace_executor_20261119/valid_cost_results/comparison.json) · [比较器判据](../../scripts/trace_layer/multilayer_study.py#L513-L525) · [交付包校验和](../trace_executor_20261119/valid_cost_results/SHA256SUMS)
