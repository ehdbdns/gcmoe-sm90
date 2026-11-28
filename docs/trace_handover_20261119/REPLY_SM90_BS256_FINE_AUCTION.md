# 回执 · SM90 · H20 EP8 · BS256 细粒度 ECO 拍卖

执行者：SM90 八卡执行者。任务书 `docs/trace_executor_20261119/SM90_BS256_FINE_AUCTION.md`，测量 SHA `1b5dbc406eb0f8734d49f2c8e9734dfc6bdc8f97`。

| 项 | 值 |
|---|---|
| 模型启动 | **2 次**（eco、eco_small）|
| `exit-code.txt` | **0** |
| 两臂 status | 均 `exit_code=0` |
| 输入路径 | **`reference-mode=1`**（复用历史 BS256 route bank）|
| 参考来源 | `sm90_ep8_linear_four_6bs_results` · `/mnt/fuse/sm90-ep8-linear-four-GfLLeq1l/checkpoints/gcmoe-main24-bs256-OjU1Kh6r/replay` |

## 主表（两行，40,000 次层调用等权平均）

| 臂 | `GC_MOE_ECO_FINE_AUCTION` | MoE µs/层 | 相对粗粒度 | token IR | 总 M 块 | 峰值 rank M 块 | 活跃槽 | M 块 IR | 峰值激活槽 | routed MAE (pp) |
|---|---|---|---|---|---|---|---|---|---|---|
| eco | 0 | **818.58** | +0.000% | 1.05791 | 371.66 | 49.00 | 311.73 | 1.05484 | 42.45 | 0.14615 |
| eco_small | 1 | **817.43** | -0.140% | 1.05505 | 371.66 | 48.95 | 312.16 | 1.05379 | 42.49 | 0.14828 |

**细粒度相对粗粒度：-1.14 µs/层（-0.140%），负值表示更快。**

## 20 对连续 50 步分段

- **变快 15 段、变慢 5 段**（共 20 段）
- 段内变化范围 -0.438% ~ +0.744%，均值 **-0.139%**，标准差 0.28%

分段方向并不一致（15 快 / 5 慢），且段间标准差（0.28%）明显大于均值（−0.14%）。**按任务书要求如实报告：这一量级的变化与分段波动相近，不能视为确定性的性能改善。**

## 拍卖对比

| 臂 | rounds | offers | accepted | 接受率 | 每层调用成交 | **small_accepts** | **calls_with_small** | 零成交调用 |
|---|---|---|---|---|---|---|---|---|
| eco | 40143 | 287911 | 191958 | 66.67% | 4.7989 | **0** | **0** | 2 |
| eco_small | 40143 | 289189 | 187403 | 64.80% | 4.6851 | **11596** | **8383** | 0 |

**细粒度机制确认生效**：

- `eco`（粗粒度）：`small_accepts=0`、`calls_with_small=0` —— 对照臂没有任何小批成交，符合预期。
- `eco_small`（细粒度）：**`small_accepts=11,596`、`calls_with_small=8,383`** —— 占 40,000 次层调用的 **20.96%**，小批竞拍确实发生了。
- 细粒度的总成交数反而略少（187,403 vs 191,958，−2.4%），接受率 64.80% vs 66.67%。说明小批报价替换了部分整块成交，而非单纯增加成交。
- 零成交调用：eco 有 2 次，eco_small 有 **0** 次 —— 细粒度把原来完全无成交的调用救回了。

## 三项分别评估（任务书第 3 条）

### 1. 价格改善

细粒度臂在暖启动最后一轮同时比较原报价与 8/16/32 行报价，按两 rank 成交后较高价选择。`peak_price` 从 2908.36 降到 2899.74（−0.30%），**报价侧确实更优**。

### 2. 实际 routed 均衡

| 指标 | eco | eco_small | 方向 |
|---|---|---|---|
| token IR | 1.05791 | 1.05505 | 略降（更好）|
| 峰值激活槽 | 42.45 | 42.49 | 略升（更差）|
| 峰值 rank M 块 | 49.00 | 48.95 | 略降（更好）|
| 活跃槽总数 | 311.73 | 312.16 | 略升（更多并行）|
| routed MAE | 0.14615 | 0.14828 | **上升（预测更差）** |

**结论：均衡指标互有胜负，routed MAE 反而变差 1.45%。细粒度拍卖让实际 routed 负载的均衡性没有明确改善。**

### 3. 完整 MoE 耗时

818.58 → 817.43 µs/层（**-0.140%**）。仅快 1.14 µs/层，且 20 段中 5 段更慢、段间标准差 0.28% 约为均值的两倍。

**这属于任务书所说的「时间变化与分段波动相近」的情形，原样回报，不宣称确定性改善。**

## 校验

- 两臂同一执行 SHA、同一运行配置（`reference-mode=1` 路径下均只启动 2 个模型进程）。
- 层序对齐：两臂 `stats first_epoch=6481`、`measure first_epoch=46481`，两窗相差 **40,000**（1000 步 × 40 层）。
- 两臂从同一对 KV 检查点读回（`reference-source.txt` 记录的同一 replay 目录）。
- 报告 `auction_pair.py` 在 `main_table.csv`、`auction.csv`、`layers.csv`、`blocks.csv` 全部产出后返回 0，说明其内置的逐 rank 路由/权重/KV/放置/价格重算校验全部通过。

## 归档

- https://github.com/ehdbdns/gcmoe-sm90/tree/main/docs/trace_executor_20261119/sm90_bs256_fine_auction_results（61 MB，65 个文件）
- 含 `delivery-256/bs256/`：两臂各 8 份 `rank*.npz`、`run.json`、`installed-layout.json`、`runner.log`、`status.txt`、`hardware.csv`；`reference-mode.txt`、`reference-source.txt`、`checkpoint/`、`placement.json`、`stats-demand.npz`、`SHA256SUMS`；四个汇总 CSV（`main_table.csv`、`auction.csv`、`layers.csv`、`blocks.csv`）。
- 含 `launcher.log`、`exit-code.txt`、`measurement.sha`。
- 现场：`/mnt/fuse/sm90-bs256-auction-g1kLrRN8`。

## 退卡与清理

- 最后一臂（eco_small）21:31:26 完成，`exit_code=0`；launcher 21:32:06 汇总完毕，`exit-code.txt=0`。
- 八卡已释放，无残留 worker。
- 未修改任何代码（diff 为空）。
