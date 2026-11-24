# 回执 · SM90 第二版 · BS512 / BS1024 整模四臂复测

执行者：SM90 八卡执行者。任务书 `docs/trace_executor_20261119/SM90_BS512_BS1024_FULL_MODEL_FOUR.md`（第二版）。

| 项 | 值 |
|---|---|
| 测量 SHA | `a1bc4476c03c55931088f8b308f9034db7c0026c` |
| `exit-code.txt` | **0** |
| `bs512.exit` | **0** |
| `bs1024.exit` | **0** |
| 模型启动次数 | **8 次**（两档各四臂，逐臂独立加载）|
| main_table / pairs / layers / auction 行数 | **8 / 12 / 320 / 2**（与任务书要求逐项一致）|

## 第一版的故障与第二版的修复

第一版（`74a5395`）两档的 vanilla 臂均在统计窗口收尾失败，`route-diag` 显示 `pos/epochs/ordinals/layer0/counts` 全部通过、仅 `layers_eq=False`，即层序相对 `(epoch-1)%40` 有固定相位偏移；对齐守卫因 `run_full_model_four.sh` 清除 `SM90_EIGHT_ARM` 而未生效。第一版 BS512 与 BS1024 各失败一次，无任何测量行。

第二版新增四臂自身的层序对齐，八臂全部一次通过，未出现任何 `route-diag`（全臂 `diag=0`）。

## 层序对齐证据（逐臂）

| BS | 臂 | `[warm] sizes/padding` | `stats first_epoch` | `(f-1)%40` | `measure first_epoch` | 差 |
|---|---|---|---|---|---|---|
| 512 | vanilla | 3/37 | 11081 | 0 | 51081 | 40000 |
| 512 | random | 3/37 | 11081 | 0 | 51081 | 40000 |
| 512 | rr | 3/37 | 11081 | 0 | 51081 | 40000 |
| 512 | eco | 3/37 | 11081 | 0 | 51081 | 40000 |
| 1024 | vanilla | 3/37 | 26201 | 0 | 66201 | 40000 |
| 1024 | random | 3/37 | 26201 | 0 | 66201 | 40000 |
| 1024 | rr | 3/37 | 26201 | 0 | 66201 | 40000 |
| 1024 | eco | 3/37 | 26201 | 0 | 66201 | 40000 |

八臂 `sizes=3 padding=37` 与任务书预期一致；`(first-1)%40` 全为 0；两窗口首 epoch 差恒为 40,000（1000 步 × 40 层）。

## 主表（八行，µs/层，1000 步 × 40 层等权平均）

| BS | 臂 | MoE ms/步 | vs vanilla | E2E ms/步 | token IR | 总 M 块 | 峰值 rank M 块 | 活跃槽 | M 块 IR |
|---|---|---|---|---|---|---|---|---|---|
| 512 | vanilla | 71.356 | +0.00% | 162.71 | 1.394 | 564.2 | 89.4 | 334.7 | 1.268 |
| 512 | random | 76.055 | +3.42% | 168.28 | 1.481 | 608.6 | 98.6 | 391.0 | 1.296 |
| 512 | rr | 59.486 | -6.71% | 151.79 | 1.061 | 595.9 | 78.7 | 398.7 | 1.056 |
| 512 | eco | 57.212 | -8.24% | 149.30 | 1.033 | 564.2 | 73.2 | 369.7 | 1.038 |
| 1024 | vanilla | 136.473 | +0.00% | 316.33 | 1.396 | 944.1 | 156.0 | 359.9 | 1.322 |
| 1024 | random | 144.252 | +2.75% | 325.01 | 1.480 | 984.0 | 168.6 | 420.1 | 1.370 |
| 1024 | rr | 110.076 | -8.36% | 289.88 | 1.044 | 976.5 | 127.1 | 423.9 | 1.042 |
| 1024 | eco | 106.860 | -9.93% | 284.93 | 1.019 | 944.1 | 120.9 | 404.2 | 1.024 |

### ECO 相对 EPLB + RR

| BS | rr MoE | eco MoE | 差值 | 百分比 |
|---|---|---|---|---|
| 512 | 59.486 | 57.212 | **-2.274 ms/步** | **-3.82%** |
| 1024 | 110.076 | 106.860 | **-3.216 ms/步** | **-2.92%** |

## 公平性：12 对全部通过

| BS | 臂对 | native_diff | 样本 | warm_native_equal | strict_pair_comparable | MoE 变化 | E2E 变化 |
|---|---|---|---|---|---|---|---|
| 512 | vanilla→random | 0 | 512000 | True | True | +6.59% | +3.42% |
| 512 | vanilla→rr | 0 | 512000 | True | True | -16.63% | -6.71% |
| 512 | vanilla→eco | 0 | 512000 | True | True | -19.82% | -8.24% |
| 512 | random→rr | 0 | 512000 | True | True | -21.79% | -9.80% |
| 512 | random→eco | 0 | 512000 | True | True | -24.78% | -11.28% |
| 512 | rr→eco | 0 | 512000 | True | True | -3.82% | -1.64% |
| 1024 | vanilla→random | 0 | 1024000 | True | True | +5.70% | +2.75% |
| 1024 | vanilla→rr | 0 | 1024000 | True | True | -19.34% | -8.36% |
| 1024 | vanilla→eco | 0 | 1024000 | True | True | -21.70% | -9.93% |
| 1024 | random→rr | 0 | 1024000 | True | True | -23.69% | -10.81% |
| 1024 | random→eco | 0 | 1024000 | True | True | -25.92% | -12.33% |
| 1024 | rr→eco | 0 | 1024000 | True | True | -2.92% | -1.71% |

所有 12 对：`native_diff=0`、`warm_native_equal=True`、`strict_pair_comparable=True`；样本数 BS512 为 512,000、BS1024 为 1,024,000，与任务书一致。

## ECO 拍卖与价格

| BS | rounds | offers | accepted | 接受率 | 每层调用成交均值 | 零成交调用 | 无报价轮 |
|---|---|---|---|---|---|---|---|
| 512 | 40159 | 345915 | 226980 | 65.62% | 5.6745 | 0 | 37 |
| 1024 | 40165 | 433250 | 293457 | 67.73% | 7.3364 | 0 | 33 |

两档 `calls=40000`（1000 步 × 40 层），零成交调用均为 0。

## 结论

- 两档 EPLB+RR 与 EPLB+fine ECO 均显著快于原生：BS512 rr −16.63%、eco −19.82%（MoE）；BS1024 rr −19.34%、eco −21.70%。
- **ECO 相对 RR 在 MoE 上再快 3.82%（BS512）与 2.92%（BS1024）**；端到端 −1.64% / −1.71%（MoE 只占整步的一部分）。
- 随机放置+RR 反而比原生慢 6.59% / 5.70%，说明收益来自 EPLB 的逐层重排而非单纯多副本。
- token IR 从 vanilla 1.394/1.396 降到 rr 1.061/1.044、eco 1.033/1.019，是主要机制。

## 权重加载证据（本地 ext4 + 页缓存预热 + 关闭预取）

| 档/臂 | 48 shard 用时 | 速率 |
|---|---|---|
| 512/vanilla | 00:04 | 10.89 it/s |
| 512/random | 00:07 | 6.21 it/s |
| 512/rr | 00:17 | 2.68 it/s |
| 512/eco | 00:18 | 2.67 it/s |
| 1024/vanilla | 00:19 | 2.51 it/s |
| 1024/random | 00:04 | 9.83 it/s |
| 1024/rr | 00:15 | 3.12 it/s |
| 1024/eco | 00:09 | 4.94 it/s |

模型路径 `/mnt/fuse/models/DeepSeek-V4.1-Flash`（ext4，非网络文件系统），启动器已校验挂载类型、config、safetensors 索引、48 分片与两份数据集 SHA256（见 `input-check.log`）。

## 归档

- 全部产物：https://github.com/ehdbdns/gcmoe-sm90/tree/main/docs/trace_executor_20261119/sm90_bs512_bs1024_full_model_four_results
- 含 `delivery-512/`、`delivery-1024/` 两个完整 delivery、汇总 `main_table.csv`(8)/`pairs.csv`(12)/`layers.csv`(320)/`auction.csv`(2)、两档 exit、`launcher.log`、`bs512.log`/`bs1024.log`、`measurement.sha`、`inputs.sha256`/`input-check.log`/`model-mount.txt`。
- 每臂目录含 `runner.log`、`run.json`、`status.txt`、`execution.sha`、八 rank NPZ、`installed-layout.json`、`hardware.csv`、`rank_shares.csv`、`pairs.csv`。
- 现场完整目录（含 KV、route bank、checkpoints）：`/mnt/fuse/sm90-four-full-v2-mvuSgtVf`。原始 KV 已按脚本默认在报告通过后清除（`GC_MAIN20_KEEP_KV` 未设）；route bank 与 `stats-demand.npz` 保留在 delivery 内。

## 退卡与清理

- 最后一臂（BS1024 eco）04:15:38 完成，`exit_code=0`；launcher 04:17:01 汇总完毕，`exit-code.txt=0`。
- 八卡已释放：各 4 MiB、无 compute app；本任务 worker 全部退出；协作锁已释放。
- 未修改任何代码（diff 为空），无编译修复。
