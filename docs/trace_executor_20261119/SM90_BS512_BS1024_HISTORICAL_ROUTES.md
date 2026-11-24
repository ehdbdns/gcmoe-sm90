# SM90 · BS512 / BS1024 五臂真实路由 replay2

目标是以历史整模正式窗口的源 rank 路由和工作量，对比 vanilla、random+RR、逐层 EPLB+RR、EPLB+fine ECO、EPLB+fine ECO+SM-reuse。使用此前五臂算子 benchmark 的 SM90 配置、weights 和激活生成器。每档每臂启动一次，进程内跑 L8、L20、L32。全程无需启动 vLLM 或加载整模。单层微秒数反映当前算子版本和调度配置。

## 输入和时间线

- 固定历史正式窗口最初连续三步，文件名为 `measure-rank*.pt`。层号与步索引均从零起，行号为 `step*40+layer`。每个源 rank 原序读出整行 `[BS,6]` 的逻辑专家 ID 和 float32 路由权重。每层每步八卡应有 24,576 / 49,152 个路由项。隐藏激活与专家矩阵使用 benchmark 的确定性数据。
- CPU 提取器用已发布的 `workload.sha256` 验证 bank，按字节一致的归档哈希复用读取，再按源 rank 保存切片。两档各自八份归档在清单里具有相同哈希。抽取后，逐层逐步的 384 个专家需求须与历史 vanilla 八卡槽计数一致。
- 使用每档各臂实际安装的布局，ECO 保留 fine cost、slack 11469、通信分区 plan；仅 eco_smr 开启 role 6。预热完成后重置 ECO warm state，第一个被记录的 replay 使用历史第 1 步，从冷状态开始。后两个依次接续真实第 2、3 步。五臂一致地在 graph 外更新路由。出图固定取第二次 replay（正式步索引 1）；保留三次原始记录。
- 每档预期产出 15 份八 rank Perfetto。延续现有 replay2 轨道：count exchange 与 pull token 同行、shared expert L1/L2、wait token、实际 CTA 边界的 routed GEMM、combine，以及 ECO 的 plan。

## 在共享 H20 执行

预约同一组 **8 张空闲 H20**，有人使用时等待。保持站点设备掩码；运行器在每臂前持协作锁复查空闲。默认 bank 位于对应历史结果的 `state-path.txt` 所指目录下的 `replay/`。如现场归档已迁移，可以指定含同哈希源银行的 `SM90_BANK_512` 或 `SM90_BANK_1024` 目录。文件缺失时先在本机已经批准的实验归档内查找清单所列文件；有同哈希副本即可使用。若查无副本，报告路径、缺失哈希和已搜索归档，继续另一档，勿用随机路由或按直方图逆推顺序。

~~~bash
set -eo pipefail
git fetch origin main
book=docs/trace_executor_20261119/SM90_BS512_BS1024_HISTORICAL_ROUTES.md
rev=$(git log -1 --format=%H origin/main -- "$book")
job=$(mktemp -d /mnt/fuse/sm90-historical-routes-XXXXXXXX)
echo "$job"
git worktree add --detach "$job/code" "$rev"
cd "$job/code"
for bs in 512 1024; do
  group="docs/trace_executor_20261119/sm90_main24_stable_decode_bs${bs}_results/delivery/bs$bs"
  key=SM90_BANK_$bs
  bank=${!key:-$(cat "$group/state-path.txt")/replay}
  rc=0
  bash scripts/sm90/run_bs1024_se_profile.sh "$job/bs$bs" "$bs" "$group" "$bank" >"$job/bs$bs.log" 2>&1 || rc=$?
  echo "$rc" >"$job/bs$bs.exit-code"
done
~~~

就绪时直接完成两个 BS。遇到编译失败仅允许等价编译修复，附 diff，记录统一测量 SHA。运行或校验失败保留首错及已有原始数据。GPU 采集完成后即可释放八卡。CPU 可在原始 trace 所在目录重新执行 `audit_se_history.py GROUP OUT`、逐层 `se_perfetto.py OUT/layerN --bs BS --matrix --historical` 和 `se_replay_report.py OUT --historical`（三个脚本在 `scripts/sm90/`），无需重跑 GPU。

## 验收与交付

每 BS 的 `history_comparison.csv` 应有 360 行数据（5 臂 × 3 层 × 3 步 × 8 rank）。每一行比对同一历史步、同层、同 rank 的物理槽向量、token 数、M 块数；ECO 另核对七项拍卖计数。展示样本的五臂逐 rank 输入、输出哈希须一致。下表是已从历史八卡 NPZ 直接核出的**第二步**数值，可用于定位取样错位；时延和排名如实报告。

| BS | 指标（顺序 L8 / L20 / L32） | vanilla | RR | ECO |
|---|---|---|---|---|
| 512 | 峰值 rank M 块 | 98 / 114 / 84 | 80 / 79 / 82 | 76 / 73 / 75 |
| 1024 | 峰值 rank M 块 | 174 / 215 / 151 | 127 / 129 / 128 | 121 / 120 / 122 |
| 512 | 成交数 | — | — | 5 / 5 / 3 |
| 1024 | 成交数 | — | — | 8 / 4 / 7 |

- 归档到 `docs/trace_executor_20261119/sm90_bs512_bs1024_historical_routes_results/{bs512,bs1024}/`。提交两份执行 SHA、退出码与启动日志、各臂日志、输入 manifest、原始 Kineto 与设备数据、历史对账 CSV、`main_table.csv`、`layers.csv`、`ranks.csv`、`moe.svg` 及全部 replay2 JSON。完整 bank 留现场，注明路径。
- 回执放 `docs/trace_handover_20261119/REPLY_SM90_BS512_BS1024_HISTORICAL_ROUTES.md`。打印十行三层均值（µs/层），保留主表各项指标、同档相对 vanilla 的有符号变化、ECO 成交三层合计/均值；列出 30 份 Perfetto 和两张图的 GitHub 地址。比较同层的历史值和本轮值。工作量或输出未通过时，列出首个具体 BS/臂/步/层/rank 和差异，把该档标作待核对。
- 推送并核验远端，清理本任务临时 worktree 和缓存，确认本任务八卡进程退出、预约已释放，保留原始 bank 和可复核产物。README 整模主表保留原口径。
