# SM90 · BS512、BS1024 各采集一次真实路由，再回放五臂

本任务接替此前要求寻找历史 bank 的任务。直接在共享 8×H20 上执行。每档只启动一次 DeepSeek‑V4.1‑Flash vanilla 整模。一次连续的纯 decode 中，前 1000 步保留每步每层的逻辑专家需求，随后 3 步记录 L8、L20、L32 各源 rank 的真实 top‑6 ID、权重和行顺序。五臂共用这份小文件。新样本的步号从统计窗口结束后开始，replay2 对应第二步（零基索引 1）。保留原 prompt 筛选、长度预算和满批条件；采集模式在第 3 个正式步后返回。

## 工作量与配置

| BS | 每源 rank 实际行数 | 每层八卡路由项 | 整模启动 | 算子启动 |
|---|---:|---:|---:|---:|
| 512 | 512 | 24,576 | vanilla × 1 | 五臂各 1 次 |
| 1024 | 1024 | 49,152 | vanilla × 1 | 五臂各 1 次 |

模型端使用 TP8/EP8、CUDA Graph、每 rank 20 GiB KV、稳定的 prefill/decode top-k；BS1024 沿用已验证的 BS512 请求批次重复两遍。模型自然生成 token。只把压缩的需求统计、选中路由切片及 vanilla 接收槽、clock、epoch 落盘，整模 KV 无需在臂间传递。

算子端五臂为 vanilla、固定随机放置+RR、逐层 EPLB+RR、逐层 EPLB+fine ECO、同一 ECO+SM-reuse。按本轮前 1000 步的逐步需求计算 DeepInfer 放置，后 3 步只用于测量；random 沿用种子 0。五臂使用相同源输入、固定种子的算子激活及专家矩阵，M64、stream、split shared、6 dispatch SM，ECO 用 fine cost / slack 11469、通信分区 plan。每臂一次进程组内跑三个层。每层预热后重置 ECO 状态，再按第 1、2、3 步顺序回放，第二步继承第一步的分配。目标是比较同批真实路由上的算子性能；这次只采到了 vanilla 整模耗时，另四臂的结果以算子回放标注。

## 一次执行

预约同一组空闲的八张 H20，保留站点设备掩码。已有其他 GPU 作业时等待。启动器在整模采集及每个算子臂前持本机协作锁复查八卡。只启动本任务；如执行者正准备旧 bank 查找，到此切换。执行命令如下，job 在持久卷并打印到终端。

~~~bash
set -eo pipefail
git fetch origin main
book=docs/trace_executor_20261119/SM90_BS512_BS1024_COLLECT_ONCE_REPLAY.md
rev=$(git log -1 --format=%H origin/main -- "$book")
job=$(mktemp -d /mnt/fuse/sm90-collect-once-XXXXXXXX)
echo "$job"
git worktree add --detach "$job/code" "$rev"
cd "$job/code"
for bs in 512 1024; do
  rc=0
  bash scripts/sm90/collect_and_replay_se.sh "$job/bs$bs" "$bs" >"$job/bs$bs.log" 2>&1 || rc=$?
  echo "$rc" >"$job/bs$bs.exit-code"
done
~~~

每档的 `collection/capture.json` 在全部八卡采完后写出；`collection/routes/manifest.json` 记录 24 个源切片的哈希，可连同切片直接归档。GPU 工作全部结束后释放预约，再处理 CPU 导出与 GitHub 交付。

## 验收

1. 每档八卡全程满批，1000 个统计步 × 40 层的总路由量均为表中数值。每源 rank 按层交付一个 `[3, BS, 6]` 切片，含 int64 ID、float32 权重。汇总源需求必须等于本次 vanilla 整模逐槽接收数。核对执行 SHA、数据集哈希、请求哈希与切片清单。
2. 在 `replay/capture_comparison.csv` 核对 **vanilla 的 72 行**（3 步 × 3 层 × 8 rank），所有 `different_slots=0`。其中 `history_*` 列是本次整模采集的值，可用于比较该层 graph 耗时。对其余四臂按相同源需求核查：random/RR 的实收物理槽要与源端算出的目标槽一致；ECO 和 eco_smr 三步的逐槽分配及七项拍卖计数一致。展示的第二步还要求五臂逐 rank 输入、输出哈希一致。只评价新样本，保留实测排名及差异。
3. 每档完整交付五行 `main_table.csv`、15 行 `layers.csv`、120 行 `ranks.csv` 和 `moe.svg`。MoE 以每层最慢 rank 的本地 graph→combine 跨度计，三层等权平均，单位 µs/层；相对变化以同档同批 vanilla 为分母，快为负、慢为正。同时给出 token IR、总/峰值 M 块、活跃槽、M 块 IR、routed/graph 份额预测误差（百分点），以及两条 ECO 臂三层拍卖成交均值和合计。
4. 每档 15 个 `replay/layer{8,20,32}/<arm>.replay2.perfetto.json`，每图 8 rank。保留 count exchange 与 pull token 同行、shared L1/L2、wait token、首末实际计算 CTA 间的 routed GEMM、combine；ECO 另有 plan。保留原始 Kineto 与设备戳。

## 恢复、交付与清理

- 已存在完整 `collection/capture.json` 时，同一 SHA 对同一 BS 目录重跑启动命令会直接使用采集文件。回放目录会核对输入清单并跳过本次已成功的臂。CPU 导出失败同样可通过这一入口重试。若某臂部分运行失败，先在现场备份该臂的首错日志和部分 trace。若采集未完成，先归档首错与 `Capture workspace:` 标记的目录，注明真实缺失阶段；仅在确认该次采集未完成且故障已处理后重试，不覆盖现场。
- 执行者只可处理保持语义的编译问题，附 diff 并固定相同测量版本；算法、映射、样本、步骤和验收公式由设计端维护。两档独立；单档失败仍交付另一档和已有证据。
- 结果提交 `docs/trace_executor_20261119/sm90_collect_once_replay_results/`，按 `bs512/{collection,replay}`、`bs1024/{collection,replay}` 保留结构，附两个退出码、启动日志和 SHA。将**真实路由切片、manifest、需求统计与实际布局一起提交**，以后可从 GitHub 重建本轮回放。回执提交 `docs/trace_handover_20261119/REPLY_SM90_BS512_BS1024_COLLECT_ONCE_REPLAY.md`。打印十行主表，给出 30 份 Perfetto 与两张图的 GitHub 地址，注明每档实际整模启动数、一致性及任何失败。保留 README 整模主表口径。
- 核查本轮 worker 已退出、站点预约已释放。核验远端提交与归档后清理本任务私有临时编译物和 worktree，记录释放情况和持久归档位置。
