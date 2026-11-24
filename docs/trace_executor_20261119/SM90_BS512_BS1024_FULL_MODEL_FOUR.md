# SM90 第二版 · BS512 / BS1024 整模四臂复测

用同一组 8×H20 跑 DeepSeek‑V4.1‑Flash 的真实 decode。每臂用前 1000 个满批纯 decode 步统计需求，按臂配置完成重排，再对连续 1000 步的全部 40 个 MoE 层计时。每档报告 fine ECO 相对逐层 EPLB + RR 的 MoE 耗时差。正常完成共启动模型八次。

| BS | 第 1 臂 vanilla | 第 2 臂 random | 第 3 臂 rr | 第 4 臂 eco |
|---:|---|---|---|---|
| 512 | 原生 | 随机放置 + RR | 逐层 EPLB + RR | 逐层 EPLB + fine ECO |
| 1024 | 原生 | 随机放置 + RR | 逐层 EPLB + RR | 逐层 EPLB + fine ECO |

## 从第一版交接

第一版 BS512 vanilla 在统计窗口结束时有 40,000 条连续 epoch，但实测模型层号未通过 `(epoch - 1) % 40`。没有正式测量行。旧回执提交时 BS1024 vanilla 仍在运行。先查看旧现场 `/mnt/fuse/sm90-four-full-JblIP9a` 的启动器、`bs*.exit`、各臂 `status.txt` 和日志尾部，将各档实际退出状态记在新回执。已有作业仍在运行就等它退出并释放 GPU。旧数据保留作诊断，第二版每档由新 SHA 的 vanilla 重新生成输入与边界快照。

第二版启用四臂自身的层序对齐。子图预热结束前，启动器所在的 worker 通过真实 GC 调用补齐一个 40 层周期，连同各 rank 的同步状态一起推进。在统计和正式窗口开始时，每个 worker 都检查层 0 的 epoch 余数；统计窗口第 1 步同时核对真实模型层号 0 到 39、40 个连续 epoch 和写入条数。这一步已包含在前 1000 步内。两窗口结束时仍须通过完整历史校验。保留启动器对 `SM90_EIGHT_ARM` 的清理，该变量还控制另一套模型路径和参考臂选择。

## 一次执行

1. 在原仓库获取第二版，只在站点允许使用且同组八张 H20 全部空闲时启动。保留调度器分配的设备掩码。启动器每臂在共享协作锁内复查占用，遇到其他作业就释放锁等待。
2. 使用节点已有的本地权重副本 `/mnt/fuse/models/DeepSeek-V4.1-Flash`，或用 `SM90_LOCAL_MODEL` 指定同版本已校验的本地副本。启动器校验挂载类型、配置、权重索引、48 个索引分片和数据集哈希，每臂用 `warm_page_cache.sh` 预热本地分片，同时设置 `GC_MOE_NO_PREFETCH=1`。checkpoints 默认放在本次 job 子目录；需要时提前指定容量足够的本地 `GC_MOE_CHECKPOINT_ROOT`。启动器在每档开始前检查可用空间。
3. 在原仓库运行下列命令。两档串行；任一臂失败即退出，记录首错。一次成功执行无需单独启动模型做冒烟、单层回放或 trace。

~~~bash
set -eo pipefail
git fetch origin main
book=docs/trace_executor_20261119/SM90_BS512_BS1024_FULL_MODEL_FOUR.md
rev=$(git log -1 --format=%H origin/main -- "$book")
job=$(mktemp -d /mnt/fuse/sm90-four-full-v2-XXXXXXXX)
echo "$job"
git worktree add --detach "$job/code" "$rev"
cd "$job/code"
grep -Fq 'requires_layer_aligned_warmup' integration/deepep_compare.py
rc=0
bash scripts/sm90/run_full_model_four.sh "$job" > "$job/launcher.log" 2>&1 || rc=$?
echo "$rc" > "$job/exit-code.txt"
exit "$rc"
~~~

4. 从 `$job/delivery-<BS>/bs<BS>/<arm>/runner.log` 提取该臂的 `[warm] sizes=`、`[layer-phase] stats first_epoch=`、`[layer-phase] stats step 1 verified` 和 `[layer-phase] measure first_epoch=`。当前单个 capture bucket 加 256、2048 两个额外大小，预期 `sizes=3 padding=37`，初值为 1 时应看到 `epoch=1->41`。prefill 和 CUDA Graph 捕获后，以实际统计窗口首 epoch 为准。它应满足 `(first-1)%40 == 0`，第 1 步覆盖 `first..first+39`；连续两窗口的首 epoch 相差 40,000。用每臂 `run.json` 中两窗口的 `initial`、`audit` 验证八 rank 各有 40,000 次调用，记录首 epoch 与调用数。
5. 遇到失败保存 `launcher.log`、该档退出码、首错栈、以上标记、GPU 占用及已启动臂数。仅可处理保持语义的编译错误，并附实际 diff 和执行 SHA；涉及修复时，同一张比较表的八臂必须可追溯至同一测量版本。层序、输入、公平性及测量失败按原样回执。完整且同 SHA 的 delivery 可由启动器校验 SHA256SUMS 后跳过。部分完成的目录保留；若仅 CPU 汇总失败，使用站点实验 Python 执行 `scripts/trace_layer/main20_report.py DELIVERY --bs BS --four-arms` 复核已有数据。

## 共同配置与公平性

- TP8/EP8、FULL_DECODE_ONLY CUDA Graph、M64、stream、fused split shared、6 个 dispatch SM，四臂 SM-reuse 均关闭。ECO 使用 fine cost、slack 11469，plan 占通信分区一个 SM。每 rank 20 GiB KV。使用固定顺序的两份 JSONL、固定请求顺序和稳定 prefill/decode top-k；BS1024 取同一 BS512 请求序列两遍。
- 每个 BS 的 vanilla 发布本档 tokens、统计和正式两套路由及两个边界的 KV，其他三臂消费同一份。逐层 EPLB 在 rank 0 GPU 上根据本档完整 1000 步需求生成布局和 RR 副本顺序。ECO 使用同一套 rank 副本归属，在 rank 内执行单副本优先排列。
- 每档四臂做六对两两核验。报告 `warm_native_equal`、`strict_pair_comparable`、正式原生 top‑1 不同数及比较数（BS512 为 512,000，BS1024 为 1,024,000）。成功对照要求两项均为真、top‑1 不同数为零。核对 prompt、硬件、配置、prefill 步数、每步满批、抢占情况、各阶段 rank 路由 ID/权重/顺序、逻辑需求、KV 读回、逻辑专家权重和正式输出。保留 RR 物理槽校验与 ECO 实际价格、拍卖计数。某对失败，回执列出臂对、首个不一致字段及原始值。

## 主表与回执

- 各次层调用先取八 rank 从本地 graph 起点到 combine 结束的最大跨度，再对 1000 步 × 40 层等权平均，单位 µs/层。相对变化按 `100 × (本臂 / 同档参考臂 − 1)`，分别相对 vanilla 和 EPLB + RR，负为快、正为慢。主表以 MoE 为准，原始整步 ms/步留作审计。
- 逐臂交付 token IR、八 rank 总 M 块、峰值 rank M 块、活跃物理槽、M 块 IR、routed 与 graph 的各 rank 耗时份额预测 MAE（百分点）。routed 从首个真实 compute CTA 开始到最后一个完成，包含末尾 push。ECO 另交付拍卖成交总数、每层调用平均成交数、报价数、接受率、零成交调用数。保留 rank 原始计数和时钟、rank_shares.csv 及逐层数据。
- 两档均成功后，汇总目录应有 `main_table.csv` 八行、`pairs.csv` 十二行、`layers.csv` 三百二十行、`auction.csv` 两行（均不含表头）。回执打印八行主表、每档 ECO 对 RR 的绝对差与有符号百分比、各臂公平性结果、层序证据、prefill 和权重加载时间（挂载点、48/48 分片、日志可得的初始化至 prefill 时间）。
- 结果提交到 `docs/trace_executor_20261119/sm90_bs512_bs1024_full_model_four_v2_results/`，回执提交到 `docs/trace_handover_20261119/REPLY_SM90_BS512_BS1024_FULL_MODEL_FOUR_V2.md`。保留第一版回执。提交两档 delivery、汇总 CSV、输入校验、测量 SHA、加载与启动日志、各档退出码。失败时列出已完成与未启动的臂，有效原始结果单列。初版与本版的退出状态分别说明。
- GPU 阶段结束后释放协作锁和站点预约，离线汇总并推送。大型原始 KV 和 route bank 留在本轮私有现场或按启动器策略清理，写明路径与保留项。核验远端归档后清理本轮临时 worktree 和可重建的私有文件。本次交付写入专属结果目录，README 主表待复核。
