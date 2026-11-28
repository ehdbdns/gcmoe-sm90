# SM90 执行任务书 · H20 EP8 · BS512、BS1024 细粒度 ECO 拍卖

沿用 BS256 粗细粒度实验的代码路径和代价函数，检验大 BS 下小批报价能否缩小实际 routed 慢 rank 的耗时差距，以及完整 MoE 能节省多少时间。只运行以下四臂。每个 BS 新测自己的粗粒度对照。

| 顺序 | 全局 BS | 均摊请求数/rank | 结果目录 | `GC_MOE_ECO_FINE_AUCTION` |
|---:|---:|---:|---|---:|
| 1 | 512 | 64 | delivery-512/bs512/eco | 0 |
| 2 | 512 | 64 | delivery-512/bs512/eco_small | 1 |
| 3 | 1024 | 128 | delivery-1024/bs1024/eco | 0 |
| 4 | 1024 | 128 | delivery-1024/bs1024/eco_small | 1 |

## 实验设置

- 同一组 8 张 H20，DeepSeek-V4.1-Flash，TP8/EP8。两臂都使用线性价格 `token 数 + 26 × M 块数 + 2 × 活跃物理槽数`，role 5 的 mode 为 0，slack 为 92。粗粒度用整块或专家原有尾块，细粒度在暖启动最后一轮额外比较 8、16、32 行报价。每卡 8 个冗余槽，保留每个专家的总 M 块数。
- 每臂运行完整模型。先统计 1,000 个满批 decode 步，按统计得到逐层 EPLB 放置，清空各层 ECO 状态，再计时 1,000 步 × 40 层。使用 M64、stream、TMA pull、fused split shared、原生 launch 顺序、FULL_DECODE_ONLY CUDA Graph、关闭 SM-reuse。plan 从 dispatch 的 6 个 SM 中拿 1 个。
- 每档优先使用该 BS 归档中校验通过的统计路由、正式路由和反馈 token。原件已清理时，由该档粗粒度臂在本次运行中录制，细粒度臂读回。无论哪条路径，两臂均读回同一对新建的 prefill/测量边界 KV。BS1024 沿用主表的固定 BS512 提示集重复规则。每档分别记录 `reference-source.txt` 和 `reference-mode.txt`。

## 执行步骤

1. 使用现有 H20 实验节点及其软件环境，取得站点预约，保留调度器分配的设备掩码。同一组八卡全部空闲再开始，有其他用户或预约则等待。启动器每臂加共用锁并复查 GPU 进程。只占这八卡。
2. 指定节点本地模型副本（默认 `/mnt/fuse/models/DeepSeek-V4.1-Flash`，其他本地路径可设 `SM90_LOCAL_MODEL`）。启动器校验模型索引、48 个分片和数据集，每臂加载前预热本地权重。临时 KV/路由保存在本作业独有的目录，启动前检查磁盘。按当前设置，两档各自开始时约需 346、368 GiB 可用容量。前一档通过校验后启动器会释放其大块 KV。
3. 在 H20 仓库执行下列命令。记录打印的 job 路径。一次启动依次执行四臂，并按 BS 单独校验。正常结束后打印两档汇总。

```bash
set -eo pipefail
git fetch origin main
book=docs/trace_executor_20261119/SM90_BS512_BS1024_FINE_AUCTION.md
rev=$(git log -1 --format=%H origin/main -- "$book")
test -n "$rev"
job=$(mktemp -d /mnt/fuse/sm90-bs512-1024-fine-auction-XXXXXXXX)
echo "$job"
git worktree add --detach "$job/code" "$rev"
cd "$job/code"
rc=0
bash scripts/sm90/run_full_model_four.sh --auction-bs512-1024 "$job" > "$job/launcher.log" 2>&1 || rc=$?
echo "$rc" > "$job/exit-code.txt"
for bs in 512 1024; do
  delivery="$job/delivery-$bs"
  if [[ -f "$delivery/SHA256SUMS" ]]; then
    (cd "$delivery" && sha256sum --quiet -c SHA256SUMS)
    cat "$delivery/bs$bs/"{main_table,auction}.csv
  elif [[ -f "$job/bs$bs.log" ]]; then
    tail -80 "$job/bs$bs.log"
  fi
done
exit "$rc"
```

4. 执行者只可修复保持语义、计时边界和配置一致的编译错误，保留首错、diff 和实际测量 SHA。同一对照的两臂必须来自同一提交。其他失败请保留现场，按 BS 回报已完成的臂和首个分歧。若只有 CPU 汇总失败，使用已归档的数据，在测量 checkout 的实验 Python 下运行 `scripts/sm90/auction_pair.py report "$job/delivery-$bs" --bs "$bs"`，无需再占卡。

## 验收和回执

1. 每档核对相同的输入、统计及测量路由 ID、权重、源行顺序、逻辑需求、边界 KV、逐层槽位图、运行配置和执行 SHA。正式原生 top-1 全量配对，BS512 为 512,000 个，BS1024 为 1,024,000 个。逐步逐层按逻辑专家检查 token 守恒及 M 块总数，按各 rank 实际配额重算代价。只有各项校验通过的 BS 才报告性能差值。
2. 每臂报告完整 MoE、8 rank 平均 routed、最慢 rank routed、后两者的 µs 差值，以及最慢 rank 超出均值的百分比。routed 用各 rank 首个 CTA 真正开始 routed GEMM 到最后一个 CTA 完成（含末尾 push）的本地跨度。对每个步层先算八卡峰值、均值、差值和百分比，再对 40,000 次调用取平均。完整 MoE 用 graph 起点到 combine 结束的最大本地跨度，每档计算细粒度相对粗粒度的带符号差值及百分比。按时间顺序列出 20 个连续 50 步配对分段，并给出快、平、慢段数。
3. 同列 token IR、八卡 M 块总数、单 rank 峰值 M 块、M 块 IR、活跃槽总数与峰值、预测峰值、routed MAE、graph 峰值超均值、最慢 rank 的拍卖耗时、报价数、成交数、接受率、小批成交数和发生小批成交的步层数。拍卖成交按步层计一次。均衡效果看实测 routed 峰值超均值，预测误差看 MAE，整体收益看完整 MoE。各档单独下结论。
4. GPU 运行结束即释放预约，确认本任务 worker 已退出。在 CPU 上整理并提交两份 delivery（两臂八 rank NPZ、运行和公平性记录、四个 CSV、校验和），以及启动日志、分档日志、退出码和测量 SHA。结果推到 `docs/trace_executor_20261119/sm90_bs512_bs1024_fine_auction_results/`，回执推到 `docs/trace_handover_20261119/REPLY_SM90_BS512_BS1024_FINE_AUCTION.md`。核验远端归档后清理本任务私有 KV、临时路由和 checkout，保留历史引用文件和共享模型。
