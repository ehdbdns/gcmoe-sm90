# SM90 执行任务 · H20 EP8 · BS256 原生 GC 的 fused shared 与 unfused shared

测量提交 `9b163a36292e4edadb23ba04d549580ff0dc3049`。现场使用上一轮八臂的同一台 H20、同一模型与本地权重路径。目标是分别回答两个问题。M64 能否缩短 M32 为 shared 付出的时间？改后完整 FFN 能否追上 vLLM unfused，整步表现如何？

## 上轮的差距在哪里

以下为 1000 个测量步 × 40 层。每次取本臂完整 FFN 最慢 rank 的同卡时钟，再对 40,000 次求均值。表中均为 fused 减 unfused，单位为 µs/层，正值表示 fused 用时更长。

| 同策略配对 | FFN 入口至 GC 图入口 | GC 图入口至首个 routed 计算 CTA | 首个至最后一个 routed 计算 CTA（含 push） | 最后 CTA 至 combine 结束 | combine 结束至 FFN 返回 | 完整 FFN |
|---|---:|---:|---:|---:|---:|---:|
| 原生，2 对 3 | −42.06 | +45.42 | +44.49 | −5.98 | +1.44 | +43.31 |
| EPLB+RR，4 对 6 | −38.39 | +48.42 | +37.82 | +8.37 | +4.44 | +60.67 |
| EPLB+ECO，7 对 8 | −36.83 | +40.66 | +36.36 | +3.92 | +4.29 | +48.40 |

GC 的 split 图要求 shared L1 完成后 routed 才能执行，shared L2 可以与 routed 在同一计算分区重叠。M32 在 256 行上对同一批 shared 权重分八个行块发起读取。M64 分四个行块，L1 的输出任务从 144 个变为 72 个，L2 从 160 个变为 80 个。L2 并发上限从 18 个 CTA 调到 9 个，使两种分块同时在途的行数上限相同。上表定位了损失发生的区间，不能把整段 routed 增量都归给 shared L2。

## 现场执行

1. 遵守站点预约与分配，仅在同一组八张 H20 均空闲且获分配时运行，遇到其他作业等待。沿用原环境，不安装依赖。下列命令从现有仓库执行，新建独立 checkout 和输出目录。
2. 只执行原生 GC 的三条计时臂，顺序为旧 fused M32、新 fused M64、vLLM unfused。优先复用上轮现场保留的发布者、路由、反馈 token、prefill KV 和测量边界 KV。启动器为当前消费者单独保存布局。若历史发布材料已被清理，新目录中先用 M32 自动发布一次，计时比较仍只取三个消费者。三臂均为 TP8/EP8、BS256、10 GiB KV/rank、1000 步统计加 1000 步测量、40 层。
3. 执行以下命令。重试同一个已开始测量的输出目录时，在 `--shared-tile-pair` 后加 `--resume`，并使用相同的测量提交。保留日志，任务作用域限于这三臂。执行者仅可修复编译问题，保留补丁并在同一修订上完成全部计时臂。语义、配额、输入、图依赖、数据校验或测量异常请原样回执。

```bash
set -euo pipefail
git fetch origin main
job=$(mktemp -d /mnt/fuse/h20-shared-tiles-XXXXXXXX)
git worktree add --detach "$job/code" 9b163a36292e4edadb23ba04d549580ff0dc3049
cd "$job/code"
prior=/mnt/fuse/h20-ir-eight-4Thp8jCg/eight/kv-10
unset SM90_SHARED_REFERENCE SM90_WA1_PROFILE_DIR
if [[ -f "$prior/arm0/complete" && -s "$prior/replay/tokens.json" && -d "$prior/kv" ]]; then
  ready=1
  for rank in {0..7}; do
    for file in "replay/stats-rank$rank.pt" "replay/measure-rank$rank.pt" \
                "kv/rank$rank.bin" "kv/prefill/rank$rank.bin"; do
      [[ -s "$prior/$file" ]] || ready=0
    done
  done
  if [[ $ready == 1 ]]; then export SM90_SHARED_REFERENCE="$prior"; fi
fi
bash scripts/sm90/run_eight.sh --shared-tile-pair "$job/result" 2>&1 | tee "$job/run.log"
```

## 如何判断

- 核对三臂相同的 prompt、反馈 token、两阶段路由及权重、物理槽 token 计数、专家权重与布局、KV 回读哈希。现场日志的 `[shared-decode]` 应显示 routed M64，shared 分别为 M32、M64、外部 vLLM。每个消费者应有 40,000 次调用/rank。
- 旧 fused 对新 fused 要求 256,000 个原生 top‑1 零差异，warm 输出哈希相同。最后一个测量步、最后一层的 shared L1 激活、激活尺度与 shared L2 BF16 输出在八个 rank 上分别按字节核对。跨 vLLM 的 top‑1 和 warm 差异另列。
- `shared-timings.csv` 和 `shared-pairs.csv` 分别给出阶段时间、FFN、整步与配对变化。分别回答 M64 是否快过 M32、是否快过 unfused，FFN 与整步各写实测值及带符号的变化百分比。
- 作为重叠证据，程序在计时结束后读取现有环形记录末六步，最多 1,920 个 rank × 层样本。列出有效数、shared L2 退出戳晚于首个 routed CTA 的比例和两者的有符号中位差。该戳来自已退出的 L2 CTA，不能代替整段重叠时长。数值回归、性能持平或变慢都应如实回报。

回执放在 `docs/trace_handover_20261119/REPLY_SM90_BS256_SPLIT_SHARED_M64.md`，CSV、测量 SHA、三臂各自的 `run.json`、八份 rank NPZ、几何日志行、产物校验和放在 `docs/trace_executor_20261119/sm90_bs256_split_shared_m64_results/`。记下现场目录和发布材料来源。完成后释放预约、退出本任务的 worker，保留上轮公共模型、KV 和路由文件。
