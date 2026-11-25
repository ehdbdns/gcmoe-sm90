# SM90 H20 EP8 · 按 token IR 选择 EPLB 冗余并续跑 BS256 Mega/GC 八臂

## 冗余扫描

六档全局 BS 为 32、64、128、256、512、1024，均摊请求数分别为 4、8、16、32、64、128/rank。DeepSeek-V4.1-Flash 有 384 个逻辑专家。先定每卡冗余 k，容量为 48+k 个物理槽；用前 1000 步每层每个专家的真实需求，运行生产用 DeepInfer 贪心复制与 rank 放置。每个 k 都按自己的容量重新放置。再用后 1000 步的固定路由跑 RR。k=0 也重排专家，原生对照使用初始 identity 放置。

对每个 step×layer，计算八卡接收 token 行数的最大值除以八卡平均值，再平均 40,000 次。该 token IR 是唯一选参指标，越接近 1 越好。另报相对原生的 IR 降幅。k 在 0 至 32 内按整数扫描。报告所有点、区间最低值和是否落在边界。BS256 的八臂实验还须在用户要求的少于 8 槽且能拍卖的区间 1 至 7 内按同一指标选最优值。并列取较小 k。所有放置臂共用这个 k。

### 本地已有结果

使用归档 identity 臂重建评价窗口 40,000 次调用的逻辑需求，按每个专家的副本数均摊后估算各 rank 的 token 数。原生列及 k=8 实测列取历史物理计数。其余列为本次放置估计。完整 198 行在同目录 h20_eplb_token_ir_screen.csv。

| BS | 原生实测 | EPLB k=0 | k=4 | k=7 | k=8 实测 | k=32 |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 1.4770 | 1.2739 | 1.2436 | 1.2309 | 1.2265 | 1.1738 |
| 64 | 1.4345 | 1.1935 | 1.1717 | 1.1621 | 1.1598 | 1.1216 |
| 128 | 1.4081 | 1.1416 | 1.1248 | 1.1181 | 1.1155 | 1.0883 |
| 256 | 1.4073 | 1.1020 | 1.0889 | 1.0850 | 1.0832 | 1.0640 |
| 512 | 1.3943 | 1.0745 | 1.0652 | 1.0615 | 1.0607 | 1.0467 |
| 1024 | 1.3957 | 1.0558 | 1.0464 | 1.0444 | 1.0435 | 1.0330 |

六档区间最低估计均在 k=32，下次八臂的受限区间最低估计均在 k=7。CPU 与 CUDA 的浮点归约会改变部分平分时的贪心放置。若副本数不整除 8，实际 RR 分配还取决于源 token 的序号。八臂启动器只读取下述有序路由实算结果。

## 执行与交接

只使用站点授权、同组八张空闲 H20。任一卡有其他作业时等待；保留站点设备掩码，与其他 SM90 任务共用现有锁。先让已运行的任务正常完成。执行者仅处理保留语义的编译错误并附首错与 diff。正式运行、选参、统计与默认配置由仓库代码负责。

~~~bash
set -eo pipefail
git fetch origin main
book=docs/trace_executor_20261119/SM90_H20_EPLB_EXTRA_SWEEP.md
rev=$(git log -1 --format=%H origin/main -- "$book")
job=$(mktemp -d /mnt/fuse/h20-ir-eight-XXXXXXXX)
echo "$job"
git worktree add --detach "$job/code" "$rev"
cd "$job/code"
bash scripts/sm90/run_h20_token_ir.sh "$job/scan" > "$job/scan.log" 2>&1
bash scripts/sm90/run_tuned_eight.sh "$job/scan/bs256/token-ir.csv" "$job/eight" > "$job/eight.log" 2>&1
~~~

扫描先处理 BS256。若新六档四臂 job 路径已知，可在启动前 export SM90_SIX_BS_JOB 指向其父目录。脚本按档查找完整有序 route bank，找不到才以该档主表设置生成原生参考和 8 槽 RR 校准臂。GPU 仅用来训练所有预算的实际放置，随后释放协作锁。CPU 校验源 ID 摘要，在每个专家的有序副本映射上按 (源 rank + 23333 × 本地 token 序号) 对副本数取模，逐次计算八个目的 rank 的 token 数。对 k=8 与历史逐槽布局一致的层还校验真实物理计数。初筛与实算均采用 1000+1000。

## BS256 八臂

| 臂 | routed 实现 | shared | 放置与分配 |
|---:|---|---|---|
| 1 | Mega，SM90 FP8×FP8 | vLLM | 原生 |
| 2 | GC，SM90 FP8×FP4 | fused split | 原生 |
| 3 | GC | vLLM unfused | 原生 |
| 4 | GC | fused split | 当选 k，EPLB + RR |
| 5 | Mega | vLLM unfused | 同一 k、同一 EPLB + RR |
| 6 | GC | vLLM unfused | 同一 k、同一 EPLB + RR |
| 7 | GC | fused split | 同一 k、同一 EPLB rank 成员 + ECO |
| 8 | GC | vLLM unfused | 同一 k、同一 EPLB rank 成员 + ECO |

arm0 是统一路由、反馈和 KV 的发布者。测量使用六臂协议的 256 条真实 256-token 前缀、TP8/EP8、512-token prefill chunk、满批 256、40 层、完整 decode graph。GC 使用 M64、按块发号。ECO 采用 T+26M+2A，slack 92，plan 的一个 SM 来自 dispatch。八臂记录完整 step ms 和完整模型 FFN 的 µs/层，后者覆盖 vLLM 外置 shared 到该层最终求和的工作。八个臂均记录 token IR、总/峰值 M 块、M 块 IR 和活跃槽。ECO 两臂还报成交次数与 routed MAE。记录后端原生 M 块尺寸、权重精度差异及所有配对的原生输出差异。

最新六臂回执 243072d 中，arm0 与 Mega 原生 arm1 已完整落盘，arm1 为 93.162 ms/步。arm2 跑完 decode 后在落盘前被终止，arm3 未启动。更早的八臂里两个原生 GC 有过数据，使用了另一组输入、KV 和 fused 实现。当前任务先用 10 GiB KV/rank、1000+1000 重新生成发布者，核对历史 SHA、原始校验和、八卡 UUID、运行环境、输入、两次 KV、路由、计数和真实输出。通过时直接导入历史 arm1，保留原测量 SHA；任一核对失败就实测。arm2、arm3 用现行 split/unfused 口径补齐。可靠的同任务完成臂在重启时校验后跳过。

首档确定 OOM 或引擎 KV 不足时，全组改为 7 GiB、1000+1000；仍失败才使用 5 GiB、500+500。各档单独比较，同一档五条放置臂保持同一个 k。减少冗余槽能节省权重显存，实际能否容纳以设备回执为准。六档选参使用主表路由，八臂使用六臂协议的短前缀，另报该批真实路由测得的 IR。中断后复用原 job、工作树和命令。若收到停止要求，等待当前臂 complete 与 artifacts.sha256 落盘。

交付到 docs/trace_executor_20261119/sm90_h20_token_ir_eight_results/，回执为 docs/trace_handover_20261119/REPLY_SM90_H20_TOKEN_IR_EIGHT.md。附六份 token-ir.csv、训练来源、每档校验和、BS256 selection.csv、8 行 summary.csv、28 对 pairs.csv、verification.json、各臂原始 rank NPZ、运行 SHA、复用及内存失败证据。真实输出或预热不一致的配对标为不可严格比较。完成后释放预约、核对 worker 已退出，清理本任务可重建缓存，保留原始归档。
