# SM90 H20 EP8 · BS256 Mega/GC 八臂，先用 4 个冗余槽

2026-11-26 重发给 SM90 执行者。请接续 BS256 Mega/GC 八臂，运行原任务的测量提交 `983f955`。独立的六档冗余扫描不作为启动条件。先检查节点上已有的 `/mnt/fuse/h20-ir-eight-*`；若同版本八臂作业正在运行，接续其回执。若中断，将 `SM90_MEGA_JOB` 设为原 job 根目录（含 `code/` 和 `eight/`），执行下列命令续跑。首次启动时让该变量留空。

执行优先级为 BS256 八臂。原生 Mega、GC fused、GC unfused 各用 48 个物理槽/rank。三个 EPLB + RR 臂和两个 GC + ECO 臂统一从每卡额外 4 槽开始，共 52 个物理槽/rank。任一放置臂因设备显存或 KV 容量不足失败时，五个放置臂统一切换到每卡额外 1 槽，共 49 个物理槽/rank。记录同一配置下的完整 step 时间和完整 FFN 时间。

## 冗余扫描

六档全局 BS 为 32、64、128、256、512、1024，均摊请求数分别为 4、8、16、32、64、128/rank。DeepSeek-V4.1-Flash 有 384 个逻辑专家。先定每卡冗余 k，容量为 48+k 个物理槽；用前 1000 步每层每个专家的真实需求，运行生产用 DeepInfer 贪心复制与 rank 放置。每个 k 都按自己的容量重新放置。再用后 1000 步的固定路由跑 RR。k=0 也重排专家，原生对照使用初始 identity 放置。

对每个 step×layer，计算八卡接收 token 行数的最大值除以八卡平均值，再平均 40,000 次。token IR 越接近 1，表示负载越均衡。另报相对原生的 IR 降幅。k 在 0 至 32 内按整数扫描。报告所有点、区间最低值和是否落在边界。该扫描用于研究冗余与均衡度的关系。八臂按已指定的 4 槽、1 槽顺序启动。

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

六档在整个扫描区间内的最低估计均在 k=32。在 1 至 7 槽内，最低估计均在 k=7。CPU 与 CUDA 的浮点归约会改变部分平分时的贪心放置。若副本数不整除 8，实际 RR 分配还取决于源 token 的序号。BS256 每卡 4 槽和 1 槽的估算 IR 分别为 1.0889 和 1.0964。

## 执行与交接

只使用站点授权、同组八张空闲 H20。任一卡有其他作业时等待。保留站点设备掩码，与其他 SM90 任务共用现有锁。若旧任务仍在运行，保留其日志和已完成结果，在安全的臂边界切换到本版本。执行者仅处理保留语义的编译错误并附首错与 diff。

~~~bash
set -eo pipefail
git fetch origin main
rev=983f955f9e2953b35c5578dcb26a26f87501c2c6
if [[ -n ${SM90_MEGA_JOB:-} ]]; then
  job=$(realpath -e "$SM90_MEGA_JOB")
  test -d "$job/eight"
  test "$(git -C "$job/code" rev-parse HEAD)" = "$rev"
else
  job=$(mktemp -d /mnt/fuse/h20-ir-eight-XXXXXXXX)
  git worktree add --detach "$job/code" "$rev"
fi
echo "$job"
cd "$job/code"
bash scripts/sm90/run_tuned_eight.sh "$job/eight" >> "$job/eight.log" 2>&1
~~~

六档有序路由扫描可独立执行。已在运行的扫描可保留各档结果。补做扫描时执行 `bash scripts/sm90/run_h20_token_ir.sh "$job/scan"`，先处理 BS256。若新六档四臂 job 路径已知，可在扫描前 export SM90_SIX_BS_JOB 指向其父目录。脚本按档查找完整有序 route bank，找不到才以该档主表设置生成原生参考和 8 槽 RR 校准臂。GPU 仅用来训练所有预算的实际放置，随后释放协作锁。CPU 校验源 ID 摘要，在每个专家的有序副本映射上按 (源 rank + 23333 × 本地 token 序号) 对副本数取模，逐次计算八个目的 rank 的 token 数。对 k=8 与历史逐槽布局一致的层还校验真实物理计数。初筛与实算均采用 1000+1000。

## BS256 八臂

| 臂 | routed 实现 | shared | 放置与分配 |
|---:|---|---|---|
| 1 | Mega，SM90 FP8×FP8 | vLLM | 原生 |
| 2 | GC，SM90 FP8×FP4 | fused split | 原生 |
| 3 | GC | vLLM unfused | 原生 |
| 4 | GC | fused split | 每卡加 4 槽，容量回退时加 1 槽，EPLB + RR |
| 5 | Mega | vLLM unfused | 同一 k、同一 EPLB + RR |
| 6 | GC | vLLM unfused | 同一 k、同一 EPLB + RR |
| 7 | GC | fused split | 同一 k、同一 EPLB rank 成员 + ECO |
| 8 | GC | vLLM unfused | 同一 k、同一 EPLB rank 成员 + ECO |

arm0 是统一路由、反馈和 KV 的发布者。测量使用六臂协议的 256 条真实 256-token 前缀、TP8/EP8、512-token prefill chunk、满批 256、40 层、完整 decode graph。GC 使用 M64、按块发号。ECO 采用 T+26M+2A，slack 92，plan 的一个 SM 来自 dispatch。八臂记录完整 step ms 和完整模型 FFN 的 µs/层，后者覆盖 vLLM 外置 shared 到该层最终求和的工作。八个臂均记录 token IR、总/峰值 M 块、M 块 IR 和活跃槽。ECO 两臂还报成交次数与 routed MAE。记录后端原生 M 块尺寸、权重精度差异及所有配对的原生输出差异。

最新六臂回执 243072d 中，arm0 与 Mega 原生 arm1 已完整落盘，arm1 为 93.162 ms/步。arm2 跑完 decode 后在落盘前被终止，arm3 未启动。更早的八臂里两个原生 GC 有过数据，使用了另一组输入、KV 和 fused 实现。当前任务先用 10 GiB KV/rank、1000+1000 重新生成发布者，核对历史 SHA、原始校验和、八卡 UUID、运行环境、输入、两次 KV、路由、计数和真实输出。通过时直接导入历史 arm1，保留原测量 SHA；任一核对失败就实测。arm2、arm3 用现行 split/unfused 口径补齐。可靠的同任务完成臂在重启时校验后跳过。

默认每卡 KV 为 10 GiB，统计 1000 步、测量 1000 步。4 槽放置臂首次确认显存或 KV 容量不足时，脚本把该档的 4 槽结果、首错、布局和放置文件归档到 `kv-*/failed-k4/`，保留同档发布者、已有完整原生臂、路由、反馈和 KV，然后重建 1 槽放置并完成五个放置臂。最终汇总中的五个放置臂须使用相同的实际冗余，八臂须使用相同的 KV 预算和窗口。

若 1 槽放置臂仍出现容量失败，沿用 1 槽，全组再降为每卡 7 GiB KV、1000+1000，必要时 5 GiB、500+500。发布者或原生臂出现容量失败时，直接调整全组 KV 预算，因为这几臂使用原生 48 槽。切换 KV 档位后重建该档的发布者并重新比较。编译、路由、输出校验或其他错误按原配置报告。中断后复用同一工作树、同一 job 和八臂启动命令。停止时等待当前臂 complete 与 artifacts.sha256 落盘。

交付到 docs/trace_executor_20261119/sm90_h20_token_ir_eight_results/，回执为 docs/trace_handover_20261119/REPLY_SM90_H20_TOKEN_IR_EIGHT.md。附最终档的 tuned-extra、tuned-kv、six-steps、8 行 summary.csv、28 对 pairs.csv、verification.json、各臂原始 rank NPZ、运行 SHA 和复用记录。若触发回退，附触发臂、原始错误与各次尝试的退出码。已有的扫描产物可单独附上。真实输出或预热不一致的配对标为不可严格比较。完成后释放预约、核对 worker 已退出，清理本任务可重建缓存，保留原始归档。
