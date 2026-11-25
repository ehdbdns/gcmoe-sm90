# SM90 · shared L1 先启动的 fine cost 补测

在同组八张 H20 上，对 BS32、64、128、256、512、1024 各补一条正式 ECO fine cost。每 rank 均摊请求数依次为 4、8、16、32、64、128。主对照为 `d6b5be9` 任务书下同 BS 的 shared-first 线性 wa=1，另与测量提交 `2f70314` 的 shared-first 线性 wa=2 对照。既有线性臂取各自的正式 GPU 记录。新增正式测量合计六次。原任务的 BS256 replay 2 Perfetto 和正在执行的线性实验继续按原任务交付。

## 价格与计时

| 设置 | 线性 wa=1 | 线性 wa=2 | 本次 fine |
|---|---:|---:|---:|
| 设备 plan 模式 | 0 | 0 | 2（H20 valid cost） |
| 满 M 块进入空槽的 slack | 91 | 92 | 11469 |
| rank 价格 | T + 26M + A | T + 26M + 2A | 各物理槽内每个 M 块按有效行数计价，再求和 |

fine 使用已有 H20 校准表。每槽有 q 行，价格为 `(q // 64) × 11469 + C(q % 64)`。空尾块价格为零。非空尾块按有效行数落入上界为 8、16、32、40、48、64 的六档，价格为该档截距加 `17 × 有效行数`；L2 末尾的 push 已计入。fine 在启动环境中沿用主对照的 wb=26、wa=1，mode 2 使用 valid 表和满块单位，自身公式中没有额外的激活槽项。

沿用 DeepSeek-V4.1-Flash、TP8/EP8、每 rank 8 个冗余槽、逐层 DeepInfer 放置、FULL_DECODE_ONLY、M64、按 M 块就绪发号、pull、fused split shared 和原 KV 预算。shared L1 的 CTA 全部开始后放行 plan 与 dispatch 入场门，routed 等待 shared L1 完成。计划占用通信分区的 1 个 SM，dispatch 用其余 5 个 SM 的 20 个 CTA，compute 分区有 72 个 SM。确认每档测量图的 `[ep8]` 日志包含角色 `(4,5,7,1,8,2,3)` 与 `shared_launch=((4, 2, 0, 1),)`。

每档先统计 1000 个满批纯 decode 步，重排后连续测量 1000 步 × 40 层，正式窗口关闭 torch.profiler。主指标取同一步同一层八 rank 各自 graph 起点到 combine 结束的最大跨度，平均后报告 µs/层。routed MAE 为八 rank 的预测耗时占比与首个真实 routed CTA 开算至最后一个 CTA 完成（含 push）的实测占比之间的平均绝对误差，单位百分点。

## 执行

1. 等站点授权的同组八张 H20 全部空闲，保留调度器设备掩码和原协作锁。正在运行的实验先完成。使用独立 job 和 detached worktree。执行者仅可修复保持语义的编译问题，并回报原始报错、diff 和实际测量 SHA。公平性、运行时错误或资源问题按 BS 保留现场并报告，其他可运行档继续。
2. 下列命令优先使用已发布的六档 wa=1 归档。若线性回执还在节点本地，在执行 shell 中将 `SM90_WA1_RESULTS` 指向原线性任务打印的 job 根目录（含 `delivery-32` 至 `delivery-1024`）。命令也会寻找本机已完成的六档目录。两个来源均未就绪时，fine 可以先运行并与 wa=2 比较。

~~~bash
set -eo pipefail
git fetch origin main
book=docs/trace_executor_20261119/SM90_SHARED_FIRST_FINE_COST.md
rev=$(git log -1 --format=%H origin/main -- "$book")
test -n "$rev"
job=$(mktemp -d /mnt/fuse/sm90-shared-first-fine-XXXXXXXX)
echo "$job"
git worktree add --detach "$job/code" "$rev"
rel=docs/trace_executor_20261119/sm90_shared_first_wa1_results
wa1=${SM90_WA1_RESULTS:-"$PWD/$rel"}
if [[ -z ${SM90_WA1_RESULTS:-} && ! -s $wa1/comparison.csv ]]; then
  for previous in /mnt/fuse/sm90-shared-first-wa1-*; do
    if [[ -s $previous/comparison.csv && -f $previous/delivery-1024/SHA256SUMS ]]; then
      wa1=$previous
      break
    fi
  done
fi
if [[ -z ${SM90_WA1_RESULTS:-} && ! -s $wa1/comparison.csv ]] && git cat-file -e "origin/main:$rel/comparison.csv" 2>/dev/null; then
  mkdir "$job/inputs"
  git archive origin/main -- "$rel" | tar -x -C "$job/inputs"
  wa1="$job/inputs/$rel"
fi
wa1=$(realpath -m "$wa1")
cd "$job/code"
rc=0
bash scripts/sm90/run_full_model_four.sh --shared-first-fine "$wa1" "$job" > "$job/launcher.log" 2>&1 || rc=$?
echo "$rc" > "$job/exit-code.txt"
exit "$rc"
~~~

3. 脚本优先复用已验收线性运行保留的 route bank、反馈 token 和两套边界 KV，并对参考副本做只读校验。如果本档 KV 已清理，脚本调用一次 vanilla publisher 重建参考。这次运行单列为准备开销。ECO 会基于统计重新计算并核对归档 EPLB 放置。
4. 每个 rank 检查两窗口的 prompt、反馈、逻辑 top-k ID/权重、需求、KV 读回、权重、层/步顺序、放置和原生输出，验证各 40,000 次层调用、正式 1000 次完整图回放。每档每对要求预热 native 摘要相同、正式 top-1 差异为零、同组 GPU UUID/驱动/运行时一致。每次调用从物理槽计数独立计算 fine 分数，与设备实际导出的分数逐项比对，八个 rank 的 prepared 记录均须为 mode=2、slack=11469。价格改变所导致的物理配额和成交次数是实际测量项。
5. 若日志为 `Fine measurements saved. Waiting for wa=1`，记录各档 `bs*.exit` 和 SHA256SUMS，先释放 GPU。wa=1 交付后，只执行下面的 CPU 汇总。沿用上面保存的 job 路径，`wa1` 取节点真实 job 或最新 GitHub 归档的本地目录。此命令也可独立复验已完成的整批比较。

~~~bash
cd "$job/code"
source scripts/env.sh
"$GC_ENV/bin/python" scripts/sm90/shared_first_fine_report.py --finish "$job" "$wa1"
~~~

## 回执

- 先按 BS 给出三组完整 MoE µs/层，重点列出 `fine − wa1` 的 µs 与 `100 × (fine/wa1 − 1)`，另列 fine 相对 wa=2 的两项差值。负值表示 fine 更快。每对列出 40 层中 fine 更快的层数和 20 个对应的连续 50 步段中 fine 更快的段数。各权重采用独立运行，微小差异结合这些配对结果说明。
- 主表同时保留相对同档 vanilla/RR 的变化、token IR、总与峰值 M 块、总与峰值活跃槽、M 块 IR、routed MAE、每次层调用成交及峰值命中计数。附逐层、逐 rank、拍卖总数/接受率、每调用原始计数及慢 rank 分段。未通过对照的档次说明具体不一致项。
- `delivery-$bs` 保存本次 fine 正式八 rank NPZ、run.json、硬件及空闲记录、测量 SHA、校验和与对 wa=2 的初步核对。`paired/` 是与两组线性数据完成核对的逐档表。最终根目录验收行数：主表 18、逐层 720、rank 份额 144、拍卖 18、公平性 12、耗时对比 12。记录正式模型启动次数、参考重建次数与各档退出码。若初次启动因等待线性结果而返回非零，保留原日志，补充 CPU 汇总退出码。
- 推送本任务的结果到 `docs/trace_executor_20261119/sm90_shared_first_fine_results/`，回执放 `docs/trace_handover_20261119/REPLY_SM90_SHARED_FIRST_FINE_COST.md`。GPU 任务结束即释放预约并检查 worker 已退出。交付校验后清理本 job 的临时缓存，历史归档、路由与共享模型保留。
