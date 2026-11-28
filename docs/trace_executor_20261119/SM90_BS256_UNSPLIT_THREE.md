# SM90 · H20 BS256 · 补测上游 H20 的非 split fused shared

测量提交 `d9ad3fdcf5c8d346b946f4aa88e3c25aed80db5f`。本次排期只新增三臂。上轮八臂的测量 SHA 为 `47e9b06328336e10584ef942a710aa332538327b`。若上一个 M32/M64 专项已经启动，保留已有数据，在当前臂安全落盘后切换任务。尚未启动的旧专项臂本轮跳过。

| 新臂 | 放置与分配 | 历史 fused split 臂，FFN µs/层 | 历史 vLLM unfused 臂，FFN µs/层 |
|---|---|---:|---:|
| 2 | GC 原生，48 个物理槽/rank | 2 · 1158.96 | 3 · 1115.65 |
| 4 | GC + 逐层 EPLB + RR，52 槽/rank | 4 · 1074.04 | 6 · 1013.37 |
| 7 | GC + 同一 EPLB 放置 + ECO，52 槽/rank | 7 · 1016.04 | 8 · 967.65 |

每行新臂均由 GC 计算 shared expert。上游 Bruce-Lee-LY/gc-moe 的 `af5a7a7` 在 78 SM H20 上选择 legacy。按本轮形状，shared L1 和 L2 在同一个 M64 kernel 执行，72 CTA、256 线程/CTA、199680 字节动态共享内存/CTA。整个 shared kernel 完成后才放行 routed。移植范围是这套 shared 调度。旧 split 的 shared 为 M32，L1 和 L2 各有一个 kernel。两种实现都使用八臂工程自己的 routed、EPLB 和 ECO。[上游策略](https://github.com/Bruce-Lee-LY/gc-moe/blob/af5a7a7/gc_moe/jit_kernels/heuristics/sm90_gc_moe.py#L8-L17) · [图依赖](https://github.com/Bruce-Lee-LY/gc-moe/blob/af5a7a7/gc_moe/jit_kernels/impls/sm90_fp8_fp4_gc_moe.py#L89-L130)

## 配置与执行

1. 只在站点授权并确认上轮同一组八张 H20 全部空闲时占用，保留分配的设备掩码。有人使用或已有预约时让出整组，在 CPU 侧等待。每臂启动前复查，或与旧作业在臂边界交接。
2. 复用原发布者的 prompts、反馈 token、统计和测量路由、prefill 和测量边界 KV、逐层 placement。默认现场为 `/mnt/fuse/h20-ir-eight-4Thp8jCg/eight/kv-10`。若原件搬迁，设置 `SM90_SHARED_REFERENCE` 指向同一份归档。缺文件先查找该归档，仍缺失则回报具体路径；本轮不启动第四个模型生成参考。新臂的日志、结果和布局写入新的任务目录。
3. 沿用 DeepSeek-V4.1-Flash，TP8/EP8，BS256，32 条请求/rank，完整 decode 图，40 层，10 GiB KV/rank，1000 步统计、1000 步测量。routed 为 M64 并按 M 块发号。放置臂每卡 4 个冗余槽，ECO 为 `T + 26M + 2A`、slack 92，plan 从 dispatch 取一个 SM，关闭 SM-reuse。启动器只选择 `2 4 7` 三个消费者。
4. 从已有仓库创建独立工作树，记录打印的 job 路径。所有文件输出到工作树之外。

```bash
set -euo pipefail
git fetch origin main
job=$(mktemp -d /mnt/fuse/h20-unsplit-three-XXXXXXXX)
echo "$job"
git worktree add --detach "$job/code" d9ad3fdcf5c8d346b946f4aa88e3c25aed80db5f
cd "$job/code"
bash scripts/sm90/run_eight.sh --unsplit-three "$job/result" 2>&1 | tee "$job/run.log"
```

同一测量提交中断时从同一 checkout 运行 `bash scripts/sm90/run_eight.sh --unsplit-three --resume "$job/result"`；记录续跑日志。已完成臂校验后跳过。CPU 报告可以在该 checkout 用现场 Python 单独执行 `scripts/sm90/report_unsplit_three.py "$job/result"`。执行者仅处理不改变语义的编译错误，保留首错和最小 diff；三条计时臂使用同一最终测量修订。配置、输入或正确性发生偏差时保留现场和原始诊断。

## 结果与回执

- 对三条新臂核对 `[shared-unsplit]` 的 `shared_m=64 grid=72 threads=256 smem=199680 phases=(0, 1) routed_wait=(4,)`。原生和 RR 的角色序列为 `(1, 4, 2, 3)`，ECO 为 `(5, 7, 1, 4, 2, 3)`。
- `unsplit-summary.csv` 共九行，含三条新臂和六条历史 GC 对照。报告完整 step ms、完整 FFN ms/步与 µs/层、FFN 的五段耗时、token IR、总/峰值 M 块、M 块 IR、活跃物理槽；ECO 还报 routed MAE 和拍卖成交次数。
- `unsplit-pairs.csv` 共六行。同策略分别对照旧 split 和旧 unfused。耗时变化以历史均值为分母，正数表示新臂更慢。程序校验设备与环境、同一输入、两次 KV、路由、物理布局和各槽计数，以及 ECO 拍卖记录。逐对列出 warm 输出哈希比较和 256,000 个原生 top‑1 的差异数；新旧 fused 零差异才标为严格可比，跨 shared 路径的差异照实保留。
- 将结果放在 `docs/trace_executor_20261119/sm90_bs256_unsplit_three_results/`。提交两张 CSV、测量 SHA、硬件和输入清单、三臂各自的 run.json、8 份 rank NPZ、installed-layout.json、runtime.log、execution.sha、artifacts.sha256、complete 及日志中的几何证据。大型 KV 和路由库继续保存在原位置。
- 回执放在 `docs/trace_handover_20261119/REPLY_SM90_BS256_UNSPLIT_THREE.md`。逐个策略写明 unsplit 对 split 和 unfused 的完整 FFN 与整步变化，列出 GPU UUID、数据来源、测量 SHA、运行目录和校验结果。结束后释放八卡预约，检查本任务 worker 已退出，清理本任务的临时缓存，保存测量证据。
