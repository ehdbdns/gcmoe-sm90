# SM90 · 8×H20 · BS256 · MegaMoE 与 GC 原生及逐层 EPLB 六臂

在 DeepSeek‑V4.1‑Flash 整模上测量完整 decode step 和完整 MoE FFN。BS 固定 256，TP8/EP8，40 层，greedy，FULL_DECODE_ONLY CUDA Graph。每个正式臂使用同一批请求，先跑 1000 个满批纯 decode 步统计需求，按策略重排后连续测量 1000 步。关闭 ECO 和 SM-reuse。

| 路由策略 | Mega，vLLM shared | GC，fused split shared | GC，vLLM shared |
|---|---|---|---|
| 原生无副本，48 槽/rank | 1 | 2 | 3 |
| 逐层 DeepInfer EPLB + RR，56 槽/rank | 5 | 4 | 6 |

GC 使用本任务提交的 SM90 kernel，stream 和 M64。fused 臂由 GC 的角色 4、8 计算 shared L1、L2，unfused 臂由 vLLM 的真实 shared MLP 计算。Mega 用 DeepGEMM SM90 FP8×FP8 routed kernel、vLLM shared；GC routed 使用 FP8×FP4。三条 EPLB 臂核对同一放置、RR 路由与物理槽计数。结论须注明后端的权重量化差别。

## 为什么重新跑

上次重试的 L1/L2 cubin 已通过，nvcc 13.2.78 配合 GCC 15.3.0。两条 Mega 臂在图捕获前申请 20 GiB/rank KV 时 OOM。原生臂一条原始 OOM 日志显示余 15.96 GiB，EPLB 臂显示余 3.00 GiB；旧日志中没有 Mega 的有效整步或 FFN 耗时。[上次回执](../trace_handover_20261119/REPLY_SM90_H20_RETRY_MEGA_1_5.md)

本轮六臂统一采用每 rank 10 GiB KV、512-token prefill chunk 与通信容量、单一 BS256 decode 图。每档使用原数据文件选出的 256 条请求，各取真实前 256 个 token。前后两个千步窗口和 routed 专家数保持完整。缩短提示与缓存配置改变了绝对时延的比较条件，应在本轮内横向比较。Mega 继续用官方的对称工作区；GC 适配器只保留它在 Mega 路径上实际使用的计数、映射和审计视图，跳过四个 GC 专属 L1/L2 激活区。每 rank 实际可用显存以本机运行记录为准。

## 执行一次

1. 取得站点预约，只用同组八张空闲 H20；任何一张被占用就等待，保留调度器的设备掩码。准备本地已验证的模型副本、原数据集、已归档的私有 DeepGEMM wheel 与工具链，以及至少 250 GB 可用的结果盘。默认模型路径为 `/mnt/fuse/models/DeepSeek-V4.1-Flash`，可用 `SM90_LOCAL_MODEL` 指向节点上同版本的本地副本。每臂启动前预热本地权重页缓存。
2. 从原仓库运行。测量 checkout 与结果目录彼此独立。

~~~bash
set -eo pipefail
git fetch origin main
book=docs/trace_executor_20261119/SM90_H20_BS256_MEGA_GC_SIX.md
rev=$(git log -1 --format=%H origin/main -- "$book")
job=$(mktemp -d /mnt/fuse/sm90-bs256-six-XXXXXXXX)
echo "$job"
git worktree add --detach "$job/code" "$rev"
cd "$job/code"
rc=0
bash scripts/sm90/run_eight.sh --six-bs256 "$job/results" > "$job/launcher.log" 2>&1 || rc=$?
echo "$rc" > "$job/exit-code.txt"
exit "$rc"
~~~

3. 启动器先做 wheel/输入哈希、真实专家三投影转换和两阶段 JIT 检查。顺序为 **0 → 5 → 1 → 2 → 3 → 4 → 6**。0 只生成统一的参考 token、每步每层路由、prefill 及测量边界 KV；不进入主表。随后六个独立模型进程都是消费者。先跑显存压力最大的 Mega EPLB。记录各臂 `[six-memory]`、实际 KV token capacity、prefill 步数、测量窗口各 rank 的显存及首错。发布者失败则停止；独立消费者运行失败时脚本保留现场并继续其余臂。
4. 因断线或预约中断，保留原目录及参考状态。在原干净 checkout、相同 SHA 和同组空闲卡上运行 `bash scripts/sm90/run_eight.sh --six-bs256 --resume "$job/results"`；已完成臂按校验和跳过。若只是 CPU 报告失败，直接用实验环境的 Python 运行 `scripts/sm90/report_eight.py "$job/results"`。执行者只可修复不改变语义的编译环境问题，附 diff、工具版本与复验；运行期 OOM、数据不符或环境不满足时归档首错。不得单臂改 KV、BS、图、精度、放置、步数、前缀或换用旧 GC 结果。

## 数据与回执

- 主表六行列出臂号、后端、shared 执行方、放置、实测 block M、完整 step 的 ms/步、完整 FFN 的 ms/步和 µs/层，以及 token IR、总 M 块、峰值 rank M 块、活跃槽与 M 块 IR。整步为后 1000 次真实 engine.step 的同步墙钟均值。FFN 为模型层的 FFN 入口到返回（shared 已汇合）的设备时钟，每次层调用取最慢 rank，覆盖 40 层 × 1000 步。
- 每一对报告 256,000 个正式原生 top‑1 的差异数、预热原生输出哈希是否一致、strict 标志及两种耗时变化。百分比是 `100 × (B/A − 1)`，负值表示 B 更快。特别打印原生三对 1→2、1→3、2→3 和 EPLB 三对 5→4、5→6、4→6，统一按标题方向用原始耗时计算。加上 1→5、2→4、3→6 三个 EPLB 收益。原生输出不一致时仅报告固定反馈轨迹下的观测，不计为严格等输出的性能胜负。失败或缺失行写明原因，禁止填 0 替代。
- 汇总需要覆盖八 rank 的两次 KV 读回、完整路由需求、相同 prompt/反馈/顺序、同策略逐槽 counts、同后端逻辑权重、每 worker 1000 次 FULL replay 和 40 层 shared owner；fused 附 `[six] GC M=64 roles=(1, 4, 8, 2, 3)` 证据。附八卡硬件和空闲记录，以及每臂预热、权重加载、运行秒数和退出码。
- 结果推至 `docs/trace_executor_20261119/sm90_h20_bs256_mega_gc_six_results/`，回执推至 `docs/trace_handover_20261119/REPLY_SM90_H20_BS256_MEGA_GC_SIX.md`。交付 summary.csv、pairs.csv、verification.json、测量 SHA、JIT 与六臂日志摘要、原始八 rank NPZ 与 run.json 的归档地址及校验和。保留失败日志、原始退出码、实际路径。GPU 阶段完成立即释放预约并确认本任务进程退出，CPU 汇总可在退卡后进行。核验远端交付后清理自己的工作树、可重建缓存与大块临时 KV；共享 wheel、模型和其他用户文件保持原状。
