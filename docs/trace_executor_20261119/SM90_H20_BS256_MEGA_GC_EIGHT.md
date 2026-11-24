# SM90 执行任务 · H20 八卡 · BS256 的八臂 MoE 对比

收到后从 CPU 核验连续执行到结果回传。上一份回执已确认 NAS、两份请求文件和官方 wheel 的位置。复用已构建的 wheel。本轮只用同机八张 H20。设计端提供模型接入、计时、启动器和验收程序。执行者只修复保持语义的编译错误，保留首错、diff 和复测记录。运行时数值、显存、框架兼容或硬件问题按下面的失败流程记录。

| 臂 | Routed 后端 | Shared | 放置及分流 |
|---:|---|---|---|
| 1 | 官方 Mega SM90，FP8×FP8 | 核外 | 原生 |
| 2 | GC SM90，FP8×FP4 | 核内 | 原生，发布参考 |
| 3 | GC SM90，FP8×FP4 | 核外 | 原生 |
| 4 | GC SM90，FP8×FP4 | 核内 | 逐层 EPLB + RR |
| 5 | 官方 Mega SM90，FP8×FP8 | 核外 | 逐层 EPLB + RR |
| 6 | GC SM90，FP8×FP4 | 核外 | 逐层 EPLB + RR |
| 7 | GC SM90，FP8×FP4 | 核内 | 逐层 EPLB + ECO fine |
| 8 | GC SM90，FP8×FP4 | 核外 | 逐层 EPLB + ECO fine |

## 启动

1. 在 SM90 执行仓库取得本任务书的提交。建立独立干净 checkout，在私有大盘分配新目录。预留至少 405 GB 可用空间。任务使用本轮第 2 臂新发布的 prefill 和重排边界 KV、token 和路由；历史 retained_bs256 作为旧实验档案保留。模型为 DeepSeek-V4.1-Flash，官方 DeepGEMM 固定在 nv_dev 的 7292e7bd0ef3b497ca948a7239d55f3f01539048，私有 wheel 摘要以 7c848494 开头。
2. 遵循现场预约或调度系统取得同一组八卡的独占使用权。共享预约优先于本任务的文件锁。等待期间只使用 CPU；核验八卡全部空闲，任何一张已被占用即等待，保留他人的进程。下列是唯一的正式启动命令。运行前确认 /mnt/fuse 指向这台机器可写的私有大盘。

```bash
set -euo pipefail
repo=$(git rev-parse --show-toplevel)
git -C "$repo" fetch origin main
book=docs/trace_executor_20261119/SM90_H20_BS256_MEGA_GC_EIGHT.md
rev=$(git -C "$repo" log -1 --format=%H origin/main -- "$book")
area=$(mktemp -d /mnt/fuse/sm90-h20-eight-XXXXXXXX)
git -C "$repo" worktree add --detach "$area/code" "$rev"
cd "$area/code"
bash scripts/sm90/run_eight.sh "$area/results" 2>&1 | tee "$area/launch.log"
```

3. 启动器先核对模型配置、索引、请求文件和 wheel 摘要，CPU 转换真实第 0 层专家的三个投影，检查 HCA 的 ACTIVE 状态。每臂之前核验八卡和占用，按 2、3、4、6、7、8、1、5 串行，每臂单独的模型进程。观察日志与显存时只使用现场允许的只读工具，结束即释放整组预约。汇总脚本可在释放 GPU 之后单独重跑。

## 工作量和读数

- 每臂 TP8/EP8，BS256，40 层，真实 prefill，1000 步收集放置统计，重排边界之后测量 1000 步。使用固定请求顺序、反馈 token、两阶段逐步逐层逻辑路由与门权重、恢复后 KV 的读回摘要。各 rank 的实际接收计数用来验证物理工作量。分别报告原生 greedy token 偏差和固定反馈一致性。
- EPLB 使用同一个 DeepInfer 逐层算法及统计窗口。原生有 384 个物理槽，EPLB 有 448 个。RR 逐项核对物理计数；两条 ECO 按 rank 核对最终分配和成交计数。ECO 使用 fine cost、slack 11469、通信侧单 plan SM、stream 开启、SM 复用关闭。
- 核外臂保留模型的 shared MLP 和最终加回；核内臂用 GC shared 角色。最终 FFN 在本层内完成。官方 Mega 入口为 fp8_mega_moe，采用同一 checkpoint 转换的 E4M3 权重与 FP32 block scale。Mega 用官方实际接收计数，不具备 GC 内核的 used-ordinal 观测点。归档真实启动配置、转换误差和共享软件包来源。
- 主指标是完整 FFN 设备时间。每个 rank 从本层 FFN 入口计到输出完成，包含门控、shared、routed、combine 以及适用的最终加法。每步每层取八 rank 的**本地跨度最大值**，40 层相加，再对 1000 步平均，单位 ms/步。时间戳取同一 GPU 的 globaltimer。调用中的路由回放和设备计数开销包含在读数内。此边界比旧主表的 GC 内核跨度更宽。
- 汇总列包括 token IR、总和峰值 M 块、活跃物理槽、M 块 IR、GC fine 预测对完整 FFN 和 routed 的平均和最大误差（百分点）、ECO 接受次数、原生 top-1 差异。无对应观测的项留空。主表只比较 MoE。相对耗时变化等于 100 ×（候选时间 / 参考时间 − 1），负值表示更快。
- 比较 2→3、4→6、7→8 的 shared 融合效应，4→7、6→8 的 ECO 效应，以及 1→3、5→6 的后端和精度共同效应。原生输出有差异时，写清固定工作量下的结论与首个差异位置。

## 失败、回传和清理

第 2 臂失败会阻断依赖它的参考回放。其他单臂失败时，启动器保留日志并继续其余有输入的臂。若出现 OOM，附该臂失败阶段（权重转换、槽展开、KV、图捕获或 decode）、rank、可用/请求分配字节、对应运行日志；保持八臂共同的 KV 预算、模型、精度与 batch。每卡 20 GiB KV 外，Mega 的 40 层 routed FP8 payload 在每 rank 48/56 槽下约占 63.3/73.8 GiB，另需 dense 权重、对称缓冲区与 CUDA Graph。显存不足的臂标记 OOM，不填入性能值。单次八臂完成或独立臂均已处理后统一回执。

在 results 下检查 summary.csv、verification.json、arm*/exit、runner.log、run.json、rank0 至 rank7 的 npz、布局和 GPU 型号。计时须有 1000×40 对有效时间戳。提供 FULL_DECODE_ONLY 图捕获和正式 decode 使用图的运行时证据。检查 shared 路径和最终输出；证据缺失时在回执写明。对首次语义失败保留完整日志，不用 eager、缩短窗口或缩小模型替代正式结果。若只需重新离线汇总，运行：

```bash
source scripts/env.sh
"$GC_ENV/bin/python" scripts/sm90/report_eight.py "$area/results"
```

大型 KV、回放和逐 rank 原始包留在私有持久盘，记录绝对路径、文件大小和 SHA256。把 summary.csv、verification.json、输入与转换摘要、各臂状态和必要的首错日志推回 docs/trace_executor_20261119/sm90_h20_bs256_mega_gc_eight_results/ 新的 formal 子目录。把正式回执追加到 docs/trace_handover_20261119/REPLY_SM90_H20_BS256_MEGA_GC_EIGHT.md，注明任务 SHA、八行结果或失败原因，以及退卡状态。回执提交中只包含本轮证据，推送 main 后核对远端 SHA。结束时释放自有进程、工作树和预约；保留复核所需原始文件、历史参考、私有 wheel 和其他用户资源。

## 6a44e35 回执后的重跑

从上面的启动命令重新取任务书 revision，新建独立输出目录，执行同一套八臂。第 2 臂上次在图捕获时终止，没有可续用的本轮参考。执行者已修复的 workspace 分配保留。八臂的注意力都使用 env3 中 vLLM 自带的 DeepGEMM，只有 1、5 臂的 routed 专家加载指定的官方 wheel。不要把其 site 加到 PYTHONPATH，也不要改共享环境中的 sparse indexer。启动器先在独立 CPU 进程检查实际模块来源和两个扩展的共存情况。私有扩展若加载失败，保留 mega-import.log，继续六条 GC 臂。正式回执附 attention-import.log、mega-import.log 和各臂状态。取卡与退卡规则同上。

## 43e94dd 回执后的执行

按“启动”中的同一命令从任务书最新 revision 新建输出目录，直接重跑八臂。已有回执确认 DeepGEMM 导入共存，第 2 臂尚未发布完整参考。保留第 2 臂先行和原有依赖顺序。八臂启动时记录真实 warmup 次数、补齐次数和前后 epoch；收集日志中的这些值。第一个统计步即核对全部 40 个实测层号、epoch 及调用数。失败时原样回传这些值和首错。保留原有 1000+1000、逐层放置和完整窗口验收。整组预约、空闲核验和退卡要求同上。
