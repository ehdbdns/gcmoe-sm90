# 八卡任务书 · 按就绪 M 块发布 routed 工作

验证 SM90 流式发票。保持已验收的 BS256、逐层 EPLB、MB-CLAIM、CUDA Graph、TMA、SM-reuse=0 和固定 step 2 的 layer 8、20。每次领取仍是一个 `(slot, M block, N tile, phase)`。

## 回执证据

旧的 layer 8 / rank 6 图来自会自动选择 epoch 的绘图脚本。它有 684 个 tile，对应 rank 6 的 layer 20 的 18 个 M 块。真正的 layer 8 / rank 6 有 36 个 M 块和 1368 个 tile。其热点 local slot 48 在 319 µs 拉完，434 µs 开始 L1，618 µs 完成最后一个 L1 tile，714 µs 开始首个 L2，800 µs 完成最后一个 L2 tile。原始 CTA 活跃数据在这些区间仍有其它槽的计算。先前回执对跨 rank L1 依赖的推测需要按当前源码复核。L2 输入由本 rank 的本槽位生成。

旧 dispatch 只在整槽拉完后发票。旧 L2 领取还可能选中 L1 未完成的槽，随后等该槽的计数。新版每拉完一个满块或尾部残块，就在该块路由元数据、激活和 scale 可见后发布这一批 L1 票。同一 M 块的所有 L1 N-tile 均完成后，再发布它的 L2 票。计算 CTA 只领取已发布的任务，可以在 L1 没票时领取已就绪的 L2。

## 执行

在八卡执行机的仓库根目录确认工作区干净，GPU 空闲，使用同一模型、数据集和工具链。入口自动校验并记录执行 SHA、失败步骤、命令和日志。按顺序跑旧方案和新方案，分别保留输出目录。

```bash
set -e
git fetch origin main
git merge --ff-only origin/main
revision=$(git rev-parse origin/main)
GC_MOE_SM90_STREAM_M_BLOCKS=0 bash scripts/trace_layer/run_aligned_ep8.sh "$revision" --fine-trace
GC_MOE_SM90_STREAM_M_BLOCKS=1 bash scripts/trace_layer/run_aligned_ep8.sh "$revision" --fine-trace
```

每轮先跑 CPU 回归和角色编译冒烟。新方案还要过无需加载模型的八卡 CUDA Graph 正确性门禁。该门禁给一个单副本专家分配 135 行，覆盖两个满 M 块和一个 7 行尾块。旧的按 N-tile 领取路径作为数值参考，CUDA Graph 重放后八卡 BF16 输出的 SHA-256 必须逐 rank 一致。正式 A/B 的两轮都使用 MB-CLAIM。

执行者的代码修改只限编译器诊断明确指出的最小兼容性修复，保持原语义并单独提交。其余失败一律停止并回传，禁止换层、改实验配置、放宽检查或拿其它 run 拼图。保留入口打印的两个目录和日志。

## 验收与回执

1. 每轮都须保留八卡 source trace、`capture.json` 和 `fine.json`，并通过同一 capture ID 与 epoch 校验。保留两层七段 Perfetto、逐 rank token／M 块计数、细粒度 Perfetto 和 detail JSON。原始数据留在执行机，回执提供位置及校验值。
2. 新方案每个非空 M 块都必须有唯一的 L1、L2 发布记录。L1 只能在本块发布后开始。L2 发布必须晚于或等于本块最后一个 L1 tile 完成。同一槽位的任务数、有效行数和 L2 push 必须与原始 owned 一致。
3. 逐层列出最后退出 routed 的 rank 和最重槽。画出该槽每个 M 块的 L1 发布、首个 L1 tile、最后一个 L1 tile、L2 发布和最后一个 L2 tile。报告多少个满块的 L1 在整槽拉完前开始。对照旧方案 rank 6 的 319/434/618/714/800 µs 五个边界，测量首票提前量、L1→L2 间隔和 routed 退出时刻。
4. 比较前先检查两轮的实际逐槽分布。如果热点 rank 或 owned 分布变了，先报告采样内的因果顺序，跨轮耗时只做有注明的对照。细 trace 的时间不代表未插桩稳态性能。

在本目录提交 `REPLY_STREAM_MBLOCK.md`。成功时把 A/B 的生成产物分别放进 `stream_results/baseline/` 和 `stream_results/streamed/`。回执附实际执行 SHA、CUDA Graph 正确性校验、两轮 capture ID 和 epoch、热点块发布表、剩余等待位置。失败时仅提交回执，写明首个失败步骤、命令、报错与已有数据位置。
