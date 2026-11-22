# SM90 · BS256 异步单副本 token 直推

运行此任务书对应的提交。用原 8 张 H20，对照默认 ECO 与新版 single_push。各一次 1000 步统计、重排、1000 步测量。新臂替换了 18561e9 的单副本算法。上一版任务书锁定了旧提交，其回执单独归档。保留原固定输入、路由、源内 ordinal、EPLB、ECO plan、成本模式、stream、输出和逐 rank 审计。源 rank 间预留的先后决定物理行序；这属于传输算法的一部分。若输出或审计变化，按失败回执，保留首个分歧。

## 执行

在八卡预约内核实所有卡空闲，保存 nvidia-smi topo -m 和 nvidia-smi topo -p2p a，核实双向 P2P 原子支持。其他用户占用时等待。使用原 SM90 venv 及已校验的 KV、replay；若原样迁移则设置 SM90_RETAINED。

```bash
set -euo pipefail
git fetch origin main
book=docs/trace_executor_20261119/SM90_BS256_ASYNC_SINGLE_PUSH.md
sha=$(git log -1 --format=%H origin/main -- "$book")
checkout=$(mktemp -d /tmp/sm90-async-push-XXXXXXXX)
git worktree add --detach "$checkout" "$sha"
(cd "$checkout" && python3 tests/test_async_single_push_cpu.py && bash scripts/trace_layer/run_bs256_eco_exchange.sh single-push)
```

源端给每个单副本目的槽单独报数，一次原子预留当前源的全部行，然后直推。每个源独占一格信箱，包括零行源。目标槽的满块收到 M 行就可以发布，尾块另需收齐 8 份声明。目标 dispatch 的一个现有 CTA 扫描单副本槽；每个就绪块进入原 ntile 队列。多副本沿用 baseline 的配额和 pull。原 launch grid 保持 24/71/71/1。

## 交付

1. 主指标为剔除采样步的 999 步完整 MoE ms/步。另报含采样的 1000 步与三段关键 rank 时间，变化用 100 × (新 / 基线 − 1)，负值更快。
2. 提交 step 2、层 0/8/20/32/39 每臂的 coarse 与 absolute/detail Perfetto JSON、counts.json、detail.json、每个 rank 的 token 和 M 块数。新臂逐层给出满块总数、发布时 declared_sources 小于 8 的满块数、finished_sources 小于 8 的满块数及尾块数。字段是目标 CTA 在检查该块前最近一次读到的源状态快照。若均为 0，原样报告。目标槽 finalize 与实际 L1 publish 分开解读。
3. 将 moe.csv、每臂 run.json、rank0–7.npz、trace、校验和、版本、运行及验证日志归档到 docs/trace_executor_20261119/sm90_bs256_async_push_results/。回执到 docs/trace_handover_20261119/REPLY_SM90_BS256_ASYNC_SINGLE_PUSH.md 并推送 main。执行者仅处理不改语义的编译问题，记录 diff 和 SHA，两臂跑相同版本。其余失败保留第一处错误和完成产物，保持实验条件和校验原样。
4. 核实实验进程退出并释放八卡。归档核验后只清理本任务临时 checkout 和输出，保留历史 KV、replay 与他人作业。
