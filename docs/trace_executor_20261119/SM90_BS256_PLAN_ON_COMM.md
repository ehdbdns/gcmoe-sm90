# SM90 · BS256 将 ECO plan 移到通信分区

目标是在 8×H20 上实测一次算力预算调整。保持已有的 6 SM 通信分区和 72 SM 计算分区。实验臂中 plan 的 1 个 CTA 与 dispatch 进入同一个通信 Green Context。dispatch 启动 5 个 CTA，即使分散到不同的 SM 也能留下一个完整 SM 给 plan；不要求驱动指定某个 SM 编号。该几何会降低 dispatch 并行度，比较的是这一整套可运行配置的净效果。

| 臂 | plan 所属 context | dispatch / shared / routed / plan CTA |
|---|---|---|
| baseline | compute | 24 / 71 / 71 / 1 |
| dispatch | communication | 5 / 72 / 72 / 1 |

两臂都使用原 pull、逐块 stream、原 EPLB + ECO fine cost、同一输入、相同 KV 和路由回放。拍卖、算子和取票规则沿用上轮 baseline。本轮只跑这两臂，顺序执行，各 1000 步统计 + 重排后 1000 步测量。

## 执行

预约同一组空闲八卡。确认历史保留现场（默认 /tmp/sm90-bs256-trace-pbpMbH4u）可读；如已搬迁，SM90_RETAINED 指向同一份经校验的 KV/replay。用任务书提交启动。记录 Trial workspace 和两臂 Capture workspace。

```bash
set -euo pipefail
git fetch origin main
task=docs/trace_executor_20261119/SM90_BS256_PLAN_ON_COMM.md
sha=$(git log -1 --format=%H origin/main -- "$task")
checkout=$(mktemp -d /tmp/sm90-plan-comm-code-XXXXXXXX)
git worktree add --detach "$checkout" "$sha"
(cd "$checkout" && bash scripts/trace_layer/run_bs256_eco_exchange.sh plan-dispatch)
```

## 验收和回执

1. 实验臂核实 worker 日志的 `SM90 ECO plan context=communication communication_sms=6 compute_sms=72`；五个采样层逐 rank 核对四个角色的实际 grid。用采样轨对照 plan/dispatch 同时在途的时间、plan 发布时刻与首个 routed CTA 的实测时刻。
2. 使用脚本的严格核验：两窗口八 rank 输入和原生输出、逐槽 counts、拍卖、预测值、放置、epoch 与历史参考一致。用 moe.csv 给出 999 个非 profiler 步的完整 MoE 和 graph→首 CTA、首→末 CTA、末 CTA→combine 三段，再附 1000 步值。按同轮 baseline 计算带正负号的相对变化。
3. 在 step 2 的 L0、8、20、32、39 附两臂 absolute Perfetto 和各 rank token/M 块；同层同 rank 列 dispatch 完成、shared 完成、plan 完成、首个 routed CTA 和 combine 完成相对 graph 的时间。区分 compute 多 1 个 CTA 的收益和通信 CTA 减少的代价，重叠时段不得直接相加。
4. 执行者只做不改变语义的编译修复并附 diff；如需编译修复，使用同一版本执行配对对照。运行、正确性或 grid 失败时保留首错日志、现场、退出码和已完成结果，勿改调度或计时。回执 docs/trace_handover_20261119/REPLY_SM90_BS256_PLAN_ON_COMM.md，产物放 docs/trace_executor_20261119/sm90_bs256_plan_on_comm_results/。提交 SHA、moe.csv、两臂 run.json、NPZ、验证日志、图及 SHA256。推 main 并验证；释放 GPU，清理此次私有 checkout 和临时缓存，保留原回放及可核验的原始证据。
