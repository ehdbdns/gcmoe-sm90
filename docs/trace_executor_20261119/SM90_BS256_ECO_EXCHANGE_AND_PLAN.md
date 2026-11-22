# SM90 · BS256 ECO 分阶段交换与 plan 分区实验

在原 H20 节点顺序执行三臂，每臂一次 DeepSeek-V4.1-Flash 模型启动。用同一次任务的 baseline 分别作对照。只占用原八张空闲卡，取得预约再开始，有其他作业时等待。

| 臂 | count exchange | plan 的 SM 来源 | dispatch / shared / routed CTA |
|---|---|---|---|
| baseline | 原有 active copies 全写 | compute | 24 / 71 / 71 |
| split | 阶段一仅写单活跃副本专家索引，阶段二按最终 quota 写多活跃副本专家索引 | compute | 24 / 71 / 71 |
| dispatch | 原有 active copies 全写 | 通信分区中独立 1 SM，余下 5 SM 跑 dispatch | 20 / 72 / 72 |

三臂均为原逐层 EPLB + ECO fine cost，mode 2、M64、slack 11469，stream 开、SM reuse 关。TP8/EP8、KV 20 GiB/rank、数据集、1000 步统计和重排后 1000 步测量均沿用 BS256 成功轮。复用已保留的 KV 与路由轨迹。每一臂在同一次正式回放中对 step 2 的零基层 0、8、20、32、39 采样。无第四臂。

## 执行

在执行者主仓开干净 shell，确认保留现场仍在。以下命令在干净 detached checkout 执行任务书对应版本。保留控制台输出中的 Trial workspace 路径。

```bash
set -euo pipefail
git fetch origin main
task=docs/trace_executor_20261119/SM90_BS256_ECO_EXCHANGE_AND_PLAN.md
sha=$(git log -1 --format=%H origin/main -- "$task")
checkout=$(mktemp -d /tmp/sm90-eco-code-XXXXXXXX)
git worktree add --detach "$checkout" "$sha"
(cd "$checkout" && bash scripts/trace_layer/run_bs256_eco_exchange.sh)
```

脚本默认读取 /tmp/sm90-bs256-trace-pbpMbH4u/{kv,replay}。若同一份现场已搬迁，用 SM90_RETAINED 指定新根目录。脚本核对历史硬件、checkpoint manifest、输入/输出哈希、两窗口 audit、各 rank 的 native token、物理槽 counts、epoch、拍卖和预测值、放置结果、采样层 launch grid。split 的每个采样 rank/layer 必须有一对阶段二事件。dispatch 需在日志确认 runtime 输出 dispatch=5 plan=1 compute=72；driver 若拒绝分区，提交 CUDA 错误和驱动版本。保留成功臂证据。执行者只可修复不改变语义的编译错误，附完整 diff 与实际执行 SHA，使用同一修复版本执行配对对照。运行、正确性或资源检查失败时保留现场和首次失败位置，禁止更换分区大小、参考状态、计时边界或放宽校验。

## 回执

提交 docs/trace_handover_20261119/REPLY_SM90_BS256_ECO_EXCHANGE_AND_PLAN.md。结果放在 docs/trace_executor_20261119/sm90_bs256_eco_exchange_plan_results/，每臂一目录。提交三臂的 moe.csv、measurement.path 指向目录中的 run.json 和八份 rank NPZ、hardware.csv、运行日志、验证日志、执行 SHA、Perfetto 和 SHA256SUMS。大文件可 gzip，保留原始数据路径和哈希。每臂五层均链接粗粒度和 absolute/detail Perfetto JSON、counts JSON，列出八 rank 的 token、M 块、活跃槽与最慢 rank。

主指标是每步 MoE 毫秒数。每个 step/layer 选 graph 开始至 combine 结束跨度最大的 rank，在同 rank 拆 graph→首个有效 CTA、首 CTA→末 CTA、末 CTA→combine 结束，再跨 40 层求和。主比较剔除 torch profiler 采样步，保留 999 步；附全 1000 步含采样结果。变动百分比按 (实验臂/同轮 baseline − 1) × 100%，负值表示更快。该轮包含细粒度插桩，跨轮比较须注明。

在同层同 rank 上列出 count exchange、shared kernel、首个有效 routed CTA、pull 完成、combine 完成的实测值与相对 graph 起点时间。split 新增的 replica index exchange span 包含写索引及等待各源 rank 发布。单列该 span 与 replica 等待时长，注明其在 plan kernel 内，与其他轨道有重叠。dispatch 核查 shared 加速是否抵消 dispatch 少一个 SM 的代价。分别给出两臂的净 MoE 变化；若波动或指标不足，保留不确定性。

完成或失败后释放八卡并记录 nvidia-smi。核对归档后仅清理本任务创建的临时 checkout 和临时文件，保留旧 KV 和回放现场。推送回执及产物到 main。
