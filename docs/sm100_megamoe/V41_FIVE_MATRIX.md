# SM100 执行者任务书 · V4.1-Flash 五臂正式对照

目标是一次启动，交付 BS256、512、1024、2048 四档各五臂，共 20 组。每组一次模型冷启动，前 1000 个满批 decode 步统计，后 1000 个满批 decode 步测量，覆盖 40 层、8 个 rank。本任务取代本目录此前的阶段性 smoke 任务。

| 臂 | 后端 | 正式窗口的专家放置及副本路由 |
|---|---|---|
| mega | 原生 MegaMoE | 384 槽，无冗余 |
| gc | 原生 GCMoE | 384 槽，无冗余 |
| mega_rr | MegaMoE | 448 槽，逐层 DeepInfer EPLB + RR |
| gc_rr | GCMoE | 同一布局与副本顺序，RR |
| gc_eco | GCMoE | 同一布局，SM100 在线 ECO 拍卖 |

## 一次启动

1. 在 SM100 机器的干净 checkout 更新 main。使用上轮已验证的模型目录、venv 和 ptxas。输出放在本机持久盘，预留至少 32 GiB 给路由带、结果和 JIT 缓存。执行前查调度器或共享预约表，只在指定八张卡均无预约且无人使用时运行。有占用就等待，保留其他用户的进程和卡绑定。为整轮保留这八张卡，沿用上一轮的组锁。使用可持续运行的终端。
2. 在仓库根目录执行以下整段。输出目录位于仓库外，重复进入同一目录会校验提交并跳过已成功的组。脚本对指定八张卡的 UUID 和计算进程做启动前复核。

   ```bash
   git switch main
   git pull --ff-only origin main
   work=/data0/home/admin/workspace/megamoe/v41-five-matrix-01
   mkdir -p "$work"
   set -o pipefail
   bash scripts/sm100/run_five.sh "$work" 2>&1 | tee -a "$work/launcher.log"
   ```

3. 一直执行至脚本退出。每档的 Mega 生成该档基准，能独立进行的后续组会继续。仅允许解决编译失败时作语义等价的编译兼容修复，附修改前完整首错、diff、原 SHA 和修复提交 SHA。需要修复时在新提交和新输出目录重启同一命令，以保证每档比较使用同版代码。模型、定价、容量、采样步数、输入、路由、计时和公平性校验由仓库固定。遇到 OOM、输入不一致等情况保留实际失败与证据。

## 核验与回执

- 四档固定使用同一批 256 条八 token 前缀，按 BS 重复。每档的原生 Mega 记录自身统计窗口的逐步需求，并生成该档三个 EPLB 臂共同使用的逐层布局。迁移后逐层核对 live 映射和已搬运权重样本。各臂对齐 greedy 输出轨迹与逐 token 的 logical top-6 ID、权重、批顺序。在正式窗口按实际接收数还原逻辑专家需求，并逐步逐层核对；每层全局 routed token 数必须为 BS × 6。
- 端到端取正式窗口的同步壁钟时间，包含所有臂共同启用的采集、回放和 eager 调度开销。完整 MoE 逐层取八个 rank 的最大本地 kernel 跨度，再对层求和。routed 跨度从首个 routed CTA 数据就绪后准备发出 MMA 到最后一个 routed L2 CTA 写完 push。时钟采用设备 globaltimer。预测沿用 Blackwell 本身的价格，即 tokens + wb × M 块数 + wa × 活跃槽数，wb、wa 由实际 block M 决定。两种误差均以八 rank 耗时占比的百分点计算。拍卖接受数按层调用去除 rank 重复后累计。
- 在 `docs/sm100_megamoe/v41_five_matrix_results/` 提交 `$work/delivery/` 的内容，附启动与释放时间、八卡退出后进程检查、各失败组完整报错日志、实际工作目录与持久保存的 rank NPZ 路径。原始 NPZ 包含逐步逐层每个槽的实际接收数，以及正式窗口每个 rank 的四个时间戳，ECO 还含逐调用接受数。可用回执中的 SHA256 清单验证。保留原始日志、路由带、计数及时间戳供复查。
- 提交 `docs/sm100_megamoe/REPLY_V41_FIVE_MATRIX.md` 并推送 main。列出四档完成数、主表所有列、每档十对的严格校验结果；只对 strict 为真的配对解读性能。缺项写明首个失败阶段及原因。每臂每档一次观测，保留未舍入 CSV。报告全矩阵后一次交接。
