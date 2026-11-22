# SM100 执行任务 · BS1024 上游 GCMoE 与 MegaMoE 同输入复测

只运行 SM100 执行者的两臂，各启动一次模型，按 Mega、GC 顺序串行执行。目标为 DeepSeek V4.1 Flash、TP8/EP8、原生专家布局、融合 shared expert，在相同的 40 层输入上测上游默认配置。GC 固定 Bruce-Lee-LY/gc-moe 的 7902831ec4a6691b6268b308bcff5ec0cb15ec86，Mega 使用上轮相同的 vLLM vendored DeepGEMM。两臂共用 vLLM、Torch、CuTe DSL、模型路径、脚本和同一组物理卡。EPLB、ECO、RR 重排、额外 tracing 均关闭。

## 1. 准备源码和公共依赖

在本项目干净 main checkout 执行。以下命令默认在同一个 bash 会话中运行。保留共享模型、原 venv 和上轮输入。先准备 CPU 侧文件，GPU 运行须等指定八卡都得到预约且空闲。

```bash
set -euo pipefail
git switch main
git pull --ff-only origin main
repo=$(git rev-parse --show-toplevel)
cd "$repo"
work=$(mktemp -d /data0/home/admin/workspace/megamoe/v41-upstream-pair-XXXXXXXX)
base=/data0/home/admin/workspace/megamoe/native-v41-stsFS4w7/venv
git clone https://github.com/Bruce-Lee-LY/gc-moe.git "$work/upstream"
git -C "$work/upstream" checkout --detach 7902831ec4a6691b6268b308bcff5ec0cb15ec86
git -C "$work/upstream" rev-parse HEAD > "$work/upstream.sha"
"$base/bin/python" -m venv "$work/dsl-env"
py="$work/dsl-env/bin/python"
"$py" -m pip install --no-deps nvidia-cutlass-dsl==4.8.1 nvidia-cutlass-dsl-libs-base==4.8.1 nvidia-cutlass-dsl-libs-core==4.8.1 nvidia-cutlass-dsl-libs-cu12==4.8.1 nvidia-cutlass-dsl-libs-cu13==4.8.1 2>&1 | tee "$work/dsl-install.log"
lib=$("$py" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
base_lib=$("$base/bin/python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
printf "import site; site.addsitedir('%s')\n" "$base_lib" > "$lib/zz_v41_common.pth"
```

原 venv 的 DSL 是 4.7.1，上游 JIT 使用 4.8.1 的缓存键接口。两臂都使用上述私有解释器和相同的 4.8.1 wheel 集。如果远端没有公网，用本机包含指定对象的上游 clone 作为 git clone 来源；pip 可用可信的本地同版本 wheel。保留实际来源和安装日志。

## 2. 执行

```bash
bash scripts/sm100/run_five.sh "$work" upstream-pair 2>&1 | tee "$work/launcher.log"
```

入口复用共享八卡组锁、UUID 核对与空闲等待；其他用户占用时等待，不终止他人的进程。两臂只读上轮 step 0、8、16 × rank 0–7 的 24 份归档，运行前按上轮 SHA256 清单核验。每步按层序重放同一批 BF16 输入、top-6 ID/权重与实时参考。保留每 rank 48 个专家及 fused shared 断言。各 worker 的 JSON 记录实际后端文件；GC 同时记录设备能力、实际 dispatch/compute SM 数与 primary-context 模式。dispatch 配比使用上游 EP8 默认请求，最终值读驱动。

两臂使用相同的输出缓冲区复用方式，让 GC 的 eager 预热和父图捕获使用同一层输出地址。该方式在集成层替换输出分配，两臂均不增加设备拷贝到计时图。eager 参考单独克隆。沿用 3 次 eager、40 层一张图、10 次预热、15 组 × 20 次 replay 的计时。每组取八卡最慢 rank，再取 15 组中位数。指标为专家层图设备耗时，单位 ms。

## 3. 判定与回执

提交 docs/sm100_megamoe/REPLY_V41_UPSTREAM_NATIVE_PAIR.md 和同目录 v41_upstream_native_pair_results/。在回执表中逐步列 Mega ms、GC ms、100 × (GC/Mega − 1)、每臂 15 组分布、相对实时参考的不同元素数和最大绝对误差。保留各层各 rank 明细。两臂各自必须 eager/replay 相等且输出有限。沿用逐元素严格门槛；不满足时仍报告耗时和差异，并标记该步严格判定未通过。保留 upstream.sha、本仓 SHA、imports.txt、24 个摘要核验、GPU UUID、版本、权重索引摘要、每臂退出码及日志、mega.json、gc.json、comparison.json 和 SHA256SUMS。核对八个 rank 的后端、GC 包路径和实际分区。报告编译器版本、两臂公共依赖及与上轮的区别。

执行者可修复不改变语义的编译环境问题，记录原始错误、处理步骤及所有 diff。上游代码必须保持指定 SHA 和干净状态；禁止修改 kernel、runtime、模型、输入、框架计时逻辑或切换 eager 代替 graph。若必须改这些才能运行，提交首个完整堆栈及已完成臂，注明阻塞位置，不追加实验。

成功或失败都要先归档日志和证据，核实本次进程退出、八卡使用情况及锁释放，记录时间。清理本次私有 dsl-env、upstream、dg-cache 和本次创建的 fixture 符号链接；保留旧目录的 24 个原文件及共享环境。仅推送回执与轻量结果到 main。
