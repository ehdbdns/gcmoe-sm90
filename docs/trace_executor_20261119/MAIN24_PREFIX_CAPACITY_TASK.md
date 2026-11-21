# 【SM90】BS256 严格回放复核与大 batch 容量测量

先在原 8×H20 上复核 BS256。六种配置各最多启动一次，顺序沿用主表。每臂前 1000 步统计、重排后 1000 步正式测量。首次差异或资源错误即停止，保留日志和现场。只允许首次模型启动前的语义等价编译修复；提交修复、从干净版本重新预检。采样、分流、重排、价格、输入输出检查、计时和实验规模均由设计端维护。

## 执行命令

在执行仓库的干净 main 分支运行，保留当前机器已验证的模型、数据、HCA 和环境设置。GC_MOE_CHECKPOINT_ROOT 选择有足够空间的本地文件系统，脚本会检查双边界 KV 和两段 route bank 的空间。

```bash
set -eo pipefail
repo=$(git rev-parse --show-toplevel)
cd "$repo"
test -z "$(git status --porcelain)"
git pull --ff-only origin main
work=$(mktemp -d /tmp/gcmoe-prefix-XXXXXXXX)
git worktree add --detach "$work/code" HEAD
cd "$work/code"
export GC_MAIN24_SINGLE_BS=256 GC_MOE_KV_BYTES=21474836480
export GC_MOE_CHECKPOINT_ROOT="${GC_MOE_CHECKPOINT_ROOT:-/tmp}"
if bash scripts/trace_layer/run_main24.sh "$work"; then code=0; else code=$?; fi
echo "$code" > "$work/exit-code.txt"
```

## 验收和回执

1. 成功时检查 `delivery/bs256/pairs.csv` 共 15 对，每对都须预热原生一致、正式原生差异为零、严格判定通过。记录六臂各自两个边界八卡的最低剩余显存、最高 PyTorch reserved、日志中 vLLM 公布的 KV 容量、峰值 KV 利用率，以及每步都产生 256 个 token 的校验结果。显存采样来自各 run.json 的 stats/measured.initial 每个 rank 的 device_memory_bytes。注明 CUDA allocator 之外的占用包含在设备剩余显存读数里。
2. 失败时记录首次失败的阶段、rank、请求号、样本位置和两边 token（若异常提供），已启动配置及 Capture workspace。保存该臂 runner.log 和原始异常。只有输入、路由、快照读回、预热及正式原生输出全部通过才能发布严格对比。
3. 提交到 `docs/trace_handover_20261119/REPLY_MAIN24_PREFIX_CAPACITY.md`。将 work 下已有的 delivery、preflight、exit-code.txt、preflight.log、progress.log 复制到仓库中新目录 `docs/trace_executor_20261119/main24_prefix_capacity_results/`，提交通常规模的日志、NPZ、JSON、CSV 和清单。二进制 KV 与路由回放文件留在现场。在原 main 分支提交回执和结果并 push origin main，失败同样交付。此任务在 BS256 停止。

## 512、1024、2048 的容量依据

BS 指实际同时解码的请求数。使用 BS256 实验日志中的每卡 20 GiB KV、vLLM 报告的 1,530,922 个 token 以及 ECO+SM-reuse 末步约 11.4% KV 利用率进行规划。该模型的压缩和滑窗缓存使逻辑 token 数与物理 KV 占用不成固定比例。按 2000 步估出的逻辑 token 数只用于核算输入规模。当前 driver 的 prompt 筛选使用独立的软件 token 预算 GC_KV_BUDGET，也要与实际选中的 prompt、KV 分组和运行时容量一起检查。

| BS | 仅 2000 步生成的逻辑 token | 每卡完整路由 ID 与权重 GiB | 本版移除的两份完整 ordinal 表 GiB（约） |
|---:|---:|---:|---:|
|256|512,000|0.69|0.46|
|512|1,024,000|1.37|0.92|
|1024|2,048,000|2.75|1.83|
|2048|4,096,000|5.49|3.66|

路由大小按 40,000 次调用 × BS × top-6 × (8 字节 ID + 4 字节权重)。ordinal 检查保留当前调用的期望行号，并为每次调用保留所有行是否相等的结果。每档两段八卡磁盘 route bank 在 BS2048 合计约 88 GiB；每卡 KV 为 20 GiB 时，两份八卡 KV 快照另占约 320 GiB。

BS256 通过后用同档固定工作量逐级验证 512、1024、2048。入口支持 GC_MAIN24_SINGLE_BS 选择单档；大档仅捕获正式满 batch 的 decode graph。先用实际 prompt 总长度计算筛选预算，采集最重臂的调度、空闲显存与 KV 水位，再决定该档统一的 KV 字节数。正式验收仍要求每步满 BS、两个窗口各 1000 步、同档原生输出一致。若连续上下文仍装不下，保留最大通过档；分段重置 KV 会改变上下文长度分布，需要单独定义实验。
