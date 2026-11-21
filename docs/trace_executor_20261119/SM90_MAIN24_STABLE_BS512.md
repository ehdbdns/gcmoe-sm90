# 【SM90】BS512 六臂严格对照与容量测量

BS256 的 15 对严格验收已通过，本轮保持同一六臂、M64、stream、稳定 prefill 和 decode top-k、相同数据集筛选区间与每臂 1000+1000 步。每卡配置 20 GiB KV；软件 prompt 预算取 2,300,000 个逻辑 token，由 BS256 默认预算按请求数放大，最终以运行时满 batch、无抢占及 KV 水位验收。一次一臂，首次失败保存现场。仅允许等价编译修复。

## 数据文件

批准使用 config/experiment.env 中的 _1.jsonl 和 _2.jsonl，两文件顺序固定。六臂共同使用同一选择器、长度范围与参考；512 条请求包含重复 prompt。此档的 prompt 池与单文件 BS256 不同，跨档变化同时受输入组成和 BS 影响。已成功进入模型阶段的运行继续，不重复已完成臂；若旧 SHA 在逗号拼接路径的可读性检查处退出，请从本次修正版本重新开始。两份源文件分别记录 SHA256。

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
unset GC_MOE_PREFIX_DIAG GC_MOE_PREFIX_SPARSE_AA GC_MOE_PREFIX_BLOCK_AA GC_MOE_PREFIX_LAYER_AA GC_MOE_PREFIX_HIDDEN_AA
export GC_MOE_ORDER_PREFILL=1 GC_MOE_STABLE_PREFILL_TOPK=1 GC_MOE_STABLE_DECODE_TOPK=1
unset GC_MOE_DECODE_STATS_AA GC_MOE_VANILLA_STATS_REPLAY
dataset_dir=/mnt/nas2/nanjun.cp/MoE-Bench-Experiments/datasets
first="$dataset_dir/dsv4-flash_online_input_ids_26-10-08_1.jsonl"
second="$dataset_dir/dsv4-flash_online_input_ids_26-10-08_2.jsonl"
export DATASET="$first,$second"
sha256sum "$first" "$second" > "$work/dataset.sha256"
export GC_KV_BUDGET=2300000 GC_MAIN24_SINGLE_BS=512 GC_MOE_KV_BYTES=21474836480
export GC_MOE_CHECKPOINT_ROOT="${GC_MOE_CHECKPOINT_ROOT:-/tmp}"
if bash scripts/trace_layer/run_main24.sh "$work"; then code=0; else code=$?; fi
echo "$code" > "$work/exit-code.txt"
```

## 验收和回执

1. 成功时检查 `delivery/bs512/pairs.csv` 共 15 对，每对都须预热原生一致、正式原生差异为零、严格判定通过。记录六臂各自两个边界八卡的最低剩余显存、最高 PyTorch reserved、日志中 vLLM 公布的 KV 容量、峰值 KV 利用率，以及每步都产生 512 个 token 的校验结果。显存采样来自各 run.json 的 stats/measured.initial 每个 rank 的 device_memory_bytes。注明 CUDA allocator 之外的占用包含在设备剩余显存读数里。
2. 失败时记录首次失败的阶段、rank、请求号、样本位置和两边 token（若异常提供），已启动配置及 Capture workspace。保存该臂 runner.log 和原始异常。只有输入、路由、快照读回、预热及正式原生输出全部通过才能发布严格对比。
3. 提交到 `docs/trace_handover_20261119/REPLY_SM90_MAIN24_STABLE_DECODE_BS512.md`。将 work 下已有的 delivery、preflight、exit-code.txt、preflight.log、progress.log 复制到仓库中新目录 `docs/trace_executor_20261119/sm90_main24_stable_decode_bs512_results/`，提交通常规模的日志、NPZ、JSON、CSV 和清单。二进制 KV 与路由回放文件留在现场。在原 main 分支提交回执和结果并 push origin main，失败同样交付。此任务在 BS512 停止。

每对正式窗口应有 512,000 个样本。记录 GPU 实际余量、KV 峰值、scheduler 抢占数、15 对一致性、六臂全部主表列和 ECO 拍卖计数。每完成一臂可在对接目录提交简短进度；无需重启或干预运行。
