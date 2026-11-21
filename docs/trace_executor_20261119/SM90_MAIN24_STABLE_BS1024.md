# SM90 · BS1024 六臂严格对照

在同一组空闲的八张 H20 上执行。沿用 BS512 的完整有序 prompt 批次两遍，每条作为独立请求，固定两份源文件、长度分布和两个稳定 top-k 开关。通过 SHA256 核对实际基础批次。每臂前 1000 步统计、后 1000 步测量，每卡 20 GiB KV。5,000,000 为软件选样预算，实际容量以八卡运行结果为准。

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
export GC_STEP_BOUNDARY=1
unset GC_KV_OVERHEAD
export GC_REPEAT_BS512_PROMPT_SHA256=e7c4aee9db91a34c018bd54464e67355681b74112d515401844f473c053cfcef
export GC_KV_BUDGET=5000000 GC_MAIN24_SINGLE_BS=1024 GC_MOE_KV_BYTES=21474836480
export GC_MOE_CHECKPOINT_ROOT="${GC_MOE_CHECKPOINT_ROOT:-/tmp}"
if bash scripts/trace_layer/run_main24.sh "$work"; then code=0; else code=$?; fi
echo "$code" > "$work/exit-code.txt"
test "$code" = 0
```

## 验收和交付

只跑 BS1024，每臂一次。保留源文件哈希、满批记录、scheduler 抢占数、峰值 KV、各 rank 最低剩余显存及最高 reserved。核对每步 1024 个 token、15 对预热一致、每对正式 1,024,000 个 token 零差异，提交全部主表列、两种 plan 误差及拍卖统计。任一容量或一致性失败即记录首错、阶段、配置和已有产物。执行者仅可做保持语义的编译修复并附 diff。

在原 main 工作区提交 docs/trace_handover_20261119/REPLY_SM90_MAIN24_STABLE_DECODE_BS1024.md 和 docs/trace_executor_20261119/sm90_main24_stable_decode_bs1024_results/（目录结构同 BS512），推送 main。大体积 KV 和路由回放文件留在机器上。

## 同轮诊断新旧 cost 的跨步耗时

若已在旧测量 SHA 启动模型，请完成原六臂并保持同一 SHA，勿重启、追加模型或混用版本。交付后用含本补充的分析脚本在原 delivery/bs1024 目录运行 `python scripts/trace_layer/cost_boundary_report.py /实际路径/delivery/bs1024`，将 stdout 保存到回执产物；旧 SHA 只支持粗分段。若尚未启动模型，从更新后的干净 worktree 执行上方命令。

新版本对全部六臂采用同样的打点，保留原正式窗口及严格校验。fine 排在线性 cost 之前，与较小两档的顺序相反。一次一臂，每臂一次启动。保留实际测量 SHA、顺序、六臂原始 run.json、rank NPZ 以及 cost_boundary.csv。

GPU 使用每卡自己的 globaltimer。正式窗口取 20 对相邻 decode 步，在 execute_model 与 sample_tokens 入口、返回的当前 stream 插入时间戳。报告上一步末层 combine 到 forward 返回、到 sampler 入口、sample_tokens 入口至返回、到下次 execute_model 入口、到下步第一层 MoE 五段。它们表示流上的经过时间，可包含排队和等待。先核对五段之和与 marked_gaps，再并列未打点区间 unmarked_gaps 和全窗 between_steps。

主机逐步记录调度 schedule、update_from_output、其余 get_output（含等待 worker）、engine.step 的其余部分及收集输出耗时。它们与 GPU 区间可能重叠，请分表报告。给出各段新减旧的均值、配对中位数、样本数，以及细分段与完整窗口方向是否一致。若新 cost 在 BS1024 更快，同样如实记录。若旧 SHA、样本波动或标记缺失导致只能定位到较粗区间，注明证据边界，勿将等待直接归因为 CPU 调度或某个 GPU 算子。
