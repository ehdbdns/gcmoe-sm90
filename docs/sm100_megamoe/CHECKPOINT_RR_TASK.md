# 【SM100】DeepSeek-V4.1-Flash 第 0 层真实 routed 权重八卡 RR 验证

本轮及后续五臂统一使用与 SM90 主表相同的 DeepSeek-V4.1-Flash。上轮旧形状的合成权重结果仅作接口验证。先在 admin 可读的模型目录或共享存储中定位 V4.1-Flash，将实际目录导出为 SM100_V41_MODEL。若找不到，先回执缺失路径或所需挂载，等权重就绪再占卡。若旧任务已启动，保留其结果并标注旧模型。本轮读取第 0 层真实打包权重，按逻辑专家装载 w1/w3 至 L1、w2 至 L2，并转换 scale 布局。使用与上一轮相同的合成输入和 gate，分别运行原生 identity 与逐层 EPLB + RR，核验 16 份 rank 记录、槽计数和逐元素输出。该对照检验换槽后是否保持结果，两臂共享加载器，不能独立验证模型语义。

只在同组八卡均空闲并确认无人预约时执行，否则每 60 秒等待复查。遵循[【SM100】共享机器规则](ENV_TASK.md)。执行者只执行下列步骤，仅允许有记录的语义等价编译修复。失败保留首条异常。

```bash
set -eo pipefail
git -c http.version=HTTP/1.1 pull --ff-only origin main
repo=$(git rev-parse --show-toplevel)
work=$(mktemp -d /data0/home/admin/workspace/megamoe/rr-smoke-XXXXXXXX)
git worktree add --detach "$work/code" HEAD
cd "$work/code"
source /home/admin/workspace/megamoe/env.sh
dg=/home/admin/workspace/megamoe/DeepGEMM
test "$(git -C "$dg" rev-parse HEAD)" = 5081f8513af57e61696de4f4f04b0e707ffb503e
export PYTHONPATH="$dg:${PYTHONPATH:-}"
: "${SM100_V41_MODEL:?set to the accessible DeepSeek-V4.1-Flash checkpoint}"
python scripts/sm100/v41_shape.py "$SM100_V41_MODEL" | tee "$work/config.log"
python scripts/sm100/checkpoint_experts.py "$SM100_V41_MODEL" | tee "$work/checkpoint.log"
sha256sum "$SM100_V41_MODEL/model.safetensors.index.json" >> "$work/config.log"
export MASTER_ADDR=127.0.0.1 MASTER_PORT=8361
(
  exec 9>>/tmp/gcmoe-sm100-gpu-group.lock
  while true; do
    flock 9
    if busy=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader) && test -z "$busy" && ! fuser /dev/nvidia{0..7} >/dev/null 2>&1; then break; fi
    flock -u 9
    sleep 60
  done
  # 绑定环境回执中已登记的 UUID。启动前确认整组当前仍无人预约使用。
  export CUDA_VISIBLE_DEVICES=$(python -c 'import json,re; d=json.load(open("docs/sm100_megamoe/ENV.json")); print(",".join(re.findall(r"GPU-[0-9A-Fa-f-]{36}", d["commands"]["gpu_list"]["output"])))')
  test "$(tr , '\n' <<< "$CUDA_VISIBLE_DEVICES" | sort -u | wc -l)" = 8
  date -Is
  python scripts/sm100/mega_rr_smoke.py --checkpoint "$SM100_V41_MODEL"
  date -Is
) 2>&1 | tee "$work/smoke.log"
```

回执注明模型实际路径及可核实的来源或 revision。将 config.log、checkpoint.log、smoke.log、执行 SHA、分配及释放时间和逐 rank 结果提交至 docs/sm100_megamoe/checkpoint_rr_results/ 与 docs/sm100_megamoe/REPLY_CHECKPOINT_RR.md，推送 main。无需启动框架或测量性能。
