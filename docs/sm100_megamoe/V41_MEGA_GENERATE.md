# 【SM100】在已通过的整模环境验证 DeepGEMM MegaMoE 的生成

采用与 SM90 相同版本的 vLLM 0.31.0，使用独立 venv、已有本地权重、显式 deep_gemm_mega_moe 后端、TP8/EP8、单请求贪心解码 4 token。此轮验证整模入口和生成能力，无端到端性能验收。环境保留给后续两种 MoE 集成复用。只允许等价编译修复，其他故障附日志回执。

```bash
set -eo pipefail
git pull --ff-only origin main
repo=$(git rev-parse --show-toplevel)
work=$(mktemp -d /data0/home/admin/workspace/megamoe/native-v41-XXXXXXXX)
cd "$repo"
git rev-parse HEAD > "$work/source.sha"
source /home/admin/workspace/megamoe/env.sh
unset PYTHONPATH
source /data0/home/admin/workspace/megamoe/native-v41-stsFS4w7/venv/bin/activate
export SM100_V41_MODEL=/data0/home/admin/workspace/megamoe/native-v41-KO1hTbTj/model
test -r "$SM100_V41_MODEL/model.safetensors.index.json"
findmnt -T "$SM100_V41_MODEL" -o TARGET,FSTYPE > "$work/load-filesystem.txt"
python scripts/sm100/v41_shape.py "$SM100_V41_MODEL" | tee "$work/config.log"
export VLLM_USE_V2_MODEL_RUNNER=1 HF_HUB_OFFLINE=1
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
  # 记录与本进程同 cgroup 的 OOM/限额变化、剩余内存和本次进程退出状态。
  cgroup_path=$(awk -F: '$1=="0" {print $3}' /proc/self/cgroup)
  cg="/sys/fs/cgroup${cgroup_path:-/}"
  if [[ ! -r "$cg/memory.events" ]]; then cg=/sys/fs/cgroup; fi
  snapshot() {
    date -Is
    for f in memory.events memory.current memory.max; do
      if [[ -r "$cg/$f" ]]; then cat "$cg/$f"; fi
    done
    awk '/MemAvailable|SwapFree/ {print}' /proc/meminfo
  }
  snapshot > "$work/memory.before"
  export PYTHONFAULTHANDLER=1 VLLM_LOGGING_LEVEL=DEBUG
  python -u scripts/sm100/v41_native_smoke.py --backend deep_gemm_mega_moe > "$work/model-full.log" 2>&1 &
  model_pid=$!
  while kill -0 "$model_pid" 2>/dev/null; do
    snapshot >> "$work/memory.log"
    sleep 30
  done
  code=0
  wait "$model_pid" || code=$?
  echo "$code" > "$work/exit-code.txt"
  snapshot > "$work/memory.after"
  tail -60 "$work/model-full.log"
  test "$code" = 0
  date -Is
) 2>&1 | tee "$work/model.log"
```

共享机器只在确认这组 8 卡无人占用或预约时启动。保存安装日志、版本、模型日志、实际 native MoE backend、八卡释放时间、工作目录路径和输出 ID 到 docs/sm100_megamoe/v41_mega_generate_results/。回执 docs/sm100_megamoe/REPLY_V41_MEGA_GENERATE.md。失败时原样记录首个错误与退出码；不得更换模型、卡数、框架或后端。提交 main。

任务固定使用带 main guard 的真实文件入口。用可跨工具调用持续运行的会话让 Python 自行退出，记录执行器是否有外部定时终止设置。若 system journal 可读，提取本次时段内 OOM/Xid 及 worker 死亡证据；权限不足则标注。无论成败记录 model-full.log、cgroup 前后计数、退出码、环境和释放时间。

对照原生 auto 后端已生成 [200, 19, 200, 18]。本轮从相同权重、同一独立环境、相同输入启动一次，唯一主动改变的选项是 moe_backend。保存实际加载的 DeepGEMM 路径、后端选择日志、shared expert 是否融合的证据和 ID；如有差异，记录首个不同的位置并回执。不自动切回原生或修改框架。此次先确认整模基线路径，暂不启用 EPLB 或 ECO，不产生五臂性能数据。
