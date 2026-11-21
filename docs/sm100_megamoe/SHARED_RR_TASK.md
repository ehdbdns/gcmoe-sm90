# 【SM100】V4.1-Flash 第 0 层真实 shared + routed 权重正确性

同批合成输入与 gate、同一份真实 checkpoint，顺序运行 routed-only identity、带 shared expert 的 identity、带 shared expert 的 EPLB+RR。后两次须在八卡上逐元素相等，每卡相对 routed-only 的输出至少有一项改变。全部调用验证物理槽行数与有限非零输出。该对照检查 shared 路径产生实际贡献以及换槽一致性，尚不提供独立模型输出基准。

按原规则仅预约同组八张空闲卡，否则等待。只执行以下命令；只允许记录在案的语义等价编译修复。失败提交首条异常和日志，不修改判据。

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
export SM100_V41_MODEL=/mnt/nas1/hf/DeepSeek-V4.1-Flash
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
  python scripts/sm100/mega_rr_smoke.py --shared --checkpoint "$SM100_V41_MODEL"
  date -Is
) 2>&1 | tee "$work/smoke.log"
```

将执行 SHA、日志、全部 24 行 rank 结果、八卡占用与释放时间分别放入 docs/sm100_megamoe/shared_rr_results/ 和 docs/sm100_megamoe/REPLY_SHARED_RR.md，提交并推送 main。此轮不测性能。
