# 【SM100】真实权重下的 ECO 配额分流正确性

继承第 0 层 shared+routed 对照，增加一臂：用仓库 Blackwell UnitPlanner 的冷启动参考配额驱动 MegaMoE 物理槽。单次调用按源 rank、token 和 top-k 顺序领取配额；检查逻辑专家、全局物理直方图、本卡实际接收行数及 BF16 输出。本次 M64、wb=260、wa=111 仅用于验证接口。CPU 参考拍卖和同步操作均不用于性能测量。

仅使用同组八张确认空闲的卡，否则等待。执行者仅执行命令，只允许记录在案的等价编译修复，其他失败原样回执。

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
  python scripts/sm100/mega_rr_smoke.py --eco-reference --shared --checkpoint "$SM100_V41_MODEL"
  date -Is
) 2>&1 | tee "$work/smoke.log"
```

将全部 32 条 rank 结果、日志、SHA、占用与释放时间交至 docs/sm100_megamoe/eco_quota_results/ 和 docs/sm100_megamoe/REPLY_ECO_QUOTA.md。记录 ECO 配额路径能否逐元素复现带 shared 的 identity。沿用已有 GitHub 回执推送流程。
