# 【SM100】修正副本表结束标记后复测在线 ECO

在六组单层对照后加一组 GCMoE GPU 在线拍卖。使用相同输入与逻辑专家，按每卡 single-copy 前缀重排本卡物理槽和对应权重。核对 GPU 计划状态、真实接收行数以及含 shared 的 BF16 输出。

本轮修正主机侧 SM100 副本表空项编码。保持上一轮 7 组、输入、验收阈值、相同环境和步骤；按照上一轮回执可复用或重建同版本临时 venv。原始失败记录保留。

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
# 复用上一份回执验证通过的临时 venv。
source /data0/home/admin/workspace/megamoe/rr-smoke-ezB7Fty7/venv/bin/activate
export GC_MOE_NVCC="$(command -v nvcc)"
test -x "$GC_MOE_NVCC"
export GC_MOE_ECO_CACHE_DIR="$work/eco-cache"
export PYTHONPATH="$work/code/source:$dg:${PYTHONPATH:-}"
python -c 'import torch,cutlass,deep_gemm,gc_moe; print(torch.__version__,gc_moe.__file__,cutlass.__file__,deep_gemm.__file__)' | tee "$work/import.log"
export SM100_V41_MODEL=/mnt/nas1/hf/DeepSeek-V4.1-Flash
python scripts/sm100/v41_shape.py "$SM100_V41_MODEL" | tee "$work/config.log"
python scripts/sm100/checkpoint_experts.py "$SM100_V41_MODEL" | tee "$work/checkpoint.log"
sha256sum "$SM100_V41_MODEL/model.safetensors.index.json" >> "$work/config.log"
export MASTER_ADDR=127.0.0.1 MASTER_PORT=8363
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
  python scripts/sm100/mega_rr_smoke.py --live-eco --compare-gc --eco-reference --shared --checkpoint "$SM100_V41_MODEL"
  date -Is
) 2>&1 | tee "$work/smoke.log"
```

交付到 docs/sm100_megamoe/online_gc_eco_sentinel_results/ 与 REPLY_ONLINE_GC_ECO_SENTINEL.md。记录所用提交、依赖、整组占用和释放时间、完整 stdout/stderr 和每个执行到的 rank 结果。成功时应有 56 行。编译以外的故障由设计端诊断，保持输入、验收条件及运行参数一致。
