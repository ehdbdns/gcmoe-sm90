# 【SM100】V4.1-Flash 的 GCMoE 与 MegaMoE 单层正确性对照

使用同一次进程中相同的 BF16 输入量化结果、真实权重、路由及 shared expert，比较两个后端的 EPLB+RR 与 ECO 参考配额输出，逐 rank 核对物理槽行数。验收记录逐元素差异和最大绝对差。遇到首个差异保留原始日志并回执。此轮只验证数值与分流。

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
# 在临时 venv 中继承已通过验证的 Torch 与 DeepGEMM，额外依赖只装进临时目录。
base_site=$(python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
python -m venv "$work/venv"
source "$work/venv/bin/activate"
python -m pip install nvidia-cutlass-dsl==4.8.1 cuda-python==13.2.0 2>&1 | tee "$work/deps.log"
python -c 'import pathlib,sysconfig,sys; (pathlib.Path(sysconfig.get_path("purelib"))/"mega_base.pth").write_text(sys.argv[1]+"\n")' "$base_site"
export PYTHONPATH="$work/code/source:$dg:${PYTHONPATH:-}"
python -c 'import torch,cutlass,deep_gemm,gc_moe; print(torch.__version__,gc_moe.__file__,cutlass.__file__,deep_gemm.__file__)' | tee "$work/import.log"
export SM100_V41_MODEL=/mnt/nas1/hf/DeepSeek-V4.1-Flash
python scripts/sm100/v41_shape.py "$SM100_V41_MODEL" | tee "$work/config.log"
python scripts/sm100/checkpoint_experts.py "$SM100_V41_MODEL" | tee "$work/checkpoint.log"
sha256sum "$SM100_V41_MODEL/model.safetensors.index.json" >> "$work/config.log"
export MASTER_ADDR=127.0.0.1 MASTER_PORT=8362
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
  python scripts/sm100/mega_rr_smoke.py --compare-gc --eco-reference --shared --checkpoint "$SM100_V41_MODEL"
  date -Is
) 2>&1 | tee "$work/smoke.log"
```

交付到 docs/sm100_megamoe/gc_parity_results/ 与 REPLY_GC_PARITY.md。记录所用提交、依赖、整组占用和释放时间、完整 stdout/stderr 和每个执行到的 rank 结果。成功时应有 48 行。编译以外的故障由设计端诊断，保持输入、验收条件及运行参数一致。
