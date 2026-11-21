# 【SM100】DeepGEMM MegaMoE 接入逐层 EPLB + RR 的八卡正确性验证

对应环境回执 7c2fa40。先核实本机实际 checkpoint 的 routed 权重及 scale，再用已有 DeepGEMM 在目标形状验证物理槽接入。此轮执行一次八卡合成正确性用例，含原生槽与 EPLB+RR 两次 fused 调用。不开模型，无需安装框架。保留旧环境及其他用户的 checkout。

## 执行

在 admin 专用 checkout 中同步最新 main。下面脚本的 CPU 部分可在等卡期间执行。使用环境回执中的同一单节点 8 卡；遵守[【SM100】共用机八卡规则](ENV_TASK.md#共用机器的八卡使用规则)。预约状态不明确时继续等待。协作锁只约束遵循此约定的任务，启动前还须核实他人预约和实际 GPU 进程。

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
python scripts/sm100/checkpoint_experts.py /data0/data3/DeepSeekV4-Flash > "$work/checkpoint.log" 2>&1 || cat "$work/checkpoint.log"
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
  python scripts/sm100/mega_rr_smoke.py
  date -Is
) 2>&1 | tee "$work/smoke.log"
```

每臂每 rank 核对物理槽实际接收行数；要求 BF16 输出有限且非全零，RR 对 identity 要求元素逐一相等。此处合成 FP8 激活与 FP4 routed 权重，不包含 shared expert。出现数值差异、启动失败或版本不符就保存首条异常和日志。只可做必要的语义等价编译修复，单独提交并记录，不得更改输入、精度或通过条件。

回执提交到 docs/sm100_megamoe/REPLY_RR_SMOKE.md，附 checkpoint.log、smoke.log、执行 SHA、分配与释放时间、两次 kernel 的全部 rank 结果。补充可由 admin 合法使用的数据集路径和已有 RTP-LLM 可运行入口；如仍缺失则注明。使用[回执推送流程](../trace_executor_20261119/GITHUB_PUSH_GUIDE.md)将这些文件推送到 main。
