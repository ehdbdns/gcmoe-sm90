# 【SM100】8 卡 V4.1-Flash 整模原生 smoke

采用与 SM90 相同版本的 vLLM 0.31.0，使用独立 venv、已有本地权重、原生自动后端、TP8/EP8、单请求贪心解码 4 token。此轮验证整模入口和生成能力，无端到端性能验收。环境保留给后续两种 MoE 集成复用。只允许等价编译修复，其他故障附日志回执。

```bash
set -eo pipefail
git pull --ff-only origin main
repo=$(git rev-parse --show-toplevel)
work=$(mktemp -d /data0/home/admin/workspace/megamoe/native-v41-XXXXXXXX)
cd "$repo"
git rev-parse HEAD > "$work/source.sha"
source /home/admin/workspace/megamoe/env.sh
unset PYTHONPATH
python -m venv "$work/venv"
source "$work/venv/bin/activate"
export PIP_INDEX_URL=https://pypi.org/simple
python -m pip install vllm==0.31.0 2>&1 | tee "$work/install.log"
python -m pip freeze > "$work/freeze.txt"
export SM100_V41_MODEL=/mnt/nas1/hf/DeepSeek-V4.1-Flash
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
  python - <<'PYTHON'
import json
import os
import torch
from vllm import LLM, SamplingParams

assert torch.cuda.device_count() == 8
assert all(torch.cuda.get_device_capability(i) == (10, 3) for i in range(8))
model = LLM(model=os.environ["SM100_V41_MODEL"], tensor_parallel_size=8,
            enable_expert_parallel=True, enable_prefix_caching=False,
            enforce_eager=True, max_model_len=2048, max_num_seqs=8,
            max_num_batched_tokens=2048, seed=0, dtype="bfloat16",
            async_scheduling=False, gpu_memory_utilization=0.8)
prompt = {"prompt_token_ids": [100, 200, 300, 400]}
out = model.generate([prompt], SamplingParams(temperature=0, max_tokens=4, ignore_eos=True))
ids = list(out[0].outputs[0].token_ids)
assert len(ids) == 4
print(json.dumps(dict(prompt=prompt, generated_ids=ids)), flush=True)
PYTHON
  date -Is
) 2>&1 | tee "$work/model.log"
```

共享机器只在确认这组 8 卡无人占用或预约时启动。保存安装日志、版本、模型日志、实际 native MoE backend、八卡释放时间、工作目录路径和输出 ID 到 docs/sm100_megamoe/v41_native_results/。回执 docs/sm100_megamoe/REPLY_V41_NATIVE.md。失败时原样记录首个错误与退出码；不得更换模型、卡数、框架或后端。提交 main。
