#!/usr/bin/env bash
# Run from any directory inside the executor's private checkout.
set -euo pipefail
repo=$(git rev-parse --show-toplevel)
cd "$repo"
export CUDA_VISIBLE_DEVICES=''
export GC_MOE_SM100_TRACE=2
base=docs/sm100_megamoe
run=$(mktemp -d "$repo/.glm53-receipt.XXXXXXXX")
code=0
command -v python3 >/dev/null || { echo 'Python 3.9 or newer is required to collect the checkpoint.' >&2; exit 2; }
args=()
if [[ -n ${GLM_MODEL:-} ]]; then args+=(--model "$GLM_MODEL"); fi
if [[ -n ${GLM_PYTHON:-} ]]; then
    candidates=("$GLM_PYTHON")
    selection=specified
else
    candidates=(python python3
      /data0/home/admin/workspace/megamoe/native-v41-stsFS4w7/venv/bin/python
      /data0/home/admin/workspace/megamoe/venv/bin/python)
    selection=autodetected
fi
py=''
fallback=''
for candidate in "${candidates[@]}"; do
    if interpreter=$(command -v -- "$candidate") && "$interpreter" -c 'import sys; sys.exit(sys.version_info < (3,9))'; then
        if [[ -z $fallback ]]; then fallback=$interpreter; fi
        if "$interpreter" -c 'from importlib.metadata import distribution; distribution("vllm")' >/dev/null 2>&1; then
            py=$interpreter
            break
        fi
    fi
done
if [[ -z $py ]]; then
    py=${fallback:-$(command -v python3)}
    if [[ -n ${GLM_PYTHON:-} && -z $fallback ]]; then selection=specified_unavailable; fi
fi
git rev-parse HEAD > "$run/revision.txt"
"$py" scripts/sm100/glm53_checkpoint.py "${args[@]}" --out "$run/checkpoint.json" >"$run/probe.log" 2>&1 || code=$?
echo "$code" > "$run/probe-exit-code.txt"
# Retain raw data for every attempt, even if the interpreter could not start.
out="$base/glm53_ep8_trace2_results"
mkdir -p "$out"
attempt="$(date -u +%Y%m%dT%H%M%SZ)-${run##*.}"
mv -- "$run" "$out/$attempt"
dest="$out/$attempt"
reply="$base/REPLY_SM100_GLM53_EP8_TRACE2.md"
python3 - "$dest" "$py" "$selection" "$reply" <<'PY'
import json
from pathlib import Path
import sys

dest, interpreter, selection, reply = sys.argv[1:]
root = Path(dest)
try:
    data = json.loads((root / "checkpoint.json").read_text())
except (ValueError, OSError):
    data = {}
code = (root / "probe-exit-code.txt").read_text().strip()
version = data.get("loader", {}).get("packages", {}).get("vllm")
if code != "0" or data.get("status") != "metadata_collected":
    status = "采集失败，详见 checkpoint.json 的 error 或 probe.log"
elif not version or selection == "specified_unavailable":
    status = "权重元数据已采集，GLM 运行环境待定位"
else:
    status = "权重元数据与 vLLM 安装信息已采集，待构建者核对 GLM loader"
cfg = data.get("config", {})
cfg = cfg.get("text_config") or cfg
rows = [
    "# SM100 · GLM‑5.3 EP8 执行回执", "", status, "",
    "| 项目 | 结果 |", "|---|---|",
    ("任务提交", (root / "revision.txt").read_text().strip()),
    ("任务阶段", "CPU 采集，正式 GPU 运行 0 臂"),
    ("指定的 trace 模式", "2"),
    ("采集器退出码", code),
    ("Python", interpreter), ("解释器选择方式", selection), ("vLLM", version or "未检出"),
    ("checkpoint", data.get("model", "未解析")), ("架构", cfg.get("architectures")),
    ("hidden / MoE intermediate / top-k", [cfg.get(k) for k in ("hidden_size", "moe_intermediate_size", "num_experts_per_tok")]),
    ("routed 层数", len(data.get("configured_routed_layers", []))),
    ("分片数", data.get("shard_count")), ("本次证据目录", dest),
]
Path(reply).write_text("\n".join(f"| {r[0]} | {r[1]} |" if isinstance(r, tuple) else r for r in rows) + "\n")
PY
cat "$reply"
echo "证据与回执已就绪。按任务书提交到 main。"
