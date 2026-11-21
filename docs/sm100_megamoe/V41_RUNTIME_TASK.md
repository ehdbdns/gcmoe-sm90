# 【SM100】确认 V4.1-Flash 整模原生推理入口与安装来源

单层在线 ECO 已过。五臂端到端对比需要能启动 DeepSeekV41ForCausalLM 的同一引擎。用 CPU 查清现有 RTP-LLM 源码是否注册该模型、MegaMoE 执行器入口，以及 admin 可获得的 vLLM 版本。整个任务不占 GPU，也不修改共享机器上的任何已有环境或其他人的文件。

在 admin 项目 checkout 执行下列命令。保留路径、版本、匹配行号和 stderr；回执时去除任何私有包源凭证，不提交外部仓库源码。查不到的项目直接记为未找到，不要自行更换模型或安装依赖。

```bash
set -eo pipefail
git pull --ff-only origin main
repo=$(git rev-parse --show-toplevel)
out="$repo/docs/sm100_megamoe/v41_runtime_results"
mkdir -p "$out"
export CUDA_VISIBLE_DEVICES=''
source /home/admin/workspace/megamoe/env.sh
rtp=/data0/data3/wjh528431/work/RTP-LLM
git -c safe.directory="$rtp" -C "$rtp" rev-parse HEAD > "$out/rtp.sha"
(
  cd "$rtp"
  rg -n -i 'DeepseekV41ForCausalLM|deepseek[_-]?v4[._-]?1' rtp_llm --glob '*.py' | head -120 || true
  rg -n 'class MegaMoe|class MegaMoE|mega_moe_se' rtp_llm/models_py/modules/factory/fused_moe --glob '*.py' | head -65 || true
) > "$out/rtp.txt" 2>&1
python - <<'PY' > "$out/python.txt"
import importlib.metadata as m
import sys
print(sys.version, sys.executable)
for name in ("torch", "rtp-llm", "vllm", "sglang", "flashinfer-python", "transformers"):
    try:
        print(name, m.version(name))
    except m.PackageNotFoundError:
        print(name, "missing")
PY
# 使用公开 PyPI 索引查询 SM90 实际使用的 vLLM 0.31.0 的发行包是否存在。
python - <<'PY' > "$out/vllm-wheel.txt"
import json
import urllib.request
url = "https://pypi.org/pypi/vllm/0.31.0/json"
try:
    with urllib.request.urlopen(url, timeout=20) as response:
        data = json.load(response)
    print("version", data["info"]["version"], "python", data["info"]["requires_python"])
    for f in data["urls"]:
        print(f["filename"], f["digests"]["sha256"])
except Exception as error:
    print(type(error).__name__, str(error))
PY
```

如果 admin 已有可复用的 V4.1 推理 venv、容器镜像、启动脚本或本地 wheel，只记录 admin 可使用的路径、版本、无凭证的启动模板及该版本确实识别 V4.1 的证据。不要启动模型。提交本轮输出和 docs/sm100_megamoe/REPLY_V41_RUNTIME.md，并推送 main。由设计端根据证据编写整模集成和原生启动任务。
