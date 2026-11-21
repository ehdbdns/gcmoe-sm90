# 【SM90】取回执行环境中 V4.1 prefill 的实际调用代码

本轮只读取已安装 Python 源码，不启动模型，无需 GPU。上一轮两个 vanilla 在 step 13 / layer 20 的 indexer 返回摘要相同，而 wo_b 输入摘要不同。设计端需要核对中间的稀疏索引选择、有效长度、缓存读取、attention 与输出投影调用顺序，才能选择下一处探针。

在与上一轮相同的 shell 和仓库执行。只提交下面脚本选择的 vLLM 开源模型模块及摘要。不要上传环境变量、模型权重或其他本机文件。任何错误原样回执，不需修改环境或算法。

```bash
set -eo pipefail
git pull --ff-only origin main
source config/experiment.env
source scripts/env.sh
export CUDA_VISIBLE_DEVICES=''
out=docs/trace_executor_20261119/sm90_prefill_source
mkdir -p "$out"
"$GC_ENV/bin/python" - "$out" <<'PY'
from pathlib import Path
from importlib.metadata import distribution
import hashlib
import sys

wheel = distribution("vllm")
root = Path(wheel.locate_file("vllm/models/deepseek_v41")).resolve()
entry = root / "nvidia/flashmla.py"
if not entry.is_file():
    raise FileNotFoundError(entry)
dest = Path(sys.argv[1])
lines = [f"vllm {wheel.version}"]
for path in sorted(root.rglob("*.py")):
    data = path.read_bytes()
    if not any(key in data for key in (b"self.wo_b(", b"_forward_prefill", b"DeepseekV4FlashMLAAttention", b"flash_mla_sparse_fwd")):
        continue
    relative = path.relative_to(root)
    target = dest / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    lines.append(f"{hashlib.sha256(data).hexdigest()}  {relative}")
(dest / "manifest.txt").write_text("\n".join(lines) + "\n")
assert (dest / "nvidia/flashmla.py").is_file()
PY
cat "$out/manifest.txt"
```

在 docs/trace_handover_20261119/REPLY_SM90_PREFILL_SOURCE.md 记录任务 SHA、实际版本和文件清单，并与上述目录一同推送 main。无需运行 A/A 或性能实验。
