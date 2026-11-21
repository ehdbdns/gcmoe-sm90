# 【SM90】取得 V4.1 两级 top-k 的实际实现（CPU，和 step38 回执独立）

设计端阅读过 vllm 0.31.0 的 deepseek_v41/attention.py。该文件把选块和选 token 交给 SparseAttnIndexer/SparseMQAIndexer；原先只采集模型目录，没有这两个实现。请在上轮同一环境运行以下只读采集，不占 GPU、不启动模型、不打断进行中的 step38 任务。只取 vLLM 公开源码，不取任何本地权重、业务数据或环境配置。

```bash
set -eo pipefail
git pull --ff-only origin main
source config/experiment.env
source scripts/env.sh
export CUDA_VISIBLE_DEVICES=''
out=docs/trace_executor_20261119/sm90_indexer_source
mkdir -p "$out"
"$GC_ENV/bin/python" - "$out" <<'PY'
from pathlib import Path
from importlib.metadata import distribution
from hashlib import sha256
import json
import sys

dest = Path(sys.argv[1])
wheel = distribution("vllm")
root = Path(wheel.locate_file("vllm")).resolve()
manifest = []
for p in sorted(root.rglob("*.py")):
    data = p.read_bytes()
    if p.name not in ("sparse_attn_indexer.py", "sparse_mqa_indexer.py") and not any(
        x in data for x in (b"def _candidate_flags_kernel(", b"def _mask_candidates_kernel(",
                            b"def _block_scores_kernel(", b"def combine_topk_swa_indices(")):
        continue
    rel = p.relative_to(root)
    t = dest / rel
    t.parent.mkdir(parents=True, exist_ok=True)
    t.write_bytes(data)
    manifest.append(f"{sha256(data).hexdigest()}  {rel}")
(dest / "SHA256SUMS").write_text("\n".join(manifest) + "\n")
config = json.loads(Path("/home/admin/local_models/DeepSeek-V4.1-Flash/config.json").read_text())
fields = ("index_topk", "index_source_layer_ids", "kv_source_layer_ids",
          "candidate_source_layer_id", "candidate_block_size", "candidate_topk_blocks")
(dest / "shape.json").write_text(json.dumps({k:config.get(k) for k in fields}, indent=2))
print(wheel.version, *manifest, sep="\n")
PY
```

提交该目录、vLLM 版本及是否取全入口的回执 docs/trace_handover_20261119/REPLY_SM90_INDEXER_SOURCE.md 到 main；无需运行实验或修改模型代码。
