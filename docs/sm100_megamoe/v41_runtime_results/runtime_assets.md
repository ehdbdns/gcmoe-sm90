# Admin-readable V4.1 runtime assets

CPU/read-only inspection only. CUDA_VISIBLE_DEVICES was empty. No model was
started, no package was installed, and no shared file was modified.

## Model checkpoint

- Path: `/mnt/hf3fs/3fs/models/DeepSeek-V4.1-Flash`
- Exists and is admin-readable.
- `config.json`: `model_type = deepseek_v41`
- `config.json`: `architectures = ["DeepseekV41ForCausalLM"]`

## RTP-LLM V4.1 source and launch scripts

- Source root:
  `/data0/data3/wjh528431/work/dsv41-mega-gate-20261012/internal/github-opensource`
- Revision recorded by that run's `metadata.json`: `a5f09e3c4fd3de584f88541586e89780e11c5844`
- Exact model registration:
  - `rtp_llm/model_factory_register.py:207`
  - `rtp_llm/models/deepseek_v41.py:261`
- Model/runtime classes:
  - `rtp_llm/models/deepseek_v41.py:126`
  - `rtp_llm/models_py/model_desc/deepseek_v41_model.py:45`
  - `rtp_llm/models_py/model_desc/deepseek_v41_dspark_model.py:7`
- Existing launch scripts:
  - `/data0/data3/wjh528431/work/dsv41-mega-gate-20261012/internal/prefill-41.sh`
  - `/data0/data3/wjh528431/work/dsv41-mega-gate-20261012/internal/decode-41.sh`
- Script entry flags include `--model_type deepseek_v41`,
  `--checkpoint_path "${MODEL_PATH}"`, `--tokenizer_path "${MODEL_PATH}"`,
  `--load_method fastsafetensors`, `--tp_size 4`, `--ep_size 4`,
  and `--world_size 4`.

The source's `.git`, `bazel-bin`, and `bazel-out` links point at
`/home/wjh528431/...`, which is not mounted in this admin namespace. The source
files are readable, but a fresh build/runtime path must not assume those links
still resolve.

## Container runtime

- Image: `rtpllm-migration:dsv4_20261012_200340`
- Image ID: `sha256:82f91edbf116a42f2dadac24137b379c7481fa45860edaab285bf043015a4472`
- Python: `/opt/conda310/bin/python`, 3.10.9
- `torch`: 2.11.0
- Installed `rtp-llm`: missing
- Installed `vllm`: missing

The image alone does not contain an RTP-LLM installation. With the historical
source bind-mounted read-only at `/src`, Python found all three V4.1 modules:

- `rtp_llm` -> `/src/rtp_llm/__init__.py`
- `rtp_llm.models.deepseek_v41` -> `/src/rtp_llm/models/deepseek_v41.py`
- `rtp_llm.models_py.model_desc.deepseek_v41_model` ->
  `/src/rtp_llm/models_py/model_desc/deepseek_v41_model.py`

No model or server was launched. This is an install-source candidate, not a
fully qualified end-to-end runtime.

Credential-free launch template (template only; not executed):

```bash
SOURCE=/data0/data3/wjh528431/work/dsv41-mega-gate-20261012/internal/github-opensource
MODEL=/mnt/hf3fs/3fs/models/DeepSeek-V4.1-Flash
IMAGE=rtpllm-migration:dsv4_20261012_200340

# Template only. The complete 41 environment and a valid built/runfiles tree
# must be supplied; this abbreviated form has no registry credentials.
docker run --rm --gpus '"device=0,1,2,3"' --network host --ipc host \
  --shm-size 64g \
  -v "$SOURCE:/workspace/github-opensource:ro" \
  -v "$MODEL:/models/DeepSeek-V4.1-Flash:ro" \
  -w /workspace/github-opensource \
  -e RTP_LLM_BASE_PATH=/workspace \
  -e MODEL_PATH=/models/DeepSeek-V4.1-Flash \
  "$IMAGE" /bin/bash -lc 'source scripts/deepseek_v41_runtime.sh && \
    exec /opt/conda310/bin/python -m rtp_llm.start_server \
      --model_type deepseek_v41 \
      --checkpoint_path "$MODEL_PATH" \
      --tokenizer_path "$MODEL_PATH" \
      --load_method fastsafetensors \
      --tp_size 4 --ep_size 4 --world_size 4'
```

## vLLM source and package availability

- Local source: `/data0/data3/wjh528431/work/vllm`
- Git revision: `36f397281295f5b58852a7eaf429b90348161079`
- Package metadata version: `0.1.dev1+g36f397281.precompiled`
- Exact registry entry:
  `vllm/model_executor/models/registry.py:371` maps
  `DeepseekV41ForCausalLM` to `vllm.models.deepseek_v41`.
- V4.1 model tree: `vllm/models/deepseek_v41/` (NVIDIA and AMD backends).

The source-level V4.1 support is present, but its `.venv/bin/python` is a
dangling link to `/home/wjh528431/.local/share/uv/...`, which is absent in the
admin namespace. No local vLLM wheel was found under the searched admin and
`wjh528431` trees.

Public PyPI provides `vllm==0.31.0` wheels for x86_64 and aarch64 plus an sdist;
filenames and SHA256 values are in `vllm-wheel.txt`. They were not downloaded or
installed.
