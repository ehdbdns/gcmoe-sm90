# SM100 V4.1 runtime preflight

Date: 2026-11-21 (Asia/Shanghai)

Constraints honored: CPU-only, `CUDA_VISIBLE_DEVICES=''`, no package install, no
model launch, no shared-environment modification, and no external source body
committed. Source evidence is recorded as path and line number only.

## Key results

- Current RTP checkout `c207b52b2ddce5acf964514968717a748053973b` does **not**
  register `DeepseekV41ForCausalLM`. The requested source command also hit
  `rtp_llm: No such file or directory`, because the checkout stores code under
  `githubopensource/rtp_llm` and `internal_source/rtp_llm`, not at the root.
- The current checkout does register `DeepseekV4ForCausalLM` at
  `githubopensource/rtp_llm/model_factory_register.py:204` and
  `githubopensource/rtp_llm/models/deepseek_v4.py:1173`.
- An admin-readable historical checkout at
  `/data0/data3/wjh528431/work/dsv41-mega-gate-20261012/internal/github-opensource`
  does register the exact V4.1 entry at `rtp_llm/model_factory_register.py:207`
  and `rtp_llm/models/deepseek_v41.py:261`.
- The V4.1 checkpoint exists at
  `/mnt/hf3fs/3fs/models/DeepSeek-V4.1-Flash`; its config declares
  `model_type=deepseek_v41` and `architectures=["DeepseekV41ForCausalLM"]`.
- Local vLLM source at `/data0/data3/wjh528431/work/vllm` recognizes V4.1 at
  `vllm/model_executor/models/registry.py:371`; its package metadata reports
  `0.1.dev1+g36f397281.precompiled`, but its venv Python target is dangling.
- The current admin Python environment lacks `rtp-llm`, `vllm`, `sglang`,
  `flashinfer-python`, and `transformers`; see `python.txt`.
- Public PyPI has `vllm==0.31.0` wheels for x86_64/aarch64 and an sdist; see
  `vllm-wheel.txt`. Nothing was installed.
- The `rtpllm-migration:dsv4_20261012_200340` image has Python 3.10.9 and torch
  2.11.0 but no installed RTP-LLM/vLLM. With historical V4.1 source mounted
  read-only, `find_spec` resolved all expected V4.1 modules. The newest SM10x
  image (`2026_11_19`) has `rtp-llm 0.2.0` but no exact V4.1 registration.

## Runtime conclusion

For RTP-LLM, the exact V4.1 entry exists in admin-readable historical source and
has working launch-script flags, but the current checkout, current Python
environment, and current SM10x image do not provide it directly. For vLLM, the
exact V4.1 model registry and implementation source exist locally, and 0.31.0
wheels are publicly obtainable, but no runnable local V4.1 venv was established.

`runtime_assets.md` records paths, versions, and a credential-free launch
template. No end-to-end inference was attempted or claimed.

## Files

- `rtp.txt`: sanitized source-search result with line numbers.
- `rtp_command.txt`: exact requested source-command stdout/stderr.
- `mega_moe_matches.txt`: first 65 MegaMoE path:line matches.
- `python.txt`: current Python/package probe.
- `vllm-wheel.txt`: public PyPI 0.31.0 distribution probe.
- `vllm_local.txt`: local vLLM evidence.
- `container_inspect.txt`: read-only container/runtime evidence.
- `runtime_assets.md`: reusable asset inventory and launch template.
