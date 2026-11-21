# SM100 V4.1-Flash native smoke result

Result: **FAIL**.

The vLLM 0.31.0 environment installed successfully and the exact V4.1 architecture was resolved as `DeepseekV41ForCausalLM`. With auto backend selection, the native MoE backend resolved to `FLASHINFER_TRTLLM_MXFP4_MXFP8`, not the separate DeepGEMM MegaMoE path.

The first task-heredoc launch failed before workers started because vLLM's spawn re-import looked for `<stdin>` as a file. The second launch used an identical file-based entry point with unchanged model, sampling, TP8/EP8, eager mode and backend options. It loaded checkpoint shards to 32/48 over 57:12, then EngineCore initiated shutdown because a background worker failed. vLLM emitted no worker stack trace. The process exited 1 without producing `generated_ids`.

See `REPLY_V41_NATIVE.md` for the receipt and the files in this directory for raw evidence. `model.heredoc-failure.log` is the original first failure; `model.log` is the complete second run. `SHA256SUMS` covers every file except itself.

