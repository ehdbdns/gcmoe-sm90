# V41 GCMoE generation attempt

Outcome: `FAILED_BEFORE_INFERENCE`. The `gc_moe` worker adapter installed in four workers, then engine initialization failed during vLLM weight post-processing with the `modelopt.py` MXFP8 shape/dtype assertion. Backend and launcher exit codes are `1`; no prefix inference or top10 output was produced.

See `../REPLY_V41_GC_GENERATE.md`, `first-error.txt`, and `adapter-install-evidence.txt`. The full backend log is `model-full.log`; resource, environment and release evidence are in the remaining files.
