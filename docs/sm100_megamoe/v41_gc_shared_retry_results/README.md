# V41 GCMoE shared-weight retry

Outcome: `FAILED_DURING_ENGINE_PROFILE`. The upstream `preserve_loader_weight` fix removed the prior ModelOpt assertion, and all eight workers installed the GC adapter. Engine initialization then failed during the profile/dummy run while constructing the GCMoE JIT compiler identity: `EnvironmentVarManager` from the current `nvidia-cutlass-dsl==4.7.1` has no `cache_key_str` attribute.

No prefix inference or top10 output was produced. See `../REPLY_V41_GC_SHARED_RETRY.md`, `first-worker-traceback.txt`, `adapter-install-evidence.txt`, and `applied-fix.diff`.
