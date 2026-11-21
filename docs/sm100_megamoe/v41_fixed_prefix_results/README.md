# V41 fixed-prefix audit artifacts

- `auto.log`, `deep_gemm_mega_moe.log`: final successful backend logs.
- `prefix-audit.json`, `prefix-audit.txt`: parsed four-prefix comparison and per-rank layer evidence.
- `auto-prefix-top10.json`, `auto-fusion-counts.json`: compact readable summaries.
- `attempt1-official/`: original task harness; both backends failed at `apply_model` function serialization.
- `attempt2-serialization-retry/`: official insecure-serialization switch retry; pickle identity still failed and an external user occupied all eight GPUs.
- `attempt3-audit-fix/`: successful run after moving the introspection function into an importable module. Model, backend and inference entry were unchanged.
- `cgroup-summary.txt`, `memory.*`, `journal-*`: resource and system-evidence snapshots.
- `gpu-release.txt`, `gpu-release-verification.txt`: release timestamp and a later empty compute-app check.
- `launcher.sh`, `versions.txt`, `config.log`, `source.sha`: reproducible launch and environment record.

For `auto`, `fused_shared=null` means the observed wrapper class has no such attribute; actual fused-layer count is `N/A`, not zero. For `deep_gemm_mega_moe`, all 8 ranks report `DeepseekV4MegaMoEExperts`, `sequence_parallel=true`, and `fused_shared=true` on 40/40 layers.
