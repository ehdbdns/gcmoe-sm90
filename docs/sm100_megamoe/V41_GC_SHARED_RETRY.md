# SM100 · 修正 shared expert 权重布局后复测 GCMoE

本轮一次整模启动。沿用 V4.1-Flash、TP8/EP8、已验证 venv、同一 checkpoint 和四个固定前缀。目标为验证量化后处理、共享专家融合和首次实际推理。

1. 在 SM100 仓库执行 `git pull --ff-only origin main`，用 `git rev-parse HEAD` 记录测量版本，确认 `scripts/sm100/v41_gc_worker.py` 包含 `preserve_loader_weight`。
2. 完整执行 [V41_GC_GENERATE.md](V41_GC_GENERATE.md) 的 Bash 代码块，包括创建全新工作目录、设备绑定和资源释放记录。先确认登记的八张卡均无人使用、无人预约；有人使用或预约就等待。仅使用这八张卡。模型入口固定为 `--backend gc_moe --prefix-audit`。
3. 成功时，从 model-full.log 提取 8 个不同 PID 的安装记录、每 rank 的 40 个专家层、每层 sequence_parallel 和 fused_shared；逐一核对四个固定前缀及其生成 ID、top10 logprob，与上轮 MegaMoE 的相同前缀并列。明确标出融合回退或数值分歧。记录退出码与释放时间。
4. 失败时保留首个 worker 的完整调用栈和退出码。执行者只可做保持语义的编译修复并附 diff；其他错误原样回执。特别保留 ModelOpt 的原断言和共享专家融合设置。
5. 新产物写入 `docs/sm100_megamoe/v41_gc_shared_retry_results/`，回执写入 `docs/sm100_megamoe/REPLY_V41_GC_SHARED_RETRY.md`。附测量版本、环境、模型日志和校验和，提交并推送 main。此前回执与产物保留。
