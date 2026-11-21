# SM100 · DSL 4.7.1 缓存键修正后的整模验证

执行一次冷启动，使用原 vLLM 环境和本地 V4.1-Flash 权重，验证上次中断的 JIT 路径，再核对四个固定前缀。

1. 拉取 main，记录测量 SHA。沿用 [V41_GC_GENERATE.md](V41_GC_GENERATE.md) 的完整 Bash 代码块创建新工作目录。只取登记的八卡，确认无人预约、无人使用；否则等待，持有原任务的同一把锁直到退出。
2. 在该代码块的锁内、启动模型之前、已设置 CUDA_VISIBLE_DEVICES 后运行以下预检。沿用该代码块的 set -eo pipefail；如果预检失败，保留 traceback，直接回执。

   ```bash
   python - <<'PY' 2>&1 | tee "$work/jit-preflight.log"
   import torch
   from importlib.metadata import version
   from gc_moe.runtime.jit import device_compile_target
   from gc_moe.jit_kernels.impls.sm100_fp8_fp4_gc_moe import _compiler_identity
   assert torch.cuda.device_count() == 8
   cap = torch.cuda.get_device_capability()
   assert cap == (10, 3)
   ident = _compiler_identity(device_compile_target(cap))
   assert ident.environment
   print(version("nvidia-cutlass-dsl"), ident.target, ident.tool_version)
   PY
   ```

3. 预检通过后继续原代码块，以 gc_moe --prefix-audit 启动一次模型。核对 8 个 worker、每 rank 40 层的 sequence_parallel 和 fused_shared，逐前缀并列 GC 与 Mega 的 top1/top10。记录真实退出码、首个完整错误栈和八卡释放情况。仅允许保持语义的编译修复，附精确 diff。其他失败直接回执，保留原软件版本和测量配置。
4. 将新日志、输出、校验和写入 docs/sm100_megamoe/v41_gc_jit_retry_results/，本轮回执写入 docs/sm100_megamoe/REPLY_V41_GC_JIT_RETRY.md，提交并推送 main。
