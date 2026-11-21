# SM100 · MegaMoE 与 GCMoE 整模冗余专家 decode 验证

使用当前 vLLM 的内置 EPLB 映射验证整模的冗余权重和连续 decode。每层 384 个逻辑专家、448 个物理槽，每卡 56 槽。本次只验证初始放置，两次启动，各用相同 prompt 贪心生成连续四步。正式五臂的 DeepInfer 放置、RR/ECO 分发和计时另行验收。

1. 拉取 main 并记录 SHA。执行 [V41_GC_GENERATE.md](V41_GC_GENERATE.md) 的 Bash 代码块两次，每次一个全新 work 目录，各自保留锁、内存监测和退出码。只在已登记八卡均无人占用且无人预约时持锁运行，繁忙就等待。保留原 checkpoint、venv、TP8/EP8。
2. 第一次将模型命令的参数改为 `--backend deep_gemm_mega_moe --eplb-decode`，第二次改为 `--backend gc_moe --eplb-decode`。两次均在启动 Python 前执行以下命令。
   ```bash
   export GC_MOE_PTXAS="$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cu13/bin/ptxas"
   "$GC_MOE_PTXAS" --version
   ```
3. 分别记录完整模型日志、退出码、融合与布局检查结果、四个输出 ID、释放时间。两臂均成功时，逐层对比 replica_layers 中的 40 个哈希，逐步对比四个 ID，报告首个差异；此轮不报告性能。遇到共同的 EPLB 初始化失败即停，保留首个完整工作进程调用栈及相关安装版本源码上下文。仅允许等价编译修复并附 diff，其他故障原样回执。
4. 新产物写入 docs/sm100_megamoe/v41_eplb_decode_results/，在 REPLY_V41_EPLB_DECODE.md（同级 docs/sm100_megamoe 目录）记录实际启动数、测量 SHA 和每臂结果，提交推送 main。
