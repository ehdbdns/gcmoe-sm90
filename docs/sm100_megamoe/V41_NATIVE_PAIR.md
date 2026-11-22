# SM100 · BS1024 原生 GC 与原生 Mega 同输入对照

本轮仅启动两次模型，依次为原生 MegaMoE、原生 GCMoE。取代 BS1024 五臂复测的续跑安排。目标是判定相同 BF16 隐藏向量、top-6 ID、权重、模型权重和原生布局下，两后端完成 40 次 MoE forward 的设备耗时。结果适用于这档负载和已采样的 decode 步。

在指定八卡获得预约、均无人使用后，在干净 main checkout 的持久终端执行。沿用上轮模型、venv 和八卡组锁，确保 sparse checkout 包含 scripts/sm100、source、integration 及上轮使用的 BS256 prompt 输入归档。

```bash
git switch main
git pull --ff-only origin main
work=/data0/home/admin/workspace/megamoe/v41-native-pair-bs1024-01
mkdir -p "$work"
set -o pipefail
bash scripts/sm100/run_five.sh "$work" native-pair 2>&1 | tee "$work/launcher.log"
```

Mega 在满批 decode 的第 0、8、16 步记录各层各 rank 的实际输入与实时输出，两臂均从同一文件重新加载。每个样本把真实的 40 个专家模块按层序捕获为一张图，每层读取该层固定输入，包含输入准备、routed 和 fused shared 专家及 combine。预热后每组重放 20 次，采 15 组 CUDA event 时间，每组取最慢 rank 的每次重放耗时，再取中位数。采集、存盘、GPU 间验证、输出比对均在计时外。模型加载、attention 和采样不在此项时间内。

回执分别列三步的两臂绝对耗时、GC 相对耗时变化、各组分布、输入摘要相等情况及输出差异。两个后端各自都必须通过 eager/replay 一致性检查。相对同一实时参考逐元素校验通过的样本才作严格性能判断；三个样本分别报告，如有混合方向或接近测量波动，照实记录。保留所有 rank 的不同元素数、最大绝对误差。每个后端仅启动一次，组内离散度反映本次运行波动。

提交 `docs/sm100_megamoe/REPLY_V41_NATIVE_PAIR.md` 和同目录 `v41_native_pair_results/`，包含 comparison.json、两份臂 JSON、日志、版本、测量 SHA 和退出码。执行者仅可做保持语义的编译修复并附 diff；其他失败保留首个完整堆栈和已有证据。沿用共享八卡清理与释放检查，保留输入及参考文件用于复核，记录其位置并推送 main。
