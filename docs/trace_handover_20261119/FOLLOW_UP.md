# 回执复核后的八卡任务

请按 [新版任务书](../trace_executor_20261119/README.md) 采样。上一轮提交的图来自旧归档，旧归档的计数窗口与时间戳窗口没有交集。此次交付以本次重排后的同一个单步 capture 为依据。

Routed GEMM 从第一个 CTA 领到有效、输入已就绪的 L1 任务开始。请在新版任务目录提交 `REPLY.md` 与两层的 Perfetto 和计数 JSON。每层需要八个 rank 的 token/M 块报告，并列出实际与 pull 重叠的 rank 和时长。
