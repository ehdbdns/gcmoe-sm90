# 7 段拆分说明（已废弃 — 见 GROUPED.md）

上一版把单层拆成 7 段（dispatch 拆 2 段、routed 拆 2 段），但**wait 1st M block 的切分是错的**：
用了 `pull_tl[11]` 当分母，而 `[11]` 是 `_dispatch_grid` 的到达计数器，不是时间戳。
结果 wait 被算成 79~162 µs，图上出现"pull 已经结束、GEMM 还没开始"的假象。

正确数值与方法见 **`GROUPED.md`**。本目录下的 `_deprecated/` 保留了旧文件，仅作历史对照，
**不要引用其中的 wait / GEMM 数值**。
