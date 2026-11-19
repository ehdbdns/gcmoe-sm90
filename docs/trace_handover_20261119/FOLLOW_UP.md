# 当前八卡任务入口

请严格执行 [当前任务书](../trace_executor_20261119/README.md)。本目录保留上轮交接与回执。新回执提交到当前任务书同目录的 `REPLY.md`。

执行者的代码修复权限仅限保持原语义的最小编译兼容性修改。运行、采集或验收失败时，请按当前任务书回传证据。

## 回执 ce91bb1 后续

**当前只做原始 trace 诊断。** 五角色编译已经通过，八份 torch trace 已导出。此次无需占卡或重新加载模型，也无需改代码。请快进到包含诊断脚本的 `origin/main`，从仓库根目录运行下面这条命令，用上次保留的原始目录。

```bash
python3 scripts/trace_layer/inspect_aligned_trace.py /tmp/gcmoe-aligned-fSc3IBre/raw   | tee /tmp/gcmoe-aligned-fSc3IBre/trace-summary.json
```

将 `trace-summary.json` 和回执更新提交到 [任务书目录](../trace_executor_20261119/)，保留原始八份 trace。回执附上采样前后几行完整日志，包括最后两个 flush iteration、`[trace]` 起止和异常。若文件已不存在，回报缺失路径。若同一 capture ID 下每种角色有 40 个 kernel，说明设备快照范围比 torch trace 多记了调用，需要定位边界。若每种角色有 120 个 kernel，则继续依据 trace 判断三组调用的来源。图和计数的 40 层验收仍适用。

后续获准重跑时，新增的 `capture.invalid.json` 会在可读取的环形范围内保留不合格采样的 epoch、原始打点与每槽计数，用于定位问题。此文件只用于诊断。
