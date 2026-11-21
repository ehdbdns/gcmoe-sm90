# 【SM100】DeepSeek-V4.1-Flash 第 0 层真实 routed 权重八卡 RR 回执

执行者：SM100。未执行任何 SM90 任务书，也未混入 SM90 结果。

## 结论

本轮 V4.1 的八卡用例**未启动**：任务自带的前置 `v41_shape.py` 在占卡前失败，因此没有 V4.1 的 8-rank 结果，也没有为 V4.1 分配 GPU。失败的首条异常已原样保存在 `checkpoint_rr_results/config.log`。

在任务书更新前已经启动的旧模型 V4-Flash 用例已完成，结果单独标注并原样保留在 `checkpoint_rr_results/legacy_v4_flash/`；它不是 V4.1 结果，不能作为本轮验收依据。

执行 SHA：`0d7177b3c9ccb2e91c7e7754884963733e1fe76e`

## V4.1 模型路径与来源

- 实际路径：`/mnt/nas1/hf/DeepSeek-V4.1-Flash`
- 目录可读，README 模型卡标题为 `DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression`，并链接 `https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash`。
- `config.json` SHA-256：`8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879`
- `model.safetensors.index.json` SHA-256：`74b0686a3d2891980d5e303251b075a3bccae2c2ff650747db2620a649b98fa8`
- 架构名为 `DeepseekV41ForCausalLM`。实际维度放在 `config.json` 的 `text_config` 中：`hidden_size=5120`、`moe_intermediate_size=2304`、`n_routed_experts=384`、`num_experts_per_tok=6`、`num_hidden_layers=40`，与任务期望的 V4.1 形状一致。

## 首条异常

`python scripts/sm100/v41_shape.py "$SM100_V41_MODEL"` 退出码 1，首条异常为：

```text
ValueError: Incompatible with SM90 V4.1-Flash: ['DeepseekV41ForCausalLM'], {'hidden_size': None, 'moe_intermediate_size': None, 'n_routed_experts': None, 'num_experts_per_tok': None, 'num_hidden_layers': None}
```

原因是该脚本从根 `config.json` 直接读取五个维度键，而实际 V4.1 配置把它们放在 `text_config` 下；这不是模型路径缺失，也不是 GPU 或 DeepGEMM 问题。按任务“只执行给定步骤、仅允许有记录的语义等价编译修复”的限制，我没有修改脚本、输入、精度或通过条件，也没有绕过该 guard 启动 GPU 用例。

建议设计端将守卫改为在存在 `text_config` 时读取 `cfg['text_config']`（否则读取 `cfg`），再重新发布或明确允许该兼容修复。修复后当前模型的实际值已满足 `5120/2304/384/6/40`，可继续执行同一八卡用例。

## V4.1 checkpoint 元数据

单独运行 `checkpoint_experts.py /mnt/nas1/hf/DeepSeek-V4.1-Flash` 成功，退出码 0，得到第 0 层代表项：

| key | dtype | shape |
|---|---|---|
| `layers.0.ffn.experts.0.w1.weight` | `I8` | `[2304, 2560]` |
| `layers.0.ffn.experts.0.w1.scale` | `F8_E8M0` | `[2304, 160]` |
| `layers.0.ffn.experts.0.w2.weight` | `I8` | `[5120, 1152]` |
| `layers.0.ffn.experts.0.w2.scale` | `F8_E8M0` | `[5120, 72]` |
| `layers.0.ffn.experts.0.w3.weight` | `I8` | `[2304, 2560]` |
| `layers.0.ffn.experts.0.w3.scale` | `F8_E8M0` | `[2304, 160]` |

这说明 V4.1 checkpoint 的 routed 权重确实是对应新形状的 FP8/UE8M0 打包布局；本轮没有把它送入 GPU smoke。

## 旧模型 V4-Flash 结果（已保留，非本轮验收）

旧任务在 V4.1 更新前已经启动，使用的模型是 `/data0/data3/DeepSeekV4-Flash`（符号链接到共享盘），任务提交为 `498f21de7acd46c56f979c6f5b46aa4e4eef8e55`。该运行：

- 8 张 SM100 的分配时间：`2026-11-21T15:20:40+08:00`
- 释放时间：`2026-11-21T15:22:58+08:00`
- `smoke_exit=0`
- 16/16 条 rank 记录通过：identity 与 EPLB+RR 均为 `different_elements=0`、`valid_output=true`、`slot_count_match=true`。
- 完整 rank 行、首条 MLOPart 非致命告警、分配 UUID 与退出信息在 `checkpoint_rr_results/legacy_v4_flash/smoke.log`。

该结果仅用于保留旧任务现场，已明确标注为 V4-Flash，不可替代 V4.1。

## 附件

- `checkpoint_rr_results/config.log`：V4.1 守卫首条异常及 config/index SHA-256。
- `checkpoint_rr_results/checkpoint.log`：V4.1 第 0 层 routed 元数据。
- `checkpoint_rr_results/smoke.log`：V4.1 未占卡、未运行的明确记录。
- `checkpoint_rr_results/legacy_v4_flash/checkpoint.log`、`smoke.log`：旧 V4-Flash 运行原始日志。
- `checkpoint_rr_results/SHA256SUMS`：上述附件校验和。

## 请求

请设计端确认并发布 `v41_shape.py` 的配置读取兼容修复；在修复前我不占用八卡、不伪造 V4.1 结果。收到允许修复或更新后的任务书后，我会立即重跑 V4.1 八卡用例并推送完整回执。
