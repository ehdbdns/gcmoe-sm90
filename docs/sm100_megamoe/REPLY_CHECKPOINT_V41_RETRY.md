# 【SM100】DeepSeek-V4.1-Flash 第 0 层真实 routed 权重八卡 RR 重试回执

执行者：SM100。未执行任何 SM90 任务书，也未混入 SM90 结果。

## 结论

通过。使用同一组 8 张 SM100、DeepSeek-V4.1-Flash 第 0 层真实 packed routed 权重，完成 identity 与逐层 EPLB + RR 两臂对照：

- identity：8/8 rank 通过；BF16 输出有限且非全零；物理槽接收行数与路由计算一致。
- EPLB + RR：8/8 rank 通过；相对 identity 的元素差异为 0；BF16 输出有限且非全零；物理槽接收行数与路由计算一致。
- 整个 case 退出码 0；未做代码、输入、精度或通过条件修改。

执行 SHA：

- 测试仓库：`b0bcfb96b843c865192e2fd585d335a5daa27550`
- 任务书 blob：`17e36680eefd32fc34479df8097db821c54a0afa`
- DeepGEMM：`5081f8513af57e61696de4f4f04b0e707ffb503e`
- 测试脚本 blob：`v41_shape.py=0b655f2549cdf210e554324837877d3cc1184050`，`mega_rr_smoke.py=fe6441a7d0b9ad52a123c16b411a34d4a6678a2f`，`real_routed_weights.py=1a566ab6da988c01bb09549c601a1a687f67ccc7`，`checkpoint_experts.py=50342700435dca931c4156c0c4e16e78cc1f79c5`

## 模型与来源

- 实际路径：`/mnt/nas1/hf/DeepSeek-V4.1-Flash`
- 模型卡：`DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression`；README 链接 `https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash`。
- `config.json` SHA-256：`8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879`
- `model.safetensors.index.json` SHA-256：`74b0686a3d2891980d5e303251b075a3bccae2c2ff650747db2620a649b98fa8`
- 架构：`DeepseekV41ForCausalLM`；实际 config 维度位于 `text_config`。

`v41_shape.py` 输出：

```json
{"hidden_size": 5120, "moe_intermediate_size": 2304, "n_routed_experts": 384, "num_experts_per_tok": 6, "num_hidden_layers": 40, "config_sha256": "8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879"}
```

## 执行环境与规模

- 节点：`e01-cn-v8n4s5sfd01`，单节点 8 卡。
- 环境：`/home/admin/workspace/megamoe/env.sh`；Python 3.12.7；Torch `2.14.1+cu130`；CUDA 13.0；NCCL 2.30.7；实际 compute capability `(10, 3)`。
- 路径：`/home/admin/workspace/megamoe/DeepGEMM`，版本与任务要求一致。
- 用例规模：384 routed experts、hidden 5120、intermediate 2304、top-k 6、每 rank 256 tokens、8 ranks；全局 448 个物理槽，每 rank 56 个。
- 输入与 gate 为合成数据；routed 权重来自 V4.1 第 0 层真实 checkpoint。不含 shared expert。
- 未启动模型、未安装框架、未测性能。

## 真实 checkpoint 元数据

| key | dtype | shape |
|---|---|---|
| `layers.0.ffn.experts.0.w1.weight` | `I8` | `[2304, 2560]` |
| `layers.0.ffn.experts.0.w1.scale` | `F8_E8M0` | `[2304, 160]` |
| `layers.0.ffn.experts.0.w2.weight` | `I8` | `[5120, 1152]` |
| `layers.0.ffn.experts.0.w2.scale` | `F8_E8M0` | `[5120, 72]` |
| `layers.0.ffn.experts.0.w3.weight` | `I8` | `[2304, 2560]` |
| `layers.0.ffn.experts.0.w3.scale` | `F8_E8M0` | `[2304, 160]` |

## 八卡分配与释放

无作业调度器；使用共享锁 `/tmp/gcmoe-sm100-gpu-group.lock`。启动前确认 compute app 为空、`/dev/nvidia0..7` 无 fd 占用、8 个 UUID 与 `ENV.json` 登记一致，且 CUDA 只可见这 8 卡。

分配开始：`2026-11-21T15:33:04+08:00`  
释放完成：`2026-11-21T15:37:19+08:00`

```text
GPU-3aad6312-0e20-d01b-4249-71cb090c2127
GPU-b0ac7f5d-738f-cdad-423f-456f4c26947e
GPU-c5a93c78-be0c-2648-564a-4b4666671960
GPU-77463de9-85e0-a6f3-2c7e-f1147d6f7f49
GPU-4a493131-84b9-4c88-18ee-50fc6f213a36
GPU-045e62f4-d04d-2f9e-e456-1fb82810029f
GPU-6b44066c-6712-30f2-02d0-fa0cb91733a3
GPU-31a509fd-e741-5ab8-b837-443ad986c404
```

用例结束后复查：无 compute app、无 nvidia 设备 fd、无锁持有者。

## 全部 rank 结果

`different_elements` 是相对 identity 输出的逐元素差异数；`valid_output` 要求 BF16 输出全部有限且非全零；`slot_count_match` 要求每 rank 实际接收行数与路由期望一致。

| arm | rank | different_elements | valid_output | slot_count_match |
|---|---:|---:|---|---|
| identity | 0 | 0 | true | true |
| identity | 1 | 0 | true | true |
| identity | 2 | 0 | true | true |
| identity | 3 | 0 | true | true |
| identity | 4 | 0 | true | true |
| identity | 5 | 0 | true | true |
| identity | 6 | 0 | true | true |
| identity | 7 | 0 | true | true |
| eplb_rr | 0 | 0 | true | true |
| eplb_rr | 1 | 0 | true | true |
| eplb_rr | 2 | 0 | true | true |
| eplb_rr | 3 | 0 | true | true |
| eplb_rr | 4 | 0 | true | true |
| eplb_rr | 5 | 0 | true | true |
| eplb_rr | 6 | 0 | true | true |
| eplb_rr | 7 | 0 | true | true |

`smoke.log` 共包含 16 条 JSON rank 记录，`smoke_exit=0`。

## 警告与限制

启动阶段每个 rank 仍打印 `MLOPart is unavailable; localization has been disabled` 和 `1 MLOPart devices` 断言告警；这是可选本地化能力不可用，进程未失败，所有内核结果如上通过。首条告警和完整输出保留在 `smoke.log`。

本用例验证真实 routed 权重在换槽后的数值保持，不覆盖 shared expert、完整模型前向、端到端精度或性能。

## 旧 V4-Flash 结果

任务书更新前的 V4-Flash 运行已完成并保留在 `docs/sm100_megamoe/checkpoint_rr_results/legacy_v4_flash/`（见 `REPLY_CHECKPOINT_RR.md`）。该结果的 16/16 rank 通过，但属于旧模型的 256/4096/2048 形状，不是 V4.1 验收依据，本回执没有用它替代本轮结果。

## 附件

- `checkpoint_v41_retry_results/config.log`：V4.1 shape 校验与 index SHA-256。
- `checkpoint_v41_retry_results/checkpoint.log`：第 0 层真实 routed 权重元数据。
- `checkpoint_v41_retry_results/smoke.log`：分配、启动警告、16 条 rank JSON、退出码与释放时间。
- `checkpoint_v41_retry_results/SHA256SUMS`：附件校验和。
