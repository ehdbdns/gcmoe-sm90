# 【SM100】DeepGEMM MegaMoE 物理槽 RR 八卡正确性回执

执行者：SM100。未执行任何 SM90 任务书，也未混入 SM90 结果。

## 结论

通过。本轮在环境回执 `7c2fa40` 之后的 SM100 检查中，使用同一组 8 张 SM100 完成一次 8-rank 合成 routed-only 正确性用例：

- 原生 identity 槽：8/8 rank 通过，输出有限且非全零，物理槽接收行数全部匹配。
- EPLB + RR 物理槽：8/8 rank 通过，相对 identity 的元素差异为 0，输出有限且非全零，物理槽接收行数全部匹配。
- 未发现数值差异、启动失败或版本不符；未做语义等价范围内的代码修复。

执行 SHA：

- 任务/测试仓库：`e3a84793a5de82cb6af7ba0cc2899795eb248b71`
- 环境回执基线：`7c2fa408fc65d45b1c2d1756bdc2ee84642c7119`
- DeepGEMM：`5081f8513af57e61696de4f4f04b0e707ffb503e`
- 测试脚本 blob：`mega_rr_smoke.py=8ade380c4e0aabf252e059a4922ca90f542ca34a`，`mega_eplb.py=e9b27371da8683bb132b5276d682c3179f62e224`，`checkpoint_experts.py=50342700435dca931c4156c0c4e16e78cc1f79c5`

## 环境与执行

- 节点：`e01-cn-v8n4s5sfd01`，单节点 8 卡。
- 环境：`/home/admin/workspace/megamoe/env.sh`；Python 3.12.7；Torch `2.14.1+cu130`；CUDA 13.0；NCCL 2.30.7；实际 compute capability `(10, 3)`。
- checkpoint：`/data0/data3/DeepSeekV4-Flash`。
- checkpoint 检查命令：`python scripts/sm100/checkpoint_experts.py /data0/data3/DeepSeekV4-Flash`，退出码 0。
- GPU 用例命令：`python scripts/sm100/mega_rr_smoke.py`，退出码 0。
- 用例规模：256 routed experts、hidden 4096、intermediate 2048、top-k 6、每 rank 256 tokens、8 ranks；全局 320 个物理槽，每 rank 40 个。合成 FP8 激活与合成 FP4 routed 权重，不含 shared expert。
- 未启动模型、未安装框架、未做性能实验。

checkpoint 实际 routed 专家头与 scale（layer 0 代表项）：

| key | dtype | shape |
|---|---|---|
| `layers.0.ffn.experts.0.w1.weight` | `I8` | `[2048, 2048]` |
| `layers.0.ffn.experts.0.w1.scale` | `F8_E8M0` | `[2048, 128]` |
| `layers.0.ffn.experts.0.w2.weight` | `I8` | `[4096, 1024]` |
| `layers.0.ffn.experts.0.w2.scale` | `F8_E8M0` | `[4096, 64]` |
| `layers.0.ffn.experts.0.w3.weight` | `I8` | `[2048, 2048]` |
| `layers.0.ffn.experts.0.w3.scale` | `F8_E8M0` | `[2048, 128]` |

即本机 checkpoint 的 routed 权重是 FP8 `I8 + F8_E8M0`；上述 smoke 按任务要求另行合成 FP4 routed 权重，不声称 checkpoint 本身是 FP4。

## 八卡分配与释放

无作业调度器；使用机器现有共享预约锁 `/tmp/gcmoe-sm100-gpu-group.lock`。启动前确认：

- `nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader` 为空；
- `/dev/nvidia0` 至 `/dev/nvidia7` 无进程占用；
- 8 个 UUID 与 `docs/sm100_megamoe/ENV.json` 登记一致，且仅这 8 卡对 CUDA 可见。

分配开始：`2026-11-21T15:15:02+08:00`  
释放完成：`2026-11-21T15:15:53+08:00`

使用的 8 个 UUID：

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

用例退出后重新检查：无 compute app、无 nvidia 设备 fd、无锁持有者。

## 全部 rank 结果

`different_elements` 是相对 identity 输出的元素差异数；`valid_output` 要求 BF16 输出全部有限且非全零；`slot_count_match` 要求本 rank 各物理槽实际接收行数与按路由计算的行数一致。

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

原始输出见 `rr_smoke_results/smoke.log`，共 16 条 JSON rank 记录。

## 警告与限制

启动阶段每个 rank 都打印了 `MLOPart is unavailable; localization has been disabled`，并伴随 `1 MLOPart devices` 的断言告警；这是 DeepGEMM 的可选本地化功能不可用告警，进程未因此失败，所有内核结果均为上述通过状态。原始首条告警与完整输出已保存在 `rr_smoke_results/smoke.log`。

本用例验证的是合成 routed-only 路径，不覆盖 shared expert、完整模型前向、吞吐或端到端精度；这些未在本轮验证。

## 数据集与 RTP-LLM 入口

可由 admin 读取的现有数据集文件：

- `/mnt/nas1/hf/datasets/Salesforce/wikitext/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-2-raw-v1/test-00000-of-00001.parquet`：mode 644，732610 bytes，Parquet magic 已检查，admin 可读。
- `/data0/data3/wjh528431/work/RTP-LLM/internal_source/rtp_llm/test/smoke/data/prompt_candidates.json`：150K，admin 可读；元数据标注来源为 ShareGPT 与 LongBench，是 RTP-LLM smoke 的标准化 prompt 候选集。
- `/mnt/nas1/hf/rtp-llm-benchmarks` 与 `/mnt/nas1/hf/vllm-benchmarks-yuanzhenhao` 可读，但内容是已有 benchmark/report，不是本轮验证过的通用数据集。

现有 RTP-LLM checkout（只读检查，未修改、未启动模型）：

- 外层路径：`/data0/data3/wjh528431/work/RTP-LLM`，HEAD `c207b52b2ddce5acf964514968717a748053973b`。
- 内层开源工作树：`/data0/data3/wjh528431/work/RTP-LLM/githubopensource`，HEAD `e59977f84d34dbdb9695ebf4a4afa8886ef8e364`；仅有 2 个未跟踪的生成 `*_pb2*.py` 文件，未改动。
- MegaMoE executor：`rtp_llm/models_py/modules/factory/fused_moe/impl/cuda/executors/mega_moe.py` 的 `MegaMoeExecutor`，调用 `deep_gemm.fp8_fp4_mega_moe`；策略注册见 `impl/cuda/strategy/fp8_fp4.py` 的 `CudaMegaMoeStrategy`（`mega_moe`）。
- 现有测试入口：`rtp_llm/models_py/modules/factory/fused_moe/impl/cuda/executors/test/mega_moe_routed_only_execution_test.py`（2-rank routed-only/standalone/fused 覆盖）和 `mega_moe_end_to_end_perf_test.py`（手动性能 guard）；Bazel smoke 目标引用 `//rtp_llm/test/smoke:smoke_sm100_moe`。

上述 RTP-LLM 入口在本轮未运行，因此只能确认源码入口存在，不能声明其当前已可直接在当前 admin venv 跑通；环境回执中已记录当前 admin venv 缺少 `deep_ep` 等框架依赖。

## 附件

- `rr_smoke_results/checkpoint.log`：checkpoint 检查原始输出。
- `rr_smoke_results/smoke.log`：分配、启动告警、全部 16 条 rank JSON、退出码与释放时间。
- `rr_smoke_results/SHA256SUMS`：上述日志的 SHA-256 校验和。

`checkpoint.log` SHA-256：`895e22a07c8799a538794f1d9cccc32f3e1c05cad7e10fbb9f9234e342b8e668`  
`smoke.log` SHA-256：`9c58ae6f8ecd9397c56d9fa099ffa2a595f551c1edf634c9b4cc81e8744106ef`
