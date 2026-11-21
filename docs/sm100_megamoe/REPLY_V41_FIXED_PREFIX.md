# 【SM100】同前缀下 Native auto 与 DeepGEMM MegaMoE 数值审计回执

执行者：**SM100**。本轮未执行任何 SM90 任务书，也未混入 SM90 结果。

## 结论

两种后端在最终等价 harness 中均完整跑完并退出 `0`：

- `auto`：退出 `0`，`2026-11-21T22:02:46+08:00` 至 `22:05:46+08:00`
- `deep_gemm_mega_moe`：退出 `0`，`2026-11-21T22:05:46+08:00` 至 `22:08:46+08:00`
- launcher：退出 `0`，`external_timeout=none`
- 两后端均为 TP8/EP8、单请求贪心、每次固定前缀只做一步推理
- 四组前缀的 top10 均已取得；`prefix-audit.json` 断言每后端 8×40 层、四组前缀一一对应

首个首选 token 分歧出现在任务规定的基础前缀 `[100,200,300,400]`：`auto=200`，`deep_gemm_mega_moe=75`。使用原生前 1/2 个输出扩展前缀时首选仍不同；使用原生前 3 个输出扩展后，两者首选恢复一致。后三次是**固定前缀重新 prefill**，不是把某一后端的分歧 token 继续自回归；因此不能据此断言分歧会或不会在真实自回归链中被放大。

本轮不计时，不构成性能结论；也没有 bit-exact 或数值容差验收要求，因此只报告观测差异，不把差异归因为单一实现原因。

## 运行信息

- 任务提交：`bb6b8fe5ab6e677d9b8739621fb1e44b66a8ba1d`
- 任务书 blob：`75508bff91b8cb0ae1ba2d13b153698812a2d495`
- 最终工作目录：`/data0/home/admin/workspace/megamoe/native-v41-s8u3defd`
- checkpoint：`/data0/home/admin/workspace/megamoe/native-v41-KO1hTbTj/model`
- `config.json` SHA-256：`8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879`
- 权重索引 SHA-256：`74b0686a3d2891980d5e303251b075a3bccae2c2ff650747db2620a649b98fa8`
- GPU：8 × `NVIDIA L20D`，每卡 `275040 MiB`，capability `(10, 3)`
- 八卡释放：`2026-11-21T22:08:46+08:00`；`22:12:04` 复核 compute-apps 与 `/dev/nvidia0..7` 均空

## 四次固定前缀结果

`gap` 为该后端 top1 与 top2 的 logprob 差（top1 − top2）。最后两列是双方首选 ID 在对方 top10 中的名次。

| 固定前缀 | auto top1 / logprob / gap | deep top1 / logprob / gap | auto top1 在 deep 名次 | deep top1 在 auto 名次 |
|---|---|---|---:|---:|
| `[100,200,300,400]` | `200` / `-0.846716` / `0.250000` | `75` / `-0.921821` / `1.500000` | 2 | 7 |
| `[100,200,300,400,200]` | `18` / `-0.761771` / `0.750000` | `300` / `-0.084430` / `2.875000` | 2 | 3 |
| `[100,200,300,400,200,19]` | `200` / `-0.059021` / `3.500000` | `200` / `-0.024628` / `4.500000` | 1 | 1 |
| `[100,200,300,400,200,19,200]` | `19` / `-0.707521` / `1.000000` | `19` / `-0.382357` / `1.750000` | 1 | 1 |

四次请求的首选 ID 依次为：

```text
auto: [200, 18, 200, 19]
deep: [ 75,300, 200, 19]
```

### 同一候选的跨后端 logprob 差（deep − auto）

| 前缀 | 候选 | 差值 |
|---|---|---:|
| `[100,200,300,400]` | `200` | `-1.575105` |
| `[100,200,300,400]` | `75` | `+4.487395` |
| `[100,200,300,400,200]` | `18` | `-2.197659` |
| `[100,200,300,400,200]` | `300` | `+2.677341` |
| `[100,200,300,400,200,19]` | `200` | `+0.034393` |
| `[100,200,300,400,200,19]` | `300` | `+0.284392` |
| `[100,200,300,400,200,19,200]` | `19` | `+0.325165` |
| `[100,200,300,400,200,19,200]` | `18` | `-0.924835` |

完整 top10 见 `auto.log`、`deep_gemm_mega_moe.log` 与 `prefix-audit.json`。

## apply_model 实际层信息

对每个 rank 的 `language_model.model.layers.0..39.ffn` 在 `model.apply_model(moe_runtime)` 之后读取属性，两后端都恰好报告 8 个 rank、每 rank 40 层，共 320 层。

| 后端 | 每 rank 观测 expert class | `sequence_parallel=true` | `fused_shared=true` | 每 rank 实际融合层数 |
|---|---|---:|---:|---:|
| `auto` | `MoERunner` | 0/40 | 0/40 | **N/A** |
| `deep_gemm_mega_moe` | `DeepseekV4MegaMoEExperts` | 40/40 | 40/40 | **40** |

该结论对 8 个 rank 全部相同；逐 rank 机器可读结果在 `prefix-audit.json`，简化计数在 `auto-fusion-counts.json`。

**`auto` 的融合层数不能写成语义上的 0。** 它的实际对象是 wrapper `MoERunner`，该对象上没有 `has_fused_shared_experts` / `fused_shared` 属性，读取结果为 `null`，所以本轮只能判为 `N/A`。`auto` 的 `sequence_parallel` 则确实是可读的布尔值 `false`。`deep_gemm_mega_moe` 的实际对象是 `DeepseekV4MegaMoEExperts`，40/40 层的两个属性均为 `true`，因此每 rank 实际融合层数为 40。

## 启动失败与等价 harness 修复

任务固定要求真实文件入口和跨工具调用持续会话。原脚本先遇到的是 Python 函数序列化问题，不是模型数值错误：

1. **Attempt 1（原任务书 harness）**：两后端都在 `model.apply_model(moe_runtime)` 处退出 `1`。首个错误为
   `TypeError: Object of type <class 'function'> is not serializable`，提示可设置 `VLLM_ALLOW_INSECURE_SERIALIZATION=1`。
2. **Attempt 2（仅加官方提示开关）**：设置 `VLLM_ALLOW_INSECURE_SERIALIZATION=1` 后，pickle fallback 仍无法解析 `__main__.moe_runtime`，首个错误为
   `PicklingError: Can't pickle <function moe_runtime ...>: it's not the same object as __main__.moe_runtime`。
3. **Attempt 3（等价 harness 修复）**：把仅用于 introspection 的 `moe_runtime` 移到可导入的 `v41_prefix_probe.py`，使父进程与 worker 引用同一模块对象；推理入口、模型、TP8/EP8、采样参数和两个 MoE 后端均未改变。该改动只是修复探针函数的 pickle identity，不是模型或 kernel 编译修复。

Attempt 2 的 deep 后端还遇到外部进程占卡：

- 用户 `ruixuan+` 在 `21:52:50` 启动 8 个 `rtp_llm_rank-*`，占用同组 8 卡。
- 这不是 SM100 进程，未终止；launcher 在锁内等待，约 `22:02:46` 外部进程退出后才启动 Attempt 3。
- Attempt 2 的 deep 首个资源错误为 `ValueError: Free memory on device cuda:3 (179.36/267.69 GiB) ... less than desired ... (0.8, 214.15 GiB)`；该错误发生在外部占卡期间，不属于 Attempt 3 的最终结果。

三次运行的日志、退出码与时间线均保留在结果目录。

## cgroup、journal 与释放

- cgroup：`/sys/fs/cgroup/user.slice/user-1384661.slice/session-1882009.scope`，`memory.max=max`
- `memory.current`：前 `431,676,358,656` bytes；峰值 `871,110,451,200` bytes（`22:05:16`）；后 `742,066,520,064` bytes
- `oom` / `oom_kill` / `oom_group_kill`：前、观测峰值、后均为 `0 / 0 / 0`
- `MemAvailable` 最低：`884,125,480 kB`；结束：`886,194,032 kB`
- `memory.current` 包含整个 session cgroup 与可回收页缓存，不是单模型 RSS
- launcher 未设置外部 `timeout`，`external_timeout=none`；模型进程与 launcher 均自行退出 `0`
- system journal 为 `PERMISSION_LIMITED`，无法据此完成 system-wide Xid/OOM 审计；cgroup 与进程退出证据单独保留

## 环境

- vLLM `0.31.0`
- torch `2.13.0+cu130`，CUDA `13.0`
- flashinfer `0.7.0.post1`
- transformers `5.17.0`
- triton `3.7.1`
- Python `3.12.7`
- 独立 venv：`/data0/home/admin/workspace/megamoe/native-v41-stsFS4w7/venv`
- `VLLM_USE_V2_MODEL_RUNNER=1`，`HF_HUB_OFFLINE=1`
- `enable_flashinfer_autotune=False`，`gpu_memory_utilization=0.8`，`enforce_eager=True`

## 产物

`docs/sm100_megamoe/v41_fixed_prefix_results/` 包含最终 `auto.log`、`deep_gemm_mega_moe.log`、`model.log`、`prefix-audit.json`、`prefix-audit.txt`、逐 rank 融合计数、三次 attempt 的失败/成功日志、launcher、版本与环境、cgroup 前后计数及峰值、八卡释放复核、journal 结果和 `SHA256SUMS`。未提交模型权重、venv 或编译缓存。
