# REPLY_FIXED_STREAM — **失败：baseline 臂在构造 LLM 时崩溃**

## 0. 首个失败点

**第 1 个臂（`step02 / baseline / paired`），在 `real_decode_100_100.py:725` 的 `LLM(...)` 构造处。**

```
ValueError: ReferenceTokens is not a subclass of
  vllm.v1.worker.gpu.sample.logits_processor.LogitsProcessor
```

`exit_code=1`，`step=3: capture`。模型**尚未加载**（崩溃发生在 `LLMEngine` 的
`InputProcessor` 初始化阶段，早于任何 GPU 权重分配，显存始终 4 MiB）。
按任务书 §执行「本轮最多两次模型启动，部分失败时保留已完成的臂和原始目录，直接回执」，
**我已停止，未启动第二次**。

## 1. 根因：同一个功能有两套互斥的实现

`76793f6` 同时加了两个东西：

**(a) monkey-patch 路径**（主体实现）
- `scripts/real_decode_100_100.py:824` → `engine.collective_rpc('_gc_moe_fixed_tokens')`
- → `integration/deepep_compare.py:2370` → `from fixed_tokens import install`
- → `integration/fixed_tokens.py:56 install()` 里包装 `GPUModelRunner.sample`，
  在原生 greedy 采样后把 `sampled_token_ids` 就地改写成参考 token。

**(b) `logits_processors` 参数**（多余入口）
- `scripts/real_decode_100_100.py:726`：
  ```python
  **({'logits_processors': ['fixed_tokens:ReferenceTokens']} if fixed_dir else {}),
  ```
- 这会让 vLLM 去 `fixed_tokens` 模块取 `ReferenceTokens` 类，并**强制要求**
  它是 `vllm.v1.worker.gpu.sample.logits_processor.LogitsProcessor` 的子类
  （见 `env3/.../logits_processor/loader.py:127`）。
- 但 `ReferenceTokens` 根本不是 LogitsProcessor：它没有 `apply()`，
  只有 `feed()` / `native_sample()`，是给 (a) 用的辅助类。

**两者互斥**：只要 `GC_MOE_FIXED_DIR` 存在，(b) 就会在构造阶段直接抛异常，
(a) 永远没有机会执行。这是**确定性失败**，不是环境或数据问题。

### 验证

```
$ python -c "import inspect; from vllm.v1.worker.gpu.sample.logits_processor import LogitsProcessor as L; print(L.__abstractmethods__)"
frozenset({'apply'})

$ python -c "... fixed_tokens.py ..."
has apply(): False        ← ReferenceTokens 不实现 apply
has feed(): True          ← 它是 (a) 用的辅助类
class ReferenceTokens:    ← 也不是任何 LogitsProcessor 的子类
```

vLLM 自己的插件加载器（`loader.py:120-128`）会把每个 FQCN 展开成类，
只要 `not issubclass(obj, LogitsProcessor)` 就 `raise ValueError`。

## 2. 为什么我没有自行修复

任务书 §执行 明确限定：

> 执行者只可对**编译器点名的**语法或类型做语义等价的最小修改。

本条是**运行时**（Python `ValueError`）而非编译器诊断；而且两种修法都会改变测量协议：

| 修法 | 后果 |
|---|---|
| 删掉 `logits_processors` 参数 | 只保留 (a) monkey-patch。**功能等价**，但属于改代码 |
| 给 `ReferenceTokens` 补 `LogitsProcessor` 子类 | 会**同时**启用 (b)，与 (a) 重复改写同一批 token，语义未知 |

第二种显然不能做。第一种虽看似等价，但仍超出“编译器点名”的授权边界，
所以我停下来回传，而不是自行决定。

## 3. 版本与环境

| 项 | 值 |
|---|---|
| 测量代码 SHA | `76793f613a1039e7fec6d510bf62c54dfd4134fe` |
| 两臂内核 | 同一 SHA（本任务书要求同提交、同工作树）|
| GPU | 8×H20，driver 580.105.08（全程未加载模型，显存 4 MiB）|
| 软件 | cutlass 4.8.1 / torch 2.13.0+cu130 / CUDA 13.0 / vllm 0.31.0 |
| 模型 | `/home/admin/local_models/DeepSeek-V4.1-Flash` |
| 数据集 | `/mnt/nas2/nanjun.cp/MoE-Bench-Experiments/datasets/dsv4-flash_online_input_ids_26-10-08_1.jsonl` |
| 工作目录 | `/tmp/gcmoe-two-arm-ktrh9fqe` |
| 失败轮原始目录 | `/tmp/gcmoe-aligned-AJybv0Ak` |
| 共享路由目录（已生成 tokens.json） | `/tmp/gcmoe-route-replay-bOF1m83d` |

**未做任何代码修改**，工作树保持原 SHA。

## 4. 已完成的部分

| 步骤 | 状态 |
|---|---|
| 参考 result 校验（`/tmp/gcmoe-aligned-GIbymGGQ/result.json`） | ✅ `64c19ebf7c5e50c3acdf354eb4f9d11fd93125d964b59a2849f0d7508d5d17b9` 校验通过 |
| 生成 `tokens.json`（256 请求 × 参考轨迹） | ✅ `/tmp/gcmoe-route-replay-bOF1m83d/tokens.json` |
| 四项 CPU 门禁 | ✅ 全部 PASS（见下） |
| baseline 臂 | ❌ 首个失败 |
| stream 臂 | 未启动 |

```
ALIGNED_STEP_CPU PASS
PULL_GEMM_DETAIL_CPU PASS
MULTILAYER_STUDY_CPU PASS
FIXED_WORKLOAD_CPU PASS
```

`FIXED_WORKLOAD_CPU` 能通过，说明 `RouteTape`（路由回放）本身没问题；
它没有覆盖到 `fixed_tokens.py` 的 vLLM 插件契约，所以这个缺陷逃过了 CPU 门禁。

## 5. 证据保留

| 内容 | 路径 |
|---|---|
| 失败轮完整目录 | `/tmp/gcmoe-aligned-AJybv0Ak`（`status.txt` / `failed-command.txt` / `execution.sha` / `run.log`）|
| 完整调用栈 | `delivery/step02/baseline/paired/runner.log` |
| 证据副本 | `delivery/step02/baseline/paired/evidence/` |
| 共享路由文件 | `/tmp/gcmoe-route-replay-bOF1m83d/`（tokens.json；placement.json 尚未生成，因为没跑到重排）|

## 6. 需要授权的最小修复

**推荐**：删除 `scripts/real_decode_100_100.py:726` 的 `logits_processors` 关键字，
只保留 (a) 的 monkey-patch 路径。理由：

- (a) 是在 `GPUModelRunner.sample()` 内部改写 `sampled_token_ids`，
  位置在原生 greedy 采样之后、KV 记账之前 —— 正是任务书要求的
  「vLLM 仍做原生 greedy 采样，随后将参考 token 送进下一步」。
- (a) 还负责 `native` 记录，供 `native_sample()` 报出 step 2 的
  `native_top1_different_of_256`（任务书 §回执 4 要求的判据）。
- (b) 无论如何都不会提供这些能力，它只会让进程起不来。

改完后 `FIXED_WORKLOAD_CPU` 的行为不变，`fixed_tokens.install()` 的契约不变，
**唯一变化是不再向 vLLM 注册一个非法的 LogitsProcessor 插件**。

若发布者确认，请按 §执行 的「单独提交补丁并回传首个诊断、受影响的 SHA、
完整日志与等价性说明」流程给我一个修后 SHA，我用它重跑完整的两次启动。
