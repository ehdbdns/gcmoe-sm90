# 【SM90】REPLY_SM90_PREFILL_SOURCE — V4.1 prefill 实际调用代码采集完成

> 本回执由 **SM90 执行者**（8×H20）提交。本轮**不启动模型、不占 GPU**。

## 0. 结论

已按任务书脚本采集 `vllm` 安装包中 `vllm/models/deepseek_v41/` 下
**命中四个关键字**的全部 Python 模块，共 **6 个文件**，并附 `manifest.txt`。

| 项 | 值 |
|---|---|
| vllm 版本 | **0.31.0** |
| 采集文件数 | **6** |
| 环境 GPU | 未分配（`CUDA_VISIBLE_DEVICES=''`）|
| 模型启动数 | **0** |

## 1. 文件清单与 SHA-256

```
vllm 0.31.0
54c3fb91b4b6741c757bf26022c8ce9f417c2354e823ede58ed2324d3cc449a4  amd/rocm.py
12c183b169b134fc909d598d6f99bd4f8da83269a4589a5bc3162211d9364c77  attention.py
a8be566e4407265db7fd109c7ad9e8a5e61265dd2813e6465a590b8e41399c5d  nvidia/flash_mla_mega_attn.py
2922c952a038dac85c526a95b0e06c159c16360e9180c412ec626b05e8614421  nvidia/flashinfer_sparse.py
6bfcc91c278bae8f9c95d9ec4da12ef98112b9ccf9d22547743485759c258348  nvidia/flashmla.py
f3b109e980d12652e024fd9b20396e11d9810544c0f4a285fccb6206ce8a3bc8  nvidia/model.py
```

筛选关键字：`self.wo_b(`、`_forward_prefill`、`DeepseekV4FlashMLAAttention`、`flash_mla_sparse_fwd`。

> 注：`amd/rocm.py` 是 AMD 后端实现，因同样命中 `self.wo_b(` 与 `_forward_prefill`
> 而被脚本收录。不适用本机（NVIDIA），仅供对照。

## 2. 关键调用链（供设计端选择下一处探针）

### 2.1 `_wo_b_proj` —— `wo_b` 的三个入口

`attention.py:696` 起：

```python
def _wo_b_proj(self, z):
    if self.gemm_rs is None:
        return self.wo_b(z)                      # ← 路径 A
    if isinstance(z, torch.Tensor) and self.gemm_rs.should_run(z):
        return self.gemm_rs.apply(z, self.wo_b)  # ← 路径 B（GEMM+RS 融合）
    return sp_reduce_scatter(self.wo_b(z))       # ← 路径 C（小 batch）
```

**上一轮观测到 `wo_b` 的 `in` 边界摘要不同**，说明分叉发生在 **`z` 的构造**，
即在进入 `_wo_b_proj` 之前。三条路径都接收同一个 `z`，与选择哪条无关。

### 2.2 prefill 路径：`_forward_prefill` 的完整步骤

`nvidia/flashmla.py:248` 起，每个 chunk 依次执行：

| 步骤 | 调用 | 作用 |
|---|---|---|
| 1 | `workspace_manager.get_simultaneous(...)` | 分配 `kv`、`combined_indices_out`、`combined_lens_out` |
| 2 | `dequantize_and_gather_k_cache(kv, compressed_k_cache, ...)` | **压缩 KV 反量化并 gather**（`swa_only` 时跳过）|
| 3 | `dequantize_and_gather_k_cache(kv, swa_k_cache, offset=chunk_N, ...)` | **SWA KV gather**，写到同一 `kv` 的 `offset=chunk_N` 之后 |
| 4 | `combine_topk_swa_indices(topk_indices[...], ..., chunk_M, chunk_N, out=(...))` | **把 topk 索引与 SWA 索引合并** |
| 5 | `flash_mla_sparse_fwd(q=..., kv=..., indices=..., topk_length=..., out=output[...])` | **实际 attention，结果原地写入 `output`** |

**第 5 步的 `out=output[query_start:query_end]` 是原地写入** ——
这正是 `wo_b` 输入 `z`（形状 `[2048, 1024]` bf16）的来源。

### 2.3 prefill / decode 的分派

`nvidia/flashmla.py:145` 起：

```python
if num_prefills > 0:
    self._forward_prefill(q=q[num_decode_tokens:], ..., output=output[num_decode_tokens:], ...)
if num_decodes > 0:
    self._forward_decode(q=q[:num_decode_tokens], ..., output=output[:num_decode_tokens], ...)
```

**两者写入同一个 `output` 的不同切片**。本轮 prefill 阶段 `num_decodes == 0`，
所以只走 `_forward_prefill`。

## 3. 对上一轮结论的支持

上一轮的边界对照是：

| 边界 | 两次是否相同 |
|---|---|
| `attn.indexer` `out` | ✅ 相同 |
| `attn.wo_b` `in` | ❌ 不同 |

从源码看，`indexer` 的输出是 **`topk_indices`**（第 4 步的输入之一），
而 `wo_b` 的输入是 **第 5 步 `flash_mla_sparse_fwd` 的 `out`**。
两者之间正是「合并索引 → 稀疏 attention」这一段。

因此候选分叉点可以进一步收窄为：

1. **`combine_topk_swa_indices`** —— 合并 topk 与 SWA 索引
   （`combined_indices` / `combined_lens`）。
   注意索引集合一致但**排列顺序**可能不同，这与 R7 观测到的
   `extra_order` 差异属于同一类现象。
2. **`dequantize_and_gather_k_cache`** —— 压缩 KV 与 SWA KV 的反量化/gather
   （两者写入同一 `kv` 缓冲区的不同 offset，`offset=chunk_N`）。
3. **`flash_mla_sparse_fwd` 内部** —— 稀疏 attention 的归约顺序。

## 4. 版本与环境

| 项 | 值 |
|---|---|
| 任务 SHA | `c5fc99c` |
| vllm | **0.31.0** |
| 采集根目录 | `vllm/models/deepseek_v41/`（`dist.locate_file` 解析）|
| `CUDA_VISIBLE_DEVICES` | **`''`**（未分配 GPU）|
| 模型启动数 | **0** |
| 代码修改 | **无** |

## 5. 交付物

`docs/trace_executor_20261119/sm90_prefill_source/`：
- `manifest.txt`（含 vllm 版本 + 6 条 sha256）
- `attention.py`
- `nvidia/flashmla.py`
- `nvidia/flash_mla_mega_attn.py`
- `nvidia/flashinfer_sparse.py`
- `nvidia/model.py`
- `amd/rocm.py`

未上传环境变量、模型权重或任何其他本机文件。
