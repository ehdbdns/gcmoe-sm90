# V41 host-drift source and graph-coverage locations

本轮只读检查。代码提交为 `b74b2eca60bedffe41e232b1659ad78d8831d464`。
安装环境为 vLLM `0.31.0`，commit `gdb9527a46`，torch `2.13.0`。

## 当前 SM100 执行路径

| 位置 | SHA256 | 摘录 | 含义 |
|---|---|---|---|
| `scripts/sm100/five_matrix.py:83` | `11e96251fb3f859c85df670f6d0381f07acaa0a5dc66518dda2c0e7af118ac5c` | `enforce_eager=True` | vLLM 自带的 full CUDA graph 被关闭，后续由 worker overlay 手工捕获局部图。 |
| `scripts/sm100/five_matrix.py:96` | 同上 | `max_num_batched_tokens=max(2048, bs)` | BS512 仍按 2048 token 预算建 runner，但与局部 MoE 图地址无关。 |
| `scripts/sm100/five_worker.py:97-154` | `15bae5844b8a6cbddc1ae9ff3fd029893737f11f6a1fcb0c1ad5a50708ed0d1d` | `scratch.copy_(self.recv[call])`; `with torch.cuda.graph(graph): original_launch(...)`; `graph.replay()`; `y.copy_(target)`; `auction[call].copy_`; `recv[call].copy_` | 每层独立捕获单次 MoE，replay 后执行输出 copy、auction copy 与计数回写。 |
| `scripts/sm100/five_worker.py:250-286` | 同上 | `layer_graphs[layer] = graph, target`; `assert len(self.layer_graphs) == LAYERS` | 40 层各自持有独立图，不是完整 decode graph。 |
| `scripts/sm100/native_dense_replay.py:27-39` | `f8cb58ec0aace4816f044e1369a8ebbdcef1aab12f07d8638b7fef5c0d712fa4` | `with torch.cuda.graph(graph): target = eager(...)`; `source.copy_(x)`; `graph.replay()` | dense MXFP8 projection 另有按 layer/shape/stream 捕获的局部图。 |
| `vllm/v1/worker/gpu_worker.py:880-883` | `43bd01925cd84f7d8a1d4a2ec30e289014a485fe4492a5343cbc893e8ed2f518` | `if not self.model_config.enforce_eager: self.model_runner.capture_model()` | full graph 的真实入口；本实验 `enforce_eager=True`，此处不执行。 |
| `vllm/v1/worker/gpu_model_runner.py:6755-6825` | `bb903dffcb5a6e5b8a477daf0854b3812e135349d61fc20c6133f91cd3c9752a` | `def capture_model`; `for runtime_mode, batch_descs in ... get_capture_descs()`; `self._capture_cudagraphs(...)` | 按 batch descriptor 捕获完整图。 |
| `vllm/v1/worker/gpu_model_runner.py:6861-6907` | 同上 | `_warmup_and_capture`; `force_attention = runtime_mode == CUDAGraphMode.FULL`; `self._dummy_run(..., is_graph_capturing=True)` | FULL decode 在捕获前强制 attention，并跑 dummy full forward。 |
| `vllm/v1/worker/gpu_model_runner.py:6909-6958` | 同上 | `_capture_cudagraphs`; `for batch_desc in batch_descriptors`; `self._warmup_and_capture(...)` | 每个 decode batch shape 捕获一张完整图。 |
| `vllm/compilation/cuda_graph.py:281-321` | `e09f694f4f50cda891b2db09d48993a9248f5a1318795b3798cd5478c268ca84` | `entry.input_addresses = ...`; `with torch.cuda.graph(cudagraph, ...): output = self.runnable(*args, **kwargs)` | 图重放要求输入地址稳定；动态 Python tensor 必须搬入持久 staging。 |
| `vllm/compilation/cuda_graph.py:345-363` | 同上 | `entry.cudagraph.replay()` | replay 路径不执行 Python `runnable`。 |
| `vllm/config/compilation.py:53-63` | `9e005aee4e8d74b23636312e9c9be796d8fc3b43ce483937cf3795fcbcaf18e5` | `FULL_DECODE_ONLY = (FULL, NONE)` | FULL 只用于 decode；mix/prefill 退回 NONE。 |
| `vllm/models/deepseek_v4/nvidia/model.py:1594-1606` | `5d5a74e339cd110def9b5e37a56fa602a42cdece98811013fb69700a0274a224` | `for idx, layer in enumerate(islice(self.layers, ...)): hidden_states, ... = layer(...)` | 40 层 forward 由 Python 循环展开；full graph 捕获时必须一次展开完。 |

## SM90 FULL_DECODE_ONLY 基线

路径：`docs/trace_executor_20261119/sm90_main24_stable_decode_bs512_results/delivery/bs512/vanilla/run.json`
（`main` 中文件 SHA256 `2dcc9aed821594ae4e5927d5deb5cbc32d7a2cd941538391d6c97fc42b50f1f2`）。

关键字段：`batch=512`、`layers=40`、`controls.ENFORCE_EAGER=0`、
`controls.CUDAGRAPH_MODE=FULL_DECODE_ONLY`、`controls.MOE_BACKEND=gc_moe`、
`measured.ms_per_step=166.4659915100783`。

因此 SM90 基线的一次 decode 由 vLLM full-decode graph 覆盖；本轮 SM100
的 `enforce_eager=True` 与 40 张单层 MoE 图不是同一覆盖层级。

## 整图回放需调整的按层状态

| 状态 | 当前行为 | full decode 图下的问题 |
|---|---|---|
| `five_worker.py:101` `self.current=(layer,call)` | Python 每次 `_five_forward` 写入 | replay 不调用 Python，捕获时只能保留展开后的 40 个静态副本；运行期不能再靠它选 buffer。 |
| `five_worker.py:137-139` `host_submit[call]` + `graph.replay()` | 每层一次 MoE replay，记录逐层 host 时间 | full graph 每步只有一次 replay，逐层 host 提交点和 correlation 消失；需要新的 per-step 标记或设备内时钟。 |
| `five_worker.py:102-106` `kernel_scratch.copy_(recv[call])` | 40 次动态 call 下标 copy 到同一 scratch | 捕获后地址固定；需要 40 份持久 staging，或图外一次批量复制到连续 40 层输入区。 |
| `five_worker.py:110,112` `audit_scratch.zero_()` | 每层清零同一审计张量 | 同一地址在 40 层展开后会被重复清零；需要按层持久 scratch，或显式定义跨层累加语义。 |
| `five_worker.py:140-148` `y.copy_`, `auction[call].copy_`, `recv[call].copy_` | replay 后 Python 回写 | full graph replay 不执行这些 Python copy；必须移入图内固定地址，或图外由一次输入/输出交换完成。 |
| `five_worker.py:142-146` `plan_start`, `quota_view`, `plan_counts[call]` | 每层读取动态视图并写回 | 动态 view 和 call 下标在图中会冻结；需固定每层 plan/quota 槽位并在捕获前绑定。 |
| `five_worker.py:107-110` `eco_meta`, `eco_state`, `buf.eco_auction_audit` | 每层不同地址 | full capture 会把这些指针固化到第一层；需按层建立静态地址表，整图内按层索引。 |
| `native_dense_replay.py:22-33` graph key | 包含 layer/shape/stream/权重地址 | full decode 捕获时这些 key 只出现一次展开；不能再在 replay 路径依赖 Python key 查表。 |

## 本轮 L19→L20 实际覆盖

两臂八 rank 的 L19 MoE graph launch 到 L20 MoE graph launch 之间均为：
3 次 `cudaGraphLaunch`（1 次 MoE + 2 次 dense local）、30 次直接 kernel launch、
8 次 `cudaMemcpyAsync`、9 次 `aten::copy_`、1 个逻辑 `aten::zero_`（内层
`aten::fill_` 2 条记录）。GC MoE 图有 3 个 kernel 节点，Mega MoE 图有 1 个；
3 张图合计分别记录 7 / 5 个 graph-node kernel 事件。

计数审计部分为 replay 后 3 个输出/计数 copy，以及下一次 replay 前 1 个 scratch copy
和 1 个 zero/fill；其余 5 个 copy、4 个 memcpy API 与 29 个直接 kernel launch 属于
模型 forward 或 dense local graph 包装。详见 `graph_coverage_compare.csv` 与
`graph_launch_details.csv`。

