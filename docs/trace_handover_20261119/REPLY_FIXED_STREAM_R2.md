# REPLY_FIXED_STREAM_R2 — **采集成功，但 collector 在 step 0 完整性断言处停止**

## 0. 结论速览

`1d6f1d4` 的修复**有效**：baseline 臂这次完整跑完了（上一轮在 `LLM()` 构造处就崩）。
采集本身 `exit_code=0`，40 份 counts、11 份 absolute、40 份总览 Perfetto 全部产出。

但 `multilayer_study.py collect` 在**验收断言**处失败：

```
ValueError: incomplete routed rows at step 0 layer 1
（multilayer_study.py:154，要求 105 × 40 每格都等于 256×6×8 = 12288）
```

**这不是回放失败**——诊断显示只有 **step 0（第一个 flush 步）** 的
`owned_totals` 不完整，**step 1–104 全部精确等于 12288**，包括采样步 step 2。
按任务书 §执行「运行、校验、解析或数值异常在首个失败步骤回传证据」，我停下回执，
未启动 stream 臂（第二次模型启动）。

## 1. 修复验证

`1d6f1d4` 改动与上轮回执的建议一致：

```diff
-        **({'logits_processors': ['fixed_tokens:ReferenceTokens']} if fixed_dir else {}),
+        model=os.environ['MODEL'], tensor_parallel_size=8,
```

不再向 vLLM 注册非法的 LogitsProcessor 插件，只保留
`_gc_moe_fixed_tokens` → `fixed_tokens.install()` 的 monkey-patch 路径。
四项 CPU 门禁全部 PASS（`FIXED_WORKLOAD_CPU` 这次还会模拟模型构造与首次请求，
并检查 worker 的采样后回填路径）。

**baseline 实跑确认**：模型正常加载、200 步预热完成、重排执行、
40 层采集与 11 层 fine 导出全部完成，全程 **0 次 Traceback**。

## 2. step 0 不完整的实测数据

| step | 角色 | 是否完整 |
|---:|---|---|
| 0 | flush 1 | ❌ **39/40 层不足**，总量 437,504 vs 应为 491,520 |
| 1 | flush 2 | ✅ 全部 12288 |
| **2** | **采样步（profiler）** | **✅ 全部 12288** |
| 3 | 恢复 1 | ✅ |
| 4 | 恢复 2 | ✅ |
| 5–104 | 计时 100 步 | ✅ 全部 12288 |

step 0 逐层合计（八 rank 汇总，期望 12288）：

```
L0: 12288  L1: 11904  L2: 9920  L3: 11520  L4: 10432  L5: 11200 ...
```

各 rank 在 step 0 的总量：47,920–59,028（期望 61,440），
缺口分布在整个 batch 上，不是个别 rank 掉线。

## 3. 根因：step 0 是路由环的首次访问

从 `integration/fixed_workload.py` 的实现看：

```python
def select(self, live_ids, live_weights, layer):
    torch.remainder(self.position, len(self.ids), out=self.index)
    ...
    torch.index_select(bank, 0, self.index, out=scratch)
    torch.where(self.replaying, scratch, live.reshape_as(scratch), out=scratch)
        bank.index_copy_(0, self.index, scratch)
```

`owned_totals` 来自 `finish_call()` 里的
`self.counts.index_copy_(0, self.index, buf.profile_slot_counts.index_select(0, row))`，
而 `row = epoch % 256` 指向 GPU 侧那张 256 行的 profile 槽计数表。

回放的 step 0 是重排后的**第一次完整 batch 调用**，
此时 `profile_slot_counts` 的 `epoch % 256` 那一行**还没有被本次回放的
40 层全量写过**（该表由 dispatch 角色按 epoch 递增写入，回放刚开始时
只写到了前若干层）。于是 step 0 读回的是**部分填充**的计数，
行数偏少；从 step 1 起环形表已经稳定，每一步都精确 12288。

**这与回放正确性无关**：路由本身由 `audit()` 独立校验——
`epochs[1:] - epochs[:-1] == 1`、`layer_ids == expected_layers`、
`digest(ids)/digest(weights)` 等于参考哈希——这些在 `collect` 里都通过的
（失败发生在它们**之后**的 `owned_totals` 断言）。也就是说
**送进 dispatch 的路由与参考完全一致**，只是 step 0 的"实测槽计数"快照不完整。

## 4. 为什么我没有自行放宽断言

任务书 §执行 明确：

> 禁止调整协议、门限、采样点、调度和解析口径。

`256*6*8` 这个断言属于**验收口径**。把 `for t in range(105)` 改成
`for t in range(1, 105)` 会改变验收条件；即使它不影响采样步 step 2，
也属于我方无权决定的改动。

另外，如果 step 0 的不完整是**设计上就该容忍的预热效应**，
那正确做法是由发布者把断言改成从 step 1 开始（或改成只对 step 2 严格），
而不是我在回执里悄悄跳过一格。

**需要发布者确认**：step 0 的 `profile_slot_counts` 不完整是否是已知的预热行为。
如果是，请给出修后的断路口径与新 SHA，我用它重跑两臂；
如果否，则说明 `RouteTape` 与 `profile_slot_counts` 的对齐有真实缺陷，
需要修回放实现。

## 5. 版本与环境

| 项 | 值 |
|---|---|
| 测量代码 SHA | `1d6f1d4a04ca11cdb06295ad0dc024ca9c1984ca` |
| 两臂内核 | 同一 SHA（同工作树） |
| baseline capture ID | `274641a56fb9452fae4c484bf95b3385`（first_epoch=14561）|
| 重放起点 epoch | 14481（回放 step 0）|
| GPU | 8×H20，driver 580.105.08 |
| 软件 | cutlass 4.8.1 / torch 2.13.0+cu130 / CUDA 13.0 / vllm 0.31.0 |
| 工作目录 | `/tmp/gcmoe-two-arm-IWEYaM5C` |
| 原始目录 | `/tmp/gcmoe-aligned-yoiKsWiA` |
| 共享路由目录 | `/tmp/gcmoe-route-replay-Q6W0KywL` |

## 6. 已完成的产出（baseline 臂，全部可用）

| 内容 | 数量 |
|---|---:|
| 层总览 Perfetto | **40** |
| counts | **40** |
| absolute Perfetto | **11** |
| `layers.csv` / `ranks.csv` | 已生成 |

这些文件本身是完整且经过 `exit_code=0` 校验的；
只是 `collect` 的后置断言让 `run.json` 没有写出来。

## 7. 建议

最小、且不改变语义的两种修法（供选择）：

1. **`for t in range(1, 105)`** —— 认为 step 0 是预热，不参与完整性校验。
   采样步 step 2 仍被严格检查，`audit()` 的路由哈希校验不变。
2. **只对采样步严格** —— `t == step` 时要求 12288，其余步骤只记录不强制。
   与任务书「此核对只覆盖给定回放负载的一个采样步」的表述更贴合。

两者都不改动 `RouteTape` 与内核，只调整 collector 的断言范围。

如果发布者认为 step 0 就应该是完整的，那么问题在
`profile_slot_counts` 的填充时序（回放起点早于该表首轮写满），
需要改 `fixed_workload.py` 的预热或 `finish_call` 的取值行。
