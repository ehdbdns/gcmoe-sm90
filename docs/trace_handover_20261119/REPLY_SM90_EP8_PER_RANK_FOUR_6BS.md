# 回执 · SM90 · 8×H20 · 每源 rank 独立请求的六档四臂复测

执行者：SM90 八卡执行者。任务书 `docs/trace_executor_20261119/SM90_EP8_PER_RANK_FOUR_6BS.md`，测量 SHA `774d077`。

## 结论：第一档第一臂在 prompt 选择之后立即失败，24 次运行全部未启动

`exit-code.txt = 1`、`bs64.exit = 1`。**0 个臂完成**，`delivery-64/bs64/` 下没有任何 `complete` 标记。

## 失败点

```
[data] 64 prompts (8/rank x 8), len 257..2017
[budget] decode_budget=2119 (1+1000warm+1000measure+118slack) prefill_tokens=55141 chunk=2048 kv_margin=256
Traceback (most recent call last):
  File ".../real_decode_100_100.py", line 1439, in <module>
    main()
  File ".../real_decode_100_100.py", line 840, in main
    raise ValueError(f"BS{nseq}: prompt selection differs from the archived cohort")
ValueError: BS64: prompt selection differs from the archived cohort
```

注意：prompt 选择**本身是成功的**（`64 prompts (8/rank x 8)`，长度 257–2017，符合任务书「长度保留在 256–2048」），失败发生在随后的**归档一致性校验**。

## 根因：`--per-rank` 分支错误地继承了 `GC_MOE_EP8_LINEAR=1`

`scripts/sm90/run_full_model_four.sh`：

```bash
line  5:  unset ... GC_MOE_EP8_LINEAR SM90_LINEAR_WA_PAIR ...
line 12:  export GC_MOE_EP8_LINEAR=1 GC_PREFILL_MARGIN=256     # ← 无条件执行
line 13:  batches=(32 64 128 256 512 1024)
line 14:  if [[ ${1:-} == --per-rank ]]; then
line 15:    source_sharded=1
line 16:    batches=(64 128 256 512 1024 2048)                # ← per-rank 用自己的 batch 表
line 17:    shift
...
line 112:   export GC_MOE_SOURCE_BATCH=$((bs / 8))            # ← 这里才设置 per-rank 语义
```

`real_decode_100_100.py` 第 827–840 行：

```python
if os.getenv("GC_MOE_EP8_LINEAR") == "1":
    if nseq < 256:
        group = f"main24_cost_results/bs{nseq}"
    elif nseq == 256:
        group = "sm90_bs256_main_five_plan_comm_results/delivery/bs256"
    else:
        group = f"sm90_bs512_bs1024_full_model_four_results/delivery-{nseq}/bs{nseq}"
    archive = ... / group
    if prompt_hash != json.loads((archive / "vanilla/run.json").read_text())["prompt_hash"]:
        raise ValueError(f"BS{nseq}: prompt selection differs from the archived cohort")
```

这段校验的语义是「**全局 BS = nseq 的历史档**」，对 `--linear-ep8` 成立。

但 per-rank 任务里 `nseq = 64` 是**每源 rank 的流数**（全局 512 个流），且 `select_decode_prompts` 走的是 `GC_MOE_SOURCE_BATCH` 分支（第 159 行，产生"每源 rank 独立、互不重叠"的请求集），与 `main24_cost_results/bs64` 那套全局 64 请求的归档必然不同。

因为 `GC_MOE_EP8_LINEAR=1` 在第 12 行就被设置，而 `GC_MOE_SOURCE_BATCH` 直到第 112 行才设置，这段归档校验在 per-rank 路径上被误触发。**它对 per-rank 语义永远不可能通过。**

## 已启动次数与状态

| 项 | 值 |
|---|---|
| 模型启动 | **1 次**（`delivery-64/bs64/vanilla`，失败）|
| 完成的臂 | **0 / 24** |
| `delivery-64/bs64/` | `vanilla/` 仅有 `runner.log`、`hardware.csv`、`preexisting-processes.csv` |
| 其余五档 | 未启动 |
| 八卡 | 已释放（各 4 MiB、无 compute app）|
| 代码修改 | **无**（diff 为空）|

## 为什么没有自行修复

任务书规定「执行者只可修保持语义的编译问题」。这条不属于编译问题，而是**校验口径与分支语义冲突**，修复需要改动其中一处：

1. 在 `--per-rank` 分支不导出 `GC_MOE_EP8_LINEAR=1`（但该变量可能还被下游用于选择线性价格/放置路径，贸然去掉会改变实验语义）；或
2. 在 `real_decode_100_100.py` 的归档校验前增加 `and not os.getenv("GC_MOE_SOURCE_BATCH")`（明确的语义豁免）。

两者都会影响实验行为，超出「保持语义」的范围，因此我只固化现场。

## 建议的修复方向（供设计端裁定，未执行）

**最小、最贴合任务书意图的修法**：在 `real_decode_100_100.py` 的归档 prompt 校验前加 per-rank 豁免：

```python
if os.getenv("GC_MOE_EP8_LINEAR") == "1" and not os.getenv("GC_MOE_SOURCE_BATCH"):
```

理由：`GC_MOE_SOURCE_BATCH` 只在 `--per-rank` 分支被设置（第 112 行），此时 prompt 集由「每源 rank 独立、互不重叠」的新规则产生，本就不应和任何按全局 BS 归档的 prompt 比对。任务书第 4 条也明确要求「六档各自建立新参考」「旧表的源路由重复，旧数据不能改列名或除以 8 后并入本表」。

若设计端认为 `GC_MOE_EP8_LINEAR` 本身也不该在 per-rank 路径上生效，则需先确认它在 ECO 价格/放置逻辑里承担的用途，再决定是否移除。

## 现场与归档

- 现场：`/mnt/fuse/sm90-ep8-per-rank-SmxFr1ik`
  - `delivery-64/bs64/vanilla/runner.log`（完整首错，含 `[data] 64 prompts` 与 `[budget]` 行）
  - `delivery-64/bs64/vanilla/{hardware.csv,preexisting-processes.csv}`
  - `bs64.log`、`bs64.exit`、`exit-code.txt`、`launcher.log`、`measurement.sha`、`input-check.log`、`per-rank`
  - 未生成任何 KV、route bank 或 NPZ（失败发生在模型加载之前）
- 已归档到 `docs/trace_executor_20261119/sm90_ep8_per_rank_four_6bs_results/`（仅首错与启动证据）。
- 八卡已释放；未占用站点预约；无残留 worker。
