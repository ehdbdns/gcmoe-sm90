# 八卡执行任务书 · EPLB 重排后单步采样

> 当前执行[R5 共同 KV 快照双臂任务书](FIXED_STREAM_R5_TASK.md)。仅启动两次模型。到回执为止。[H20 valid 耗时模型对照](VALID_COST_TASK.md)暂缓，待任务一验证收益并收到交接更新后再执行两次。上轮[逐 M 块领票四组采样](BATCH_CLAIM_TASK.md)保留供查阅。

下文保留此前的单轮采样流程。该流程使用首次有效 CTA 领取打点提交 `cffe4835270f0d0bcc048b502a0b5716eac910b1` 及包含本任务书的后续提交，依次执行步骤 1 至 5。该轮验收禁止使用 `prof_phases.py`、`FINAL_V2` 或任何旧 run。

原单轮入口从仓库根目录执行。命令运行前人工核对八张 H20 已分配并空闲，核对同一个模型、数据集与工具链位置。脚本将保留执行版本、失败步骤、命令及日志。

```bash
git fetch origin main
git merge --ff-only origin/main
bash scripts/trace_layer/run_aligned_ep8.sh "$(git rev-parse origin/main)"
```

下面步骤 1–4 说明入口执行的命令和验收标准。诊断历史见 [交接说明](../trace_handover_20261119/FOLLOW_UP.md)。

## 执行权限

- 执行者负责检查环境、编译、采集、运行现有解析器以及回传产物。代码修改仅限步骤 2 中由编译器诊断定位的最小兼容性修复，例如 CuTe DSL 不支持的表达式写法。修复前保留原始错误，修复后按步骤 2 重测。任何编译修复单独提交，在回执中列出前后 SHA、诊断位置和修复依据。
- 冻结实验配置、内核行为、打点位置、调度、数据定义、解析器、测试、验收阈值和层选择。禁止关闭功能绕过编译、切换 graph/eager、改 batch、换层、换模型或数据集、改时间线、调整或手填 JSON。环境路径可指向执行机上的同一个模型、数据集和工具链。若必须改变实验配置或源码语义才能继续，按步骤 5 回传失败。
- 除上述编译兼容性修复外，任何一步失败都停在该步，按步骤 5 回执。保留首次失败日志和原始数据，等下一版明确指令。禁止改用旧归档交付、跨 run 拼接、补估计值或提交一半结果作为成功。

## 1. 固定版本并检查执行机

确保工作区干净。同步 `main` 时只快进。入口验证 HEAD 与指定 SHA 一致，并记录驱动、显存、PyTorch、CUDA 和 vLLM 版本。核对八张 H20 的分配及空闲情况。发现占用时回报，保留现有进程。

## 2. 运行 CPU 检查和两套编译冒烟

先确认 `ALIGNED_STEP_CPU PASS`。随后编译常规路径和插桩路径。以退出码和日志里的 `compiled roles` 为准。常规路径应包含角色 1、2、5、3。插桩路径应包含角色 1、4、2、5、3。

若遇到编译失败，先保留本次目录，再修复编译器指出的最小问题。提交修复后以新 SHA 重跑单一入口，确保常规路径和插桩路径都通过。如果无法在维护原语义的前提下修通，直接回传。模型运行时若出现 JIT 编译错误，也只允许这一类修复。修复后以新 SHA 重跑单一入口，脚本会新建目录并从头采集。

## 3. 重新采集一次 BS256

入口通过步骤 2 后预热权重，并执行固定的采集命令。运行顺序为 prefill、200 步 warmup、逐层 EPLB 重排、2 个未采样 decode 步、1 个正式采样步。采样步从重排后零基编号，所以报告里的 `step` 应为 2。记录八个 worker 的 trace 导出结果。采集耗时用于排错。

检查日志里的逐层重排、八个 worker 导出成功和 256 条序列各生成一个 token。缺文件、采集失败或环境异常时进入失败回执。

## 4. 解析固定的两层并验收

只用步骤 3 同一个 `raw/` 下的 `capture.json` 和八份 torch trace。选择该步零基 MoE 层序号 8 和 20，以解析器退出码为第一道验收。

逐份核对以下结果。任何一项失败都按步骤 5 回报。

1. 八个 rank 共享 capture ID。每个 rank 的 torch trace 必须有 40 层、每层五个角色。设备快照作为候选范围，按整步层间 dispatch 间隔与逐层设备打点严格配对，八卡必须共同匹配唯一的一组连续 40 个 epoch。每层报告与对应 Perfetto 相同 step、layer、epoch，原始窗口和时钟残差校验通过。
2. 每个 Perfetto 有 56 个区段，即八个 rank 各含 count exchange、pull token、shared L1+L2、ECO plan、wait token、GEMM、combine。count exchange 到最早的 pull CTA 进入 slot 循环为止，覆盖 routing、push 与 peer readiness rendezvous。count exchange 与 pull 合计等于 dispatch，wait token 与 GEMM 合计等于 routed。
3. GEMM 起点是首个 routed CTA 取得有效 L1 任务且输入已就绪时的设备打点。每个交付层至少有一个 rank 同时满足 `pull_after_first_claim_us > 0` 和 `drawn_pull_gemm_overlap_us > 0`。在 Perfetto 中核对这些 rank 的 pull 与 GEMM 区段确实重叠。
4. 每层有 0–7 共八条负载记录。逐 rank 回报 `tokens`、`routed_rows`、`m_blocks` 和 `block_m`。`tokens` 按实测输入复制倍数折算，`m_blocks` 逐槽对 `owned / block_m` 向上取整后求和。八个 rank 的 `routed_rows` 总和等于全部输入行数乘 top-k。沿用解析器给出的 plan 代价和归一化 L1 差值。
5. 如导出 PNG，直接使用同一份已验收数据。这是可选交付。

## 5. 回传到 GitHub

- **成功**。把两份 `*.perfetto.json` 和两份 `*.counts.json` 原样放入本目录的 `results/`，提交并推送。在本目录新建 `REPLY.md`，填写起始与执行 SHA、是否有编译修复、机器与运行配置、capture ID、step、两层及 epoch、八卡负载和重叠时长。附原始 `capture.json`、八份 torch trace 和日志的可访问位置及校验值。
- **失败**。仍提交并推送本目录的 `REPLY.md`。填写首个失败步骤、执行 SHA、完整命令、退出码、错误原文和日志位置。能获得 rank、epoch、trace 行号或残差时一并附上。保留已有的原始采样目录和校验值。失败或部分结果写明缺项，留在诊断目录中。
- 提交时只暂存本任务相关文件，编译兼容性修复单独提交。推送冲突时保留本地提交，回报冲突。

成功时把 `out` 设置成入口输出的实际目录，再复制和暂存四份产物。失败时跳过复制，只暂存回执。推送前检查暂存文件。

```bash
task=docs/trace_executor_20261119
mkdir -p "$task/results"
cp "$out/deliver"/*.perfetto.json "$out/deliver"/*.counts.json "$task/results/"
git add -- "$task/results/" "$task/REPLY.md"
git diff --cached --stat
git commit -m "Report aligned EP8 profile results"
git push origin HEAD:refs/heads/main
```

回执同时适用于成功和失败。回复提交到本目录后，设计端会核对，并决定是否需要新的步骤。
