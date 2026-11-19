# 八卡任务书 · 查清 pull 与 routed GEMM 的慢 rank

本轮以 `08b8481` 收录的 absolute trace 为细粒度事件来源。保持已验收的 BS256、逐层 EPLB、MB-CLAIM 和 CUDA Graph 配置。对重排后 step 2 的零基 MoE 层 8、20 同时采集八卡 torch profile、逐槽路由行数和细粒度事件。所有分析由本轮同一 capture ID 的 40 层记录定位，旧的 epoch 14729、14741 只用于对照。

上次 layer 8 的 rank 6 有 189 个折算 token、31 个 M 块、11 个活跃槽，其中 local slot 47 有 1040 行。请在新采样中找出实际最后完成 routed 的 rank，并验证热点槽是否真的拖后了任务发布、是否导致计算 CTA 等工作，以及最后退出的是哪一阶段。若本轮重排或 trace 开销改变了慢 rank 或热点槽，按本轮实测标注，并说明旧现象是否重现。

## 1. 在八卡机从唯一入口执行

确认工作区干净、八张 H20 空闲、模型与数据集路径可访问。快进到本任务所在的 `origin/main`。

```bash
git fetch origin main
git merge --ff-only origin/main
bash scripts/trace_layer/run_aligned_ep8.sh "$(git rev-parse origin/main)" --fine-trace
```

入口先跑两项 CPU 回归，再编译常规四角色与插桩五角色。两套角色验收通过后才加载模型。记录入口打印的 `Capture workspace` 路径。若失败，保留该目录的 `status.txt`、执行 SHA、报错和原始文件。

执行者只可做保持原语义的最小编译兼容性修复，且单独提交后从入口重跑。其他失败停在原步骤并回执。禁止改层、换 batch、改计时或重排配置、放宽检查、补画数据，或把不同 run 的事件拼接。

## 2. 核对配对与时钟

- `--fine-trace` 在测量步前启用按 rank 独立的设备时钟原点。细粒度环容量固定为每 rank 4,194,304 条（160 MiB）。采集完一整个 decode 步后，从环形缓冲中只导出已经与 torch trace 对齐的两个目标 epoch。检查 `raw/fine.json` 的 capture ID、八个 rank、两个 epoch 及各 rank 的 `count/capacity`。若回绕或任一目标缺记录，回报失败。
- 区段用当前内核的 10 字记录，epoch 在索引 8。type 1／2 分别是 slot 开始／完成，`id` 为全局槽号，`a0` 为本地槽号。type 1 的 `a1` 是 dispatch CTA，type 2 的 `a1` 是 owned 行数。type 3 记录副本在 plan 上的等待。type 6／7 记录同一个计算 CTA 的 M 块／N-tile 开始与结束，type 8／9 记录同一 L2 tile 的 push。
- 旧画图脚本会自动挑一个 epoch，CSV 示例也有旧布局。此次以入口调用的 `pull_gemm_detail.py` 为准。它用完整事件键配对，从每槽 owned 和 block M 重建预期任务集合，逐 rank 检查 pull、L1/L2 和 L2 push 的完整性。有错的 epoch 留在诊断目录。
- 粗粒度 `GEMM` 仍从首个有效 CTA 领任务计时。细粒度 type 6 是该计算 tile 的开始，type 6 到 7 可能包含加载、同步和 L2 push。type 4／5 是队列观察的边沿，两个边沿间没有 tile 只能证明这段未观测到 tile，结合就绪、票号和依赖判断等待原因。L2 push 不是独立的 combine kernel。

## 3. 回答慢 rank 为什么慢

每个目标层分别给出八卡表，包含 token 数、实际 routed rows、M 块数、活跃槽数、最重的槽及其 CTA、首个有效 claim、最后 pull 完成、最后 L1/L2 tile 和最后 push、routed exit。时间以本 rank dispatch 入口为零，数值用微秒。用同一份细粒度 Perfetto 核对表。

每层以 **routed 绝对结束时间最晚**的 rank 为主，另选 token 或 M 块数接近但退出更早的一张卡作对照。分别回答：

1. **拉取是否拖后。** 写出最后三个完成 pull 的槽、各自的实际行数、所属 CTA、该 CTA 先前处理过的槽、plan wait 和完成时刻。核对最重槽是否也是最后就绪槽，并量化它与次晚槽的差距。
2. **后续 CTA 有无工作。** 对照热点槽 pull 完成、该槽第一个 L1 tile、当时活跃 CTA 数和队列边沿。给出至少一段可复核的时间区间，注明有哪些任务在跑、有哪些 CTA 报告了空队列。若空队列证据缺失，写明观测范围。
3. **计算尾部拖在哪。** 列出最后退出的几个 L1/L2 tile 和最后推送的 L2 tile，包含本地槽号、CTA、M 块、N-tile、valid 行数、起止时间。量化从最后 pull 到最后 tile、从最后 tile 到 routed exit 的时差，说明最慢 rank 是被迟到的输入、L1/L2 依赖、tile／CTA 分配、L2 push，还是 kernel 退出阶段拖慢。证据不足的环节标成待定位。
4. **跨卡等待落在哪。** 用粗粒度时间线观察其它 rank 的 combine 何时进入、何时结束。把本 rank 的 pull/GEMM 拖尾与其它 rank 等待 peer 的时间分开。细打点带来额外原子开销，当前测量用于定位机制与顺序。

## 4. 提交产物和回执

把两层的原始 `*.perfetto.json`、`*.counts.json` 以及对应的 `*.detail.perfetto.json`、`*.detail.json` 原样提交到本目录的 `fine_results/`。在 `REPLY_FINE_TRACE.md` 写入实际执行 SHA、capture ID、step／layer／epoch、上面的对照表和结论。附原始 torch trace、`capture.json`、`fine.json`、日志的可访问位置及 `sha256.txt`。无需把整份原始环全部提交到 Git。

失败时仅提交回执，包含失败步骤、命令、错误原文、相关 rank／epoch 和已保留的原始数据路径。以本回执区分已确认的瓶颈与仍需额外打点的环节。
