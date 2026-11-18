# 单层 7 段时序（最终版）

文件：`one_layer_final.perfetto.json` / `one_layer_final.png`
拖进 <https://ui.perfetto.dev>：8 个 process（rank 0-7），每个 7 条 thread（阶段）。

来源：`FINAL_V2` 一次运行，BS256、原始耗时函数（wb=26 wa=0 +token）、CUDA graph、
mb-claim、shared 由 gc_moe role 4 计算、1000 warm + 1000 measure。

## 7 段

| 段 | 数据来源 |
|---|---|
| role1 count exchange | trace 的 dispatch kernel 段内，按 `[10]→[4]` 与 `[4]→[5]` 的比例切分 |
| role1 pull token | 同上 |
| role4 shared L1+L2 | trace（grid=71 block=256）|
| role5 ECO plan | trace（grid=1 block=512）|
| role2 wait 1st M block | trace 的 compute kernel 段内，按 `[10]→[15]` 与 `[15]→[11]` 的比例切分 |
| role2 GEMM | 同上 |
| role3 combine | trace（grid=288）|

## 数值（µs）

| rank | count exch | pull token | shared | plan | wait 1st M | GEMM | combine |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 20 | 408 | 194 | 112 | 153 | 588 | 154 |
| 1 | 22 | 500 | 193 | 109 | 161 | 634 | 98 |
| 2 | 20 | 332 | 198 | 102 | 157 | 584 | 144 |
| 3 | 21 | 513 | 192 | 108 | 160 | 716 | 16 |
| 4 | 19 | 234 | 193 | 111 | 109 | 218 | 566 |
| 5 | 30 | 402 | 194 | 110 | 162 | 424 | 306 |
| 6 | 23 | 347 | 194 | 110 | 153 | 394 | 345 |
| 7 | 21 | 171 | 193 | 107 | 78 | 115 | 698 |

## 上一版为什么错

之前把 `pull_tl` 的设备打点**当作绝对时刻**去定位各段。`pull_tl` 是每 rank 独立的
uint32 微秒计数器（还会回绕），8 个 rank 的基准不同，混进 trace 的时间轴后：

- role4 shared 被画到 dispatch 之后很多（实际是同时开始，甚至会早 1-2 µs）
- routed GEMM 结束画到 combine 之后（实际 GEMM 结束 → combine 开始，间隔 2-5 µs）

现在**只用 trace 的 kernel 起止**做绝对定位，`pull_tl` 只用来在同一 rank 内算**比例**
（`[10]→[4]` 占 dispatch 的多少、`[10]→[15]` 占 compute 的多少），比例与时钟无关。

## 不变量校验（全部 8 rank 通过）

| 检查 | 结果 |
|---|---|
| role4 shared 起点 − role1 dispatch 起点 | −3.4 ~ +2.7 µs（同时开始）|
| role2 GEMM 结束 ≤ role3 combine 开始 | 全部成立，间隔 2-5 µs |
| 负时间戳 / 非正时长 | 0 |
