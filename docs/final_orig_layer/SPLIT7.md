## 7 段拆分（dispatch 与 routed GEMM 各自再拆两段）

同一层（trace idx 39608，pull_row #184），全部来自 `FINAL_V2` 一次运行。

| 段 | 起 | 止 | 数据来源 |
|---|---|---|---|
| role1 **count exchange** | `[10]` dispatch 入口 | `[4]` 即将处理第一个本地槽 | 设备打点 |
| role1 **pull token** | `[4]` | `[5]` 最后一个本地槽拉完 | 设备打点 |
| role4 shared L1+L2 | kernel 起 | kernel 止 | torch trace |
| role5 ECO plan | kernel 起 | kernel 止 | torch trace |
| role2 **wait 1st M block** | r2 launch | `[15]` 首个 CTA 领到任务 | trace + 设备打点 |
| role2 **GEMM** | `[15]` | `[11]` compute 退出 | 设备打点 |
| role3 combine | kernel 起 | kernel 止 | torch trace |

| rank | count exch | pull token | shared | plan | wait 1st M | **GEMM** | combine |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 22 | 449 | 194 | 112 | 195 | **749** | 154 |
| 1 | 22 | 502 | 193 | 109 | 197 | **775** | 98 |
| 2 | 22 | 362 | 198 | 102 | 198 | **738** | 144 |
| 3 | 22 | 534 | 192 | 108 | 195 | **872** | 16 |
| 4 | 21 | 262 | 193 | 111 | 191 | **380** | 566 |
| 5 | 24 | 325 | 194 | 110 | 198 | **518** | 307 |
| 6 | 25 | 381 | 194 | 110 | 204 | **526** | 345 |
| 7 | 23 | 191 | 193 | 107 | 197 | **289** | 698 |

### 三个量化结论

**1. count exchange 恒定 21–25 µs（差 1.19 倍）**

它是 routing + count 交换 + 一次跨 rank `wait_epoch`，与负载无关。

**2. pull token 差 2.80 倍（191–534 µs）—— 这是 dispatch 差异的唯一来源**

`[4]` 的注释写明：单副本槽立即拉，**副本槽必须等 plan 发布 quota 才能拉**。
所以 pull 长短取决于「本 rank 有多少副本槽」以及「plan 何时发布」。
rank7 的 pull 只有 191 µs（最短），rank3 是 534 µs（最长）。

**3. routed 的差异 100% 在计算，不在等待**

`wait 1st M block` 全部 191–204 µs（1.07 倍），`GEMM` 是 289–872 µs（3.02 倍）。

### 一个问题：pull 和 role2 的关系

`[5]`（pull 结束）不是 role2 的启动条件——role2 在 `[5]` 之前就已 launch 并开始领任务。
role2 的 `[15]`（首个任务）≈ `[10] + 195 µs`，而 pull 结束在 `[10] + 213~559 µs`。

也就是说 **role2 的第一个 M 块在 pull 还没全部结束时就已经到位**（`MB_CLAIM=1` 按块发布），
这正是 `wait 1st M block`（195 µs）远小于 `pull token`（最长 534 µs）的原因。
