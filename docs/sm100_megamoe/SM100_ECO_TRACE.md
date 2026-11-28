# SM100 EP8 的 ECO 和测量口径

本实现针对 DeepSeek-V4.1-Flash 的通用 fused GC 入口。原生、逐层 EPLB + RR、同放置 EPLB + ECO 使用 Bruce-Lee-LY/gc-moe 的 `af5a7a750800954f52b9ff28cc55d945ead48da6`。ECO 从 Natureal/gc-moe 的 `fb8b65fa829c97962eb2dbc9a4ad8b759b9cd736` 接入冷启动、按层暖状态、拍卖、配额、每个副本在源序列中的起点以及实际 pull。原生和 RR 继续使用物理路由 ID，ECO 在设备上消费逻辑路由 ID。通信分区留 1 SM 给 plan，计算分区保留双 CTA cluster，图中 plan、dispatch、compute 可并行。ECO 的 combine 需要获知物理写回方就绪。当前实现每个有输出任务的 warp 在读取前取得所有 peer 的完成 epoch，这部分开销计入 ECO。旧版单独强制 shared 优先的图依赖没有沿用。

默认价格取 Natureal 的实际块高对应参数。BM16 的价格为 `T + 249M + 140A`，BM32 为 `T + 272M + 117A`。显式设置 `GC_MOE_ECO_UNIT_CFG=26,2,1,6,0` 可重现 H20 的线性系数。报告使用该次运行的设备分数与实际参数。三臂正式入口在历史 EP8 启动器设置 `SM100_BRUCE_ECO=1`，每个运行使用独立目录。

## 与 SM90 的对照

依据 `docs/SM90_EP8_DECODE.docx` 的整模主表和 `sm90_bs1024_se_bench_results/{rr,eco}.replay2.perfetto.json` 核对。

| 项目 | SM100 采集与汇总 |
|---|---|
| BS、臂、MoE µs/层 | 每步每层先算每 rank 的首个 graph 角色到本 rank combine 结束，再取八 rank 最大值，最后平均。包含最早启动的 ECO plan。 |
| 相对原生、相对 RR | 同档耗时之比减一，以百分比显示，负数代表更快。通过既有配对公平性校验后填写。 |
| token IR、总 M 块、峰值 rank M 块、活跃物理槽、M 块 IR | pull 完成后按目的物理槽记录实际条数。按实际 BM 对每个槽分别向上取整。IR 每次先计算八 rank 最大值除以均值。另留峰值活跃物理槽。 |
| ECO routed MAE、成交/层调用 | 比较八个设备价格占比与实际 routed 时长占比，取绝对百分点差均值。保留报价、成交、接受率、零成交调用及 quota。八份拍卖副本核对一致后只计一次。 |
| 各 rank 原始数据 | 整模记录 step、layer、各 rank 物理槽实收、graph/routed 边界。采样层另存 BM、设备时钟、plan 分数及时间标定误差。 |
| replay 概览 | count exchange 与 pull token 在同一行，前者按历史口径从 dispatch 入场至首次真实 pull，包含路由、索引和等待。shared L1/L2 分轨。wait token 从首次 routed L1 进入数据就绪检查到首个真实 routed MMA。routed 含最后 L2 epilogue/push。combine 单列，ECO 另列 plan。 |
| absolute 明细 | 记录 dispatch CTA/warp 的逐 token pull 区间、实际凑齐 M 块的发布时间，及 compute cluster 的 shared/routed L1/L2 任务。另有每个 activation loader 等待 M 块的区间和逐 CTA 的 routed L2 epilogue/push 区间。每个任务带 SM、CTA、槽、M 块、N cluster 和 valid M。双 CTA 共用 leader 的 MMA 开始点，各用自己的 epilogue 完成点。 |

SM100 的 shared 与 routed 同属 compute role。shared 区间取实际 MMA 到对应 epilogue 的包络。SM90 历史图中独立 shared kernel 的驻留跨度与这里的起止定义有区别。所有 rank 通过各自两次 Kineto 时钟锚点对齐。区间重叠可直接读取，空白包含等待和未被对应事件覆盖的工作，不能单凭图判定 SM 在空转。

## 一个开关

在创建 workspace 和捕获 graph 前设置 `GC_MOE_SM100_TRACE`。

| 值 | 行为 |
|---:|---|
| 0 | 普通 GC/ECO 调用的默认值。设备时钟、计数和明细采集在编译时关掉，调用不分配、清零或拷出测量区。 |
| 1 | 采主表所需的设备时间和实际槽计数。三臂主表启动器默认选此档。 |
| 2 | 加入逐任务记录。三臂整模入口在重排后第 2 次测量 replay 采第 8、20、32 层，保存每 rank 的 torch.profiler 原件、设备数组、概览 Perfetto、absolute Perfetto 和 rank 计数。明细容量默认每 rank 每层 65,536 条，溢出会报错。 |

正式主表与详细 trace 使用不同运行目录。明细模式每个 replay 都有设备插桩，采样层另有图内快照，观测到的时长必须标注模式。导出的 `step0002-layer20.perfetto.json` 是第 20 层概览，`step0002-layer20.absolute.perfetto.json` 是其 CTA 明细。

## 配对测量开关开销

`scripts/sm100/eco_trace_overhead.py` 读取相同的已归档逐层放置、逻辑 top-k ID 和 gate 权重，复用同一组确定性算子权重和激活。它测选定层的图 replay。两个模式使用相同的重置、预热和 replay 次数，逐对交替顺序，均由图外 CUDA Event 计时。比较 BF16 输出、逐层暖状态及采集臂实际槽计数和 quota。输出逐对八卡最大耗时、配对变化和 bootstrap 区间。是否可以视为等效，按业务可接受的误差范围与该区间判断。整模主表另行测量。

在获准使用的空闲八卡环境，沿用三臂的 Python 环境、已打补丁的 GC、模型配置及固定放置。以下命令分别测概要和详细插桩自身的开销。报告保存到私有运行目录。详细插桩 A/B 不含导出文件或 Kineto 会话耗时。

~~~bash
torchrun --standalone --nproc-per-node=8 scripts/sm100/eco_trace_overhead.py "$group" --level 1 --out "$group/trace-0-1.json"
torchrun --standalone --nproc-per-node=8 scripts/sm100/eco_trace_overhead.py "$group" --level 2 --out "$group/trace-0-2.json"
~~~

本地验证覆盖 CPU 导出和离线 cubin 构建。多卡跨 rank 同步、连续 replay 的正确性、原生输出公平性和测量开销均须由 SM100 实机验证。
