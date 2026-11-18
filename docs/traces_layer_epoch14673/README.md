# 单层 8-rank torch profile 切片（epoch 14673）

从 BS256 + `GC_MOE_ECO_COSTFN=1` + EPLB 重排 + `ENFORCE_EAGER=1` 的 torch profile
中，切出**同一层**（epoch tag `14673`，trace 内 call #128）在 8 个 rank 上的全部事件。

- 源：`prof/prof_eco_bs256_v3/traces/rankN.json`（每 rank 1.18 GB 全量 chrome trace）
- 切片窗口：绝对 µs `4077691964997 .. 4077691966987`（约 1990 µs）
- 每文件约 58–64 KB

## 这个切片修正了什么

上一版切片**在 Perfetto / chrome tracing 里打开是空的**，原因有三：

1. **缺 `process_name` / `thread_name` 元数据（`ph:"M"`）** —— 没有这些，查看器不建轨道。
   现在每个文件都补了 8 条：host 进程 + 5 条设备 stream（`stream 23` 是主 stream，
   `2082/2083/2084` 是 gc_moe 的 green-context stream）。
2. **时间戳是 4.08e12 绝对 µs** —— 查看器时间轴会缩放到不可用。
   现在全部 **rebase 到 0**（相对本层第一次 dispatch）。
3. **`ac2g` flow 事件只剩一端** —— 按窗口过滤时成对事件会被拆开。现在成对保留。

另外附了 `layer_timeline_from_slice.png`：**直接由这些切片文件画出来**，
所以 PNG 与 JSON 内容必然一致，可用它交叉校验查看器的显示。

## 怎么打开

任选其一，把 `rankN.json` 拖进去：

- <https://ui.perfetto.dev>（推荐，chrome://tracing 已在新版 Chrome 移除）
- 本地 `python -m perfetto` 或任意支持 chrome trace JSON 的查看器

打开后应看到 5 条轨道：`stream 23` 上有 shared（marlin/nvjet）和大部分 elementwise；
`2082` = dispatch（grid 24）、`2083` = routed compute（grid 71）+ combine（grid 288）、
`2084` = ECO plan（grid 1）。

## 这一层的时间结构（µs）

| 阶段 | 定义 | 实测（rank0..7 范围） |
|---|---|---|
| count exchange | `[10]` → `[21]`（本 rank 24 CTA 齐） | 16–19 |
| wait for peers | `[21]` → `[4]`（跨 rank rendezvous） | **1–439** |
| token pull | `[4]` → `[5]` | 49–97 |
| shared L1+L2 | marlin/nvjet 首末 kernel | 193–219 |
| **wait token** | compute kernel launch → `[15]` | **30–500** |
| routed GEMM | `[15]` → compute exit | 42–120 |
| combine | `[12]` → `[13]` | 8–56 |

两个容易误读的点：

- **`count exchange` 的结尾在 rank 间不对齐**（散 438 µs）。真正同步点是 `[4]`，对齐在 1 µs。
  各 rank 的 `wait for peers` 长度恰好吸收了这个错峰。
- **`wait token` 不是 GEMM**。routed compute kernel 是 graph 根节点，很早就被 launch，
  但 CTA 停在 mbarrier 上等第一个 M 块落地。把它算进 GEMM 会让 GEMM 看起来长 5–10 倍。
