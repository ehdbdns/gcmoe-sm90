# 【SM100】V41 BS512 dense 同图提交复测回执

执行者：**SM100**。本轮只执行 `V41_BS512_DENSE_SUBMIT_FIX.md`，未执行任何 SM90
任务书。Mega、GC 各在预约的同一组八张 L20D 上启动一次，串行运行；CPU 用于启动、
RPC、导出和离线分析，GPU 仅用于两臂模型。未改模型、MoE kernel、dense 计算、
输入、逐层计时或密集回放语义。

## 结论

- 两臂退出码均为 **0**。每臂先跑 1000 个统计步，最后两个真实 decode 步完成
  dense 图 capture/validate，再跑 7 个测量步；dense 回放开关实际为
  `off,on,on,off,on,off,on`。八个 rank 在每一步都是 **82/82** 张校验过的图，
  每 rank `validated_layer_graphs=40`、`block_m=16`。验收通过。
- **dense 局部间隔确实缩短。** L20 的 quantize→dense 同流配对在 8 个 rank ×
  2 个调用上共 32 条，间隔为 Mega `-0.704..+0.159 us`、GC `-0.863..+0.321 us`；
  旧归档的 4 条对应记录为 Mega `223.842/269.537 us`、GC `204.162/217.505 us`。
  4/4 条旧记录的 quantize 名、dense 名、形状和调用顺序均精确匹配。
- **跨 rank 到达时差和完整 MoE 跨度有改善，但不是两臂全层一致。** L20 的两次
  `关→开` 转换在 Mega、GC 都缩短；Mega 的五层中位数在两次转换都缩短；GC 的
  3→4 五层中位数缩短，但 5→6 只有 2/5 层缩短，L0/L8 反而变大。
- **局部空档缩短没有被解释成到达错峰的充分原因。** 本轮只报告两者并存和逐层
  差异，不把 dense 同图提交当作跨 rank 收益的唯一因果解释。

## 执行与环境

| 项目 | 值 |
|---|---|
| 任务书 | `docs/sm100_megamoe/V41_BS512_DENSE_SUBMIT_FIX.md` |
| 任务书 SHA256 | `230ddc53f59ea592d976b1544027d5582bdcb2a52f4769efca7ec0111a9862e3` |
| 执行源码提交 | `17154241bb76fab4c80cbbc7f47c4982e01ba167` |
| upstream | `fb8b65fa829c97962eb2dbc9a4ad8b759b9cd736` |
| 输入 fixture SHA256 | `ceaa52dc3f3b09fa95e08c59b83b56b97d90b8335b3437a506cb5dbd42fe09f2` |
| CPU | 256 × Intel Xeon 6776P（2 socket × 64 core，2 thread/core） |
| GPU | 8 × NVIDIA L20D；UUID 见 `gpus.txt` |
| Python / Torch / vLLM | 3.12.7 / 2.13.0 / 0.31.0 |
| DSL / FlashInfer | `nvidia-cutlass-dsl==4.8.1` 五个包 / `flashinfer-python==0.7.0.post1` |
| ptxas | CUDA 13.5.37 |
| GPU 锁 | `2026-11-22T22:54:50+08:00` 取得 |
| Mega | `22:55:01`–`23:01:16`，exit 0 |
| GC | `23:01:17`–`23:07:12`，exit 0 |
| 释放 | `23:07:44` 写出 `released.log`；之后复验 GPU 与锁均空闲 |

RR 与 physical-load preflight 通过。PyPI 的 91 MB CU12 wheel 下载停滞，改从本机
pip HTTP 缓存取五个 wheel；五个 SHA256 全部对旧 manifest 校验通过
（`wheel-check.txt`），随后以 `--no-index` 安装。

## Dense graph 与验收

| 检查 | Mega | GC |
|---|---|---|
| 1000 统计步 + 7 测量步 | 是 | 是 |
| step 0–6 dense 模式 | off,on,on,off,on,off,on | 同左 |
| 每步 8 rank `graphs=validated` | 82/82 | 82/82 |
| 每 rank 40 层图 | 40 | 40 |
| `block_m` / RR 调用 | 16 / 0 | 16 / 0 |
| 8 个 measure NPZ | 8 | 8 |
| Perfetto / overview | 30 / 30 | 30 / 30 |

step 1 保留真实 Kineto kernel；step 2–6 只使用外锚点与设备内时间戳。每个
step/layer 的八 rank token 合计均为 3072。`validation.txt` 全部通过。

## L20 局部空档：旧证据同调用匹配

新 CSV 的 32 条记录覆盖两臂 8 个 rank 各 2 个调用。L19→L20 窗口内调用顺序为
**#1 shape 5120、#2 shape 1024**，两臂所有 rank 的 stream 都是 23。旧归档只保留
两臂早晚 entry 各一条，因此只对这 4 条做逐条匹配；旧 correlation ID 在本轮图回放
中已重新编号，不能按 correlation 匹配，改名以外均精确对上。

| Arm | 旧角色 | Rank | Shape | 旧 gap (us) | 新 gap (us) | 差 (us) | 新调用位置 |
|---|---|---:|---:|---:|---:|---:|---:|
| GC | early | 0 | 1024 | 204.162 | 0.159 | 204.003 | #2 |
| GC | late | 2 | 5120 | 217.505 | -0.640 | 218.145 | #1 |
| Mega | early | 6 | 1024 | 223.842 | -0.031 | 223.873 | #2 |
| Mega | late | 4 | 5120 | 269.537 | -0.576 | 270.113 | #1 |

按形状汇总全部 rank 的中位数：

| Arm | Shape | n | 中位数 (us) | 范围 (us) |
|---|---|---:|---:|---:|
| Mega | 1024 | 8 | -0.031 | -0.032..+0.159 |
| Mega | 5120 | 8 | -0.592 | -0.704..-0.384 |
| GC | 1024 | 8 | +0.128 | +0.033..+0.321 |
| GC | 5120 | 8 | -0.608 | -0.863..-0.320 |

负值是同一 stream 相邻 Kineto event 边界的亚微秒重叠，物理串行含义是**没有可观测
正间隔**；不要把负值当成额外收益。可确认的是旧 204.162–269.537 us 的 CPU 提交型
空档已消失，局部缩短量约 204–270 us。旧归档没有其余 6 个 rank 的同调用记录，
缺失覆盖已在 `step001_dense_gap_match.csv` 与本表中明确。

## L20 五步逐项

`host` 是跨 rank 的 Python `graph.replay` 调用前时差；`entry` 是 GPU MoE 入口时差；
`maxex` 是最长交换；`global` 是完整八卡 MoE 跨度。单位均为 us。

| Arm | Step | Dense | 早 rank | 晚 rank | host | entry | maxex | global | max tokens | max M blocks |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Mega | 2 | on | 4 | 0 | 28163.003 | 290.556 | 307.296 | 517.861 | 987 | 88 |
| Mega | 3 | off | 1 | 0 | 4377.426 | 287.750 | 303.072 | 513.391 | 968 | 90 |
| Mega | 4 | on | 0 | 7 | 3541.715 | 269.888 | 287.392 | 499.338 | 979 | 92 |
| Mega | 5 | off | 4 | 0 | 3869.264 | 301.756 | 319.136 | 543.836 | 991 | 95 |
| Mega | 6 | on | 3 | 0 | 3431.042 | 260.311 | 277.312 | 488.057 | 978 | 89 |
| GC | 2 | on | 3 | 0 | 21846.830 | 441.104 | 453.376 | 690.437 | 962 | 87 |
| GC | 3 | off | 7 | 0 | 24004.984 | 311.372 | 322.112 | 566.816 | 955 | 90 |
| GC | 4 | on | 2 | 3 | 2195.480 | 270.719 | 281.568 | 525.540 | 991 | 90 |
| GC | 5 | off | 4 | 7 | 3127.286 | 311.724 | 322.112 | 570.731 | 985 | 88 |
| GC | 6 | on | 5 | 0 | 6397.186 | 281.537 | 291.712 | 536.245 | 977 | 89 |

五个采样层按开关分组的中位数：

| Arm / 范围 | Dense | entry | maxex | global |
|---|---|---:|---:|---:|
| Mega 全部 on | on | 275.840 | 292.224 | 472.188 |
| Mega 全部 off | off | 300.958 | 319.184 | 506.947 |
| GC 全部 on | on | 321.418 | 333.440 | 526.927 |
| GC 全部 off | off | 301.233 | 311.200 | 501.758 |
| Mega step4/6 | on | 272.864 | 289.808 | 470.448 |
| GC step4/6 | on | 286.979 | 297.184 | 496.223 |

GC 的“全部 on”包含第一个外锚点测量步 step 2；该步 host spread 为 21.847 ms，
entry/global 为 441.104/690.437 us，是明显的首步异常。排除 step 2 后，GC 的
step4/6 on 中位数低于 step3/5 off。

相邻转换的中位数（五层，on 减 off）：

| Arm | 转换 | entry Δ | maxex Δ | global Δ | 下降层数 |
|---|---|---:|---:|---:|---:|
| Mega | 3→4 | -22.670 | -24.672 | -28.590 | 5/5 |
| Mega | 5→6 | -41.445 | -41.824 | -55.779 | 4/5 |
| GC | 3→4 | -40.653 | -40.544 | -41.276 | 4/5 global，5/5 entry/maxex |
| GC | 5→6 | +6.271 | +5.792 | +13.844 | 2/5 |

L20 本身：Mega 3→4 为 `-17.862/-15.680/-14.053 us`，5→6 为
`-41.445/-41.824/-55.779 us`；GC 3→4 为 `-40.653/-40.544/-41.276 us`，
5→6 为 `-30.187/-30.400/-34.486 us`。因此 L20 两臂两次转换均下降；GC 的
五层 5→6 混合结果不能概括为整体改善。

## Step 2：旧归档与新测量

| Arm | changed slots | 旧 host (CUDA API) | 新 host (Python replay) | 旧→新 entry | 旧→新 maxex | 旧→新 global | 对齐半宽 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Mega | 281/384 | 9461.480 | 28163.003 | 383.436→290.556 | 401.920→307.296 | 607.488→517.861 | 0.832→0.912 |
| GC | 276/384 | 2973.841 | 21846.830 | 349.609→441.104 | 358.976→453.376 | 590.534→690.437 | 0.817→0.929 |

单位 us。两列 host 一个是旧 CUDA API 入口时差，一个是新 Python `graph.replay`
调用前时差，不能当成同一 API 的精确时长。Step 2 的 Mega 三项分别改善
`92.880/94.624/89.627 us`；GC 三项分别恶化 `91.495/94.400/99.902 us`。
GC step 2 同时有 21.847 ms 的 Python replay 调用时差，且它是第一外锚点测量步；
因此不能把 GC step 2 的差值单独归因于 dense 回放本身。

L20 旧/新路由量：Mega rank token 范围 `158–977 → 158–987`、M 块
`40–88 → 42–88`；GC `159–944 → 162–962`、M 块 `43–85 → 40–87`。两臂总额
均为 3072 token。五层 changed slot 数为 268–304/384，L20 为 281/384 与
276/384；路由实例并不完全相同，逐层解释需保留这一限制。

## 时钟锚点与后处理修复

两臂的 `clock-before` 短会话都出现 PyTorch 导出缺陷：锚点 kernel 为
`ts=0,dur=0`，但 GPU globaltimer 读数完好；`clock-after` 锚点有效。首轮
`native_phase_perfetto.py` 因而报
`rank 0: unstable clock alignment 4502618961800.102 us`，`launcher.log` 保留
该失败，模型两臂本身均为 exit 0。

分析器改为：保留前后两个 GPU 读数作为测量窗口，只用有效的 after 锚点做设备到
Kineto 的线性映射；`alignment_drift_us=null`，不伪造漂移。该改动不触碰设备时钟、
模型或测量数据，diff 为 `analyzer-fallback.diff`。回退核验把 before 的 GPU 锚点
映射回其 profiler 会话，16/16 rank 均落在会话 span 内（距开始 1.973–3.343 ms，
距结束前 76.9–118.0 us）。after 锚点半宽为 Mega `0.480–0.912 us`、GC
`0.432–0.929 us`。回退后 Perfetto 与 skew 脚本均 exit 0。

## 图与文件

- 三张必需 CSV：`step001_dense_gaps.csv`、`within_run_skew.csv`、`step002_skew.csv`。
- 每臂 `result.json`、8 个 `measure-rank*.npz`、30 张 `perfetto/`、30 张 `overview/`。
- 释放证据：`gpu_release.txt`、`lock_release.txt`、`processes-stopped.txt`；当时无
  compute app，八卡利用率 0%，每卡仅 11–138 MiB 显示占用；两个 flock 均可立即取得。
- 原始 profile、前后校准会话和锚点 NPY 未提交；`raw-evidence.sha256` 共 112 条，
  全部 `sha256sum -c` 通过，路径见 `raw-evidence-paths.txt`。

L20 GitHub 链接（SM100 结果目录）：

- Mega step 1：[overview](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/sm100_megamoe/v41_dense_submit_results/mega/overview/step001_layer20.perfetto.json) · [详细](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/sm100_megamoe/v41_dense_submit_results/mega/perfetto/step001_layer20.perfetto.json)
- Mega step 2：[overview](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/sm100_megamoe/v41_dense_submit_results/mega/overview/step002_layer20.perfetto.json) · [详细](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/sm100_megamoe/v41_dense_submit_results/mega/perfetto/step002_layer20.perfetto.json)
- GC step 1：[overview](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/sm100_megamoe/v41_dense_submit_results/gc/overview/step001_layer20.perfetto.json) · [详细](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/sm100_megamoe/v41_dense_submit_results/gc/perfetto/step001_layer20.perfetto.json)
- GC step 2：[overview](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/sm100_megamoe/v41_dense_submit_results/gc/overview/step002_layer20.perfetto.json) · [详细](https://github.com/ehdbdns/gcmoe-sm90/blob/main/docs/sm100_megamoe/v41_dense_submit_results/gc/perfetto/step002_layer20.perfetto.json)

## 限制

- 旧 L20 host spread 是 CUDA API 入口差，新值是 Python graph.replay 调用前差；
  两者只作方向参考，不作精确 API 时长比较。
- `step001_dense_gaps.csv` 只有 L20；负 gap 是 Kineto 边界重叠，按 0 解释。
- 单锚点回退无法独立测量 before/after 漂移；只给出有效 after 锚点半宽和 16/16
  的 before 会话内映射核验。
- 路由 slot 与每 rank token/M 块存在变化；不能把 step 2 的差值视为同路由实例的
  纯净 A/B。
- `wall_ms` 含准备、RPC 和导出，不代表长窗吞吐。
