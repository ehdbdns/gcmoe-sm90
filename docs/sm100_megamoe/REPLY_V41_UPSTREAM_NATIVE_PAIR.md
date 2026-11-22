# 【SM100】BS1024 上游 GCMoE 与 Vendored MegaMoE 同输入复测回执

执行者：**SM100**。本轮仅执行 SM100 任务书，未执行任何 SM90 任务书。

## 结论

两臂各成功启动一次并完成三个样本。Mega 图输出对实时 Mega 参考均为 0 差异；
GC 图输出在步骤 0、8、16 分别有 21、4、11 个元素不同。因此 **三个步骤均未通过
严格逐元素判定**，下表耗时只作本次观测，不构成严格性能胜负结论。

任务书：`docs/sm100_megamoe/V41_UPSTREAM_NATIVE_PAIR.md`，
SHA256 `987acc7a099d446ccc98ad29c02ffff6c8e236823612a9da99f48a07e73c0971`。
执行仓 SHA：`ba462195bac57a1241a13f45033d07c60979baff`，任务分支基线
`8f52fb0`；四个修复提交随后 rebase 到最新 main 为 `ae82e37`、`10ecef5`、
`1f17204`、`89b8577`。上游 GCMoE 固定为
`7902831ec4a6691b6268b308bcff5ec0cb15ec86`，
运行前该 checkout 为干净状态。

## 配置与执行

- 模型：`/data0/home/admin/workspace/megamoe/native-v41-KO1hTbTj/model`；
  DeepSeek V4.1 Flash，TP8 / EP8，原生专家布局，融合 shared expert。
- 每样本 40 个真实专家层，top-6，每 rank 48 个本地专家。
- 输入：上轮持久化的
  `step{0,8,16}-rank{0..7}.pt`，共 24 份；运行前 24/24 通过 SHA256 核验，
  两臂加载同一组文件。摘要见 `measurements.sha256`、`fixture-check.txt`。
- Mega 后端：共享 venv 中的 `vllm.third_party.deep_gemm.mega`。
- GC 后端：指定上游 checkout 中的 `gc_moe.gc`，包版本 0.6.2。
- 两臂共用模型、输入、八张物理卡、启动器与计时路径。EPLB、ECO、RR 重排和
  额外 tracing 均关闭。
- 输出缓冲区按任务书在集成层复用；eager 参考单独克隆。两臂均未向计时图
  增加设备拷贝。
- 计时：3 次 eager、40 层一张图、10 次预热、15 组 x 20 次 replay。每组取
  八卡最慢 rank，再取 15 组中位数，单位为 ms。

## 三步耗时

正号表示 GC 比 Mega 慢。

| Step | Mega (ms) | GC (ms) | 100 x (GC/Mega - 1) | 严格判定 |
|---:|---:|---:|---:|:---:|
| 0 | 7.887888 | 7.895562 | +0.0973% | 否 |
| 8 | 7.852768 | 7.827145 | -0.3263% | 否 |
| 16 | 7.899139 | 7.869527 | -0.3749% | 否 |

严格判定要求 GC 对该步实时 Mega 参考逐元素无差异。GC 三步均有差异，故
**没有任何一步可标记 strict=true**。

## 15 组分布

表中每组值是该组八卡最慢 rank 的单次 replay 耗时，中位数即上表结果。

| Arm | Step | Min (ms) | Median (ms) | Max (ms) | Max/Min-1 |
|---|---:|---:|---:|---:|---:|
| Mega | 0 | 7.885387 | 7.887888 | 7.891906 | 0.0827% |
| Mega | 8 | 7.850117 | 7.852768 | 7.857034 | 0.0881% |
| Mega | 16 | 7.896449 | 7.899139 | 7.901536 | 0.0644% |
| GC | 0 | 7.889677 | 7.895562 | 7.912362 | 0.2875% |
| GC | 8 | 7.822422 | 7.827145 | 7.836041 | 0.1741% |
| GC | 16 | 7.861742 | 7.869527 | 7.878930 | 0.2186% |

全部 90 个逐组值见 `v41_upstream_native_pair_results/groups.csv`。

## 正确性

- eager/replay：两臂各 960/960 个层输出
  （8 rank x 3 step x 40 layer）通过 `torch.equal`，且所有输出有限。
- Mega 图对实时 Mega 参考：三步均为 0 个不同元素，最大绝对误差 0。
- GC 图对实时 Mega 参考：

| Step | 不同元素 / 总数 | Max abs | 各 rank 不同元素数 | 各 rank 最大绝对误差 |
|---:|---:|---:|---|---|
| 0 | 21 / 209,715,200 | 2.0 | `[1,0,1,1,9,4,4,1]` | `[2.98e-8,0,1.49e-8,4.88e-4,1,2,2,3.73e-9]` |
| 8 | 4 / 209,715,200 | 0.015625 | `[0,0,1,0,0,0,0,3]` | `[0,0,6.10e-5,0,0,0,0,1.5625e-2]` |
| 16 | 11 / 209,715,200 | 8.0 | `[0,2,0,9,0,0,0,0]` | `[0,4.88e-4,0,8,0,0,0,0]` |

GC 合计 36 / 629,145,600 个元素不同，占 5.7220e-6%。逐层不同元素数为
step 0: 21（涉及 12 层）、step 8: 4（涉及 4 层）、step 16: 11（涉及 4 层）。
逐 rank 明细在 `output_diff_by_rank.csv`，全部 960 条逐层明细在
`output_diff_by_layer.csv`，原始元数据在 `mega.json`、`gc.json`。

GC 的 21 / 4 / 11 差异与此前 vendored-GC 对照结果完全一致；本轮改用固定上游
包后，数值差异模式未变。

## 后端与实际分区

- Mega：8/8 rank 为 `vllm.third_party.deep_gemm.mega`，capability `[10,3]`，
  包版本 0.6.2。
- GC：8/8 rank 为 `gc_moe.gc`，capability `[10,3]`，包版本 0.6.2。
- GC 每个 rank 均为 `dispatch_sms=12`、`compute_sms=136`、`primary=False`。
- 实际 import 路径见 `imports.txt`；八 rank 完整后端记录见 `backends.txt`。

## 编译器、依赖与上轮区别

- `ptxas`：CUDA 13.5，V13.5.37，build
  `cuda_13.5.r13.5/compiler.39125733_0`。
- `nvcc`：CUDA 13.0，V13.0.48，build
  `cuda_13.0.r13.0/compiler.36260728_0`。
- 驱动：580.105.08。
- 两臂公共环境：vLLM 0.31.0、Torch 2.13.0+cu130、FlashInfer 0.7.0.post1、
  NCCL 2.29.7、Triton 3.7.1。
- 本轮私有解释器使用 CuTe DSL 4.8.1；上轮共享环境为 4.7.1。两臂使用同一套
  4.8.1 wheel。完整版本见 `versions.txt`，来源与安装日志见 `dsl-*`。
- 与上轮区别：GC 改为指定干净上游 `7902831`，私有 DSL 从 4.7.1 升到 4.8.1；
  Mega vendored 后端、模型、输入、八卡和计时路径未变。

## 输出复用修复与失败记录

任务书给出的旧输出复用方式在 Mega 启动时连续四次报
`AssertionError: expert output allocation changed`，未产生可用计时样本。
四个失败尝试的 launcher/worker 日志和退出码均已保留，退出码都是 1：

- attempt 0：`launcher-attempt0.log`、`mega-attempt0.log`。
- attempt 1：`launcher-attempt1.log`、`mega-attempt1.log`。
- attempt 2：`launcher-attempt2.log`、`mega-attempt2.log`。
- attempt 3：`launcher-attempt3.log`、`mega-attempt3.log`；日志与 attempt 2
  相同，为同一失败的重复归档。

最终修复栈：

- `ae82e37`（执行时为 `69de893`）：按源张量分配匹配复用。
- `10ecef5`（执行时为 `c3718f6`）：直接包装专家的 `torch.empty_like` 分配。
- `1f17204`（执行时为 `cc9343f`）：按 shape、dtype、device、layout 匹配复用。
- `89b8577`（执行时为 `ba46219`）：增加由环境变量门控的复用诊断。

完整 diff 为 `output_reuse_fix.diff`。修改只涉及输出分配复用适配层，未改
模型、kernel、runtime 算术、输入、参考或计时逻辑。

## 退出、释放与清理

- 正式测量退出码：`mega.exit-code=0`、`gc.exit-code=0`。
- 最终 GPU 释放时间：`2026-11-22T16:37:10+08:00`。
- 释放后于 `2026-11-22T08:38:21Z` 检查：无 compute application，
  `/dev/nvidia0..7` 无持有者；证据为 `gpu_release_after_run.txt`。
- 八卡组锁在成功测量前取得、结束后释放；`lock-release.txt` 记录
  `run_lock_acquired=yes`。
- 已删除本轮私有 `dsl-env`（498M）、`upstream`（21M）、`dg-cache`（1.2M）、
  `wheels`（180M）和 24 个 fixture 符号链接；共享环境和 24 个原文件保留。
  详见 `cleanup.txt`。
- 未终止或清理他人 GPU 作业。

## 提交内容

结果目录 `docs/sm100_megamoe/v41_upstream_native_pair_results/` 包含：
`comparison.json`、`mega.json`、`gc.json`、两臂成功日志、四次失败日志与退出码、
`groups.csv`、逐 rank/逐层差异 CSV、`upstream.sha`、`upstream-inspect.txt`、
`imports.txt`、`fixture-check.txt`、`compiler.txt`、`versions.txt`、
`weights.sha256`、`gpus.txt`、获取/释放/锁记录、`cleanup.txt`、
`environment.txt`、`backends.txt`、`task.md`、`task.sha256`、
`measurements.sha256`、DSL 来源日志、修复 diff 和 `SHA256SUMS`。

24 份原 `.pt` 文件未复制入 Git，保留在
`/data0/home/admin/workspace/megamoe/v41-native-pair-bs1024-01/`，SHA256 已随
`measurements.sha256` 提交。
