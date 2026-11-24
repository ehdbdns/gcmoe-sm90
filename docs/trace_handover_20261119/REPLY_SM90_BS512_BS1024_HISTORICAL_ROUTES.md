# 回执 · SM90 · BS512 / BS1024 五臂真实路由 replay2

执行者：SM90 八卡执行者。任务书 `docs/trace_executor_20261119/SM90_BS512_BS1024_HISTORICAL_ROUTES.md`（commit `b2be6f1`）。

## 结论：本档无法执行 —— 历史源路由 bank 在机器重启中丢失

任务书要求以历史整模正式窗口的 `measure-rank*.pt`（每源 rank 一个文件，内存 taped `ids`/`weights`，形状 `(40000, BS, 6)`）作为唯一合法路由输入，并明确禁止用随机路由或按直方图逆推顺序代替。这两个 bank 不在任何现存位置。

| 档 | 需求哈希（`workload.sha256`） | 文件 | 期望字节 | 默认路径（任务书） | 现状 |
|---|---|---|---|---|---|
| BS512 | `325366496aefe6dd2472b29134ffa37a2da76e7f4c6aa1561861483b8df133d2` | `measure-rank0..7.pt` | 1 474 560 000 | `/tmp/gcmoe-main24-bs512-zQ4Vytw6/replay` | 目录不存在 |
| BS1024 | `8482a712e0ae93d2e540088202c350ff27d1b00ce03beb53c091f23ff7abc0d5` | `measure-rank0..7.pt` | 2 949 120 000 | `/tmp/gcmoe-main24-bs1024-fhSPzc57/replay` | 目录不存在 |

两档的 `state-path.txt` 分别指向上述两个 `/tmp` 目录，重启后 `/tmp` 已清空（同轮 `/tmp` 内容此前只抢救了 `retained_bs256/`、`captures/`、`rt/`，见 `app_source/sm90_persist_20261123/README_PERSISTED.md`）。

### 已搜索范围与结果

搜过 `/tmp`、`/mnt/fuse`、`/mnt/dsv41`（含 `nanjun.cp`、`.Trash-500`）、`/home/admin/workspace`、`/home/admin/workspace/aop_lab/app_source`，按文件名 `measure-rank*`、以及按两个目标字节数（1 474 560 000 / 2 949 120 000）各扫了一遍；另查过 git 全部 blob（无 >100 MB 对象）。**没有任何同哈希或同尺寸副本。**

现存可用的 bank 只有三个，均不匹配：

| bank | `measure-rank0.pt` SHA256 | 尺寸 | 判定 |
|---|---|---|---|
| `app_source/sm90_persist_20261123/retained_bs256/replay/` | `b728a47ae1e4253be7d165958ed9154d932fc6111144e2c9f8b7b6e76c094deb` | 737 281 829 | BS256，非本档 |
| `/mnt/fuse/sm90-h20-eight-CF3A1Zkp/results/replay/` | 同上（同一份） | 737 281 829 | BS256，非本档 |
| `/mnt/fuse/sm90-h20-bs32-34Erv5Jb/results/replay/` | `4773a870843aaeec65c1277be21d317707177c1033de3f7e35b7f1d933732b82` | 92 161 829 | BS32，非本档 |

其余前置材料齐备，缺的只有 bank：两档的 `stats-demand.npz`、`placement.json`、`workload.sha256`、五臂 `installed-layout.json` 与 `vanilla/random/rr/eco/eco_smr` 的 `rank*.npz` 都在仓库 `delivery/bs{512,1024}/` 下且完好。

## 已完成的部分：参考表 CPU 核验（不依赖 bank）

任务书给出的“第二步（step 索引 1）峰值 rank M 块”与“ECO 成交数”直接取自历史八卡 NPZ，可用仓库内归档复核。已按 `row = 40*1 + layer` 逐层逐 rank 重算，**全部逐项吻合**：

| BS | 指标（L8 / L20 / L32） | vanilla 核验 | RR 核验 | ECO 核验 | 任务书 | 判定 |
|---|---|---|---|---|---|---|
| 512 | 峰值 rank M 块 | 98 / 114 / 84 | 80 / 79 / 82 | 76 / 73 / 75 | 同左 | 一致 |
| 1024 | 峰值 rank M 块 | 174 / 215 / 151 | 127 / 129 / 128 | 121 / 120 / 122 | 同左 | 一致 |
| 512 | ECO 成交数 | — | — | 5 / 5 / 3 | 5 / 5 / 3 | 一致 |
| 1024 | ECO 成交数 | — | — | 8 / 4 / 7 | 8 / 4 / 7 | 一致 |

（ECO 成交数取 `auction[..., 2]`，即历史 NPZ 第二字段的 accepted 计数；`auction[..., 0]` 全为 1、`[..., 5:7]` 全为 0，符合七项拍卖记录格式。）

另核：vanilla 每层每步八卡路由项合计 = BS×8×6（512 → 24 576，1024 → 49 152），与任务书一致；这也正是 `extract_se_routes.py` 用来校验 bank 正确性的那条不变量。

## 未做的事（按任务书纪律）

- 未启动任何八卡 GPU 作业：没有合法路由输入，跑出来的时延不满足“历史真实路由”口径，任务书禁止用它替代。
- 未用随机路由、未按直方图逆推顺序、未改动任何实验参数或脚本（diff 为空）。
- 未生成 `history_comparison.csv`、`main_table.csv`、`layers.csv`、`ranks.csv`、`moe.svg` 与 30 份 Perfetto，因为都需要真实路由。
- 八卡当前全空（各 4 MiB、无 compute app），无本任务残留进程。

## 需要补回什么才能继续

把两个 bank 恢复到任意可读路径即可（不必是原 `/tmp` 路径，任务书支持 `SM90_BANK_512` / `SM90_BANK_1024` 指定目录）：

- `/mnt/dsv41/.../replay/measure-rank{0..7}.pt`（BS512，哈希 `325366a…`）
- `/mnt/dsv41/.../replay/measure-rank{0..7}.pt`（BS1024，哈希 `8482a71…`）

恢复后我按任务书原样执行：两档各 5 臂 × 3 层，`extract_se_routes.py` 先过哈希与 vanilla 槽计数校验，再跑 `run_bs1024_se_profile.sh OUT BS GROUP BANK`，最后 `audit_se_history.py` + 逐层 `se_perfetto.py` + `se_replay_report.py`，并交付 `history_comparison.csv`（360 行）、30 份 Perfetto、两张图与十行主表。

## 现场与退卡

- 未占用 GPU；无本任务 worktree（未创建），无可清理缓存。
- 本回执提交到 `docs/trace_handover_20261119/REPLY_SM90_BS512_BS1024_HISTORICAL_ROUTES.md`。
