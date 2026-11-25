# 【SM100】EP8 线性四臂单 BS 回执：GPU 条件未满足，未启动测量

执行者：**SM100**。本轮只执行 SM100 任务书，未执行或改写任何 SM90 任务书。

## 结论

按指挥官要求只做 **BS256**，并在该档结束后立即停止。实际在八卡组门外被资源条件
阻塞：4 张卡空闲，另外 4 张持续被外部 `rtp_llm_rank-0..3` 占用，因此没有启动任何
臂、没有加载模型、没有产生测量行，也没有 BS512/BS1024 尝试。**本回执不给出性能或
公平性结论。**

- 任务书：`docs/sm100_megamoe/SM100_EP8_LINEAR_FOUR_3BS.md`
- 任务 SHA256：`7f0d9281edf818eef1e5c52e8994166f2dd554a5b410632195ea82c3ab79abf1`
- 任务提交：`55ef4ff27176c9c839d2feaa44348e2055ceb2fa`
- 实际执行提交：`d16bf90047024504d0c974a70dc2da13b6d3102c`
- 固定上游：`fb8b65fa829c97962eb2dbc9a4ad8b759b9cd736`
- 计划范围：BS256 的 `vanilla / random / rr / eco`
- 实际启动臂数：0
- 测量行数：0

## 阻塞证据

任务要求八张卡整组无预约、无占用、无设备句柄。最终检查如下：

| GPU | UUID | 占用 |
|---:|---|---|
| 0 | `GPU-3aad6312-0e20-d01b-4249-71cb090c2127` | 空闲 |
| 1 | `GPU-b0ac7f5d-738f-cdad-423f-456f4c26947e` | 空闲 |
| 2 | `GPU-c5a93c78-be0c-2648-564a-4b4666671960` | 空闲 |
| 3 | `GPU-77463de9-85e0-a6f3-2c7e-f1147d6f7f49` | 空闲 |
| 4 | `GPU-4a493131-84b9-4c88-18ee-50fc6f213a36` | 外部 `rtp_llm_rank-0`，约 262 GiB，100% |
| 5 | `GPU-045e62f4-d04d-2f9e-e456-1fb82810029f` | 外部 `rtp_llm_rank-1`，约 262 GiB，100% |
| 6 | `GPU-6b44066c-6712-30f2-02d0-fa0cb91733a3` | 外部 `rtp_llm_rank-2`，约 262 GiB，100% |
| 7 | `GPU-31a509fd-e741-5ab8-b837-443ad986c404` | 外部 `rtp_llm_rank-3`，约 262 GiB，100% |

四个外部进程的 PID 为 `3351983`、`3351987`、`3351989`、`3351991`，均从
`2026-11-25T16:37:34+08:00` 开始运行，到最终检查仍在。任务书明确要求任一卡被
他人占用时整组等待；我没有终止、迁移或干扰这些外部作业。

本会话 `nvidia-smi` 把设备标成 `NVIDIA L20D`，而任务书写 B300。硬件身份也需要
指挥官在重试前确认；这不是本轮唯一阻塞点，外部占用已足以阻止启动。

原始记录：`ep8_linear_four_3bs_results/gpu-occupancy.csv`、
`gpu-compute-apps.csv`、`gpu-final-check.txt`。

## 预检与提示词

历史 JSONL 中分别只有 187 和 158 条落在 [256, 2048] token 区间，原选择器因此拒绝
三档输入。已按 SM90 的实际 cohort 方式修正：按长度排序、等步长抽取、短池再次等
步长补齐、BS512 两份交错、BS1024 复用 BS512 两份；BS256 只用第一份文件。

三档提示词摘要现与 SM90 完全一致：

```text
254af9d8a2752eab68f6419f085c1973e22e582a3f50a3c99a9853efeffdaec8  bs256/prompts.json
e7c4aee9db91a34c018bd54464e67355681b74112d515401844f473c053cfcef  bs512/prompts.json
d7eeeb50bb0428af010e6cad5ed56598f8c559a06525070a4d09ebd023d7e74c  bs1024/prompts.json
```

BS512/BS1024 的摘要只是预检证据，两档均未执行。原始 JSONL、模型配置、模型索引和
四个补丁的 SHA256 均记录在 `preflight.log` 与 `VALIDATION.md` 中。

## 停止与释放

- 队列在 `2026-11-25T18:07:42+08:00` 停止，tmux 会话已结束。
- 任务启动器 PID `2153811` 最终已不存在。
- 未生成 `run/acquired.log`，说明本任务从未越过八卡获取门。
- 最终扫描不到 `ep8_matrix.py` 或本任务启动器 worker。
- 本任务没有取得 GPU、没有启动模型，也没有需要释放的本任务显存。
- 外部四个 `rtp_llm` 进程保持原状，未被我终止。

证据：`queue-stop.log`、`launcher.log`、`gpu_release_after_run.txt`、
`gpu-final-check.txt`。

## 交付状态

结果目录：`docs/sm100_megamoe/ep8_linear_four_3bs_results/`。

保留了任务与执行版本、提示词摘要、完整上游补丁、启动器、环境版本、任务前置检查、
GPU 占用快照、停止记录和校验和。由于没有臂启动，以下文件按预期不存在：

```text
main.csv
layers.csv
pairs.csv
auction.csv
*/result.json
*/model.log
*-routes-*.pt
kv-*/rank*.bin
```

这些不是零耗时结果。只有八张卡全部空闲且硬件身份确认后，才可从 BS256 重新执行
四臂；BS512/BS1024 未获本轮授权，保持不执行。
