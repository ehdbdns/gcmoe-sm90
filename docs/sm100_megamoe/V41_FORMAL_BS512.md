# SM100 · V4.1 Flash 正式 BS512 五臂

仅 SM100 执行。按表序在同一节点、预约的同一组八张空闲 SM103 卡上串行运行，每臂独立启动模型一次。每臂正式解码前用 1000 个满批步统计，随后测量 1000 个满批步。每步 512 个请求、40 个专家层、TP8/EP8、top-6、融合 shared expert。只有三个 EPLB 臂在窗口边界应用同一份 DeepInfer 逐层放置。

| 臂 | 内核与分流 |
|---|---|
| mega | vLLM 所带 DeepGEMM MegaMoE，原生布局 |
| gc | Bruce 上游 GCMoE 7902831，原生布局 |
| mega_rr | 同一 MegaMoE，EPLB + RR |
| gc_rr | 同一上游 GCMoE，EPLB + RR |
| gc_eco | 本仓 SM100 GCMoE 的 ECO 实现，EPLB + 拍卖，记录实际 cost 配置、block M 和成交次数 |

Bruce 上游缺少本项目的 ECO 接口。第五臂代表完整的仓库实现；比较 gc_rr 与 gc_eco 时，内核版本和分流同时变化。两条 GC 实现统一申请 8 个 dispatch SM，回执记录实际分区。Mega 使用既有私有计时头文件；两条上游 GC 臂只应用本任务随附的设备时钟采集补丁。

## 执行

在原 SM100 主仓 CPU 侧准备一个私有目录，保留共享模型与环境。预留约 550 GiB 本地空间用于两个 KV 边界、路由和结果；按实际空闲空间选择挂载点。命令在同一个 bash 会话执行。

```bash
set -euo pipefail
git switch main
git pull --ff-only origin main
repo=$(git rev-parse --show-toplevel)
cd "$repo"
work=$(mktemp -d /data0/home/admin/workspace/megamoe/v41-formal-bs512-XXXXXXXX)
base=/data0/home/admin/workspace/megamoe/native-v41-stsFS4w7/venv
git clone https://github.com/Bruce-Lee-LY/gc-moe.git "$work/upstream"
git -C "$work/upstream" checkout --detach 7902831ec4a6691b6268b308bcff5ec0cb15ec86
"$base/bin/python" -m venv "$work/dsl-env"
py="$work/dsl-env/bin/python"
"$py" -m pip install --no-deps nvidia-cutlass-dsl==4.8.1 nvidia-cutlass-dsl-libs-base==4.8.1 nvidia-cutlass-dsl-libs-core==4.8.1 nvidia-cutlass-dsl-libs-cu12==4.8.1 nvidia-cutlass-dsl-libs-cu13==4.8.1 2>&1 | tee "$work/dsl-install.log"
lib=$("$py" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
base_lib=$("$base/bin/python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
printf "import site; site.addsitedir('%s')\n" "$base_lib" > "$lib/zz_v41_common.pth"
bash scripts/sm100/run_five.sh "$work" formal-bs512 2>&1 | tee "$work/launcher.log"
```

上游 clone 和 DSL wheels 也可使用上轮核实的本地同版本来源。启动器取得共享八卡组锁、检查 UUID 和空闲状态。任何卡有占用就等待，全程不终止他人进程。沿用 SM100 五臂的 256 条固定八 token 前缀，按顺序重复两次到 BS512。模型外层 eager，正式窗口各专家层预热后 CUDA Graph replay。仅新起这五臂；模型或测量异常保存首栈，禁止更换内核或重排策略凑齐结果。

## 公平性与交付

1. Mega 发布原生轨迹、逻辑专家 ID、gate 权重与 1000 步路由需求。八卡分别在 prefill 后、正式窗口前写入/恢复同一 KV backing store，比较请求、物理块表、KV 布局及 SHA256 读回。每臂使用相同每卡 32 GiB KV 配额。主表保留同配置、同放置验证和完整图边界的证据。
2. 对 10 对组合报告 controlled_replay、原生 top-1 一致性、严格判定。原生指覆盖反馈前记录的 greedy top-1；分别核对 prefill、1000 步统计和 1000 步正式窗口，每窗口覆盖 512,000 个请求位置。原始 gate 尾数差异另列，实际送入内核的回放 ID、权重及批顺序必须相同。失败时按窗口/rank 给出不同 token 数、首个请求/步和两边数值。隐藏向量逐元素相等不作为跨后端比较条件。各臂仍须通过 40 层本后端 eager/replay 校验。
3. 五臂主表列完整 MoE ms/步（每步逐层取八卡最大 graph 起点至 combine 完成跨度，再累加 40 层并对步取均值）和相对本轮原生 Mega 的有符号耗时变化，负值为更快。列 token IR、总/峰值 M 块、活跃物理槽、M 块 IR、routed/graph 预测 MAE 和 ECO 接受次数。严格通过的配对才注明原生轨迹一致；其余照实给出固定工作量计时与失败原因。
4. 将结果提交 docs/sm100_megamoe/v41_formal_bs512_results/，回执提交 docs/sm100_megamoe/REPLY_V41_FORMAL_BS512.md，推送 main。包含 delivery、每臂版本/后端/实际分区、KV 清单、运行日志、退出码、SHA256SUMS 及释放记录。检查全部任务进程退出、八卡状态与组锁释放。核验回执证据后清理本次私有 KV 二进制、路由、编译缓存、overlay、dsl-env 与上游 checkout，保留共享模型和历史产物。执行者仅可修复保持语义的编译故障并附 diff。
