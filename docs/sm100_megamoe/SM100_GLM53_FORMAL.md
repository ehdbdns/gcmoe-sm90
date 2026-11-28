# SM100 · GLM-5.3 BS256 三臂整模实验

**只在获准使用的同组 8 张 Blackwell GPU 全部空闲时启动每一臂。** 有任一张卡被占用或被他人预约，就整组等待。每一臂结束释放八卡，再独立检查下一臂。CPU 安装和 EPLB 放置计算均在 GPU 锁外。执行者只运行下面的命令；如有编译器兼容性修复，保留原错误和补丁。模型加载、拓扑、路由、数值、采样和指标差异都交回构建者处理。

## 执行

在完成 CPU 回执 `0415075` 的同一台机器运行。复用回执确认的 vLLM 0.31.0 环境，脚本新建私有 CuTe DSL 4.8.1 环境。确认调度分配/预约有效，再在分配的 shell 执行整段。若调度器提供 `CUDA_VISIBLE_DEVICES`，直接沿用；在无调度器且共享超过 8 卡的机器，请事先用 `SM100_GPU_UUIDS` 填入本任务获准使用的八个逗号分隔 UUID。两者均未设置时，仅在节点恰好可见本任务获准使用的八卡时执行。卡忙时脚本每 60 秒复查，不占空闲卡。

默认工作目录沿用上轮私有盘。可事先把 `GLM_WORK_ROOT` 指向同机更大的私有存储。48 GiB/rank 的两份逐字节 KV 快照需要 768 GiB，脚本检查卷上至少 800 GiB 可用；实际存储建议预留更多空间给环境、原始 trace 和日志。若此前环境已被清理，先按运行环境任务书在同机恢复，并通过 `GLM_RUNTIME_PYTHON` 指向新回执中的解释器。

~~~bash
(
set -eu
umask 077
storage=${GLM_WORK_ROOT:-/data0/home/admin/workspace/glm53-work}
mkdir -p "$storage"
job=$(mktemp -d "$storage/formal.XXXXXXXX")
echo "本次私有目录 $job"
git clone --depth=1 --filter=blob:none --sparse --branch main https://github.com/ehdbdns/gcmoe-sm90.git "$job/repo"
cd "$job/repo"
git sparse-checkout set --no-cone '/scripts/sm100/*.py' '/scripts/sm100/run_glm53_three.sh' '/scripts/sm100/bruce_sm100_eco.patch' '/integration/*.py' '/source/gc_moe/eco/' '/source/gc_moe/eplb/' '/docs/sm100_megamoe/v41_native_phase_results/dsl-wheelhouse.sha256' '/docs/sm100_megamoe/v41_full_decode_pair_results/compat/sitecustomize.py' '/docs/sm100_megamoe/glm53_formal_results/'
mkdir "$job/results"
rc=0
bash scripts/sm100/run_glm53_three.sh "$job/results" > "$job/results/run.log" 2>&1 || rc=$?
dest="docs/sm100_megamoe/glm53_formal_results/$(basename "$job")"
python3 scripts/sm100/glm53_receipt.py "$job/results" "$dest" "$rc"
git add -- "$dest"
git -c user.name="SM100 executor" -c user.email="sm100-executor@users.noreply.github.com" commit -m "[SM100] GLM-5.3 BS256 trace-2 formal execution (exit $rc)"
git push origin HEAD:refs/heads/main
git fetch --depth=8 origin main
git merge-base --is-ancestor HEAD FETCH_HEAD
echo "回执已到 main：$(git rev-parse HEAD)；运行退出码 $rc"
)
~~~

从另一终端用 `tail -f 私有目录/results/run.log` 查看进展。若 push 仅因另一个执行者先提交被拒，在本次 checkout 运行 `git pull --rebase origin main` 后再次 push。退出码非零时保留日志和首个失败现场；CPU 校验或离线 cubin 均不作为本轮性能结果。

## 运行口径与验收

- 三臂依次为原生 GC、逐层 EPLB + DeepInfer RR、同放置 EPLB + Natureal ECO。原生每卡 32 槽，两条 EPLB 臂每卡 40 槽。TP8/EP8、BS256、每 rank 32 个 token、全部 75 个 routed 层。CUDA Graph 模式与 trace 2 对三臂相同。ECO 从 dispatch 分区拿 1 个 SM。GLM 的 shared FP8 为 128×128 尺度，routed FP4 为每行 K32 尺度。私有 HF 配置显式选择 vLLM 的 routed MXFP4 loader。
- 使用固定的 256 条中文研究问题和 GLM 自己的 tokenizer，属于受控工作负载。这批问题与历史 DS4.1 的线上请求来源不同。每臂同一批请求连续跑 1000 个统计步和 1000 个测量步。统计结束，按原生臂的逐步逐层专家计数计算 EPLB，冻结放置。两个 consumer 在原图地址安装物理槽对应的权重、映射和 ECO 状态。两处 KV 边界字节校验，反馈、批内顺序、原生及强制逻辑 gate 分别留证。
- 从重排后第 2 个 replay 取层 8、20、32 的逐 rank counts、torch.profiler 原件、概览和 absolute Perfetto。各样本与对应步层主表原始 clocks/counts 对照。主表覆盖后 1000 步 × 75 层；完整 MoE 是每层每步八 rank 最大的首角色到 combine 结束跨度，routed 从真实 routed MMA 开始到最后 L2 push。记录 token IR、总 M 块、峰值 M 块、活跃槽、M 块 IR，以及 ECO 的 routed MAE、报价、成交与接受率。计时代表 trace 2 开启时的性能。
- GPU 进程退出即释放。报告含主表、逐臂校验结果、小型 Perfetto、各原始文件路径/大小/摘要。大文件保存在私有目录。严格配对不通过时，主表仍保留绝对测量，配对百分比留空。核对回执后可清理本任务的安装包和编译缓存，保留 KV/路由参考、结果与日志以便复核。
