# SM100 · GLM‑5.3 真实权重校验与运行环境

适用于首次接手的 SM100 执行者。上一份 CPU 回执已经通过，实际模型为 `/mnt/hf3fs/3fs/models/GLM-5.3-FP4Moe-FP8ue8m0`。本次在挂载该目录的机器上运行以下整段，**全程隐藏 GPU**。后续原生 GC、EPLB/RR、EPLB/ECO 每臂只有在获准使用的同组八张 GPU 全部空闲后才可开始；每臂结束释放。本轮无需预约或等待卡。

脚本先检查现有 Python、Conda 和历史工作区。若选不到带 CUDA 构建的 torch、vLLM 和 safetensors，就在新建的个人目录安装基线 vLLM 0.31.0。可事先设置 `GLM_PYTHON` 指向本机已有 GLM 环境，`GLM_WORK_ROOT` 指向个人可写、有空间容纳 PyTorch/CUDA wheels 的磁盘。沿用机器已有 pip 镜像配置。保存成功的私有环境供下一阶段复用。执行者无需修改模型配置、权重或仓库代码。

~~~bash
(
set -eu
umask 077
job=$(mktemp -d "${TMPDIR:-/tmp}/glm53-run.XXXXXXXX")
echo "私有 checkout 位置 $job/repo"
git clone --depth=1 --filter=blob:none --sparse --branch main https://github.com/ehdbdns/gcmoe-sm90.git "$job/repo"
cd "$job/repo"
git sparse-checkout set --no-cone '/scripts/sm100/glm53_*.py' '/scripts/sm100/real_routed_weights.py' '/docs/sm100_megamoe/glm53_runtime_results/'
out=docs/sm100_megamoe/glm53_runtime_results
rc=0
CUDA_VISIBLE_DEVICES='' python3 scripts/sm100/glm53_runtime.py --out "$out" || rc=$?
git add -- "$out"
git -c user.name="SM100 executor" -c user.email="sm100-executor@users.noreply.github.com" commit -m "[SM100] GLM-5.3 runtime and real weight check (exit $rc)"
git push origin HEAD:refs/heads/main
git fetch --depth=8 origin main
git merge-base --is-ancestor HEAD FETCH_HEAD
echo "回执已在 main：$(git rev-parse HEAD)，校验退出码 $rc"
)
~~~

成功时 `runtime.json` 的状态为 `cpu_verified`，`weight-checks.json` 含 54 条记录。实际 decoder 层 8、20、32 各校验 routed 专家 0、255 和 shared 专家，包含 gate、up、down 的 payload 与 scale。校验对照独立 safetensors 读取器逐字节比较，其中 shared 的每行尺度按 128 行广播，routed 的尺度保留逐行。回执仅含形状、摘要和环境信息。该校验覆盖 GC 权重变换的输入；实际设备执行、模型输出和性能由后续整模任务验证。失败时照常提交 `runtime.json`，详细安装日志留在其中列出的私有目录，无需提交含镜像地址的原始日志。若仅因并发 push 被拒，在该 checkout 执行 `git pull --rebase origin main` 后重新 push，回传提交号。
