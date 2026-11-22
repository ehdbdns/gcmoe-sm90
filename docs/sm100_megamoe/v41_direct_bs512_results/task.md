# SM100 · DeepSeek-V4.1-Flash BS512 原生解码五臂

取代停跑的正式任务。本轮只在原预约的八张 SM103 上执行，同一组卡串行跑 mega、gc、mega_rr、gc_rr、gc_eco，每臂启动一次。五臂同模型、同提示、同批大小、32 GiB/rank KV、40 层融合 shared expert，每臂 1000 步统计 + 1000 步正式测量。由本轮 mega 统计生成一份逐层 DeepInfer 放置，三条 EPLB 臂使用同一份。两条 RR 臂用 DeepInfer RR；gc_eco 用 Natureal 原始 ECO 分支 fb8b65f 的默认拍卖、索引交换和 compute 侧 plan。使用已有的计时及成交数插桩。

每臂自行 greedy 解码，直接使用本臂的 gate，不写读 KV 边界、token 表或路由档案，不比较跨臂 top-1 或十组配对。各臂实际 token 和专家需求可能不同。保留每臂正式窗口前 40 层有限输出及 eager/graph 自检、每步满批检查、放置安装检查和设备时间戳检查。

## 运行命令

在 SM100 的实验仓库使用干净 main。命令先取齐前次稀疏 checkout 遗漏的提示输入，再建新的私有目录以免重复应用补丁。八卡锁、UUID 和空闲检查由启动器负责，任何占用时等待。

```bash
set -euo pipefail
git switch main
git pull --ff-only origin main
cd "$(git rev-parse --show-toplevel)"
fixture=docs/trace_executor_20261119/sm90_main24_stable_decode_bs256_results/delivery/bs256/vanilla
if [[ $(git config --bool core.sparseCheckout || true) == true ]]; then
  git sparse-checkout add "$fixture" scripts/sm100 source/gc_moe integration docs/sm100_megamoe
fi
test -r "$fixture/rank0.npz"
work=$(mktemp -d /data0/home/admin/workspace/megamoe/v41-direct-bs512-XXXXXXXX)
base=/data0/home/admin/workspace/megamoe/native-v41-stsFS4w7/venv
git clone --branch eco https://github.com/Natureal/gc-moe.git "$work/upstream"
git -C "$work/upstream" checkout --detach fb8b65fa829c97962eb2dbc9a4ad8b759b9cd736
"$base/bin/python" -m venv "$work/dsl-env"
py="$work/dsl-env/bin/python"
"$py" -m pip install --no-deps nvidia-cutlass-dsl==4.8.1 nvidia-cutlass-dsl-libs-base==4.8.1 nvidia-cutlass-dsl-libs-core==4.8.1 nvidia-cutlass-dsl-libs-cu12==4.8.1 nvidia-cutlass-dsl-libs-cu13==4.8.1
lib=$("$py" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
base_lib=$("$base/bin/python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
printf "import site; site.addsitedir('%s')\n" "$base_lib" > "$lib/zz_v41_common.pth"
bash scripts/sm100/run_five.sh "$work" direct-bs512 2>&1 | tee "$work/launcher.log"
```

## 回执及释放

提交 docs/sm100_megamoe/REPLY_V41_DIRECT_BS512.md 和 v41_direct_bs512_results/。附 delivery、日志、退出码、版本、硬件、放置 SHA、各臂启动结束时间和释放记录。发布五行 metrics.csv，包含逐层八卡最大跨度累加的完整 MoE ms/步、相对同轮 mega 的有符号变化（负数为更快）、token IR、总/峰值 M 块、活跃槽、M 块 IR、routed 与 graph 的预测误差及 ECO 成交次数。列实际 block M、上游 ECO 参数和分区。该表表示同提示下各臂自然生成时的性能。执行者仅可修复不改变语义的编译故障并附 diff；失败保留已完成臂和首栈。

结束后核对本任务进程退出、八卡及组锁释放。将测量 NPZ 归档或保留在执行机并记录路径与哈希，核对回执后清理本轮私有 checkout、venv、overlay 和编译缓存。保留共享模型、环境及历史回执，推送新回执到 main。
