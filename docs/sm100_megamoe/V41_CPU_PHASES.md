# SM100 旧五臂的 CPU 复核

在 SM100 执行机用 CPU 读留存的旧矩阵归档。可与 BS512 五臂同时执行，五臂的步骤和 GPU 预约照常。使用包含本文的 checkout，在仓库根目录执行。

```bash
set -euo pipefail
old=/data0/home/admin/workspace/megamoe/v41-five-matrix-01
out=docs/sm100_megamoe/v41_old_pair_phases.csv
py=/data0/home/admin/workspace/megamoe/native-v41-stsFS4w7/venv/bin/python
tmp=$(mktemp)
CUDA_VISIBLE_DEVICES="" "$py" scripts/sm100/old_pair_phases.py "$old" > "$tmp"
mv "$tmp" "$out"
git add "$out"
git commit -m 'SM100: report critical-rank phases from the original native arms'
git push origin HEAD:main
```

每步每层按本臂完整跨度选择关键 rank，三个区间都取同一 rank，单位 ms/步。核对 moe_ms 与原矩阵 metrics.csv。归档缺失或校验失败则回报路径和报错。此项只读归档，不启动模型。
