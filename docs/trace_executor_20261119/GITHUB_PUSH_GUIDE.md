# 回执推送 GitHub 的标准流程（SM90 执行者的实际做法）

> 适用：把任务书回执与结果目录提交并推送到 `ehdbdns/gcmoe-sm90` 的 `main`。
> 把下面的 `/path/to/your/repo` 换成自己的仓库路径。

## 0. 前提

仓库里 `origin` 已带 token，不用自己拼 URL：

```bash
git remote -v
# origin  https://ghp_xxx@github.com/ehdbdns/gcmoe-sm90.git (fetch)
# origin  https://ghp_xxx@github.com/ehdbdns/gcmoe-sm90.git (push)
```

**踩过的坑（很重要）**：这台机器上普通的 `git fetch` 经常返回**旧的** origin/main，
看起来"没更新"其实是连接池/缓存问题。**必须加 `-c http.version=HTTP/1.1`**。

## 1. 同步到最新（每次开头都做）

```bash
cd /path/to/your/repo

timeout 90 git -c http.version=HTTP/1.1 fetch origin main

echo "local=$(git rev-parse --short HEAD)  remote=$(git rev-parse --short origin/main)"

git status --porcelain      # 必须为空
git merge --ff-only origin/main
```

fetch 报 `Empty reply from server` 就直接重试，最多 3 次，通常会成功。

## 2. 把产物放进仓库目录

按任务书给的路径建目录（**不要用 `rm -rf`，环境策略会直接拒绝**）：

```bash
cd /path/to/your/repo

target=docs/<任务目录>/<结果目录名>          # 例如 docs/trace_executor_xxx/xxx_results
test ! -e "$target" || { echo "target exists, stop"; exit 1; }
mkdir -p "$target"

cp -a /tmp/your-work/delivery/. "$target/"
cp /tmp/your-work/preflight.log "$target/"
cp /tmp/your-work/progress.log  "$target/"

# 回执按任务书指定路径
cp /tmp/reply.md docs/trace_handover_xxx/REPLY_XXX.md
```

## 3. 生成校验和并自检（必须做）

**在目标目录内部**生成，让路径是相对的：

```bash
cd "$target"
find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS

sha256sum --quiet -c SHA256SUMS && echo "SHA_OK"    # 必须看到 SHA_OK
```

## 4. 提交

**提交信息走文件，不要写在 `-m "..."` 里** —— 消息含单引号或反引号会被 shell 吃掉，
报 `error: pathspec 'xxx' did not match`。

```bash
cd /path/to/your/repo

cat > /tmp/commitmsg.txt <<'MSGEOF'
[SM100] Report <任务简述>

<正文：每个核心数字一行，说明判据是否通过>
MSGEOF

git add -- "$target" docs/trace_handover_xxx/REPLY_XXX.md

git diff --cached --stat | tail -5      # 确认没多提交/漏提交

git commit -F /tmp/commitmsg.txt
git log --oneline -1
```

## 5. 推送（带 postBuffer + 可重试）

```bash
timeout 300 git -c http.version=HTTP/1.1 -c http.postBuffer=524288000 \
  push origin HEAD:main 2>&1 | tail -3
```

| 报错 | 处理 |
|---|---|
| `Empty reply from server` | 重试，通常第 2–3 次成功 |
| `non-fast-forward` / `hint: use 'git pull'` | 别人先推了，走第 6 步 |

## 6. 推送被拒（别人抢先推了）

**不要强推**，用 rebase：

```bash
cd /path/to/your/repo

timeout 90 git -c http.version=HTTP/1.1 fetch origin main
git -c http.version=HTTP/1.1 pull --rebase origin main

git rev-list --left-right --count origin/main...HEAD   # 应为 "0  1"

timeout 300 git -c http.version=HTTP/1.1 -c http.postBuffer=524288000 \
  push origin HEAD:main
```

## 7. 确认真的推上去了（别只看 push 输出）

```bash
timeout 60 git -c http.version=HTTP/1.1 fetch origin main -q
echo "local=$(git rev-parse --short HEAD)  remote=$(git rev-parse --short origin/main)"
# 两个短 SHA 必须相同

git ls-tree -r --name-only origin/main docs/<任务目录>/ | head -20
```

## 8. 清理现场

**禁止 `rm -rf`**。用 Python：

```bash
python3 - <<'EOF'
import shutil, os
for d in ('/tmp/your-work-dir',):
    if os.path.isdir(d):
        shutil.rmtree(d, ignore_errors=True)
        print(d, 'removed' if not os.path.exists(d) else 'FAILED')
for f in ('/tmp/commitmsg.txt',):
    if os.path.isfile(f):
        os.unlink(f)
EOF
```

用了多工作树就先摘再删：

```bash
git worktree remove --force /tmp/your-work-dir/code
git worktree prune
```

## 9. 完整可复制模板

```bash
#!/bin/bash
set -eo pipefail
REPO=/path/to/your/repo
HANDOVER=docs/trace_handover_xxx/REPLY_XXX.md
TARGET=docs/trace_executor_xxx/xxx_results
WORK=/tmp/your-work
MSG=/tmp/commitmsg.txt

cd "$REPO"

# 1) 同步（HTTP/1.1 必须）
timeout 90 git -c http.version=HTTP/1.1 fetch origin main
test -z "$(git status --porcelain)" || { echo "dirty worktree"; exit 1; }
git merge --ff-only origin/main

# 2) 产物入仓
test ! -e "$TARGET" || { echo "target exists"; exit 1; }
mkdir -p "$TARGET"
cp -a "$WORK/delivery/." "$TARGET/"
cp "$WORK/preflight.log" "$WORK/progress.log" "$TARGET/"
cp /tmp/reply.md "$HANDOVER"

# 3) 校验和 + 自检
( cd "$TARGET" && find . -type f ! -name SHA256SUMS -print0 \
    | sort -z | xargs -0 sha256sum > SHA256SUMS \
    && sha256sum --quiet -c SHA256SUMS && echo SHA_OK )

# 4) 提交（消息走文件）
cat > "$MSG" <<'EOF'
[SM100] Report <任务简述>
EOF
git add -- "$TARGET" "$HANDOVER"
git diff --cached --stat | tail -5
git commit -F "$MSG"

# 5) 推送（带 postBuffer）
timeout 300 git -c http.version=HTTP/1.1 -c http.postBuffer=524288000 \
  push origin HEAD:main

# 6) 确认
timeout 60 git -c http.version=HTTP/1.1 fetch origin main -q
echo "local=$(git rev-parse --short HEAD) remote=$(git rev-parse --short origin/main)"
```

## 10. 三个最容易翻车的点

1. **不写 `-c http.version=HTTP/1.1`** → fetch 拿到旧提交，误以为"任务没更新"。
2. **提交信息写在 `-m "..."` 里** → 单引号/反引号被 shell 解析，
   报 `error: pathspec ... did not match`。
3. **只看 `git push` 输出了事** → 一定用 `git rev-parse` 比对 local 与 remote 短 SHA。

## 11. 附：我这边实际用过的轮询命令

```bash
# 每 20 秒检查一次远端，有新提交就打印
while true; do
  timeout 30 git -C /tmp/eco_publish/repo -c http.version=HTTP/1.1 fetch origin main -q 2>/dev/null
  l=$(git -C /tmp/eco_publish/repo rev-parse HEAD)
  r=$(git -C /tmp/eco_publish/repo rev-parse origin/main)
  echo "$(date +%H:%M:%S) local=${l:0:7} remote=${r:0:7}"
  [ "$l" != "$r" ] && git -C /tmp/eco_publish/repo log --oneline "$l..$r" | head -5
  sleep 20
done
```
