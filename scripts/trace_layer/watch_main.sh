#!/usr/bin/env bash
# From the checkout to keep current: bash scripts/trace_layer/watch_main.sh [poll-seconds]
set -eu -o pipefail
interval=${1:-60}
if [[ $# -gt 1 || ! $interval =~ ^[1-9][0-9]*$ ]]; then
    echo "Usage: $0 [positive poll interval in seconds]" >&2
    exit 2
fi

root=$(git rev-parse --show-toplevel)
cd "$root"
branch=$(git symbolic-ref --short HEAD)
state=$(git rev-parse --absolute-git-dir)/watch-main
mkdir -p "$state"
exec 9>>"$state/lock"
if ! flock -n 9; then
    echo "A watcher is already running in $root."
    exit 0
fi
export GIT_TERMINAL_PROMPT=0
exec >>"$state/watch.log" 2>&1
printf '%s\n' "$$" > "$state/pid"
trap 'rm -f -- "$state/pid"' EXIT
last_status=''

status() {
    if [[ $1 != "$last_status" ]]; then
        printf '%s %s\n' "$(date -u +%FT%TZ)" "$1"
        last_status=$1
    fi
}

poll() {
    local local_head remote_head
    if ! git fetch --quiet --no-tags --no-recurse-submodules --no-write-fetch-head \
            origin '+refs/heads/main:refs/remotes/origin/main'; then
        status "Fetch failed. Will retry."
        return
    fi
    local_head=$(git rev-parse HEAD) || return 1
    remote_head=$(git rev-parse refs/remotes/origin/main) || return 1
    date -u +%FT%TZ > "$state/last-fetch"
    if [[ $(git symbolic-ref -q --short HEAD) != "$branch" ]]; then
        status "Waiting for checkout of $branch."
    elif [[ $local_head == "$remote_head" ]]; then
        status "Up to date at $local_head."
    elif ! git merge-base --is-ancestor "$local_head" "$remote_head"; then
        status "No fast-forward from $local_head to origin/main $remote_head."
    elif ! git diff --quiet HEAD -- || ! git diff --cached --quiet; then
        status "Waiting for local edits before updating to $remote_head."
    elif git merge --quiet --ff-only --no-autostash --no-overwrite-ignore "$remote_head"; then
        status "Updated $branch from $local_head to $remote_head."
        git diff --name-status "$local_head" "$remote_head" --
    else
        status "Fast-forward to $remote_head deferred. Will retry."
    fi
}

status "Watching origin/main on $branch every ${interval}s."
while true; do
    poll || status "Git check failed. Will retry."
    sleep "$interval" 9>&-
done
