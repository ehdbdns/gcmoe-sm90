"""Publish the packaged GLM receipt atomically using only api.github.com."""
import argparse
import base64
import hashlib
import json
import os
import re
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

REPO = "ehdbdns/gcmoe-sm90"
PARENT = "docs/sm100_megamoe/glm53_formal_results"


def request(method, path, data=None):
    token = os.environ["GH_TOKEN"]
    req = Request(
        f"https://api.github.com/repos/{REPO}/{path}",
        method=method,
        data=json.dumps(data).encode() if data is not None else None,
        headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
                 "Content-Type": "application/json", "User-Agent": "glm53-executor"},
    )
    for attempt in range(4):
        try:
            with urlopen(req, timeout=180) as response:
                return json.load(response)
        except HTTPError as exc:
            if exc.code not in (408, 429, 500, 502, 503, 504) or attempt == 3:
                raise
        except (URLError, TimeoutError):
            if attempt == 3:
                raise
        time.sleep(2 ** attempt)


def lookup(tree, path):
    for part in path.split("/"):
        children = request("GET", f"git/trees/{tree}")
        if children.get("truncated"):
            raise RuntimeError("incomplete tree listing")
        entry = next((e for e in children["tree"] if e["path"] == part), None)
        if entry is None:
            return None
        tree = entry["sha"]
    return tree


def publish(folder):
    if not re.fullmatch(r"formal\.[a-zA-Z0-9_-]+", folder.name):
        raise ValueError("pass the formal.<run-id> directory produced by the receipt packager")
    receipt = json.loads((folder / "receipt.json").read_text())
    target = f"{PARENT}/{folder.name}"
    entries = []
    for f in sorted(folder.rglob("*")):
        if not f.is_file():
            continue
        if f.is_symlink() or f.stat().st_size >= 100 << 20:
            raise ValueError(f"unexpected receipt file {f}")
        data = f.read_bytes()
        expected = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        blob = request("POST", "git/blobs", {"encoding": "base64", "content": base64.b64encode(data).decode()})
        if blob["sha"] != expected:
            raise ValueError(f"GitHub blob hash differs for {f.name}")
        entries.append({"path": f.relative_to(folder).as_posix(), "mode": "100644", "type": "blob", "sha": expected})
    subtree = request("POST", "git/trees", {"tree": entries})["sha"]
    for _ in range(5):
        head = request("GET", "git/ref/heads/main")["object"]["sha"]
        base = request("GET", f"git/commits/{head}")["tree"]["sha"]
        existing = lookup(base, target)
        if existing == subtree:
            return head
        if existing:
            raise ValueError(f"refusing to replace existing receipt {target}")
        tree = request("POST", "git/trees", {"base_tree": base, "tree": [
            {"path": target, "mode": "040000", "type": "tree", "sha": subtree}]})["sha"]
        commit = request("POST", "git/commits", {
            "message": f"[SM100] GLM-5.3 BS256 {len(receipt['complete_arms'])}/3 arms (exit {receipt['exit_code']})",
            "tree": tree, "parents": [head]})["sha"]
        try:
            request("PATCH", "git/refs/heads/main", {"sha": commit, "force": False})
        except HTTPError as exc:
            if exc.code not in (409, 422):
                raise
            if request("GET", "git/ref/heads/main")["object"]["sha"] == head:
                raise
            continue
        tip = request("GET", "git/ref/heads/main")["object"]["sha"]
        tip_tree = request("GET", f"git/commits/{tip}")["tree"]["sha"]
        if lookup(tip_tree, target) == subtree:
            return commit
    raise RuntimeError("main advanced repeatedly; rerun the publish command with this same receipt")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("receipt", type=Path)
    dest = p.parse_args().receipt.resolve()
    sha = publish(dest)
    print(f"https://github.com/{REPO}/tree/{sha}/{PARENT}/{dest.name}")
