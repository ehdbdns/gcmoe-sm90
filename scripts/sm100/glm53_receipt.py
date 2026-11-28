"""Package the GLM measurement or its failure without publishing prompts, KV or weights."""

import argparse
import gzip
import hashlib
import json
import re
import shutil
from pathlib import Path


def collect(root, out, rc):
    out.mkdir(parents=True)
    receipt = {
        "private_archive": str(root),
        "exit_code": rc,
        "revision": (root / "revision").read_text().strip() if (root / "revision").exists() else None,
        "complete_arms": [],
        "files": [],
    }
    for item in ("main.csv", "placement.npy", "revision", "arch-proof.json", "deepgemm-proof.json",
                 "resume.json", "resume-attempts.txt", "gpu-admission.csv"):
        if (root / item).is_file():
            shutil.copy2(root / item, out / item)
    for arm in ("vanilla", "rr", "eco"):
        src = root / arm
        if (src / "result.json").exists():
            receipt["complete_arms"].append(arm)
        if not src.is_dir():
            continue
        for f in sorted(src.rglob("*")):
            if not f.is_file():
                continue
            rel = f.relative_to(root)
            with f.open("rb") as stream:
                h = hashlib.sha256()
                for block in iter(lambda: stream.read(1 << 20), b""):
                    h.update(block)
                digest = h.hexdigest()
            row = {"path": str(rel), "bytes": f.stat().st_size, "sha256": digest}
            if f.name == "result.json" or f.name.endswith((".perfetto.json", ".ranks.json")):
                dst = out / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                if row["bytes"] < 45 << 20:
                    shutil.copyfile(f, dst)
                    row["github"] = str(rel)
                else:
                    dst = dst.with_suffix(".json.gz")
                    with f.open("rb") as ip, gzip.open(dst, "wb", compresslevel=5) as op:
                        shutil.copyfileobj(ip, op)
                    if dst.stat().st_size < 90 << 20:
                        row["github"] = str(dst.relative_to(out))
                    else:
                        dst.unlink()
            receipt["files"].append(row)
    if rc:
        path = root / "run.log"
        if path.exists():
            with path.open("rb") as f:
                f.seek(max(0, path.stat().st_size - 48000))
                tail = f.read().decode(errors="replace").splitlines()[-100:]
            (out / "failure-tail.txt").write_text(re.sub(r"\S+://\S+", "[URL]", "\n".join(tail)) + "\n")
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("code", type=int)
    a = p.parse_args()
    collect(a.root, a.out, a.code)
