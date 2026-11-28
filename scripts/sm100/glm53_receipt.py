"""Package the GLM measurement or its failure without publishing prompts, KV or weights."""

import argparse
import gzip
import hashlib
import json
import re
import shutil
from pathlib import Path


def first_errors(log, output):
    # Teardown on the other seven ranks often fills the last hundred lines.
    if not log.is_file():
        return
    issue = re.compile(r"Traceback \(most recent call last\)|\b(?:RuntimeError|ValueError|CUDAError|"
                       r"OutOfMemoryError|AcceleratorError):|CUDA error:|illegal (?:memory|instruction)|"
                       r"GC .*timeout|ECO plan assertion|device-side assert|SIGTERM|SIGKILL", re.I)
    lines = []
    with log.open(errors="replace") as stream:
        for n, line in enumerate(stream, 1):
            if issue.search(line):
                clean = re.sub(r"\S+://\S+|(?:gh[pousr]_|github_pat_)\w+", "[redacted]", line.strip())
                lines.append(f"{n}: {clean[:1200]}")
                if len(lines) == 80:
                    break
    if lines:
        output.write_text("\n".join(lines) + "\n")


def collect(root, out, rc):
    out.mkdir(parents=True)
    receipt = {
        "private_archive": str(root),
        "exit_code": rc,
        "revision": (root / "revision").read_text().strip() if (root / "revision").exists() else None,
        "complete_arms": [],
        "files": [],
    }
    arms = ("vanilla", "rr", "eco")
    if (root / "scheduled-arms.json").exists():
        arms = json.loads((root / "scheduled-arms.json").read_text())
        if arms != ["vanilla", "rr", "eco8", "eco12"]:
            raise ValueError("invalid GLM slot-budget schedule")
        receipt["scheduled_arms"] = arms
        for path in root.glob("placement-extra*.npy"):
            shutil.copy2(path, out / path.name)
    for item in ("scheduled-arms.json", "main.csv", "reference-main.csv", "placement.npy", "revision", "arch-proof.json", "deepgemm-proof.json",
                 "resume.json", "resume-attempts.txt", "gpu-admission.csv", "launch-comparison.json"):
        if (root / item).is_file():
            shutil.copy2(root / item, out / item)
    for name in ("historical-price", "new-price"):
        if (root / name / "scores.json").exists():
            shutil.copytree(root / name, out / name)
    if (root / "resume.json").is_file():
        previous = json.loads((root / "resume.json").read_text()).get("rr_source")
        if previous:
            first_errors(Path(previous) / "eco-extra8-try0.log", out / "previous-eco-errors.txt")
    for arm in arms:
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
            if f.name in ("setup.json", "result.json") or f.name.endswith((".perfetto.json", ".ranks.json")):
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
        for log in root.glob("*-extra*-try*.log"):
            first_errors(log, out / (log.stem + "-errors.txt"))
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
