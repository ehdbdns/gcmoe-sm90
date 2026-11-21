"""Assemble the small GitHub delivery and checksum the retained rank evidence."""

import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys


def collect(root):
    dest = root / "delivery"
    dest.mkdir(exist_ok=True)
    for name in (
        "revision",
        "weights.sha256",
        "versions.txt",
        "gpus.txt",
        "ptxas.txt",
        "header.log",
        "acquired.log",
        "rr-preflight.log",
    ):
        if (root / name).exists():
            shutil.copy2(root / name, dest / name)
    totals = {"metrics": [], "pairs": []}
    status = []
    with (dest / "rank-files.sha256").open("w") as manifest:
        for bs in (256, 512, 1024, 2048):
            group = root / f"bs{bs}"
            sub = dest / group.name
            sub.mkdir(exist_ok=True)
            summary = group / "summary.json"
            status.append(
                json.loads(summary.read_text())
                if summary.exists()
                else {"bs": bs, "failures": ["missing summary"]}
            )
            for name, rows in totals.items():
                path = group / f"{name}.csv"
                if path.exists():
                    with path.open() as f:
                        rows.extend(dict(row, bs=bs) for row in csv.DictReader(f))
            for path in [
                group / "plan.npz",
                summary,
                *group.glob("*/result.json"),
                *group.glob("*/exit-code"),
                *group.glob("*/start"),
                *group.glob("*/end"),
            ]:
                if path.exists():
                    target = sub / path.relative_to(group)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, target)
            for path in sorted(group.glob("*/*-rank*.npz")):
                with path.open("rb") as src:
                    digest = hashlib.file_digest(src, "sha256").hexdigest()
                manifest.write(f"{digest}  {path.relative_to(root)}\n")
    for name, rows in totals.items():
        if rows:
            with (dest / f"{name}.csv").open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
    (dest / "status.json").write_text(json.dumps(status, indent=2))


if __name__ == "__main__":
    collect(Path(sys.argv[1]))
