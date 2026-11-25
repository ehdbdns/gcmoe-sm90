"""Manage the BS256 eight-arm fallback and verify legacy Mega provenance."""
import csv
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np
from report_eight import NATIVE_SHA, check, read_arm

ARCHIVE = Path(__file__).resolve().parents[2] / "docs/trace_executor_20261119"
OLD = ARCHIVE / "sm90_h20_bs256_mega_gc_six_v2_results/results"


def select(path):
    with path.open() as f:
        rows = [r for r in csv.DictReader(f) if int(r["bs"]) == 256]
    check(len(rows) == 33 and {int(r["k"]) for r in rows} == set(range(33)), "incomplete BS256 sweep")
    check(all(r["method"] == "ordered_rr" and r["device"] == "cuda" and
              int(r["training_steps"]) == 1000 and int(r["measurement_calls"]) == 40000 and
              math.isfinite(float(r["token_ir"])) and float(r["token_ir"]) >= 1 for r in rows),
          "expected exact H20 placement and ordinal RR counters")
    eligible = [(float(r["token_ir"]), int(r["k"])) for r in rows if 1 <= int(r["k"]) < 8]
    return min(eligible)[1]


def reduce(root):
    """Archive four-slot placements after a capacity failure; keep the publisher and native arms."""
    check((root/"tuned-extra").read_text().strip() == "4", "expected a four-slot attempt")
    archive = root/"failed-k4"
    archive.mkdir(exist_ok=True)
    saved = archive/"capacity-failed-arm"
    marker = root/"capacity-failed-arm"
    arm = int((saved if saved.is_file() else marker).read_text())
    check(4 <= arm <= 8, "replication fallback requires a placement arm failure")
    status = Path(f"arm{arm}")/"exit"
    evidence = archive/status if (archive/status).is_file() else root/status
    check(int(evidence.read_text()) != 0, "failed arm has a successful exit")
    if not saved.exists():
        marker.rename(saved)

    (archive/"replay").mkdir(exist_ok=True)
    files = [Path("replay/placement.json"), Path("replay/placement.tmp"),
             Path("summary.csv"), Path("pairs.csv"), Path("verification.json")]
    for arm in range(4, 9):
        files.extend((Path(f"arm{arm}"), Path(f"replay/{arm}-layout.json")))
    for rel in files:
        if (root/rel).exists():
            check(not (archive/rel).exists(), f"fallback archive already contains {rel}")
            (root/rel).rename(archive/rel)
    (archive/"tuned-extra").write_text("4\n")
    next_budget = root/"tuned-extra.tmp"
    next_budget.write_text("1\n")
    next_budget.replace(root/"tuned-extra")


def reuse(root):
    # The publisher was regenerated. Carry the native consumer's original SHA
    # only if the two publishers match on this hardware at both boundaries.
    try:
        for key in ("input.sha256", "input.sha256.log", "conversion.jsonl", "gpus.csv"):
            check((root/key).read_bytes() == (OLD/key).read_bytes(), f"different {key}")
        check((root/"arm0/runtime.log").read_bytes() == (OLD/"arm1/runtime.log").read_bytes(), "runtime changed")
        for arm in (0, 1):
            folder = OLD / f"arm{arm}"
            check((folder/"complete").is_file() and (folder/"execution.sha").read_text().strip() == NATIVE_SHA,
                  "legacy arm incomplete or unpinned")
            subprocess.run(["sha256sum", "--quiet", "-c", "artifacts.sha256"], cwd=folder, check=True)
        old_run = json.loads((OLD/"arm0/run.json").read_text())
        new_run = json.loads((root/"arm0/run.json").read_text())
        for key in ("GC_MOE_KV_BYTES","CUDAGRAPH_MODE","GC_MOE_BLOCK_M_FORCE","MAX_BATCHED","NCCL_IB_HCA"):
            check(new_run["controls"].get(key) == old_run["controls"].get(key), f"different {key}")
        _, old, old_ids = read_arm(OLD, 0)
        _, new, new_ids = read_arm(root, 0, kv_gib=10, tuned=True)
        check(old == new and np.array_equal(old_ids, new_ids), "publisher audit or actual outputs differ")
        read_arm(OLD, 1)
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as exc:
        print(f"Measure native Mega on the current checkout: {exc}")
        return
    with tempfile.TemporaryDirectory(dir=root) as tmp:
        dst = Path(tmp)/"arm1"
        shutil.copytree(OLD/"arm1", dst)
        (dst/"imported-native").write_text(NATIVE_SHA+"\n")
        dst.rename(root/"arm1")
    print(f"Accepted complete archived native Mega from {NATIVE_SHA}")


if __name__ == "__main__":
    action, path = sys.argv[1:]
    if action == "select":
        print(select(Path(path)))
    elif action == "reduce":
        reduce(Path(path))
    elif action == "reuse":
        reuse(Path(path))
    else:
        raise ValueError(action)
