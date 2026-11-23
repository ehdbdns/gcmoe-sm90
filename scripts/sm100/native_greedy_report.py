"""Compare independently decoded streams, preserving differing native routes and outputs."""
import json
from pathlib import Path
import sys

import numpy as np

from full_decode_report import graph_evidence


def differences(a, b):
    if a.shape != b.shape:
        raise ValueError("incompatible observation shapes")
    mask = a != b
    count = int(mask.sum())
    first = None
    if count:
        pos = np.unravel_index(mask.argmax(), mask.shape)
        first = dict(position=list(map(int, pos)), mega=a[pos].item(), gc=b[pos].item())
    return dict(count=count, first=first)


def report(root):
    arms = ("mega", "gc")
    info = [json.loads((root / arm / "result.json").read_text()) for arm in arms]
    assert info[0]["prompt_sha256"] == info[1]["prompt_sha256"]
    result = dict(prompt_sha256=info[0]["prompt_sha256"], graph={}, moe_ms={}, phases={})
    prefixes = [json.loads((root / arm / "prefill.json").read_text()) for arm in arms]
    result["prefill_mismatched_requests"] = sum(prefixes[0][k] != prefixes[1][k] for k in prefixes[0])
    for arm in arms:
        evidence = [graph_evidence(root / arm / f"profile-rank{r}.json", arm) for r in range(8)]
        assert all(record["passed"] for record in evidence)
        result["graph"][arm] = evidence
    for phase in ("stats", "measure"):
        tokens = [np.load(root / arm / f"{phase}-tokens.npy").T for arm in arms]
        assert all(t.shape == (1000, 512) for t in tokens)
        summary = dict(tokens=differences(*tokens), ranks=[])
        result["phases"][phase] = summary
        totals = np.zeros((2, 1000, 40), dtype=np.int64)
        critical_ns = np.zeros_like(totals)
        for r in range(8):
            row = {}
            for name in ("ids", "weights"):
                values = [np.load(root / arm / f"{phase}-{name}-rank{r}.npy", mmap_mode="r")
                          .reshape(1000, 40, 64, 6) for arm in arms]
                row[name] = differences(*values)
            counts = []
            for i, arm in enumerate(arms):
                rank = info[i][phase]["ranks"][r]
                assert rank["full_graph_observed_calls"] == 40000
                with np.load(root / arm / f"{phase}-rank{r}.npz") as f:
                    c = f["counts"]
                    assert c.shape == (1000, 40, 48) and (c >= 0).all()
                    counts.append(c)
                    totals[i] += c.sum(-1)
                    clocks = f["clocks"][..., :4]
                    assert (clocks[..., 0] > 0).all() and (np.diff(clocks, axis=-1) >= 0).all()
                    np.maximum(critical_ns[i], clocks[..., 3] - clocks[..., 0], out=critical_ns[i])
            row["slot_counts"] = differences(*counts)
            summary["ranks"].append(row)
        assert (totals == 3072).all()
        if phase == "measure":
            for i, arm in enumerate(arms):
                result["moe_ms"][arm] = float(np.delete(critical_ns[i].sum(1), 2).mean() / 1e6)
    times = result["moe_ms"]
    result["gc_time_change_percent_vs_mega"] = 100 * (times["gc"] / times["mega"] - 1)
    (root / "native_greedy_pair.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    report(Path(sys.argv[1]))
