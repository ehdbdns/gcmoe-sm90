"""Read completed SM90 native GC arms 2/3. All interval arithmetic uses one GPU clock."""
import csv
import json
import sys
from pathlib import Path

import numpy as np

NAMES = ("ffn", "before_gc", "gc_to_first", "first_to_last", "last_to_combine", "after_gc")


def require(ok, msg):
    if not ok:
        raise ValueError(msg)


def intervals(stamps, clocks, counts):
    a, b = stamps.reshape(-1, 2).T
    g, t, u, c = clocks[:, :4].T
    active = counts.sum(1) > 0
    require(active.all(), "empty rank-layer: routed interval undefined, inspect separately")
    require(((0 < a) & (a <= g) & (g <= t) & (t < u) & (u <= c) & (c <= b)).all(),
            "timestamps not nested on the same rank")
    return np.stack((b-a, g-a, t-g, u-t, c-u, b-c), axis=-1)


def audit_keys(run):
    rows = sorted(run["measured"]["audit"], key=lambda r: r["rank"])
    initial = sorted(run["measured"]["initial"], key=lambda r: r["rank"])
    require([r["rank"] for r in rows] == [r["rank"] for r in initial] == list(range(8)), "rank coverage")
    require(all(r["full_graph_replays"] == 1000 and r["calls"] == 40000 for r in rows), "replay coverage")
    checks = [(r["ids"], r["weights"], r["demand"], r["batch_order"], b["checkpoint"]["logical_weights"],
               b["checkpoint"]["placement"], b["checkpoint"]["snapshot"]["readback_sha256"])
              for r, b in zip(rows, initial)]
    return run["prompt_hash"], run["output_hash"], checks


def analyze(root):
    dirs = [root / f"arm{i}" for i in (2, 3)]
    sha = (root / "measurement.sha").read_text().strip()
    for d in dirs:
        require((d / "complete").is_file() and (d / "execution.sha").read_text().strip() == sha, str(d))
    runs = [json.loads((d / "run.json").read_text()) for d in dirs]
    require(audit_keys(runs[0]) == audit_keys(runs[1]), "measured workload or boundary differs")
    arrays = [[], []]
    samples = None
    for rank in range(8):
        with np.load(dirs[0] / f"rank{rank}.npz") as f, np.load(dirs[1] / f"rank{rank}.npz") as v:
            q = f["counts"]
            require(q.shape == (40000, 48) and (q >= 0).all() and np.array_equal(q, v["counts"]),
                    f"physical counts differ at rank {rank}")
            native = [src["native"] for src in (f, v)]
            require(all(x.shape == (1000, 256) and (x >= 0).all() for x in native), "native coverage")
            if samples is None:
                samples = native
            require(all(np.array_equal(x, y) for x, y in zip(samples, native)), "native disagreement within TP")
            for i, src in enumerate((f, v)):
                c, epochs = src["clocks"], src["epochs"]
                require(np.array_equal(c[:, 4], epochs) and np.all(np.diff(epochs) == 1)
                        and (epochs[0]-1) % 40 == 0, "epoch coverage")
                require(src["ffn_ns"].shape == (1000, 40, 2), "FFN coverage")
                arrays[i].append(intervals(src["ffn_ns"], c, q))
    times = [np.stack(a, axis=1) for a in arrays]  # call, rank, interval in ns
    selected = [a[np.arange(len(a)), a[:, :, 0].argmax(1)] for a in times]
    means = [a.reshape(1000, 40, -1).sum(1).mean(0) / 1e6 for a in selected]
    f, v = means
    require(np.isclose((f-v)[0], (f-v)[1:].sum()), "phase deltas do not close")
    writer = csv.writer(sys.stdout)
    writer.writerow(["metric", "fused_ms_per_step", "unfused_ms_per_step", "fused_minus_unfused_ms_per_step"])
    for name, x, y in zip(NAMES, f, v):
        writer.writerow([name, x, y, x-y])
    x, y = [r["measured"]["ms_per_step"] for r in runs]
    writer.writerow(["decode_host", x, y, x-y])
    print("measurement_sha", sha, "native_top1_diff", np.count_nonzero(samples[0] != samples[1]),
          "ffn_change_pct", 100*(f[0]/v[0]-1), file=sys.stderr)
    writer.writerow(["layer", "rank", *["fused_minus_unfused_us_"+name for name in NAMES]])
    delta = (times[0]-times[1]).reshape(1000, 40, 8, -1).mean(0) / 1000
    for layer in range(40):
        for rank in range(8):
            writer.writerow([layer, rank, *delta[layer, rank]])


if __name__ == "__main__":
    analyze(Path(sys.argv[1]))
