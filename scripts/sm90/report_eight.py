"""Validate eight-arm workload controls and aggregate rank-local complete-FFN spans."""
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "trace_layer"))
from main20_report import metrics, slot_cost


def check(condition, message):
    if not condition:
        raise ValueError(message)


def read_arm(root, arm):
    path = root / f"arm{arm}"
    run = json.loads((path / "run.json").read_text())
    check((run["steps"], run["layers"], run["batch"], run["stream"]) == (1000, 40, 256, 1), "window")
    check(run["rearranged"] == (arm >= 4), "placement treatment")
    data, widths, spans, routed = [], [], [], []
    audit = {}
    greedy = None
    accepted = None
    for phase in ("stats", "measured"):
        rows = sorted(run[phase]["audit"], key=lambda r: r["rank"])
        check([r["rank"] for r in rows] == list(range(8)), "audit ranks")
        check(all(r["calls"] == 40000 for r in rows), "call coverage")
        if phase == "measured":
            check(all(r["full_graph_replays"] == 1000 and len(r["shared_modules"]) == 40 for r in rows),
                  "graph or shared workload coverage")
        audit[phase] = [[r[k] for k in ("ids", "weights", "demand")] for r in rows]
    initial = sorted(run["measured"]["initial"], key=lambda r: r["rank"])
    check([r["rank"] for r in initial] == list(range(8)), "checkpoint ranks")
    boundaries = [r["checkpoint"] for r in initial]
    audit["kv"] = [b["snapshot"]["readback_sha256"] for b in boundaries]
    audit["weights"] = [b["logical_weights"] for b in boundaries]
    audit["placement"] = [b["placement"] for b in boundaries]
    audit["trajectory"] = [run["prompt_hash"], run["output_hash"]]
    mega = arm in (1, 5)
    block_m = 64
    if mega:
        log = (path / "runner.log").read_text(errors="replace")
        # Official config line includes runtime m/ep shape. Only full-decode batch is measured.
        configs = [line for line in log.splitlines() if "num_tokens=256, num_topk=6)" in line and "MegaMoESM90Config(" in line]
        sizes = {int(m) for line in configs for m in re.findall(r"MegaMoESM90Config\(block_m=(\d+)", line)}
        check(len(sizes) == 1, f"ambiguous official Mega tile M {sizes}")
        block_m = sizes.pop()
    for rank in range(8):
        with np.load(path / f"rank{rank}.npz", allow_pickle=False) as f:
            q = f["counts"]
            check(q.shape == (40000, 48 if arm <= 3 else 56) and np.all(q >= 0), "physical counts")
            epoch = initial[rank]["epoch"] + np.arange(40000)
            check(np.array_equal(f["epochs"], epoch), "epoch gap")
            bounds = f["ffn_ns"]
            check(bounds.shape == (1000, 40, 2) and np.all(bounds[..., 0] > 0)
                  and np.all(bounds[..., 1] > bounds[..., 0]), "FFN timeline")
            spans.append((bounds[..., 1] - bounds[..., 0]).ravel() / 1000)
            data.append(q)
            widths.append(((q + block_m - 1) // block_m).sum(1))
            if not mega:
                clocks = f["clocks"]
                check(np.array_equal(clocks[:, 4], epoch), "GC graph clocks")
                active = q.sum(1) > 0
                check(np.all(clocks[active, 2] > clocks[active, 1]), "GC routed clocks")
                routed.append(np.where(active, clocks[:, 2] - clocks[:, 1], 0))
            x = f["native"]
            check(x.shape == (1000, 256) and np.all(x >= 0), "native output")
            if greedy is None:
                greedy = x
            check(np.array_equal(x, greedy), "rank disagreement on native output")
            if arm in (7, 8):
                a = f["auction"]
                check(a.shape == (40000, 7) and np.all(a >= 0), "auction records")
                if accepted is None:
                    accepted = a
                check(np.array_equal(accepted, a), "replicated auction disagreement")
    q = np.stack(data, axis=1)
    check(np.all(q.sum((1, 2)) == 8 * 256 * 6), "route conservation")
    elapsed = np.stack(spans, axis=1)
    row = metrics(q, elapsed[..., None])
    blocks = np.stack(widths, axis=1)
    row.update(total_m_blocks=blocks.sum(1).mean(), peak_rank_m_blocks=blocks.max(1).mean(),
               m_block_ir=(blocks.max(1) / blocks.mean(1)).mean(), block_m=block_m,
               accepted=int(accepted[:, 2].sum()) if accepted is not None else None)
    if routed:
        price = np.array([slot_cost(k) for k in range(int(q.max()) + 1)])[q].sum(-1)
        for key, target in (("graph", elapsed), ("routed", np.stack(routed, axis=1))):
            err = np.abs(price / price.sum(1, keepdims=True) - target / target.sum(1, keepdims=True))
            row[key + "_mae_pp"] = 100 * err.mean()
            row[key + "_max_pp"] = 100 * err.max()
    audit["counts"] = hashlib.sha256(q.tobytes()).hexdigest()
    audit["auction"] = hashlib.sha256(accepted.tobytes()).hexdigest() if accepted is not None else None
    return row, audit, greedy


def report(root):
    rows, audits, natives, failures = {}, {}, {}, {}
    for arm in (2, 3, 4, 6, 7, 8, 1, 5):
        try:
            rows[arm], audits[arm], natives[arm] = read_arm(root, arm)
            for key in ("trajectory", "stats", "measured", "kv"):
                check(audits[arm][key] == audits[2][key], f"publisher mismatch in {key}")
            for peer in rows:
                if peer == arm or peer in failures:
                    continue
                if (arm in (1, 5)) == (peer in (1, 5)):
                    check(audits[arm]["weights"] == audits[peer]["weights"], "logical weight mismatch")
                if (arm >= 4) == (peer >= 4):
                    check(audits[arm]["placement"] == audits[peer]["placement"], "placement mismatch")
                if (arm <= 3 and peer <= 3 or 4 <= arm <= 6 and 4 <= peer <= 6 or arm >= 7 and peer >= 7):
                    check(audits[arm]["counts"] == audits[peer]["counts"], "physical workload mismatch")
                    check(audits[arm]["auction"] == audits[peer]["auction"], "auction mismatch")
            rows[arm]["native_top1_diff"] = int(np.count_nonzero(natives[arm] != natives[2]))
        except (OSError, ValueError, KeyError) as exc:
            failures[arm] = str(exc)
    valid = {a: r for a, r in rows.items() if a not in failures}
    for arm, row in valid.items():
        row["arm"] = arm
        row["vs_2_pct"] = 100 * (row["moe_ms_per_step"] / rows[2]["moe_ms_per_step"] - 1)
    if valid:
        with (root / "summary.csv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in valid.values() for k in r)))
            writer.writeheader()
            writer.writerows(valid[a] for a in sorted(valid))
    pairs = {f"{a}_to_{b}": 100 * (valid[b]["moe_ms_per_step"] / valid[a]["moe_ms_per_step"] - 1)
             for a, b in ((2, 3), (4, 6), (7, 8), (1, 3), (5, 6), (4, 7), (6, 8)) if a in valid and b in valid}
    (root / "verification.json").write_text(json.dumps(dict(failures=failures, audits=audits, pair_change_pct=pairs), indent=2))
    return int(bool(failures))


if __name__ == "__main__":
    sys.exit(report(Path(sys.argv[1])))
