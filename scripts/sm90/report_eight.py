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


NATIVE_SHA = "3c2d8e8556fa7ffddf26a1f51e979dec34d7f61e"


def read_arm(root, arm, batch=256, steps=1000, kv_gib=None, extra=8, tuned=False):
    path = root / f"arm{arm}"
    run = json.loads((path / "run.json").read_text())
    check((run["steps"], run["layers"], run["batch"], run["stream"]) == (steps, 40, batch, 1), "window")
    check(run["rearranged"] == (arm >= 4), "placement treatment")
    data, widths, spans, routed = [], [], [], []
    audit = {}
    greedy = None
    accepted = None
    calls = steps * 40
    if kv_gib is not None:
        check(int(run["controls"]["GC_MOE_KV_BYTES"]) == kv_gib * 1024**3, "common KV budget")
        check(int(run["controls"]["GC_MOE_EPLB_WINDOW"]) == steps, "EPLB statistics window")
        if arm == 1 and (path/"imported-native").exists() and tuned:
            check((path/"execution.sha").read_text().strip() == NATIVE_SHA and kv_gib == 10 and steps == 1000,
                  "invalid legacy native provenance")
        else:
            cache = run["kv_capacity"]
            required = batch * (705 + 2 * steps)
            check(cache["tokens"] >= cache["required"] == required and cache["concurrency"] >= batch,
                  "insufficient initialized KV capacity")
    if tuned and arm >= 4:
        check(int(run["controls"]["GC_MOE_EPLB_EXTRA_PER_RANK"]) == extra, "wrong EPLB slot budget")
        if arm >= 7:
            check(all(r["checkpoint"]["plan_price"] == {"mode": 0, "slack": 92}
                      for r in run["measured"]["initial"]), "wrong linear ECO price")
    for phase in ("stats", "measured"):
        if kv_gib is not None:
            check(len(run[phase]["ten_step_ms"]) == steps // 10, "incomplete host timing window")
        rows = sorted(run[phase]["audit"], key=lambda r: r["rank"])
        check([r["rank"] for r in rows] == list(range(8)), "audit ranks")
        check(all(r["calls"] == calls for r in rows), "call coverage")
        if phase == "measured":
            check(all(r["full_graph_replays"] == steps and len(r["shared_modules"]) == 40 for r in rows),
                  "graph or shared workload coverage")
        audit[phase] = [[r[k] for k in ("ids", "weights", "demand")] for r in rows]
    audit["warm_native"] = [r.get("warm_native") for r in rows]
    audit["order"] = [r["batch_order"] for r in rows]
    audit["shared"] = [r["shared_modules"] for r in rows]
    audit["prefill_kv"] = [r["checkpoint"]["prefill_snapshot"]["readback_sha256"]
                           for r in sorted(run["stats"]["initial"], key=lambda r: r["rank"])]
    initial = sorted(run["measured"]["initial"], key=lambda r: r["rank"])
    check([r["rank"] for r in initial] == list(range(8)), "checkpoint ranks")
    boundaries = [r["checkpoint"] for r in initial]
    audit["kv"] = [b["snapshot"]["readback_sha256"] for b in boundaries]
    audit["weights"] = [b["logical_weights"] for b in boundaries]
    audit["installed_layout"] = [b.get("installed_layout") for b in boundaries]
    audit["placement"] = [b["placement"] for b in boundaries]
    audit["trajectory"] = [run["prompt_hash"], run["output_hash"]]
    mega = arm in (1, 5)
    block_m = 64
    if mega:
        log = (path / "runner.log").read_text(errors="replace")
        # Official config line includes runtime m/ep shape. Only full-decode batch is measured.
        configs = [line for line in log.splitlines() if f"num_tokens={batch}, num_topk=6)" in line and "MegaMoESM90Config(" in line]
        sizes = {int(m) for line in configs for m in re.findall(r"MegaMoESM90Config\(block_m=(\d+)", line)}
        check(len(sizes) == 1, f"ambiguous official Mega tile M {sizes}")
        block_m = sizes.pop()
    for rank in range(8):
        with np.load(path / f"rank{rank}.npz", allow_pickle=False) as f:
            q = f["counts"]
            check(q.shape == (calls, 48 if arm <= 3 else 48 + extra) and np.all(q >= 0), "physical counts")
            epoch = initial[rank]["epoch"] + np.arange(calls)
            check(np.array_equal(f["epochs"], epoch), "epoch gap")
            bounds = f["ffn_ns"]
            check(bounds.shape == (steps, 40, 2) and np.all(bounds[..., 0] > 0)
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
            check(x.shape == (steps, batch) and np.all(x >= 0), "native output")
            if greedy is None:
                greedy = x
            check(np.array_equal(x, greedy), "rank disagreement on native output")
            if arm in (7, 8):
                a = f["auction"]
                check(a.shape == (calls, 7) and np.all(a >= 0), "auction records")
                if accepted is None:
                    accepted = a
                check(np.array_equal(accepted, a), "replicated auction disagreement")
    q = np.stack(data, axis=1)
    check(np.all(q.sum((1, 2)) == 8 * batch * 6), "route conservation")
    elapsed = np.stack(spans, axis=1)
    row = metrics(q, elapsed[..., None], steps=steps)
    blocks = np.stack(widths, axis=1)
    step_ms = float(run["measured"]["ms_per_step"])
    check(np.isfinite(step_ms) and step_ms > 0, "invalid full step timer")
    row.update(batch=batch, step_ms=step_ms, total_m_blocks=blocks.sum(1).mean(), peak_rank_m_blocks=blocks.max(1).mean(),
               m_block_ir=(blocks.max(1) / blocks.mean(1)).mean(), block_m=block_m,
               accepted=int(accepted[:, 2].sum()) if accepted is not None else None)
    if routed and (not tuned or arm >= 7):
        if tuned:
            price = q.sum(2) + 26 * ((q+63)//64).sum(2) + 2 * np.count_nonzero(q, axis=2)
        else:
            price = np.array([slot_cost(k) for k in range(int(q.max()) + 1)])[q].sum(-1)
        for key, target in (("graph", elapsed), ("routed", np.stack(routed, axis=1))):
            err = np.abs(price / price.sum(1, keepdims=True) - target / target.sum(1, keepdims=True))
            row[key + "_mae_pp"] = 100 * err.mean()
            row[key + "_max_pp"] = 100 * err.max()
    audit["counts"] = hashlib.sha256(q.tobytes()).hexdigest()
    audit["auction"] = hashlib.sha256(accepted.tobytes()).hexdigest() if accepted is not None else None
    return row, audit, greedy


def report_six(root):
    """All six measured arms consume arm 0's banks. Compare the actual native tokens separately."""
    from itertools import combinations

    valid, audits, native, failures = {}, {}, {}, {}
    sha = (root / "measurement.sha").read_text().strip()
    marker = root / "six-steps"
    steps = int(marker.read_text()) if marker.is_file() else 1000
    check(steps in (500, 1000), "six-arm window")
    kv_gib = (7 if steps == 1000 else 5) if marker.is_file() else None
    tuned = (root/"tuned-extra").is_file()
    extra = int((root/"tuned-extra").read_text()) if tuned else 8
    if tuned:
        kv_gib = int((root/"tuned-kv").read_text())
    for arm in range(9 if tuned else 7):
        try:
            folder = root / f"arm{arm}"
            provenance = (folder / "execution.sha").read_text().strip()
            imported = tuned and arm == 1 and (folder/"imported-native").is_file()
            check(provenance == (NATIVE_SHA if imported else sha), "measurement revision")
            check((folder / "complete").is_file(), "incomplete arm")
            row, audit, tokens = read_arm(root, arm, steps=steps, kv_gib=kv_gib, extra=extra, tuned=tuned)
            owner = "_ZeroShared" if arm in (0, 2, 4, 7) else "DeepseekV4MLP"
            check(audit["shared"] == [[owner] * 40] * 8, "shared ownership")
            check(all(audit["warm_native"]), "missing warm native-output hash")
            if arm not in (1, 5):
                roles = ("(4, 1, 8, 2, 3)", "(1, 4, 8, 2, 3)") if arm in (0, 2, 4) else ("(1, 2, 3)",)
                if arm in (7, 8):
                    roles = ("(4, 5, 7, 1, 8, 2, 3)",) if arm == 7 else ("(5, 7, 1, 2, 3)",)
                log = (folder / "runner.log").read_text()
                check(any(f"[six] GC M=64 roles={order}" in log for order in roles), "GC role geometry")
            if arm:
                check(0 in audits, "reference validation failed")
                for k in ("trajectory", "order", "prefill_kv", "stats", "measured", "kv"):
                    check(audit[k] == audits[0][k], f"reference mismatch in {k}")
            for peer, prior in audits.items():
                if (peer in (1, 5)) == (arm in (1, 5)):
                    check(audit["weights"] == prior["weights"], "logical weights differ within backend")
                if (peer >= 4) == (arm >= 4):
                    check(audit["placement"] == prior["placement"], "rank placement mismatch")
                    if (arm >= 7) == (peer >= 7):
                        for k in ("counts", "installed_layout", "auction"):
                            check(audit[k] == prior[k], f"policy mismatch in {k}")
            audits[arm], native[arm] = audit, tokens
            if arm:
                valid[arm] = dict(arm=arm, window_steps=steps, kv_gib_per_rank=kv_gib,
                                  step_ms=row["step_ms"], ffn_ms_per_step=row["moe_ms_per_step"],
                                  ffn_us_per_layer=25 * row["moe_ms_per_step"],
                                  **{k: row[k] for k in ("block_m", "token_ir", "total_m_blocks",
                                     "peak_rank_m_blocks", "m_block_ir", "active_physical_slots")})
                if tuned:
                    valid[arm].update(provenance=provenance, extra_per_rank=extra if arm >= 4 else 0,
                                      routed_mae_pp=row.get("routed_mae_pp"), accepted=row["accepted"])
        except (OSError, KeyError, ValueError) as exc:
            failures[arm] = str(exc)
    pairs = []
    for a, b in combinations(valid, 2):
        diff = int(np.count_nonzero(native[a] != native[b]))
        warm_equal = audits[a]["warm_native"] == audits[b]["warm_native"]
        pairs.append(dict(arm_a=a, arm_b=b, native_diff=diff, samples=native[a].size,
                          warm_native_equal=warm_equal, strict_pair_comparable=warm_equal and diff == 0,
                          step_change_pct=100 * (valid[b]["step_ms"] / valid[a]["step_ms"] - 1),
                          ffn_change_pct=100 * (valid[b]["ffn_ms_per_step"] / valid[a]["ffn_ms_per_step"] - 1)))
    for name, records in (("summary", list(valid.values())), ("pairs", pairs)):
        with (root / (name + ".csv")).open("w") as f:
            if records:
                writer = csv.DictWriter(f, fieldnames=records[0])
                writer.writeheader()
                writer.writerows(records)
    (root / "verification.json").write_text(json.dumps(dict(failures=failures, audits=audits), indent=2))
    return int(bool(failures) or len(pairs) != (28 if tuned else 15) or not all(p["strict_pair_comparable"] for p in pairs))


def report(root):
    if (root / "six-mode").is_file() and (root / "six-mode").read_text().strip() == "1":
        return report_six(root)
    rows, audits, natives, failures = {}, {}, {}, {}
    marker = root / "compact-mode"
    compact = marker.exists() and marker.read_text().strip() == "1"
    for arm in ((2, 3, 6, 1, 5) if compact else (2, 3, 4, 6, 7, 8, 1, 5)):
        try:
            rows[arm], audits[arm], natives[arm] = read_arm(root, arm, 32 if compact else 256)
            if compact:
                check(2 not in failures, "reference publisher failed validation")
                check(audits[arm]["shared"] == [["DeepseekV4MLP"] * 40] * 8, "unfused shared ownership")
                for key in ("order", "prefill_kv"):
                    check(audits[arm][key] == audits[2][key], f"publisher mismatch in {key}")
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
    if compact:
        valid.pop(2, None)  # Producer runs live routing. Only consumers enter timing comparisons.
    for arm, row in valid.items():
        row["arm"] = arm
        if not compact:
            row["vs_2_pct"] = 100 * (row["moe_ms_per_step"] / rows[2]["moe_ms_per_step"] - 1)

    if valid:
        with (root / "summary.csv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in valid.values() for k in r)))
            writer.writeheader()
            writer.writerows(valid[a] for a in sorted(valid))
    pairs = {f"{a}_to_{b}": 100 * (valid[b]["moe_ms_per_step"] / valid[a]["moe_ms_per_step"] - 1)
             for a, b in ((2, 3), (4, 6), (7, 8), (1, 3), (5, 6), (4, 7), (6, 8)) if a in valid and b in valid}
    step_pairs = {f"{a}_to_{b}": 100 * (valid[b]["step_ms"] / valid[a]["step_ms"] - 1)
                  for a, b in ((1, 3), (5, 6)) if a in valid and b in valid}
    (root / "verification.json").write_text(json.dumps(dict(failures=failures, audits=audits, pair_change_pct=pairs, step_change_pct=step_pairs), indent=2))
    return int(bool(failures))


if __name__ == "__main__":
    sys.exit(report(Path(sys.argv[1])))
