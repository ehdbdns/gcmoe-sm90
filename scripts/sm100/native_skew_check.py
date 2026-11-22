"""Compare step 2 across acquisitions using the same arm, layer and clock boundaries."""
import csv
import gzip
import json
import sys
from pathlib import Path

import numpy as np

from native_phase_perfetto import LAYERS, require


def old_submissions(path):
    launches = {}
    with gzip.open(path, "rt") as f:
        for row in csv.DictReader(f):
            key = row["arm"], int(row["layer"]), int(row["rank"])
            value = int(row["launch_host_ns"])
            if launches.setdefault(key, value) != value:
                raise ValueError(f"inconsistent archived host launch for {key}")
    return launches


def read_layer(folder, arm, layer, step=2):
    stem = folder / arm / "perfetto" / f"step{step:03}_layer{layer:02}"
    meta = json.loads(stem.with_suffix(".counts.json").read_text())
    ranks = sorted(meta["ranks"], key=lambda row: row["rank"])
    events = json.loads(stem.with_suffix(".perfetto.json").read_text())["traceEvents"]
    moe = sorted((e for e in events if e["ph"] == "X" and e["name"] == "MoE"), key=lambda e: e["pid"])
    require([r["rank"] for r in ranks] == [e["pid"] for e in moe] == list(range(8)), "missing rank")
    require((meta["arm"], meta["step"], meta["layer"]) == (arm, step, layer), "wrong trace")
    entry = np.array([e["ts"] for e in moe])
    early, late = int(entry.argmin()), int(entry.argmax())
    clocks = np.array([r["device_clocks_ns"] for r in ranks])
    exchange = (clocks[:, 5] - clocks[:, 4]) / 1000
    duration = (clocks[:, 3] - clocks[:, 0]) / 1000
    require(np.all(duration > 0) and np.all(exchange >= 0), "invalid phase boundaries")
    return dict(early_rank=early, late_rank=late, entry_spread_us=np.ptp(entry),
                early_exchange_us=exchange[early], late_exchange_us=exchange[late],
                max_exchange_us=exchange.max(), global_moe_us=max(entry + duration) - min(entry),
                max_alignment_half_width_us=max(r["alignment_half_width_us"] for r in ranks)), ranks


def dense_gaps(folder, arm, rank):
    # Step 1 is the live CUDA-only capture. Compare adjacent launches on the
    # same CUDA stream inside the inter-layer window preceding L20.
    raw = json.loads((folder / f"profile-rank{rank}.json").read_text())
    kernels = sorted((e for e in raw["traceEvents"] if e.get("cat") == "kernel"
                      and e.get("ph") == "X"), key=lambda e: e["ts"])
    marks = [e["ts"] + e["dur"] / 2 for e in kernels if "sm100_phase_clock_anchor" in e["name"]]
    require(len(marks) == 2, "missing CUDA sample anchors")
    a, b = np.load(folder / f"anchor-rank{rank}.npy")
    with np.load(folder / f"measure-rank{rank}.npz") as data:
        clocks = data["clocks"][1]
    lo, hi = marks[0] + (np.array([clocks[19, 3], clocks[20, 0]]) - a) * ((marks[1] - marks[0]) / (b - a))
    previous = {}
    rows = []
    for e in kernels:
        if not lo <= e["ts"] < hi:
            continue
        stream = e["args"]["stream"]
        prev = previous.get(stream)
        if prev and "mxfp8_quantize" in prev["name"].lower() and "dense_blockscaled_gemm" in e["name"].lower():
            rows.append(dict(arm=arm, step=1, layer=20, rank=rank, stream=stream,
                             quant=prev["name"], gemm=e["name"],
                             gap_us=e["ts"] - (prev["ts"] + prev["dur"])))
        previous[stream] = e
    return rows


def compare(root, docs):
    archive = docs / "v41_native_phase_results"
    launches = old_submissions(docs / "v41_host_submit_results/host_segments.csv.gz")
    rows, within, gaps = [], [], []
    for arm in ("mega", "gc"):
        old = json.loads((archive / arm / "result.json").read_text())
        new = json.loads((root / arm / "result.json").read_text())
        require(new["phase_acquisition"] == "step1_cuda_steps2_to_6_anchors_only", "wrong acquisition")
        require(all(new[k] == old[k] for k in ("arm", "bs", "prompt_sha256", "workload")), "workload mismatch")
        for key in ("block_m", "validated_layer_graphs", "sm_split"):
            require([r[key] for r in old["measure"]["ranks"]] ==
                    [r[key] for r in new["measure"]["ranks"]], f"graph geometry mismatch: {key}")
        controls = new["dense_control"]
        require([r["mode"] for r in controls] == ["off", "on", "on", "off", "on", "off", "on"],
                "unexpected dense replay schedule")
        replays = [r["ranks"] for r in controls[1:]] + [new["dense_final"]]
        for control, counts in zip(controls, replays, strict=True):
            require(len(counts) == 8 and all(r["graphs"] == r["validated"] > 0 and
                    ((r["previous_replays"] > 0) == (control["mode"] == "on")) for r in counts),
                    f"missing validated dense graph coverage at step {control['step']}")
        replay = []
        for rank in range(8):
            with np.load(root / arm / f"measure-rank{rank}.npz") as data:
                stamps = data["host_submit_mono_ns"]
                require(stamps.shape == (7, 40) and np.all(np.diff(stamps.ravel()) > 0),
                        "invalid host replay timestamps")
                replay.append(stamps)
            gaps.extend(dense_gaps(root / arm, arm, rank))
        for layer in LAYERS:
            baseline, before = read_layer(archive, arm, layer)
            quiet, after = read_layer(root, arm, layer)
            changed = sum(x != y for a, b in zip(before, after, strict=True)
                          for x, y in zip(a["slot_rows"], b["slot_rows"], strict=True))
            host_old = [launches[arm, layer, r] for r in range(8)]
            host_new = [times[2, layer] for times in replay]
            row = dict(arm=arm, layer=layer, changed_slot_counts=changed,
                       archived_host_launch_spread_us=np.ptp(host_old) / 1000,
                       clean_host_replay_call_spread_us=np.ptp(host_new) / 1000)
            for name, values in (("archived", baseline), ("clean", quiet)):
                row.update({f"{name}_{k}": v for k, v in values.items()})
            for k in ("entry_spread_us", "max_exchange_us", "global_moe_us"):
                row[f"clean_minus_archived_{k}"] = quiet[k] - baseline[k]
            rows.append(row)
            for step in range(2, 7):
                measurement, slots = read_layer(root, arm, layer, step)
                within.append(dict(arm=arm, layer=layer, step=step, dense=controls[step]["mode"],
                    host_replay_spread_us=np.ptp([t[step, layer] for t in replay]) / 1000,
                    max_rank_tokens=max(r["tokens"] for r in slots),
                    max_rank_m_blocks=max(r["m_blocks"] for r in slots), **measurement))
    with (root / "step001_dense_gaps.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=("arm", "step", "layer", "rank", "stream", "quant", "gemm", "gap_us"))
        writer.writeheader()
        writer.writerows(gaps)
    with (root / "within_run_skew.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=within[0])
        writer.writeheader()
        writer.writerows(within)
    with (root / "step002_skew.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    compare(*map(Path, sys.argv[1:]))
