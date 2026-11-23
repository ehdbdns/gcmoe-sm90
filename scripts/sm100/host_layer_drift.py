"""Separate inherited MoE host submission skew from new work on the launch thread."""
import argparse
from collections import Counter
from pathlib import Path
import sys

import numpy as np

from native_host_submit import partition

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "docs/sm100_megamoe/v41_l20_host_device_results"))
from analyze_l20_host_device import API_CATS, CPU_CATS, alias, device_mapping, write_csv


def read_rank(folder, arm, rank):
    m = device_mapping(folder, rank, 1)
    raw = m["raw"]
    events = [e for e in raw["traceEvents"] if e.get("ph") == "X"]
    device = [e for e in events if e.get("cat") == "kernel" and alias(e["name"]) == arm + "_moe"]
    api = {e["args"]["correlation"]: e for e in events if e.get("cat") in API_CATS
           and e["name"] == "cudaGraphLaunch" and "correlation" in e.get("args", {})}
    base_us = raw["baseTimeNanoseconds"] / 1000
    stamps = m["map"](m["clocks"][1])
    launches = []
    for entry, _, _, end, *_ in stamps:
        correlations = {e["args"].get("correlation") for e in device if entry-5 <= e["ts"] <= end+5}
        hits = [api[c] for c in correlations if c in api]
        if len(hits) != 1:
            raise ValueError(f"{arm} rank {rank} layer {len(launches)} has {len(hits)} matching launches")
        launches.append(hits[0])
    submitters = {(e["pid"], e["tid"]) for e in launches}
    if len(submitters) != 1 or any(a["ts"] >= b["ts"] for a, b in zip(launches, launches[1:])):
        raise ValueError("ambiguous submitting thread or graph order")
    host = [e for e in events if (e.get("pid"), e.get("tid")) in submitters and e.get("cat") in CPU_CATS]
    with np.load(folder / f"measure-rank{rank}.npz") as data:
        mono = data["host_submit_mono_ns"]
    rows, accounts = [], []
    for layer, launch in enumerate(launches):
        end = launch["ts"]
        start = launches[layer-1]["ts"] if layer else m["marks"][0]["ts"] + m["marks"][0]["dur"]/2
        costs = Counter()
        for _, duration, event, parent in partition(host, start, end):
            costs[event.get("cat", "unobserved"), event.get("name", "unobserved"), parent] += duration
        if abs(sum(costs.values()) - (end-start)) > 0.1:
            raise ValueError("host window does not close")
        accounts.append(costs)
        rows.append(dict(arm=arm, rank=rank, layer=layer, host_api_us=end+base_us,
                         device_entry_us=stamps[layer, 0]+base_us, host_window_us=end-start,
                         launch_to_entry_us=stamps[layer, 0]-end, correlation=launch["args"]["correlation"]))
    return rows, accounts, mono


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    ranks, windows, costs, controls = [], [], [], []
    for arm in ("gc", "mega"):
        data = [read_rank(args.raw / arm, arm, rank) for rank in range(8)]
        for rows, _, _ in data:
            ranks.extend(rows)
        # Keep the same pair across layers, selected by measured L20 device entry.
        order = sorted(range(8), key=lambda r: data[r][0][20]["device_entry_us"])
        early, late = order[0], order[-1]
        for layer in range(40):
            a, b = data[early][0][layer], data[late][0][layer]
            key = dict(arm=arm, early=early, late=late, layer=layer)
            delta = b["host_window_us"]-a["host_window_us"]
            windows.append(dict(**key, launch_skew_us=b["host_api_us"]-a["host_api_us"],
                                new_window_us=delta, device_skew_us=b["device_entry_us"]-a["device_entry_us"],
                                early_queue_us=a["launch_to_entry_us"], late_queue_us=b["launch_to_entry_us"]))
            counter = data[late][1][layer].copy()
            counter.subtract(data[early][1][layer])
            for (cat, name, parent), value in counter.items():
                costs.append(dict(**key, category=cat, name=name, parent=parent, delta_us=value))
            if abs(sum(counter.values())-delta) > 0.1:
                raise ValueError("signed contributions do not close")
        for step in (1, 4, 6):
            for rank, (_, _, mono) in enumerate(data):
                for layer, stamp in enumerate(mono[step]):
                    controls.append(dict(arm=arm, rank=rank, step=step, layer=layer, submit_mono_ns=int(stamp)))
    for name, records in (("ranks", ranks), ("windows", windows), ("exclusive", costs), ("mono", controls)):
        write_csv(args.out / (name + ".csv"), records)


if __name__ == "__main__":
    main()
