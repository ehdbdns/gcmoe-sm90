"""Join per-rank Kineto kernels to the corresponding step-2 device milestones."""
import json
from pathlib import Path
import sys

import numpy as np

from native_phase_overview import write_overview

LAYERS = (0, 8, 20, 32, 39)
PHASES = (("MoE", 0, 3), ("entry to first routed MMA", 0, 1),
          ("dispatch: source indices and count exchange incl. barrier", 4, 5),
          ("dispatch: pull loop incl. metadata and readiness", 6, 7),
          ("shared L1", 8, 9), ("shared L2", 10, 11),
          ("routed L1", 12, 13), ("routed L2 incl. push", 14, 15),
          ("routed L2 output epilogue / push", 16, 15),
          ("combine incl. readiness wait", 18, 3))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def rank_trace(folder, rank, epoch_ns, step=2, anchors_only=False):
    kineto, device, widths, kernels = [], [], [], []
    stems = ("clock-before", "clock-after") if anchors_only else ("profile",)
    for stem in stems:
        raw = json.loads((folder / f"{stem}-rank{rank}.json").read_text())
        offset = (int(raw["baseTimeNanoseconds"]) - epoch_ns) / 1000
        spans = [e for e in raw["traceEvents"] if e.get("cat") == "kernel" and e.get("ph") == "X"]
        marks = sorted((e for e in spans if "sm100_phase_clock_anchor" in e["name"]), key=lambda e: e["ts"])
        clock_file = stem if anchors_only else "anchor"
        readings = np.load(folder / f"{clock_file}-rank{rank}.npy").tolist()
        require(len(marks) == len(readings) == (1 if anchors_only else 2), "missing Kineto calibration")
        kineto.extend(offset + e["ts"] + e["dur"] / 2 for e in marks)
        device.extend(readings)
        widths.extend(e["dur"] / 2 for e in marks)
        if not anchors_only:
            kernels = spans
    require(device[1] > device[0] and kineto[1] > kineto[0], "unordered Kineto calibration")
    drift = (kineto[1] - kineto[0]) - (device[1] - device[0]) / 1000
    uncertainty = max(widths)
    require(abs(drift) <= 2 * uncertainty + 5, f"rank {rank}: unstable clock alignment {drift} us")

    def aligned(t):
        return kineto[0] + (t - device[0]) / 1000 + drift * (t - device[0]) / (device[1] - device[0])

    with np.load(folder / f"measure-rank{rank}.npz") as f:
        stamps, counts = f["clocks"][step], f["counts"][step]
    require(stamps.shape == (40, 19) and counts.shape == (40, 48), "unexpected native trace geometry")
    require(np.all(stamps[1:, 0] >= stamps[:-1, 3]), "layer calls out of order")
    for layer in LAYERS:
        clocks = stamps[layer].tolist()
        start, end = clocks[0], clocks[3]
        require(device[0] <= start < end <= device[1], "clocks outside profiled step")
        events = []
        for lane, (name, a, b) in enumerate(PHASES, 1):
            require(start <= clocks[a] <= clocks[b] <= end, f"rank {rank}, layer {layer}: {name}")
            events.append(dict(ph="X", pid=rank, tid=lane, name=name, cat="device phase envelope",
                               ts=aligned(clocks[a]), dur=(clocks[b]-clocks[a])/1000,
                               args=dict(step=step, layer=layer, rank=rank)))
        raw_count = 0
        for e in kernels:
            ts = offset + e["ts"]
            if ts <= aligned(end) and ts + e["dur"] >= aligned(start):
                raw_count += 1
                events.append(dict(ph="X", pid=rank, tid=1000 + int(e["args"]["stream"]),
                                   name=e["name"], cat="Kineto kernel", ts=ts, dur=e["dur"]))
        require(anchors_only or raw_count, f"rank {rank} layer {layer}: missing Kineto kernel")
        yield layer, events, dict(rank=rank, slot_rows=counts[layer].tolist(),
                                  tokens=int(counts[layer].sum()), device_clocks_ns=clocks,
                                  alignment_half_width_us=uncertainty, alignment_drift_us=drift)


def build(root):
    for arm in ("mega", "gc"):
        folder = root / arm
        info = json.loads((folder / "result.json").read_text())
        require(info["arm"] == arm and info["bs"] == 512, "wrong run")
        quiet = info.get("phase_acquisition") == "step1_cuda_steps2_to_6_anchors_only"
        for step in (range(1, 7) if quiet else (2,)):
            anchors_only = quiet and step >= 2
            clock_stem = "clock-before" if anchors_only else "profile"
            bases = [int(json.loads((folder / f"{clock_stem}-rank{r}.json").read_text())["baseTimeNanoseconds"])
                     for r in range(8)]
            base = min(bases)
            samples = {l: ([], []) for l in LAYERS}
            for r in range(8):
                bm = info["measure"]["ranks"][r]["block_m"]
                for layer, events, row in rank_trace(folder, r, base, step, anchors_only):
                    row["block_m"] = bm
                    row["m_blocks"] = sum((x + bm - 1) // bm for x in row["slot_rows"])
                    samples[layer][0].extend(events)
                    samples[layer][1].append(row)
            dest = folder / "perfetto"
            dest.mkdir(exist_ok=True)
            for layer, (events, rows) in samples.items():
                require(sum(r["tokens"] for r in rows) == 512 * 6, "incomplete routed rows")
                prefix = dest / f"step{step:03}_layer{layer:02}"
                for r in range(8):
                    events.append(dict(ph="M", pid=r, tid=0, name="process_name", args={"name": f"{arm} rank {r}"}))
                trace = dict(traceEvents=events, displayTimeUnit="ns")
                counts = dict(step=step, layer=layer, arm=arm, baseTimeNanoseconds=base, phases=PHASES, ranks=rows)
                counts["acquisition"] = "device milestones with external anchors" if anchors_only else "Kineto kernels and device milestones"
                prefix.with_suffix(".perfetto.json").write_text(json.dumps(trace))
                prefix.with_suffix(".counts.json").write_text(json.dumps(counts, indent=2))
                write_overview(folder, prefix.name, counts, trace)



if __name__ == "__main__":
    build(Path(sys.argv[1]))
