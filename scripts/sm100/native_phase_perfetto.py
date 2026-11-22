"""Join per-rank Kineto kernels to the corresponding step-2 device milestones."""
import json
from pathlib import Path
import sys

import numpy as np

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


def rank_trace(folder, rank, epoch_ns):
    raw = json.loads((folder / f"profile-rank{rank}.json").read_text())
    base = int(raw["baseTimeNanoseconds"])
    kernels = [e for e in raw["traceEvents"] if e.get("cat") == "kernel" and e.get("ph") == "X"]
    marks = sorted((e for e in kernels if "sm100_phase_clock_anchor" in e["name"]), key=lambda e: e["ts"])
    device = np.load(folder / f"anchor-rank{rank}.npy").tolist()
    require(len(marks) == len(device) == 2 and device[1] > device[0], "missing Kineto calibration")
    kineto = [(base - epoch_ns) / 1000 + e["ts"] + e["dur"] / 2 for e in marks]
    drift = (kineto[1] - kineto[0]) - (device[1] - device[0]) / 1000
    uncertainty = max(e["dur"] for e in marks) / 2
    require(abs(drift) <= 2 * uncertainty + 5, f"rank {rank}: unstable clock alignment {drift} us")

    def aligned(t):
        return kineto[0] + (t - device[0]) / 1000 + drift * (t - device[0]) / (device[1] - device[0])

    with np.load(folder / f"measure-rank{rank}.npz") as f:
        stamps, counts = f["clocks"][2], f["counts"][2]
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
                               args=dict(step=2, layer=layer, rank=rank)))
        raw_count = 0
        for e in kernels:
            ts = (base - epoch_ns) / 1000 + e["ts"]
            if ts <= aligned(end) and ts + e["dur"] >= aligned(start):
                raw_count += 1
                events.append(dict(ph="X", pid=rank, tid=1000 + int(e["args"]["stream"]),
                                   name=e["name"], cat="Kineto kernel", ts=ts, dur=e["dur"]))
        require(raw_count, f"rank {rank} layer {layer}: missing Kineto kernel")
        yield layer, events, dict(rank=rank, slot_rows=counts[layer].tolist(),
                                  tokens=int(counts[layer].sum()), device_clocks_ns=clocks,
                                  alignment_half_width_us=uncertainty, alignment_drift_us=drift)


def build(root):
    for arm in ("mega", "gc"):
        folder = root / arm
        info = json.loads((folder / "result.json").read_text())
        require(info["arm"] == arm and info["bs"] == 512, "wrong run")
        bases = [int(json.loads((folder / f"profile-rank{r}.json").read_text())["baseTimeNanoseconds"])
                 for r in range(8)]
        base = min(bases)
        samples = {l: ([], []) for l in LAYERS}
        for r in range(8):
            bm = info["measure"]["ranks"][r]["block_m"]
            for layer, events, row in rank_trace(folder, r, base):
                row["block_m"] = bm
                row["m_blocks"] = sum((x + bm - 1) // bm for x in row["slot_rows"])
                samples[layer][0].extend(events)
                samples[layer][1].append(row)
        dest = folder / "perfetto"
        dest.mkdir(exist_ok=True)
        for layer, (events, rows) in samples.items():
            require(sum(r["tokens"] for r in rows) == 512 * 6, "incomplete routed rows")
            prefix = dest / f"step002_layer{layer:02}"
            for r in range(8):
                events.append(dict(ph="M", pid=r, tid=0, name="process_name", args={"name": f"{arm} rank {r}"}))
            prefix.with_suffix(".perfetto.json").write_text(json.dumps(dict(traceEvents=events, displayTimeUnit="ns")))
            prefix.with_suffix(".counts.json").write_text(json.dumps(dict(step=2, layer=layer, arm=arm,
                baseTimeNanoseconds=base, phases=PHASES, ranks=rows), indent=2))


if __name__ == "__main__":
    build(Path(sys.argv[1]))
