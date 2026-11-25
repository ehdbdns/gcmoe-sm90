"""Export one measured full-model layer at a time, with explicit clock provenance."""
import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np
from se_perfetto import check, decode

LAYERS = (8, 20, 32)
# Lane IDs and labels from the EP8 Word report's eco.replay2.perfetto.json.
LANES = {"combine": (3, "combine"), "shared expert L1": (4, "shared expert L1"),
         "ECO plan": (5, "ECO plan"), "shared expert L2": (7, "shared expert L2"),
         "count exchange": (8, "count exchange + pull token"), "pull token": (8, "count exchange + pull token"),
         "wait token": (11, "wait token"), "routed GEMM": (12, "routed GEMM")}


def overlap(a, b):
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def export(group, dest, profile, arm_name="eco"):
    dest.mkdir(exist_ok=True, parents=True)
    arm = group / ("eco-profile" if profile else arm_name)
    run = json.loads((arm / "run.json").read_text())
    wa = int(run["controls"]["GC_MOE_ECO_WA"])
    check(run["batch"] == 256 and run["cost"] == 0 and run["layers"] == 40, "expected linear BS256 ECO")
    events = {layer: [] for layer in LAYERS}
    bases = {}
    audit = []
    step = 2  # historical device-clock-only exports
    if profile:
        raw = group / "profile-raw"
        capture = json.loads((raw / "capture.json").read_text())
        workers = {r["rank"]: r for r in capture["ranks"]}
        step = capture["step"]
        check(wa == 1 and step in (1, 2) and sorted(workers) == list(range(8)), "incorrect capture")
    for rank in range(8):
        with np.load(arm / f"rank{rank}.npz") as arrays:
            sl = slice(step * 40, (step + 1) * 40)
            epochs, clocks, counts = (arrays[k][sl] for k in ("epochs", "clocks", "counts"))
        check(np.array_equal(epochs, clocks[:, 4]) and np.all(np.diff(epochs) == 1), "stale clocks")
        check(epochs[0] == run["measured"]["initial"][0]["epoch"] + step * 40, "wrong step")
        if profile:
            source = raw / f"rank{rank}.json"
            if source.exists():
                trace = json.loads(source.read_text())
            else:
                with gzip.open(source.with_suffix(".json.gz"), "rt") as fp:
                    trace = json.load(fp)
            check(trace["gcmoe_capture"] == {"id": capture["id"], "rank": rank}, "wrong worker trace")
            bases[rank] = int(trace["baseTimeNanoseconds"])
            worker = workers[rank]
            check(worker["next_epoch"] == epochs[-1] + 1 and capture["first_epoch"] == epochs[0] and
                  len(worker["calls"]) == 40, "step or layer mismatch")
            calls = []
            for row, clock, count in zip(worker["calls"], clocks, counts):
                check(row["epoch"] == clock[4] and np.array_equal(row["owned"], count), "wrong physical rows")
                calls.append(dict(epoch=row["epoch"], stamps=[s % 2**32 for s in row["stamps"]],
                                  clocks=clock.tolist(), rank_tokens=int(count.sum()),
                                  m_blocks=int(((count + 63) // 64).sum())))
            spans = list(decode(trace, dict(calls=calls, roles=worker["roles"]), True, tokens=256))
            check(len(spans) == 40, "incomplete role coverage")
        for layer, track in events.items():
            g, first, last, end, epoch, score = map(int, clocks[layer])
            count = counts[layer]
            n, m, a = int(count.sum()), int(((count + 63) // 64).sum()), int(np.count_nonzero(count))
            check(g <= first < last <= end and score == n + 26*m + wa*a, "invalid CTA timing or score")
            meta = dict(bs=256, step_zero_based=step, layer=layer, epoch=epoch, physical_token_rows=n,
                        m_blocks=m, active_slots=a, wa=wa, wb=26, price=score)
            if profile:
                segments, stamps = spans[layer]
                selected = {name: segments[name] for name in LANES}
                meta.update(sample=step, rank_tokens=n, first_cta_raw_us=stamps["first_cta_raw_us"], last_cta_raw_us=stamps["last_cta_raw_us"])
                s0, s1 = segments["shared expert L1"]
                p0, p1 = segments["ECO plan"]
                d0 = segments["dispatch"][0]
                audit.append(dict(rank=rank, layer=layer, epoch=epoch, plan_minus_l1_start_us=p0-s0,
                                  dispatch_minus_l1_start_us=d0-s0, plan_minus_l1_end_us=p0-s1,
                                  first_routed_cta_minus_l1_end_us=stamps["first_cta_raw_us"]-s1,
                                  plan_l1_overlap_us=overlap((s0, s1), (p0, p1)),
                                  shared_l2_routed_overlap_us=overlap(segments["shared expert L2"], segments["routed GEMM"]),
                                  pull_routed_overlap_us=overlap(segments["pull token"], segments["routed GEMM"]),
                                  routed_kernel_minus_l1_end_us=segments["routed kernel"][0]-s1))
            else:
                selected = {"MoE (graph entry to combine end)": (0, (end-g)/1000),
                            "routed GEMM": ((first-g)/1000, (last-g)/1000)}
            track.append(dict(ph="M", name="process_name", pid=rank, tid=0, args={"name": f"rank {rank}"}))
            lanes = {}
            for name, (start, stop) in selected.items():
                tid, lane = LANES[name] if profile else (len(lanes) + 1, "routed" if name == "routed GEMM" else name)
                if lane not in lanes:
                    lanes[lane] = tid
                    track.append(dict(ph="M", name="thread_name", pid=rank, tid=lanes[lane], args={"name": lane}))
                track.append(dict(ph="X", cat="SM90 MoE", name=name, pid=rank, tid=lanes[lane],
                                  ts=start, dur=stop-start, args=meta))
    phase_rows = []
    for layer, track in events.items():
        timed = [e for e in track if e["ph"] == "X"]
        if profile:
            for e in timed:
                e["ts"] += (bases[e["pid"]] - min(bases.values())) / 1000
            origin = min(e["ts"] for e in timed)
            for e in timed:
                e["ts"] -= origin
                phase_rows.append(dict(layer=layer, rank=e["pid"], epoch=e["args"]["epoch"],
                                       phase=e["name"], start_us=e["ts"], end_us=e["ts"]+e["dur"],
                                       duration_us=e["dur"], physical_token_rows=e["args"]["physical_token_rows"],
                                       m_blocks=e["args"]["m_blocks"], active_slots=e["args"]["active_slots"]))
        anchors = [e for e in timed if e["name"] == "routed GEMM"]
        check(len(anchors) == 8 and sum(e["args"]["physical_token_rows"] for e in anchors) == 256*6*8,
              "missing rank or routed rows")
        info = dict(alignment="Kineto baseTimeNanoseconds" if profile else "each rank relative to its own graph entry",
                    source="torch.profiler + device clocks" if profile else "full-model device clocks only",
                    measurement_sha=(arm / "execution.sha").read_text().strip())
        (dest / f"bs256-eco-wa{wa}-step{step:03}-layer{layer:02}.perfetto.json").write_text(
            json.dumps(dict(traceEvents=track, displayTimeUnit="us", otherData=info), ensure_ascii=False) + "\n")
    if profile:
        with (dest / "phase-times.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=phase_rows[0])
            writer.writeheader()
            writer.writerows(phase_rows)
        with (dest / "launch-order.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=audit[0])
            writer.writeheader()
            writer.writerows(audit)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("group", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--arm", choices=("eco", "eco_linear_wa1"), default="eco")
    parser.add_argument("--require-replay2", action="store_true")
    opt = parser.parse_args()
    if opt.require_replay2:
        check(opt.profile and json.loads((opt.group / "profile-raw/capture.json").read_text())["step"] == 1,
              "expected second post-placement measured replay")
    export(opt.group, opt.output or opt.group / "perfetto", opt.profile, opt.arm)
