"""Compare GLM role admission on matched ECO workloads using existing device clocks."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from eco_trace import calibration
from glm53_matrix import LAYERS
from glm53_resume import RR_ARCHIVE, file_hash

ARCHIVE = RR_ARCHIVE.parent / "formal.QLfEZmKt"


def verify_reference(root):
    receipt = json.loads((ARCHIVE / "receipt.json").read_text())
    inventory = {row["path"]: row for row in receipt["files"]}
    if (root / "revision").read_text().strip() != receipt["revision"]:
        raise ValueError("wrong reference ECO revision")
    files = ["eco/result.json"]
    for r in range(8):
        files += [f"eco/measure-rank{r}.npz", f"eco/perfetto/step0002-rank{r}.kineto.json",
                  f"eco/perfetto/step0002-layer08-rank{r}.npz"]
    for name in files:
        if file_hash(root / name) != inventory[name]["sha256"]:
            raise ValueError(f"reference differs from its published receipt: {name}")
    if file_hash(root / "placement.npy") != file_hash(ARCHIVE / "placement.npy"):
        raise ValueError("reference placement mismatch")


def phase_summary(clocks, eco):
    # [step, layer, rank]; all arithmetic before the rank reduction is on one device clock.
    dispatch, compute, routed, routed_end, shared, l1_end, end = (
        clocks[..., i] for i in (0, 1, 4, 7, 8, 9, 13))
    first = np.minimum(dispatch, compute)
    if eco:
        first = np.minimum(first, clocks[..., 15])
    span = end - first
    if (np.any(first <= 0) or np.any(shared < compute) or np.any(l1_end < shared)
            or np.any(routed < compute) or np.any(routed_end < routed) or np.any(end < routed_end)):
        raise ValueError("invalid GLM shared/routed phase clocks")
    slow = span.argmax(axis=2)[..., None]

    def slow_us(x):
        return float(np.take_along_axis(x, slow, axis=2).mean() / 1000)

    def mean_us(x):
        return float(x.mean() / 1000)

    out = dict(moe_us=slow_us(span), slow_rank_to_routed_us=slow_us(routed-first),
               slow_rank_routed_us=slow_us(routed_end-routed), slow_rank_tail_us=slow_us(end-routed_end),
               compute_minus_dispatch_us=mean_us(compute-dispatch),
               first_shared_mma_minus_first_role_us=mean_us(shared-first),
               first_shared_mma_minus_compute_us=mean_us(shared-compute),
               first_shared_mma_minus_dispatch_us=mean_us(shared-dispatch))
    if eco:
        plan, plan_end, pull_end = (clocks[..., i] for i in (15, 16, 3))
        if np.any(plan_end < plan) or np.any(pull_end < dispatch):
            raise ValueError("invalid ECO plan/dispatch spans")
        overlap = np.minimum(np.minimum(l1_end, plan_end), pull_end) - np.maximum(
            np.maximum(shared, plan), dispatch)
        out.update(first_shared_mma_before_dispatch_pct=100*float((shared < dispatch).mean()),
                   first_shared_mma_before_plan_pct=100*float((shared < plan).mean()),
                   dispatch_before_plan_pct=100*float((dispatch < plan).mean()),
                   shared_l1_dispatch_plan_overlap_us=mean_us(np.maximum(overlap, 0)))
    return out


def sample_arrival(root, clocks):
    # Replay 2 contains the preceding layer too. Anchor each rank before comparing ranks.
    folder = root / "eco/perfetto"
    paths = [folder / f"step0002-rank{r}.kineto.json" for r in range(8)]
    epoch = min(json.loads(path.read_text())["baseTimeNanoseconds"] for path in paths)
    layers = {layer: [] for layer in LAYERS}
    for r, path in enumerate(paths):
        with np.load(folder / f"step0002-layer08-rank{r}.npz") as sample:
            stamp, error = calibration(path, sample["ticks"], epoch)
            if not np.array_equal(sample["clocks"], clocks[1, 8-3, r]):
                raise ValueError("replay 2 differs from measurement archive")
        for layer, rows in layers.items():
            i = layer - 3
            prev = clocks[1, i-1, r, 13]
            c = clocks[1, i, r]
            first = min(c[0], c[1], c[15])
            if prev > first:
                raise ValueError("layer starts before its predecessor combines")
            rows.append(dict(rank=r, anchor_half_width_us=error, previous_combine_end_us=stamp(prev),
                             first_role_us=stamp(first), dispatch_entry_us=stamp(c[0]),
                             compute_entry_us=stamp(c[1]), plan_entry_us=stamp(c[15]),
                             first_shared_mma_us=stamp(c[8]), combine_end_us=stamp(c[13]),
                             previous_combine_to_first_role_us=stamp(first)-stamp(prev)))
    return layers


def compare(old, new):
    if file_hash(old / "placement.npy") != file_hash(new / "placement.npy"):
        raise ValueError("ECO placement changed")
    results = [json.loads((root / "eco/result.json").read_text()) for root in (old, new)]
    for k in ("arm", "prompt_sha256", "stats", "measure"):
        if results[0][k] != results[1][k]:
            raise ValueError(f"unpaired ECO result: {k}")
    for idx, (a, b) in enumerate(zip(results[0]["setup"], results[1]["setup"], strict=True)):
        if a["rank"] != idx or any(a[k] != b[k] for k in
                ("rank", "gpu", "slots", "block_m", "trace_level", "dispatch_sms", "compute_sms", "eco_unit")):
            raise ValueError("ECO setup differs")
        if ([v[0] for v in a["roles"]] != [4, 1, 2, 3] or [v[0] for v in b["roles"]] != [2, 1, 4, 3]
                or sorted(a["roles"]) != sorted(b["roles"])):
            raise ValueError("expected reordered graph roots on the original SM partitions")
    report = {}
    unmatched = {}
    with (new / "main.csv").open() as f:
        main = {r["arm"]: r for r in csv.DictReader(f)}
    for label, root, arm in (("native", new, "vanilla"), ("rr", new, "rr"),
                             ("eco_plan_first", old, "eco"), ("eco_shared_first", new, "eco")):
        if main[arm]["full_native_match"] != "True":
            raise ValueError(f"{arm} failed the native checks")
        ranks = []
        for r in range(8):
            with np.load(root / arm / f"measure-rank{r}.npz") as data:
                clocks = data["clocks"]
                if clocks.shape != (1000, 75, 21):
                    raise ValueError("incomplete measurement")
                if label == "eco_shared_first":
                    with np.load(old / arm / f"measure-rank{r}.npz") as ref:
                        if not np.array_equal(data["counts"], ref["counts"]):
                            unmatched.setdefault("physical_slot_counts", []).append(r)
                        if not np.array_equal(clocks[..., 17:20], ref["clocks"][..., 17:20]):
                            unmatched.setdefault("auctions_or_price", []).append(r)
                ranks.append(clocks)
        clocks = np.stack(ranks, axis=2)
        row = phase_summary(clocks, arm == "eco")
        if label != "eco_plan_first" and not np.isclose(row["moe_us"], float(main[arm]["moe_us"])):
            raise ValueError("critical rank selection disagrees with main.csv")
        if arm == "eco":
            row["replay2_by_layer"] = sample_arrival(root, clocks)
        report[label] = row
    report["unmatched_ranks"] = unmatched
    (new / "launch-comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    if unmatched:
        raise ValueError(f"ECO workloads differ: {unmatched}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("new", nargs="?", type=Path)
    args = parser.parse_args()
    verify_reference(args.reference)
    if args.new:
        compare(args.reference, args.new)
