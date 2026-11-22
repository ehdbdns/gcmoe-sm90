"""Decompose the existing native Mega/GC matrix clocks on the local critical rank."""
import argparse
import csv
from pathlib import Path
import sys

import numpy as np


def phases(directory, bs):
    clocks = []
    counts = []
    for rank in range(8):
        with np.load(directory / f"measure-rank{rank}.npz") as data:
            clocks.append(data["clocks"])
            counts.append(data["counts"])
    ticks = np.stack(clocks, axis=2).astype(np.int64)
    rows = np.stack(counts, axis=2)
    if ticks.shape != (1000, 40, 8, 4) or rows.shape != (1000, 40, 8, 48):
        raise ValueError(f"{directory}: unexpected archive shape")
    if np.any(rows < 0) or np.any(rows.sum((2, 3)) != bs * 6):
        raise ValueError(f"{directory}: invalid received token count")
    span = ticks[..., 3] - ticks[..., 0]
    rank = span.argmax(axis=2)
    idx = np.indices(rank.shape)
    critical = ticks[idx[0], idx[1], rank]
    if np.any(critical[..., 0] <= 0) or np.any(np.diff(critical) < 0):
        raise ValueError(f"{directory}: invalid critical-rank edges")
    values = np.diff(critical).sum(axis=1).mean(axis=0) / 1e6
    return rows, np.append(values, values.sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    writer = csv.writer(sys.stdout)
    writer.writerow(["bs", "arm", "start_to_first_routed_ms", "first_to_last_routed_ms",
                     "last_routed_to_combine_end_ms", "moe_ms"])
    for bs in (256, 512, 1024, 2048):
        ref = None
        for arm in ("mega", "gc"):
            directory = args.root / f"bs{bs}" / arm
            rows, values = phases(directory, bs)
            if arm == "mega":
                ref, baseline = rows, values
            elif not np.array_equal(ref, rows):
                raise ValueError(f"BS{bs}: native-arm physical counts differ")
            writer.writerow([bs, arm, *values])
        writer.writerow([bs, "gc_minus_mega", *(values - baseline)])


if __name__ == "__main__":
    main()
