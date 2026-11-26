"""Compare H20 BS256 launch policies at identical linear wa=2 and audit replay 2."""
from pathlib import Path
import sys

import numpy as np

from shared_first_wa1_report import (BASELINE_SHA, arm_data, check_boundaries, check_pair,
                                     read_rows, require, slow_rank_phases, write_rows)


def report(group, baseline):
    timeline = read_rows(group / "perfetto/launch-order.csv")
    require(len(timeline) == 24 and {(int(r["layer"]), int(r["rank"])) for r in timeline}
            == {(layer, rank) for layer in (8, 20, 32) for rank in range(8)}, "incomplete timeline")
    lead = np.array([float(r["plan_minus_l1_start_us"]) for r in timeline])
    overlap = np.array([float(r["overlap_pct_of_shorter"]) for r in timeline])
    require(np.isfinite(lead).all() and np.all((0 <= overlap) & (overlap <= 100.000001)),
            "invalid overlap timing")
    write_rows(group / "overlap.csv", [dict(samples=len(lead), shared_first=int((lead > 0).sum()),
               overlap_positive=int((overlap > 0).sum()),
               shared_lead_0_to_10us_and_majority_overlap=int(((lead > 0) & (lead <= 10) & (overlap > 50)).sum()),
               min_lead_us=lead.min(), median_lead_us=np.median(lead), max_lead_us=lead.max(),
               min_overlap_pct=overlap.min(), median_overlap_pct=np.median(overlap))])

    sha = (group.parent / "measurement.sha").read_text().strip()
    phases = []
    for directory, arm, revision in ((baseline, "eco", BASELINE_SHA), (group, "eco", sha),
                                     (group, "eco-profile", sha)):
        arm_data(directory / arm, 256, "eco", revision, True, True, ep8_wa=2)
        if arm == "eco":
            phases.append(slow_rank_phases(directory / arm, "all_L1_CTAs" if directory == baseline
                                          else "independent_roots"))
    check_pair(baseline / "eco", group / "eco", same_wa=True, current_wa=2, change_launch=True)
    check_pair(group / "eco", group / "eco-profile", same_wa=True, current_wa=2)
    check_boundaries(baseline, group)
    write_rows(group / "slow_rank_phases.csv", phases)
    old, new = (p["full_moe_us"] for p in phases)
    write_rows(group / "comparison.csv", [dict(baseline_us=old, candidate_us=new, delta_us=new-old,
                                               change_pct=100*(new/old-1), calls=40000, native_diff=0)])


if __name__ == "__main__":
    report(Path(sys.argv[1]), Path(sys.argv[2]))
