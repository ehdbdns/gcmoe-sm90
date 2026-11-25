"""Audit the wa=1 ECO run against the same shared-first wa=2 cohort."""
import csv
import json
from pathlib import Path
import sys

import numpy as np

from compare_ep8_linear import workload
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "trace_layer"))
from main20_report import arm_data, auction_metrics, metrics, require, write_rows


def check_pair(reference, current, same_wa=False):
    old, new = [json.loads((d / "run.json").read_text()) for d in (reference, current)]
    require(workload(old) == workload(new), "different routes, feedback, KV, placement or native warm outputs")
    paths = {"GC_MOE_FIXED_DIR", "GC_MOE_STRICT_STATE_DIR", "GC_MOE_EXEC_ROOT", "GC_MOE_SRC", "GC_MOE_CHECKPOINT_ROOT"}
    for k in set(old["controls"]) | set(new["controls"]):
        if k not in paths and k != "GC_MOE_ECO_WA":
            require(old["controls"].get(k) == new["controls"].get(k), f"different runtime control: {k}")
    require((old["controls"]["GC_MOE_ECO_WA"], new["controls"]["GC_MOE_ECO_WA"])
            == (("1" if same_wa else "2"), "1"), "wrong treatment")
    for name in ("runtime.log", "hardware.csv", "installed-layout.json"):
        require((reference / name).read_bytes() == (current / name).read_bytes(), f"different {name}")
    for rank in range(8):
        with np.load(reference / f"rank{rank}.npz") as a, np.load(current / f"rank{rank}.npz") as b:
            for k in (("epochs", "native", "counts", "auction") if same_wa else ("epochs", "native")):
                np.testing.assert_array_equal(a[k], b[k], err_msg=f"rank {rank}: {k}")


def slow_rank_phases(arm, label):
    clocks = []
    for rank in range(8):
        with np.load(arm / f"rank{rank}.npz") as data:
            clocks.append(data["clocks"][:, :4])
    clocks = np.stack(clocks, axis=1)
    slowest = (clocks[:, :, 3] - clocks[:, :, 0]).argmax(1)
    bounds = clocks[np.arange(len(clocks)), slowest]
    require(np.all(np.diff(bounds, axis=1) >= 0), "invalid slow-rank CTA boundaries")
    before, routed, after = np.diff(bounds, axis=1).mean(0) / 1000
    return dict(arm=label, graph_to_first_cta_us=before, routed_us=routed,
                last_cta_to_combine_end_us=after, full_moe_us=before+routed+after)


def report(group, ref, bs):
    eco = group / "eco"
    sha = (group.parent / "measurement.sha").read_text().strip()
    check_pair(ref / "eco", eco)
    _, _, _, _, counts, v, auction, _ = arm_data(eco, bs, "eco", sha, True, True, ep8_wa=1)
    with (ref / "main_table.csv").open() as source:
        previous = {r["arm"]: float(r["moe_us_per_layer"]) for r in csv.DictReader(source)}
    graph, routed, price = v.transpose(2, 0, 1)
    timing = graph.max(1)
    us = float(timing.mean())
    row = metrics(counts, v)
    row.pop("moe_ms_per_step")
    row.update(bs=bs, wa=1, wb=26, slack=91, moe_us_per_layer=us, wa2_us_per_layer=previous["eco"],
               change_pct_vs_wa2=100 * (us / previous["eco"] - 1),
               change_pct_vs_archived_rr=100 * (us / previous["rr"] - 1),
               change_pct_vs_archived_vanilla=100 * (us / previous["vanilla"] - 1),
               routed_mae_pp=float(np.abs(price / price.sum(1, keepdims=True) -
                                          routed / routed.sum(1, keepdims=True)).mean() * 100),
               **auction_metrics(auction))
    write_rows(group / "wa1.csv", [row])
    write_rows(group / "layers.csv", [dict(bs=bs, layer=i, moe_us=float(x.mean()))
                                       for i, x in enumerate(timing.reshape(1000, 40).T)])
    phases = [slow_rank_phases(ref / "eco", "shared_first_wa2"), slow_rank_phases(eco, "shared_first_wa1")]
    if bs == 256:
        old = ref.parents[2] / "sm90_bs256_linear_wa_pair_results/delivery/bs256/eco_linear_wa1"
        phases.insert(0, slow_rank_phases(old, "previous_launch_wa1"))
    write_rows(group / "slow_rank_phases.csv", phases)
    if bs == 256:
        check_pair(eco, group / "eco-profile", same_wa=True)


if __name__ == "__main__":
    report(Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]))
