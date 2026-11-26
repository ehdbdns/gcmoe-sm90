"""Audit the wa=1 ECO run against the same shared-first wa=2 cohort."""
import csv
import json
from pathlib import Path
import sys

import numpy as np

from compare_ep8_linear import workload
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "trace_layer"))
from main20_report import (LAYERS, RANKS, STEPS, arm_data, auction_metrics, linear_peak_metrics,
                           metrics, require, write_auction_detail, write_rows)

BATCHES = (32, 64, 128, 256, 512, 1024)
BASELINE_SHA = "2f70314e2d1948db7e45c0066c1e9099cc3021a7"


def check_pair(reference, current, same_wa=False, *, fine=False, reference_wa=2, current_wa=1, change_launch=False):
    old, new = [json.loads((d / "run.json").read_text()) for d in (reference, current)]
    require(workload(old) == workload(new), "different routes, feedback, KV, placement or native warm outputs")
    paths = {"GC_MOE_FIXED_DIR", "GC_MOE_STRICT_STATE_DIR", "GC_MOE_EXEC_ROOT", "GC_MOE_SRC", "GC_MOE_CHECKPOINT_ROOT"}
    cost_keys = {"GC_MOE_ECO_WA", "GC_MOE_ECO_COSTFN"}
    for k in set(old["controls"]) | set(new["controls"]):
        if k not in paths and k not in cost_keys:
            require(old["controls"].get(k) == new["controls"].get(k), f"different runtime control: {k}")
    require((old["controls"]["GC_MOE_ECO_WA"], new["controls"]["GC_MOE_ECO_WA"])
            == ((str(current_wa) if same_wa else str(reference_wa)), str(current_wa)), "wrong linear weight")
    for run, mode in ((old, 0), (new, 2 if fine else 0)):
        require(run["cost"] == mode and run["controls"]["GC_MOE_ECO_COSTFN"] == str(mode), "wrong cost mode")
    marker = "shared_launch=((4, 2, 0, 1),) independent_plan=True"
    graph_modes = [marker in (d / "runner.log").read_text() for d in (reference, current)]
    require(graph_modes == [False, True] if change_launch else graph_modes[0] == graph_modes[1],
            "unexpected ECO launch policy difference")
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
    # A rank may finish last in combine even when it received no routed rows.
    active = bounds[:, 1] != -1
    require(np.all(bounds[~active, 2] == 0) and np.all(np.diff(bounds[active], axis=1) >= 0),
            "invalid slow-rank CTA boundaries")
    before, routed, after = np.diff(bounds[active], axis=1).mean(0) / 1000 if active.any() else (None,) * 3
    return dict(arm=label, calls_with_routed_work=int(active.sum()), graph_to_first_cta_us=before,
                routed_us=routed, last_cta_to_combine_end_us=after,
                full_moe_us=float((bounds[:, 3] - bounds[:, 0]).mean() / 1000))


def eco_tables(group, bs, wa, reference_times, *, fine=False, write_detail=True):
    arm = group / "eco"
    sha = (group.parent / "measurement.sha").read_text().strip()
    if wa == 2:
        require(sha == BASELINE_SHA, "unexpected baseline measurement SHA")
    _, _, _, _, counts, v, auction, _ = arm_data(arm, bs, "eco", sha, True, True, ep8_wa=wa, ep8_mode=2 if fine else 0)
    graph, routed, price = v.transpose(2, 0, 1)
    require(np.all(routed.sum(1) > 0), "empty routed span")
    graph_share, routed_share, predicted = (100 * arr / arr.sum(1, keepdims=True)
                                            for arr in (graph, routed, price))
    errors = np.abs(predicted - routed_share)
    timing = graph.max(1).reshape(STEPS, LAYERS)
    us = float(timing.mean())
    key = dict(bs=bs, arm="eco", wa=None if fine else wa)
    summary = metrics(counts, v)
    summary.pop("moe_ms_per_step")
    main = dict(**key, moe_us_per_layer=us, moe_change_pct_vs_vanilla=100*(us/reference_times["vanilla"]-1),
                moe_change_pct_vs_rr=100*(us/reference_times["rr"]-1), **summary,
                routed_mae_pp=float(errors.mean()), auction_accepted_mean=float(auction[:, 2].mean()),
                **linear_peak_metrics(counts, v, price))
    layers = [dict(**key, layer=i, mean_slowest_rank_us=float(timing[:, i].mean()),
                   routed_mae_pp=float(errors[i::LAYERS].mean())) for i in range(LAYERS)]
    ranks = [dict(**key, rank=r, predicted_pct=float(predicted[:, r].mean()),
                  routed_pct=float(routed_share[:, r].mean()), routed_mae_pp=float(errors[:, r].mean()),
                  graph_pct=float(graph_share[:, r].mean())) for r in range(RANKS)]
    if wa == 1 and write_detail:
        write_auction_detail(arm, bs, "eco", auction)
    return dict(main_table=[main], layers=layers, rank_shares=ranks,
                auction=[dict(**key, **auction_metrics(auction))]), timing


def read_rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def check_archived_tables(ref, tables):
    for name, rows in tables.items():
        directory = ref / "eco" if name == "rank_shares" else ref
        historical = [r for r in read_rows(directory / f"{name}.csv") if r["arm"] == "eco"]
        require(len(historical) == len(rows), f"archived {name} coverage differs")
        for old, row in zip(historical, rows):
            for k, value in old.items():
                if k != "arm":
                    require(np.isclose(row[k], float(value), rtol=1e-10, atol=1e-10),
                            f"archived {name} differs for {k}: {value} vs {row[k]}")


def check_boundaries(ref, group):
    for r in range(RANKS):
        for folder in ("checkpoint", "checkpoint/prefill"):
            path = Path(folder) / f"rank{r}.json"
            require(json.loads((ref / path).read_text()) == json.loads((group / path).read_text()),
                    f"changed request/KV boundary manifest: {path}")


def report(group, ref, bs):
    eco = group / "eco"
    check_pair(ref / "eco", eco)
    check_boundaries(ref, group)
    if bs == 256:
        check_pair(eco, group / "eco-profile", same_wa=True)

    reference_times = {r["arm"]: float(r["moe_us_per_layer"]) for r in read_rows(ref / "main_table.csv")}
    old, old_times = eco_tables(ref, bs, 2, reference_times)
    check_archived_tables(ref, old)
    new, new_times = eco_tables(group, bs, 1, reference_times)
    for name in old:
        write_rows(group / f"{name}.csv", old[name] + new[name])
    before = old["main_table"][0]["moe_us_per_layer"]
    after = new["main_table"][0]["moe_us_per_layer"]
    delta = after - before
    pct = 100 * (after / before - 1)
    # Use matching decode ordinals. Each block includes all 40 layers of 50 consecutive steps.
    block_deltas = (new_times - old_times).reshape(20, -1).mean(1)
    write_rows(group / "comparison.csv", [dict(bs=bs, wa2_us=before, wa1_us=after, wa1_minus_wa2_us=delta,
               change_pct_vs_wa2=pct, faster_observed_wa="tie" if delta == 0 else (1 if delta < 0 else 2),
               wa1_faster_layers=int((new_times.mean(0) < old_times.mean(0)).sum()),
               wa1_faster_50step_blocks=int((block_deltas < 0).sum()))])
    write_rows(group / "pairs.csv", [dict(bs=bs, arm_a="eco_wa2", arm_b="eco_wa1", native_diff=0,
               measured_samples=bs*STEPS, warm_native_equal=True, strict_pair_comparable=True,
               moe_change_pct=pct, moe_delta_us_per_layer=delta)])
    primary = new["main_table"][0]
    # Keep the single-arm summary used by the original three-batch handoff.
    write_rows(group / "wa1.csv", [dict(primary, wb=26, slack=91, wa2_us_per_layer=before,
               change_pct_vs_wa2=pct, change_pct_vs_archived_rr=primary["moe_change_pct_vs_rr"],
               change_pct_vs_archived_vanilla=primary["moe_change_pct_vs_vanilla"], **new["auction"][0])])
    phases = [slow_rank_phases(ref / "eco", "shared_first_wa2"), slow_rank_phases(eco, "shared_first_wa1")]
    if bs == 256:
        previous = ref.parents[2] / "sm90_bs256_linear_wa_pair_results/delivery/bs256/eco_linear_wa1"
        phases.insert(0, slow_rank_phases(previous, "previous_launch_wa1"))
    write_rows(group / "slow_rank_phases.csv", phases)


def merge(job, *, fine=False, group_root=None):
    policies = ("linear_wa2", "linear_wa1", "fine") if fine else ("2", "1")
    for name, per_policy in (("main_table", 1), ("layers", LAYERS), ("rank_shares", RANKS),
                             ("auction", 1), ("comparison", 0), ("pairs", 0)):
        merged = []
        for bs in BATCHES:
            directory = (group_root or job) / f"delivery-{bs}" / f"bs{bs}"
            rows = read_rows(directory / f"{name}.csv")
            require(all(int(r["bs"]) == bs for r in rows), f"wrong BS in {name}")
            if per_policy:
                field = "cost_model" if fine else "wa"
                require(len(rows) == len(policies)*per_policy and
                        all(sum(r[field] == policy for r in rows) == per_policy for policy in policies),
                        f"BS{bs}: incomplete {name}")
            elif fine:
                require(sorted(r["reference"] for r in rows) == ["linear_wa1", "linear_wa2"],
                        f"BS{bs}: missing cost comparison in {name}")
            else:
                require(len(rows) == 1, f"BS{bs}: incomplete {name}")
            merged.extend(rows)
        write_rows(job / f"{name}.csv", merged)


if __name__ == "__main__":
    if sys.argv[1] == "--merge":
        merge(Path(sys.argv[2]))
    else:
        report(Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]))
