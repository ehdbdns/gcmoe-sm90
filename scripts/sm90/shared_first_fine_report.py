"""Compare the new H20 valid-row price with both measured shared-first linear policies."""
from pathlib import Path
import sys
import subprocess

from shared_first_wa1_report import (BATCHES, LAYERS, STEPS, check_archived_tables, check_boundaries, check_pair,
                                     eco_tables, merge, read_rows, slow_rank_phases, write_rows)


def report(group, wa2, wa1, bs, output=None):
    reference_times = {r["arm"]: float(r["moe_us_per_layer"]) for r in read_rows(wa2 / "main_table.csv")}
    fine_tables, fine_times = eco_tables(group, bs, 1, reference_times, fine=True, write_detail=output is None)
    dest = output or group
    dest.mkdir(parents=True, exist_ok=True)
    combined = {name: [] for name in fine_tables}
    comparisons, pairs, phases = [], [], []

    def append_tables(tables, policy, mode, slack, sha):
        for name, rows in tables.items():
            combined[name].extend(dict(row, cost_model=policy, mode=mode, slack=slack,
                                       measurement_sha=sha) for row in rows)

    for ref, weight in ((wa2, 2), (wa1, 1)):
        if ref is None:
            continue
        policy = f"linear_wa{weight}"
        check_pair(ref / "eco", group / "eco", fine=True, reference_wa=weight)
        check_boundaries(ref, group)
        tables, times = eco_tables(ref, bs, weight, reference_times, write_detail=False)
        if weight == 2:
            check_archived_tables(ref, tables)
        append_tables(tables, policy, 0, 90 + weight, (ref / "eco/execution.sha").read_text().strip())
        before, after = float(times.mean()), float(fine_times.mean())
        delta = after - before
        pct = 100 * (after / before - 1)
        diffs = fine_times - times
        comparisons.append(dict(bs=bs, reference=policy, reference_us=before, fine_us=after,
                                fine_minus_reference_us=delta, fine_change_pct=pct,
                                lower_observed_mean="tie" if delta == 0 else ("fine" if delta < 0 else policy),
                                fine_faster_layers=int((diffs.mean(0) < 0).sum()),
                                fine_faster_50step_blocks=int((diffs.reshape(STEPS//50, 50*LAYERS).mean(1) < 0).sum())))
        pairs.append(dict(bs=bs, reference=policy, treatment="fine", measured_samples=bs*STEPS,
                          native_diff=0, warm_native_equal=True, strict_pair_comparable=True))
        phases.append(slow_rank_phases(ref / "eco", policy))
    append_tables(fine_tables, "fine", 2, 11469, (group / "eco/execution.sha").read_text().strip())
    for name, rows in combined.items():
        write_rows(dest / f"{name}.csv", rows)
    write_rows(dest / "pairs.csv", pairs)
    write_rows(dest / "comparison.csv", comparisons)
    write_rows(dest / "slow_rank_phases.csv", phases + [slow_rank_phases(group / "eco", "fine")])


def finish(job, wa1_job):
    archive = Path(__file__).resolve().parents[2] / "docs/trace_executor_20261119/sm90_ep8_linear_four_6bs_results"
    missing = [bs for bs in BATCHES if not (wa1_job / f"delivery-{bs}/bs{bs}/eco/run.json").is_file()]
    if missing:
        raise SystemExit(f"Fine measurements saved. Waiting for wa=1 BS {missing}. "
                         "Join on CPU with shared_first_fine_report.py --finish FINE_JOB WA1_RESULTS.")
    final = job / "paired"
    for bs in BATCHES:
        suffix = Path(f"delivery-{bs}") / f"bs{bs}"
        group, wa2, wa1 = job / suffix, archive / suffix, wa1_job / suffix
        for directory in (group.parent, wa1.parent):
            if (directory / "SHA256SUMS").is_file():
                subprocess.run(["sha256sum", "--quiet", "-c", "SHA256SUMS"], cwd=directory, check=True)
        report(group, wa2, wa1, bs, output=final / suffix)
    merge(job, fine=True, group_root=final)


if __name__ == "__main__":
    if sys.argv[1] == "--finish":
        finish(Path(sys.argv[2]), Path(sys.argv[3]))
    elif sys.argv[1] == "--measured":
        report(Path(sys.argv[2]), Path(sys.argv[3]), None, int(sys.argv[4]))
    else:
        raise SystemExit("expected --measured GROUP WA2_GROUP BS or --finish FINE_JOB WA1_RESULTS")
