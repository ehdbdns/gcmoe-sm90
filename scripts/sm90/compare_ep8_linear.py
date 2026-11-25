"""Compare the three new four-arm groups with the full-model runs used in the EP8 Word table."""
import csv
import json
from pathlib import Path
import sys


def workload(run):
    """Inputs and decode trajectory; physical slot counts and the ECO price are treatments."""
    key = [run["prompt_hash"], run["output_hash"], run["prefill_steps"]]
    for phase in ("stats", "measured"):
        audits = sorted(run[phase]["audit"], key=lambda x: x["rank"])
        bounds = sorted(run[phase]["initial"], key=lambda x: x["rank"])
        if [row["rank"] for row in audits] != list(range(8)) or [row["rank"] for row in bounds] != list(range(8)):
            raise ValueError(f"incomplete {phase} rank coverage")
        for audit, boundary in zip(audits, bounds):
            key.extend(audit[k] for k in ("rank", "ids", "weights", "demand"))
            checkpoint = boundary["checkpoint"]
            if phase == "stats":
                key.append(checkpoint["prefill_snapshot"]["readback_sha256"])
            else:
                key.extend((audit["warm_native"], audit["batch_order"], checkpoint["logical_weights"],
                            checkpoint["snapshot"]["readback_sha256"], checkpoint["placement"]))
    return key


def read_times(group):
    with (group / "main_table.csv").open() as f:
        return {r["arm"]: (float(r["moe_us_per_layer"]) if "moe_us_per_layer" in r else
                           float(r["moe_ms_per_step"]) * 25) for r in csv.DictReader(f)}


def compare(job):
    archive = Path(__file__).resolve().parents[2] / "docs/trace_executor_20261119"
    wa_pair = archive / "sm90_bs256_linear_wa_pair_results/delivery/bs256"
    with (wa_pair / "main_table.csv").open() as f:
        wa1_us = next(float(row["moe_us_per_layer"]) for row in csv.DictReader(f) if int(row["wa"]) == 1)
    wa1_workload = workload(json.loads((wa_pair / "eco_linear_wa1/run.json").read_text()))
    rows = []
    for bs in (256, 512, 1024):
        old = (archive / "sm90_bs256_main_five_plan_comm_results/delivery/bs256" if bs == 256 else
               archive / f"sm90_bs512_bs1024_full_model_four_results/delivery-{bs}/bs{bs}")
        new = job / f"delivery-{bs}/bs{bs}"
        previous, current = read_times(old), read_times(new)
        for arm in ("vanilla", "random", "rr", "eco"):
            before, after = (workload(json.loads((group / arm / "run.json").read_text())) for group in (old, new))
            old_pct = 100 * (previous[arm] / previous["rr"] - 1)
            new_pct = 100 * (current[arm] / current["rr"] - 1)
            compare_wa1 = bs == 256 and arm == "eco"
            rows.append(dict(bs=bs, arm=arm, historical_moe_us=previous[arm], current_moe_us=current[arm],
                             change_pct=100 * (current[arm] / previous[arm] - 1), controlled_inputs_match=before == after,
                             historical_vs_rr_pct=old_pct, current_vs_rr_pct=new_pct, vs_rr_delta_pp=new_pct-old_pct,
                             previous_wa1_us=wa1_us if compare_wa1 else None,
                             change_pct_vs_wa1=100 * (current[arm] / wa1_us - 1) if compare_wa1 else None,
                             wa1_inputs_match=(after == wa1_workload) if compare_wa1 else None))
    with (job / "historical.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    compare(Path(sys.argv[1]))
