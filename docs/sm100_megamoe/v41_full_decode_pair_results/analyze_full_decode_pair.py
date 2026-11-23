#!/usr/bin/env python3
"""Validate and tabulate the BS512 full-decode pair without changing raw data."""

import csv
import json
from pathlib import Path

import numpy as np


LAYERS = 40


def aligned_us(row, timestamp_ns):
    host0, host1 = row["alignment_anchor_host_ns"]
    device0, device1 = row["alignment_anchor_device_ns"]
    drift = (host1 - host0) - (device1 - device0) / 1000
    return host0 + (timestamp_ns - device0) / 1000 + drift * (timestamp_ns - device0) / (device1 - device0)


def write_csv(path, fieldnames, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_l20(root, arm):
    return json.loads((root / arm / "perfetto" / "step002_layer20.counts.json").read_text())


def main(root):
    out = Path(__file__).resolve().parent
    analysis = out / "analysis"
    analysis.mkdir(exist_ok=True)
    pair = json.loads((root / "full_decode_pair.json").read_text())

    l20_rows = []
    l20_summary = {}
    arm_anchors = {}
    for arm in ("mega", "gc"):
        data = load_l20(root, arm)
        rows = sorted(data["ranks"], key=lambda row: row["rank"])
        arm_anchors[arm] = rows
        entries = [aligned_us(row, row["device_clocks_ns"][0]) for row in rows]
        exits = [aligned_us(row, row["device_clocks_ns"][3]) for row in rows]
        base = min(entries)
        exchanges = [(row["device_clocks_ns"][5] - row["device_clocks_ns"][4]) / 1000 for row in rows]
        for row, entry, exit_ in zip(rows, entries, exits, strict=True):
            clocks = row["device_clocks_ns"]
            l20_rows.append(dict(
                arm=arm, rank=row["rank"], tokens=row["tokens"],
                m_blocks=row["m_blocks"], block_m=row["block_m"],
                entry_aligned_us=f"{entry:.6f}", entry_rel_us=f"{entry-base:.6f}",
                exit_aligned_us=f"{exit_:.6f}", exit_rel_us=f"{exit_-base:.6f}",
                local_moe_us=f"{(clocks[3]-clocks[0])/1000:.6f}",
                entry_to_first_mma_us=f"{(clocks[1]-clocks[0])/1000:.6f}",
                count_exchange_us=f"{(clocks[5]-clocks[4])/1000:.6f}",
                alignment_half_width_us=row["alignment_half_width_us"],
                alignment_drift_us=row["alignment_drift_us"],
                raw_globaltimer_entry_ns=clocks[0],
            ))
        early = entries.index(min(entries))
        late = entries.index(max(entries))
        max_exchange = exchanges.index(max(exchanges))
        l20_summary[arm] = dict(
            step=data["step"], layer=data["layer"],
            tokens=[row["tokens"] for row in rows],
            m_blocks=[row["m_blocks"] for row in rows],
            total_tokens=int(sum(row["tokens"] for row in rows)),
            total_m_blocks=int(sum(row["m_blocks"] for row in rows)),
            earliest_rank=early, latest_rank=late,
            entry_skew_us=float(max(entries) - min(entries)),
            full_span_us=float(max(exits) - min(entries)),
            exit_skew_us=float(max(exits) - min(exits)),
            max_exchange_rank=max_exchange, max_exchange_us=float(max(exchanges)),
            max_alignment_half_width_us=float(max(row["alignment_half_width_us"] for row in rows)),
            max_abs_alignment_drift_us=float(max(abs(row["alignment_drift_us"]) for row in rows)),
            alignment_method=sorted({row["alignment_method"] for row in rows}),
            note="device globaltimer timestamps are aligned per rank with the two Kineto anchors",
        )

    write_csv(analysis / "l20_ranks.csv", list(l20_rows[0]), l20_rows)
    (analysis / "l20_summary.json").write_text(json.dumps(l20_summary, indent=2) + "\n")

    boundary_rows = []
    for arm in ("mega", "gc"):
        rows = arm_anchors[arm]
        clocks = []
        for rank in range(8):
            with np.load(root / arm / f"measure-rank{rank}.npz") as handle:
                clocks.append(handle["clocks"])
        clocks = np.stack(clocks)
        for layer in range(LAYERS):
            entries = np.array([aligned_us(row, int(clocks[row["rank"], 2, layer, 0])) for row in rows])
            exits = np.array([aligned_us(row, int(clocks[row["rank"], 2, layer, 3])) for row in rows])
            local = (clocks[:, 2, layer, 3] - clocks[:, 2, layer, 0]) / 1000
            boundary_rows.append(dict(
                arm=arm, layer=layer, earliest_rank=int(entries.argmin()),
                latest_rank=int(entries.argmax()), entry_skew_us=f"{entries.max()-entries.min():.6f}",
                earliest_entry_aligned_us=f"{entries.min():.6f}",
                latest_exit_aligned_us=f"{exits.max():.6f}",
                full_span_us=f"{exits.max()-entries.min():.6f}",
                min_local_moe_us=f"{local.min():.6f}", max_local_moe_us=f"{local.max():.6f}",
            ))
    write_csv(analysis / "layer_boundaries_step002.csv", list(boundary_rows[0]), boundary_rows)

    graph_launch = {}
    for arm in ("mega", "gc"):
        graph_launch[arm] = [
            dict(rank=rank, correlations=item["correlations"],
                 matching_launches=item["matching_launches"],
                 attention_nodes=item["role_nodes"]["attention"],
                 dense_nodes=item["role_nodes"]["dense"], moe_nodes=item["role_nodes"]["moe"],
                 marker_count=item["marker_count"], foreign_kernels=item["foreign_kernels"],
                 passed=item["passed"])
            for rank, item in enumerate(pair["graph"][arm])
        ]
    (analysis / "graph_launch.json").write_text(json.dumps(graph_launch, indent=2) + "\n")

    timings = pair["sum_of_layer_max_rank_local_moe_ms"]
    perf = dict(
        clean_steps=pair["clean_steps"], strict_pair_pass=pair["strict_pair_pass"],
        mega_ms=timings["mega"], gc_ms=timings["gc"],
        gc_minus_mega_percent=100 * (timings["gc"] / timings["mega"] - 1),
        signed_gc_reduction_percent=100 * (1 - timings["gc"] / timings["mega"]),
        authoritative=bool(pair["strict_pair_pass"]),
        method="per step/layer max of eight local full-MoE durations; sum 40 layers; mean over 999 steps excluding step 2",
    )
    (analysis / "performance_summary.json").write_text(json.dumps(perf, indent=2) + "\n")

    prior = [
        dict(metric="L20 entry skew us", previous_mega=347.1162109375, previous_gc=357.8896484375,
             current_mega=l20_summary["mega"]["entry_skew_us"], current_gc=l20_summary["gc"]["entry_skew_us"],
             note="previous step 1 dense submit; current step 2 full decode graph"),
        dict(metric="longest L20 count exchange us", previous_mega=366.0498046875,
             previous_gc=368.609375, current_mega=l20_summary["mega"]["max_exchange_us"],
             current_gc=l20_summary["gc"]["max_exchange_us"],
             note="previous dense submit; current full graph has ~1-3 us entry skew"),
    ]
    write_csv(analysis / "prior_l20_comparison.csv", list(prior[0]), prior)

    calls_ok = True
    count_totals_ok = True
    counts_equal = True
    replay_id_equal = 0
    replay_weight_equal = 0
    logical_equal = 0
    results = {arm: json.loads((root / arm / "result.json").read_text()) for arm in ("mega", "gc")}
    for phase in ("stats", "measure"):
        arrays = {}
        for arm in ("mega", "gc"):
            result = results[arm]
            calls_ok &= all(row["full_graph_observed_calls"] == 40000 for row in result[phase]["ranks"])
            arrays[arm] = []
            for rank in range(8):
                with np.load(root / arm / f"{phase}-rank{rank}.npz") as handle:
                    counts = handle["counts"]
                arrays[arm].append(counts)
                count_totals_ok &= counts.shape == (1000, 40, 48) and (counts >= 0).all()
        for rank in range(8):
            replay_id_equal += int(results["mega"][phase]["ranks"][rank]["replay_ids"] ==
                                   results["gc"][phase]["ranks"][rank]["replay_ids"])
            replay_weight_equal += int(results["mega"][phase]["ranks"][rank]["replay_weights"] ==
                                       results["gc"][phase]["ranks"][rank]["replay_weights"])
            logical_equal += int(results["mega"][phase]["ranks"][rank]["logical_ids"] ==
                                 results["gc"][phase]["ranks"][rank]["logical_ids"])
        global_totals = np.stack(arrays["mega"]).sum(axis=(0, 3))
        count_totals_ok &= global_totals.shape == (1000, 40) and bool((global_totals == 3072).all())
        for rank in range(8):
            counts_equal &= np.array_equal(arrays["mega"][rank], arrays["gc"][rank])
    graph_ok = all(item["passed"] for arm in ("mega", "gc") for item in pair["graph"][arm])
    gc_result = json.loads((root / "gc" / "result.json").read_text())
    validation = [
        f"strict_pair_pass={pair['strict_pair_pass']}",
        f"strict_failure_count={len(pair['failures'])}",
        f"graph_coverage_passed={graph_ok} ({sum(len(pair['graph'][a]) for a in ('mega','gc'))}/16 ranks)",
        f"full_graph_calls_40000={calls_ok} (32/32 arm-phase-rank)",
        f"slot_counts_identical_between_arms={counts_equal} (16/16 phase-rank)",
        f"global_token_total_3072={count_totals_ok} (2000/2000 step-layer)",
        f"replay_id_hash_equal={replay_id_equal}/16",
        f"replay_weight_hash_equal={replay_weight_equal}/16",
        f"logical_id_hash_equal={logical_equal}/16",
        "stats native_diff=" + repr([row["native_diff"] for row in gc_result["stats"]["ranks"]]),
        "measure native_diff=" + repr([row["native_diff"] for row in gc_result["measure"]["ranks"]]),
        "GC first native difference=" + repr(gc_result["measure"]["ranks"][0]["first_native_diff"]),
        "GC first gate difference=" + repr(gc_result["measure"]["ranks"][0]["first_gate_difference"]),
        "strict failures: " + "; ".join(pair["failures"]),
    ]
    (out / "analysis_validation.txt").write_text("\n".join(validation) + "\n")


if __name__ == "__main__":
    import sys
    main(Path(sys.argv[1]).resolve())
