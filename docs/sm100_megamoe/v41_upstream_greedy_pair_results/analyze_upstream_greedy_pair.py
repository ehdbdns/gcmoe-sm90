#!/usr/bin/env python3
"""Tabulate independently generated greedy Mega/GC arms without cross-branching."""

import csv
import json
from pathlib import Path

import numpy as np


LAYERS = 40


def aligned_us(row, timestamp_ns):
    host0, host1 = row["alignment_anchor_host_ns"]
    device0, device1 = row["alignment_anchor_device_ns"]
    drift = (host1 - host0) - (device1 - device0) / 1000
    return host0 + (timestamp_ns - device0) / 1000 + drift * (
        timestamp_ns - device0
    ) / (device1 - device0)


def write_csv(path, fieldnames, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(root):
    out = Path(__file__).resolve().parent
    analysis = out / "analysis"
    analysis.mkdir(exist_ok=True)
    pair = json.loads((root / "native_greedy_pair.json").read_text())
    results = {arm: json.loads((root / arm / "result.json").read_text())
               for arm in ("mega", "gc")}

    l20_rows = []
    l20_summary = {}
    l20_rank_rows = {}
    for arm in ("mega", "gc"):
        data = json.loads((root / arm / "perfetto/step002_layer20.counts.json").read_text())
        rows = sorted(data["ranks"], key=lambda row: row["rank"])
        l20_rank_rows[arm] = rows
        entries = [aligned_us(row, row["device_clocks_ns"][0]) for row in rows]
        exits = [aligned_us(row, row["device_clocks_ns"][3]) for row in rows]
        base = min(entries)
        exchanges = [(row["device_clocks_ns"][5] - row["device_clocks_ns"][4]) / 1000
                     for row in rows]
        for row, entry, exit_ in zip(rows, entries, exits, strict=True):
            clocks = row["device_clocks_ns"]
            l20_rows.append(dict(
                arm=arm,
                rank=row["rank"],
                tokens=row["tokens"],
                m_blocks=row["m_blocks"],
                block_m=row["block_m"],
                entry_aligned_us=f"{entry:.6f}",
                entry_rel_us=f"{entry-base:.6f}",
                exit_aligned_us=f"{exit_:.6f}",
                exit_rel_us=f"{exit_-base:.6f}",
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
            step=data["step"],
            layer=data["layer"],
            tokens=[row["tokens"] for row in rows],
            m_blocks=[row["m_blocks"] for row in rows],
            total_tokens=int(sum(row["tokens"] for row in rows)),
            total_m_blocks=int(sum(row["m_blocks"] for row in rows)),
            earliest_rank=early,
            latest_rank=late,
            entry_skew_us=float(max(entries) - min(entries)),
            full_span_us=float(max(exits) - min(entries)),
            exit_skew_us=float(max(exits) - min(exits)),
            max_exchange_rank=max_exchange,
            max_exchange_us=float(max(exchanges)),
            max_alignment_half_width_us=float(
                max(row["alignment_half_width_us"] for row in rows)),
            max_abs_alignment_drift_us=float(
                max(abs(row["alignment_drift_us"]) for row in rows)),
            alignment_method=sorted({row["alignment_method"] for row in rows}),
        )
        (analysis / f"l20_{arm}.json").write_text(
            json.dumps(dict(data=l20_summary[arm], ranks=rows), indent=2) + "\n")

    write_csv(analysis / "l20_ranks.csv", list(l20_rows[0]), l20_rows)
    (analysis / "l20_summary.json").write_text(json.dumps(l20_summary, indent=2) + "\n")

    boundary_rows = []
    for arm in ("mega", "gc"):
        rows = l20_rank_rows[arm]
        clocks = []
        for rank in range(8):
            with np.load(root / arm / f"measure-rank{rank}.npz") as handle:
                clocks.append(handle["clocks"])
        clocks = np.stack(clocks)
        for layer in range(LAYERS):
            entries = np.array([
                aligned_us(row, int(clocks[row["rank"], 2, layer, 0])) for row in rows
            ])
            exits = np.array([
                aligned_us(row, int(clocks[row["rank"], 2, layer, 3])) for row in rows
            ])
            local = (clocks[:, 2, layer, 3] - clocks[:, 2, layer, 0]) / 1000
            boundary_rows.append(dict(
                arm=arm,
                layer=layer,
                earliest_rank=int(entries.argmin()),
                latest_rank=int(entries.argmax()),
                entry_skew_us=f"{entries.max()-entries.min():.6f}",
                earliest_entry_aligned_us=f"{entries.min():.6f}",
                latest_exit_aligned_us=f"{exits.max():.6f}",
                full_span_us=f"{exits.max()-entries.min():.6f}",
                min_local_moe_us=f"{local.min():.6f}",
                max_local_moe_us=f"{local.max():.6f}",
            ))
    write_csv(analysis / "layer_boundaries_step002.csv",
              list(boundary_rows[0]), boundary_rows)

    graph_rows = []
    for arm in ("mega", "gc"):
        for rank, item in enumerate(pair["graph"][arm]):
            graph_rows.append(dict(
                arm=arm,
                rank=rank,
                passed=item["passed"],
                correlation=",".join(item["correlations"]),
                matching_launches=item["matching_launches"],
                marker_count=item["marker_count"],
                attention_nodes=item["role_nodes"]["attention"],
                dense_nodes=item["role_nodes"]["dense"],
                moe_nodes=item["role_nodes"]["moe"],
                foreign_kernels=item["foreign_kernels"],
            ))
    write_csv(analysis / "graph_launch.csv", list(graph_rows[0]), graph_rows)
    (analysis / "graph_launch.json").write_text(json.dumps(graph_rows, indent=2) + "\n")

    config_rows = []
    for arm in ("mega", "gc"):
        for phase in ("stats", "measure"):
            for rank, row in enumerate(results[arm][phase]["ranks"]):
                split = row.get("sm_split") or {}
                config_rows.append(dict(
                    arm=arm,
                    phase=phase,
                    rank=rank,
                    block_m=row["block_m"],
                    dispatch_sm=split.get("dispatch", ""),
                    compute_sm=split.get("compute", ""),
                    backend_file=row["backend_file"],
                    full_graph_observed_calls=row["full_graph_observed_calls"],
                ))
    write_csv(analysis / "block_m_sm_split.csv", list(config_rows[0]), config_rows)

    difference_rows = []
    for phase in ("stats", "measure"):
        phase_data = pair["phases"][phase]
        for metric in ("tokens",):
            item = phase_data[metric]
            difference_rows.append(dict(
                phase=phase, rank="all", metric=metric, count=item["count"],
                first_position=json.dumps(item["first"]["position"]),
                mega_first=item["first"]["mega"], gc_first=item["first"]["gc"]))
        for rank, row in enumerate(phase_data["ranks"]):
            for metric in ("ids", "weights", "slot_counts"):
                item = row[metric]
                difference_rows.append(dict(
                    phase=phase, rank=rank, metric=metric, count=item["count"],
                    first_position=json.dumps(item["first"]["position"]),
                    mega_first=item["first"]["mega"], gc_first=item["first"]["gc"]))
    write_csv(analysis / "native_differences.csv",
              list(difference_rows[0]), difference_rows)
    (analysis / "native_differences.json").write_text(
        json.dumps(pair["phases"], indent=2) + "\n")

    times = pair["moe_ms"]
    performance = dict(
        mega_ms=times["mega"],
        gc_ms=times["gc"],
        gc_minus_mega_percent=100 * (times["gc"] / times["mega"] - 1),
        signed_gc_time_change_percent=100 * (times["gc"] / times["mega"] - 1),
        clean_steps=999,
        method=("per step/layer max of eight local full-MoE durations; "
                "sum 40 layers; mean over 999 steps excluding step 2"),
        independent_generation=True,
        route_divergence_expected=True,
        comparison_note=("Each arm decodes its own trajectory; after the first token "
                         "divergence, context, routes and per-layer work are no longer "
                         "the same even though the launch envelope remains BS512."),
    )
    (analysis / "performance_summary.json").write_text(
        json.dumps(performance, indent=2) + "\n")

    calls_ok = all(row["full_graph_observed_calls"] == 40000
                   for arm in ("mega", "gc") for phase in ("stats", "measure")
                   for row in results[arm][phase]["ranks"])
    graph_ok = all(item["passed"] for arm in ("mega", "gc") for item in pair["graph"][arm])
    token_shapes_ok = all(
        np.load(root / arm / f"{phase}-tokens.npy", mmap_mode="r").shape == (512, 1000)
        for arm in ("mega", "gc") for phase in ("stats", "measure"))
    count_totals = []
    for arm in ("mega", "gc"):
        for phase in ("stats", "measure"):
            total = np.zeros((1000, LAYERS), dtype=np.int64)
            for rank in range(8):
                with np.load(root / arm / f"{phase}-rank{rank}.npz") as handle:
                    counts = handle["counts"]
                    assert counts.shape == (1000, LAYERS, 48)
                    total += counts.sum(-1)
            count_totals.append(bool((total == 3072).all()))
    split = results["gc"]["stats"]["ranks"][0]["sm_split"]
    validation = [
        f"mega_exit={ (root / 'mega/exit-code').read_text().strip() }",
        f"gc_exit={ (root / 'gc/exit-code').read_text().strip() }",
        f"graph_coverage_passed={graph_ok} (16/16 ranks)",
        f"full_graph_calls_40000={calls_ok} (32/32 arm-phase-rank)",
        f"token_array_shape_1000x512={token_shapes_ok}",
        f"global_token_total_3072={all(count_totals)} ({sum(count_totals)}/4 arm-phases)",
        f"prefill_mismatched_requests={pair['prefill_mismatched_requests']}",
        f"stats_token_differences={pair['phases']['stats']['tokens']['count']}",
        f"measure_token_differences={pair['phases']['measure']['tokens']['count']}",
        f"stats_first_token_difference={pair['phases']['stats']['tokens']['first']}",
        f"measure_first_token_difference={pair['phases']['measure']['tokens']['first']}",
        f"stats_rank0_first_route_difference={pair['phases']['stats']['ranks'][0]['ids']['first']}",
        f"stats_rank0_first_weight_difference={pair['phases']['stats']['ranks'][0]['weights']['first']}",
        f"gc_sm_split_dispatch={split['dispatch']}",
        f"gc_sm_split_compute={split['compute']}",
        f"mega_ms={times['mega']}",
        f"gc_ms={times['gc']}",
        f"gc_time_change_percent_vs_mega={performance['gc_minus_mega_percent']}",
    ]
    (out / "analysis_validation.txt").write_text("\n".join(validation) + "\n")


if __name__ == "__main__":
    import sys
    main(Path(sys.argv[1]).resolve())
