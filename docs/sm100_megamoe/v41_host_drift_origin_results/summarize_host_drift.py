#!/usr/bin/env python3
"""Build the V41 host-drift summaries from the retained Kineto traces."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/sm100"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "v41_l20_host_device_results"))

from analyze_l20_host_device import API_CATS, CPU_CATS, alias, device_mapping, write_csv  # noqa: E402
from native_host_submit import partition  # noqa: E402


ARMS = ("gc", "mega")
DIVERGENCE_US = 100.0


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def trace_events(mapping: dict) -> tuple[list[dict], dict, dict]:
    events = [event for event in mapping["raw"]["traceEvents"] if event.get("ph") == "X"]
    by_corr = defaultdict(list)
    by_thread = defaultdict(list)
    for event in events:
        corr = event.get("args", {}).get("correlation")
        if corr is not None:
            by_corr[str(corr)].append(event)
        if "pid" in event and "tid" in event:
            by_thread[(event["pid"], event["tid"])].append(event)
    for values in by_thread.values():
        values.sort(key=lambda event: (event.get("ts", 0), event.get("dur", 0)))
    return events, by_corr, by_thread


def launches_from_rows(mapping: dict, rows: list[dict], by_corr: dict) -> list[dict]:
    launches = []
    for row in sorted(rows, key=lambda item: int(item["layer"])):
        hits = [event for event in by_corr.get(str(row["correlation"]), [])
                if event.get("cat") in API_CATS and event.get("name") == "cudaGraphLaunch"]
        if len(hits) != 1:
            raise ValueError(f"correlation {row['correlation']} has {len(hits)} graph launches")
        launches.append(hits[0])
    if len(launches) != 40:
        raise ValueError(f"expected 40 graph launches, got {len(launches)}")
    return launches


def detail_occurrences(mapping: dict, rows: list[dict], by_corr: dict, by_thread: dict,
                       rank: int, layer: int) -> list[dict]:
    launches = launches_from_rows(mapping, rows, by_corr)
    launch = launches[layer]
    start = (launches[layer - 1]["ts"] if layer else
             mapping["marks"][0]["ts"] + mapping["marks"][0]["dur"] / 2)
    end = launch["ts"]
    pid, tid = launch["pid"], launch["tid"]
    thread = by_thread[(pid, tid)]
    positions = {id(event): index for index, event in enumerate(thread)}
    host = [event for event in thread if event.get("cat") in CPU_CATS]
    counts = Counter()
    family_counts = Counter()
    result = []
    for left, duration, event, parent in partition(host, start, end):
        category = event.get("cat", "unobserved") if event else "unobserved"
        name = event.get("name", "unobserved") if event else "unobserved"
        signature = (category, name, parent)
        occurrence = counts[signature]
        counts[signature] += 1
        name_occurrence = family_counts[(category, name)]
        family_counts[(category, name)] += 1
        index = positions.get(id(event))
        previous = thread[index - 1] if index is not None and index else {}
        following = thread[index + 1] if index is not None and index + 1 < len(thread) else {}
        corr = str(event.get("args", {}).get("correlation", "")) if event else ""
        result.append(dict(
            arm=rows[0]["arm"], rank=rank, layer=layer, occurrence_index=occurrence,
            name_occurrence=name_occurrence,
            category=category, name=name, parent=parent, duration_us=duration,
            offset_us=left - start, raw_pid=event.get("pid", "") if event else "",
            raw_tid=event.get("tid", "") if event else "", correlation=corr,
            previous_event=previous.get("name", ""), next_event=following.get("name", ""),
        ))
    return result


def key_layers() -> dict[str, tuple[int, int]]:
    return {"gc": (1, 1), "mega": (2, 10)}


def build_layer_summary(windows: dict, ranks: dict, mappings: dict, out: Path) -> None:
    rows = []
    for arm in ARMS:
        arm_windows = windows[arm]
        early = int(arm_windows[0]["early"])
        late = int(arm_windows[0]["late"])
        mark_early = mappings[(arm, early)]["marks"][0]
        mark_late = mappings[(arm, late)]["marks"][0]
        anchor_skew = mark_late["ts"] - mark_early["ts"]
        candidates = [(layer, float(row["new_window_us"]))
                      for layer, row in enumerate(arm_windows[1:], 1)
                      if abs(float(row["new_window_us"])) >= DIVERGENCE_US]
        if not candidates:
            raise ValueError(f"{arm}: no divergence over {DIVERGENCE_US} us")
        first_layer, first_value = candidates[0]
        max_layer, max_value = max(enumerate(arm_windows[1:], 1),
                                   key=lambda item: float(item[1]["new_window_us"]))
        max_value = float(max_value["new_window_us"])
        l19, l20 = arm_windows[19], arm_windows[20]
        rows.append(dict(
            arm=arm, early_rank=early, late_rank=late,
            l0_anchor_skew_us=anchor_skew,
            l0_anchor_to_submit_us=arm_windows[0]["new_window_us"],
            l0_launch_skew_us=arm_windows[0]["launch_skew_us"],
            l0_device_skew_us=arm_windows[0]["device_skew_us"],
            first_divergence_layer=first_layer,
            first_divergence_new_us=first_value,
            max_new_layer=max_layer, max_new_us=max_value,
            l19_launch_skew_us=l19["launch_skew_us"],
            l19_device_skew_us=l19["device_skew_us"],
            l19_early_queue_us=l19["early_queue_us"],
            l19_late_queue_us=l19["late_queue_us"],
            l20_new_window_us=l20["new_window_us"],
            l20_launch_skew_us=l20["launch_skew_us"],
            l20_device_skew_us=l20["device_skew_us"],
            l20_early_queue_us=l20["early_queue_us"],
            l20_late_queue_us=l20["late_queue_us"],
        ))
    write_csv(out / "layer_summary.csv", rows)

    rank_rows = []
    for arm in ARMS:
        first_layer, max_layer = key_layers()[arm]
        layers = sorted({0, first_layer, max_layer, 18, 19, 20})
        early = int(windows[arm][0]["early"])
        late = int(windows[arm][0]["late"])
        for rank in range(8):
            role = "early" if rank == early else "late" if rank == late else "other"
            for layer in layers:
                row = ranks[(arm, rank, layer)]
                rank_rows.append(dict(
                    arm=arm, rank=rank, role=role, layer=layer,
                    anchor0_trace_us=mappings[(arm, rank)]["marks"][0]["ts"],
                    host_api_us=row["host_api_us"], device_entry_us=row["device_entry_us"],
                    host_window_us=row["host_window_us"],
                    launch_to_entry_us=row["launch_to_entry_us"],
                    correlation=row["correlation"],
                ))
    write_csv(out / "rank_key_layers.csv", rank_rows)


def build_closure(windows: dict, exclusive: list[dict], out: Path) -> None:
    rows = []
    for arm in ARMS:
        arm_windows = windows[arm]
        early, late = int(arm_windows[0]["early"]), int(arm_windows[0]["late"])
        for scope, layers, start_row, end_row in (
            ("L1-19", range(1, 20), arm_windows[0], arm_windows[19]),
            ("L20", range(20, 21), arm_windows[19], arm_windows[20]),
            ("L1-20", range(1, 21), arm_windows[0], arm_windows[20]),
        ):
            selected = [row for row in exclusive if row["arm"] == arm and int(row["layer"]) in layers]
            positive = sum(float(row["delta_us"]) for row in selected if float(row["delta_us"]) > 0)
            negative = sum(float(row["delta_us"]) for row in selected if float(row["delta_us"]) < 0)
            unobserved = sum(float(row["delta_us"]) for row in selected if row["category"] == "unobserved")
            total_exclusive = sum(float(row["delta_us"]) for row in selected)
            new_window = sum(float(arm_windows[layer]["new_window_us"]) for layer in layers)
            inherited = float(start_row["launch_skew_us"])
            total_launch = float(end_row["launch_skew_us"])
            early_queue = float(end_row["early_queue_us"])
            late_queue = float(end_row["late_queue_us"])
            queue_delta = late_queue - early_queue
            device_skew = float(end_row["device_skew_us"])
            rows.append(dict(
                arm=arm, scope=scope, early=early, late=late,
                inherited_launch_skew_us=inherited, new_window_us=new_window,
                total_launch_skew_us=total_launch,
                host_closure_error_us=total_launch - inherited - new_window,
                exclusive_positive_us=positive, exclusive_negative_us=negative,
                exclusive_unobserved_us=unobserved, exclusive_sum_us=total_exclusive,
                exclusive_closure_error_us=new_window - total_exclusive,
                early_queue_us=early_queue, late_queue_us=late_queue,
                queue_delta_us=queue_delta, device_skew_us=device_skew,
                device_closure_error_us=device_skew - total_launch - queue_delta,
            ))
    write_csv(out / "layer_closure.csv", rows)


def build_step_mono(raw: Path, mono: list[dict], windows: dict, out: Path) -> None:
    by_arm_step = defaultdict(list)
    for row in mono:
        by_arm_step[(row["arm"], int(row["step"]))].append(row)
    rows, summary = [], []
    for arm in ARMS:
        folder = raw / arm
        mappings = {rank: device_mapping(folder, rank, 1) for rank in range(8)}
        for step in (1, 4, 6):
            maps = {rank: (mappings[rank] if step == 1 else device_mapping(folder, rank, step))
                    for rank in range(8)}
            entries = {rank: float(maps[rank]["map"](maps[rank]["clocks"][step, 20, 0]))
                       for rank in range(8)}
            early = min(entries, key=entries.get)
            late = max(entries, key=entries.get)
            values = {(int(row["rank"]), int(row["layer"])): int(row["submit_mono_ns"])
                      for row in by_arm_step[(arm, step)]}
            for layer in range(40):
                rows.append(dict(arm=arm, step=step, early=early, late=late, layer=layer,
                                 submit_skew_us=(values[(late, layer)] - values[(early, layer)]) / 1000.0))
            summary.append(dict(
                arm=arm, step=step, early=early, late=late,
                l0_skew_us=(values[(late, 0)] - values[(early, 0)]) / 1000.0,
                l19_skew_us=(values[(late, 19)] - values[(early, 19)]) / 1000.0,
                l20_skew_us=(values[(late, 20)] - values[(early, 20)]) / 1000.0,
                l39_skew_us=(values[(late, 39)] - values[(early, 39)]) / 1000.0,
                step1_pair_early=windows[arm][0]["early"] if step == 1 else "",
                step1_pair_late=windows[arm][0]["late"] if step == 1 else "",
            ))
    write_csv(out / "step146_mono.csv", rows)
    write_csv(out / "step146_mono_summary.csv", summary)


def semantic_class(category: str, name: str, offset_us: float, window_us: float) -> str:
    if category == "unobserved":
        return "unobserved"
    if name == "cudaGraphLaunch":
        return "graph_replay"
    audit_copy = name in ("aten::copy_", "cudaMemcpyAsync") and offset_us < 100.0
    audit_state = (name in ("aten::copy_", "aten::fill_", "aten::zero_",
                            "cudaMemcpyAsync", "cudaLaunchKernel")
                   and offset_us >= max(0.0, window_us - 450.0))
    if audit_copy or audit_state:
        return "graph_wrapper_audit"
    if category in API_CATS:
        return "cuda_api"
    if category == "cpu_op":
        return "model_forward"
    return "other"


def build_exclusive(raw: Path, ranks: dict, exclusive: list[dict], mappings: dict,
                    windows: dict, out: Path) -> None:
    occurrence_rows = []
    details = {}
    compared_by_layer = {}
    for arm in ARMS:
        for rank in range(8):
            rows = [ranks[(arm, rank, layer)] for layer in range(40)]
            _, by_corr, by_thread = trace_events(mappings[(arm, rank)])
            for layer in range(40):
                values = detail_occurrences(mappings[(arm, rank)], rows, by_corr, by_thread, rank, layer)
                details[(arm, rank, layer)] = values
                occurrence_rows.extend(values)
    write_csv(out / "exclusive_occurrences.csv", occurrence_rows,
              ["arm", "rank", "layer", "occurrence_index", "name_occurrence", "category", "name", "parent",
               "duration_us", "offset_us", "raw_pid", "raw_tid", "correlation",
               "previous_event", "next_event"])

    top_rows = []
    for arm in ARMS:
        early, late = int(next(row for row in exclusive if row["arm"] == arm)["early"]), \
            int(next(row for row in exclusive if row["arm"] == arm)["late"])
        for layer in range(40):
            keys = set()
            for side in (early, late):
                for row in details[(arm, side, layer)]:
                    keys.add((row["category"], row["name"], row["parent"], row["occurrence_index"]))
            compared = []
            for key in keys:
                left = [row for row in details[(arm, early, layer)]
                        if (row["category"], row["name"], row["parent"], row["occurrence_index"]) == key]
                right = [row for row in details[(arm, late, layer)]
                         if (row["category"], row["name"], row["parent"], row["occurrence_index"]) == key]
                delta = (sum(row["duration_us"] for row in right)
                         - sum(row["duration_us"] for row in left))
                sample = (right or left)[0]
                late_window = float(ranks[(arm, late, layer)]["host_window_us"])
                early_window = float(ranks[(arm, early, layer)]["host_window_us"])
                compared.append(dict(
                    arm=arm, early=early, late=late, layer=layer,
                    category=key[0], name=key[1], parent=key[2], occurrence_index=key[3],
                    early_duration_us=sum(row["duration_us"] for row in left),
                    late_duration_us=sum(row["duration_us"] for row in right), delta_us=delta,
                    early_offset_us=left[0]["offset_us"] if left else "",
                    late_offset_us=right[0]["offset_us"] if right else "",
                    early_name_occurrence=left[0]["name_occurrence"] if left else "",
                    late_name_occurrence=right[0]["name_occurrence"] if right else "",
                    semantic_class=(semantic_class(key[0], key[1], right[0]["offset_us"], late_window)
                                    if right else semantic_class(key[0], key[1], left[0]["offset_us"], early_window)),
                    raw_pid=sample["raw_pid"], raw_tid=sample["raw_tid"],
                    correlation=sample["correlation"], previous_event=sample["previous_event"],
                    next_event=sample["next_event"],
                ))
            positives = sorted((row for row in compared if row["delta_us"] > 0),
                               key=lambda row: row["delta_us"], reverse=True)[:5]
            negatives = sorted((row for row in compared if row["delta_us"] < 0),
                               key=lambda row: row["delta_us"])[:3]
            total_positive = sum(row["delta_us"] for row in compared if row["delta_us"] > 0)
            total_negative = sum(row["delta_us"] for row in compared if row["delta_us"] < 0)
            compared_by_layer[(arm, layer)] = {
                (row["category"], row["name"], row["parent"], row["occurrence_index"]): row
                for row in compared
            }
            for direction, selected in (("positive", positives), ("negative", negatives)):
                for rank_index, row in enumerate(selected):
                    row = dict(row, direction=direction, rank_within_direction=rank_index + 1,
                               layer_positive_sum_us=total_positive,
                               layer_negative_sum_us=total_negative)
                    top_rows.append(row)
    write_csv(out / "top_exclusive.csv", top_rows,
              ["arm", "early", "late", "layer", "direction", "rank_within_direction",
               "category", "name", "parent", "occurrence_index", "early_duration_us",
               "late_duration_us", "delta_us", "semantic_class", "early_offset_us",
               "late_offset_us", "early_name_occurrence", "late_name_occurrence",
               "layer_positive_sum_us",
               "layer_negative_sum_us", "raw_pid", "raw_tid", "correlation",
               "previous_event", "next_event"])

    cumulative = defaultdict(float)
    for row in exclusive:
        cumulative[(row["arm"], row["category"], row["name"], row["parent"])] += float(row["delta_us"])
    cumulative_rows = []
    for (arm, category, name, parent), delta in sorted(cumulative.items(), key=lambda item: item[1], reverse=True):
        cumulative_rows.append(dict(arm=arm, category=category, name=name, parent=parent, delta_us=delta))
    write_csv(out / "exclusive_cumulative.csv", cumulative_rows)

    semantic_rows = []
    for arm in ARMS:
        for scope, layers in (("L1-19", range(1, 20)), ("L20", (20,))):
            totals = defaultdict(lambda: dict(positive=0.0, negative=0.0, net=0.0, occurrences=0))
            for layer in layers:
                for row in compared_by_layer[(arm, layer)].values():
                    value = totals[row["semantic_class"]]
                    delta = row["delta_us"]
                    value["net"] += delta
                    value["occurrences"] += 1
                    if delta > 0:
                        value["positive"] += delta
                    elif delta < 0:
                        value["negative"] += delta
            for semantic, value in sorted(totals.items(), key=lambda item: item[1]["net"], reverse=True):
                semantic_rows.append(dict(arm=arm, scope=scope, semantic_class=semantic,
                                          positive_us=value["positive"], negative_us=value["negative"],
                                          net_us=value["net"], matched_occurrences=value["occurrences"]))
    write_csv(out / "exclusive_semantic_summary.csv", semantic_rows)

    evidence_rows = []
    first = key_layers()
    for arm in ARMS:
        for label, layers in (("first_divergence", (first[arm][0],)),
                              ("max_new", (first[arm][1],)),
                              ("L20", (20,)),
                              ("L1-19_top", tuple(range(1, 20)))):
            totals = defaultdict(float)
            for row in exclusive:
                if row["arm"] == arm and int(row["layer"]) in layers:
                    totals[(row["category"], row["name"], row["parent"])] += float(row["delta_us"])
            selected = [key for key, value in sorted(
                totals.items(), key=lambda item: item[1], reverse=True)[:5] if value > 0]
            selected.extend(key for key, value in sorted(
                totals.items(), key=lambda item: item[1])[:3] if value < 0)
            for category, name, parent in selected:
                layer_values = []
                for layer in layers:
                    matches = [value for key, value in compared_by_layer[(arm, layer)].items()
                               if key[:3] == (category, name, parent)]
                    if matches:
                        layer_values.append((sum(value["delta_us"] for value in matches), matches, layer))
                if not layer_values:
                    raise ValueError(f"{arm} {label}: unresolved occurrence evidence")
                layer_delta, matches, layer = max(layer_values, key=lambda item: abs(item[0]))
                layer_row = max(matches, key=lambda item: abs(item["delta_us"]))
                evidence_rows.append(dict(
                    arm=arm, scope=label, category=category, name=name, parent=parent,
                    occurrence_index=layer_row["occurrence_index"],
                    cumulative_delta_us=totals[(category, name, parent)],
                    representative_layer=layer, representative_delta_us=layer_row["delta_us"],
                    representative_layer_delta_us=layer_delta,
                    semantic_class=layer_row["semantic_class"],
                    early_duration_us=layer_row["early_duration_us"],
                    late_duration_us=layer_row["late_duration_us"],
                    early_offset_us=layer_row["early_offset_us"],
                    late_offset_us=layer_row["late_offset_us"],
                    early_name_occurrence=layer_row["early_name_occurrence"],
                    late_name_occurrence=layer_row["late_name_occurrence"],
                    raw_pid=layer_row["raw_pid"], raw_tid=layer_row["raw_tid"],
                    correlation=layer_row["correlation"],
                    previous_event=layer_row["previous_event"], next_event=layer_row["next_event"],
                ))
    write_csv(out / "exclusive_evidence.csv", evidence_rows)


def build_graph_coverage(raw: Path, ranks: dict, mappings: dict, windows: dict, out: Path) -> None:
    rows = []
    details = []
    for arm in ARMS:
        early = int(windows[arm][0]["early"])
        late = int(windows[arm][0]["late"])
        for rank in range(8):
            one_rank = [ranks[(arm, rank, layer)] for layer in range(40)]
            _, by_corr, by_thread = trace_events(mappings[(arm, rank)])
            launches = launches_from_rows(mappings[(arm, rank)], one_rank, by_corr)
            start, end = launches[19]["ts"], launches[20]["ts"]
            pid, tid = launches[19]["pid"], launches[19]["tid"]
            events = [event for event in by_thread[(pid, tid)]
                      if start < event["ts"] <= end and event.get("cat") in CPU_CATS | {"cpu_op"}]
            counts = Counter((event.get("cat"), event.get("name")) for event in events)
            curves = sorted((event for event in events if event.get("cat") == "cuda_runtime"
                             and event.get("name") == "cudaGraphLaunch"), key=lambda event: event["ts"])
            graph_nodes = 0
            moe_graphs = 0
            for sequence, curve in enumerate(curves):
                corr = str(curve.get("args", {}).get("correlation", ""))
                members = by_corr.get(corr, [])
                kernels = [event for event in members if event.get("cat") == "kernel"]
                node_count = len({event.get("args", {}).get("graph node id") for event in kernels})
                aliases = [alias(event.get("name", "")) for event in kernels]
                graph_nodes += node_count
                is_moe = any(name == arm + "_moe" for name in aliases)
                if is_moe:
                    moe_graphs += 1
                details.append(dict(
                    arm=arm, rank=rank, role=(("early" if rank == int(windows[arm][0]["early"])
                                               else "late" if rank == int(windows[arm][0]["late"])
                                               else "other")),
                    graph_sequence=sequence + 1, offset_us=curve["ts"] - start,
                    correlation=corr, graph_kind="moe" if is_moe else "dense_local",
                    graph_node_events=node_count, kernel_aliases="|".join(aliases),
                ))
            direct = sum(counts[(category, name)] for category, name in (
                ("cuda_runtime", "cudaLaunchKernel"),
                ("cuda_runtime", "cudaLaunchKernelExC"),
                ("cuda_driver", "cuLaunchKernelEx"),
            ))
            copies = counts[("cuda_runtime", "cudaMemcpyAsync")]
            zeroes = counts[("cpu_op", "aten::zero_")]
            role = "early" if rank == early else "late" if rank == late else "other"
            rows.append(dict(
                arm=arm, rank=rank, role=role, window_us=end - start,
                graph_launch_calls=len(curves), moe_graph_launch_calls=moe_graphs,
                dense_graph_launch_calls=len(curves) - moe_graphs,
                direct_kernel_launch_calls=direct, total_kernel_launch_calls=direct + len(curves),
                cuda_memcpy_async_calls=copies,
                aten_copy_ops=counts[("cpu_op", "aten::copy_")],
                aten_fill_ops=counts[("cpu_op", "aten::fill_")],
                aten_zero_ops=zeroes,
                graph_node_events=graph_nodes,
                audit_post_replay_copies=3,
                audit_pre_replay_copies=1,
                audit_pre_replay_zero_fill_ops=1,
                model_copy_ops=counts[("cpu_op", "aten::copy_")] - 4,
                model_memcpy_async_calls=copies - 4,
                model_direct_kernel_launch_calls=direct - 1,
                l19_early_queue_us=windows[arm][19]["early_queue_us"],
                l19_late_queue_us=windows[arm][19]["late_queue_us"],
                l20_early_queue_us=windows[arm][20]["early_queue_us"],
                l20_late_queue_us=windows[arm][20]["late_queue_us"],
                early_replay_ahead_l19_us=windows[arm][19]["early_queue_us"],
                early_replay_ahead_l20_us=windows[arm][20]["early_queue_us"],
            ))
    write_csv(out / "graph_coverage_compare.csv", rows)
    write_csv(out / "graph_launch_details.csv", details)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    windows_rows = read_rows(args.out / "windows.csv")
    rank_rows = read_rows(args.out / "ranks.csv")
    exclusive_rows = read_rows(args.out / "exclusive.csv")
    mono_rows = read_rows(args.out / "mono.csv")
    windows = {arm: [row for row in windows_rows if row["arm"] == arm] for arm in ARMS}
    ranks = {(row["arm"], int(row["rank"]), int(row["layer"])): row for row in rank_rows}
    mappings = {(arm, rank): device_mapping(args.raw / arm, rank, 1)
                for arm in ARMS for rank in range(8)}

    build_layer_summary(windows, ranks, mappings, args.out)
    build_closure(windows, exclusive_rows, args.out)
    build_step_mono(args.raw, mono_rows, windows, args.out)
    build_exclusive(args.raw, ranks, exclusive_rows, mappings, windows, args.out)
    build_graph_coverage(args.raw, ranks, mappings, windows, args.out)


if __name__ == "__main__":
    main()
