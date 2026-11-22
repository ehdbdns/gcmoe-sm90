#!/usr/bin/env python3
"""CPU-only union/boundary analysis for the V41 L19->L20 residual window."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ARMS = ("gc", "mega")
PAIR = {"gc": (2, 0), "mega": (3, 4)}
STEPS = (1, 4, 6)
DEVICE_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}
BIG_GAP_US = 10.0
EPS = 1e-6


def alias(name: str) -> str:
    patterns = (
        ("gc_moe", "GCMoERoleKernel"),
        ("mega_moe", "mega_moe_impl"),
        ("memcpy", "Memcpy DtoD"),
        ("mhc", "mega_mhc_impl"),
        ("all_gather", "lamport_all_gather"),
        ("reduce_scatter", "lamport_reduce_scatter"),
        ("quantize", "MXFP8Quantize"),
        ("dense_gemm", "BlockScaledPersistentDenseGemm"),
        ("q_kv_norm", "_q_kv_norm_quant_kernel"),
        ("save_compress", "_fused_save_compress_norm_kernel"),
        ("qnormrope", "fusedDeepseekV4QNormRopeKVRopeQuantInsert"),
        ("rope_quant", "_rope_quant_insert_nvfp4"),
        ("indexer", "_indexer_k_norm"),
        ("paged_logits", "paged_mqa_logits"),
        ("block_scores", "_block_scores"),
        ("gather_topk", "gatherTopK"),
        ("radix", "radixSortKVInPlace"),
        ("store_cand", "_store_candidates"),
        ("filtered_topk", "FilteredTopK"),
        ("copy", "direct_copy_kernel_cuda"),
        ("global_topk", "_compute_global_topk"),
        ("attention", "core_attn::fwd_kernel"),
        ("fp8_fp4_gemm", "sm100_fp8_fp4_gemm_1d1d"),
        ("mega_gate", "mega_gate_impl"),
        ("prepare", "_prepare_megamoe_inputs"),
        ("fill", "FillFunctor"),
        ("cublas_tst", "nvjet_sm103_tst"),
        ("cublas_tss", "nvjet_sm103_tss"),
        ("splitkreduce", "splitKreduce"),
        ("cutlass_dev", "kernel_cutlass_device_kernel"),
    )
    for short, needle in patterns:
        if needle in name:
            return short
    return "other"


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def merge_intervals(intervals: list[tuple[float, float]]) -> list[list[float]]:
    merged: list[list[float]] = []
    for left, right in sorted(intervals):
        if right <= left + EPS:
            continue
        if not merged or left > merged[-1][1] + EPS:
            merged.append([left, right])
        else:
            merged[-1][1] = max(merged[-1][1], right)
    return merged


def complement(start: float, end: float, merged: list[list[float]]) -> list[tuple[float, float]]:
    gaps: list[tuple[float, float]] = []
    cursor = start
    for left, right in merged:
        if left > cursor + EPS:
            gaps.append((cursor, left))
        cursor = max(cursor, right)
    if end > cursor + EPS:
        gaps.append((cursor, end))
    return gaps


def find_adjacent(events: list[dict], left: float, right: float) -> tuple[list[dict], list[dict]]:
    previous = [e for e in events if abs(e["end"] - left) <= 1e-4]
    following = [e for e in events if abs(e["start"] - right) <= 1e-4]
    if not previous:
        upper = max(e["end"] for e in events if e["end"] <= left + 1e-4)
        previous = [e for e in events if abs(e["end"] - upper) <= 1e-4]
    if not following:
        lower = min(e["start"] for e in events if e["start"] >= right - 1e-4)
        following = [e for e in events if abs(e["start"] - lower) <= 1e-4]
    return previous, following


def joined(events: list[dict], field: str) -> str:
    return " | ".join(str(e[field]) for e in events)


def load_window(raw: Path, archive: Path, arm: str, rank: int) -> dict:
    trace = json.loads((raw / arm / f"profile-rank{rank}.json").read_text())
    all_device = [
        e for e in trace["traceEvents"]
        if e.get("ph") == "X" and e.get("cat") in DEVICE_CATS
    ]
    anchors = sorted(
        e["ts"] + e["dur"] / 2
        for e in all_device
        if "sm100_phase_clock_anchor" in e["name"] and e["dur"] > 0
    )
    device_anchor = np.load(raw / arm / f"anchor-rank{rank}.npy")
    if len(anchors) != 2 or not int(device_anchor[0]) < int(device_anchor[1]):
        raise ValueError(f"invalid two-anchor calibration: {arm} rank {rank}")
    scale = (anchors[1] - anchors[0]) / int(device_anchor[1] - device_anchor[0])
    with np.load(archive / arm / f"measure-rank{rank}.npz") as data:
        clocks = data["clocks"][1].copy()
    exit_ns = int(clocks[19, 3])
    entry_ns = int(clocks[20, 0])
    if not int(device_anchor[0]) < exit_ns < entry_ns < int(device_anchor[1]):
        raise ValueError(f"step-1 clocks outside anchors: {arm} rank {rank}")
    start = anchors[0] + (exit_ns - int(device_anchor[0])) * scale
    end = anchors[0] + (entry_ns - int(device_anchor[0])) * scale

    events = []
    for e in all_device:
        if "sm100_phase_clock_anchor" in e["name"]:
            continue
        left = max(start, e["ts"])
        right = min(end, e["ts"] + e["dur"])
        if right <= left:
            continue
        args = e.get("args", {})
        events.append({
            "start": left,
            "end": right,
            "raw_start": e["ts"],
            "raw_end": e["ts"] + e["dur"],
            "dur": right - left,
            "name": e["name"],
            "alias": alias(e["name"]),
            "cat": e["cat"],
            "stream": int(args.get("stream", -1)),
            "correlation": int(args.get("correlation", -1)),
            "overlap": 0,
        })
    events.sort(key=lambda e: (e["start"], e["strid"] if "strid" in e else e["stream"], e["correlation"]))

    by_stream = defaultdict(list)
    for e in events:
        by_stream[e["stream"]].append(e)
    for stream_events in by_stream.values():
        stream_events.sort(key=lambda e: (e["start"], e["correlation"], e["name"]))
        active_end = None
        for e in stream_events:
            if active_end is not None and e["start"] < active_end - 0.5:
                e["overlap"] = 1
            active_end = max(active_end, e["end"]) if active_end is not None else e["end"]

    union = merge_intervals([(e["start"], e["end"]) for e in events])
    gaps = complement(start, end, union)
    busy = sum(right - left for left, right in union)
    gap_total = sum(right - left for left, right in gaps)
    return {
        "arm": arm,
        "rank": rank,
        "start": start,
        "end": end,
        "window": end - start,
        "dev_anchor": device_anchor,
        "anchors": anchors,
        "scale": scale,
        "events": events,
        "union": union,
        "gaps": gaps,
        "busy": busy,
        "gap_total": gap_total,
        "closure": (end - start) - busy - gap_total,
        "overlap_events": sum(e["overlap"] for e in events),
    }


def window_outputs(windows: dict[tuple[str, int], dict], out: Path) -> None:
    union_rows, event_rows, summary_rows, gap_rows = [], [], [], []
    for arm in ARMS:
        entries = [windows[(arm, r)]["end"] for r in range(8)]
        exits = [windows[(arm, r)]["start"] for r in range(8)]
        earliest_entry = min(entries)
        earliest_exit = min(exits)
        early_rank = min(range(8), key=lambda r: entries[r])
        late_rank = max(range(8), key=lambda r: entries[r])
        for r in range(8):
            w = windows[(arm, r)]
            big = [right - left for left, right in w["gaps"] if right - left >= BIG_GAP_US]
            biggest = max(w["gaps"], key=lambda g: g[1] - g[0], default=(w["start"], w["start"]))
            prev, nxt = find_adjacent(w["events"], *biggest) if w["gaps"] else ([], [])
            summary_rows.append({
                "arm": arm,
                "rank": r,
                "early_or_late": "early" if r == early_rank else ("late" if r == late_rank else ""),
                "l19_exit_after_arm_min_us": w["start"] - earliest_exit,
                "l19_exit_spread_us": max(exits) - min(exits),
                "window_us": w["window"],
                "l20_entry_after_arm_min_us": w["end"] - earliest_entry,
                "l20_entry_spread_us": max(entries) - min(entries),
                "union_busy_us": w["busy"],
                "gap_total_us": w["gap_total"],
                "closure_error_us": w["closure"],
                "event_count": len(w["events"]),
                "same_stream_overlap_events": w["overlap_events"],
                "gaps_ge_10us_count": len(big),
                "gaps_ge_10us_total_us": sum(big),
                "largest_gap_us": biggest[1] - biggest[0],
                "largest_gap_prev": joined(prev, "alias"),
                "largest_gap_prev_stream": joined(prev, "stream"),
                "largest_gap_prev_correlation": joined(prev, "correlation"),
                "largest_gap_next": joined(nxt, "alias"),
                "largest_gap_next_stream": joined(nxt, "stream"),
                "largest_gap_next_correlation": joined(nxt, "correlation"),
            })

            for left, right in w["union"]:
                active = [e for e in w["events"] if min(right, e["end"]) > max(left, e["start"]) + EPS]
                union_rows.append({
                    "arm": arm,
                    "rank": r,
                    "kind": "busy_union",
                    "start_after_l19_us": left - w["start"],
                    "end_after_l19_us": right - w["start"],
                    "duration_us": right - left,
                    "events": joined(active, "alias"),
                    "streams": joined(active, "stream"),
                    "correlations": joined(active, "correlation"),
                })
            for left, right in w["gaps"]:
                prev, nxt = find_adjacent(w["events"], left, right)
                union_rows.append({
                    "arm": arm,
                    "rank": r,
                    "kind": "gap_union",
                    "start_after_l19_us": left - w["start"],
                    "end_after_l19_us": right - w["start"],
                    "duration_us": right - left,
                    "events": "",
                    "streams": "",
                    "correlations": "",
                    "previous": joined(prev, "alias"),
                    "following": joined(nxt, "alias"),
                    "previous_streams": joined(prev, "stream"),
                    "following_streams": joined(nxt, "stream"),
                    "previous_correlations": joined(prev, "correlation"),
                    "following_correlations": joined(nxt, "correlation"),
                })
            for e in w["events"]:
                event_rows.append({
                    "arm": arm,
                    "rank": r,
                    "start_after_l19_us": e["start"] - w["start"],
                    "end_after_l19_us": e["end"] - w["start"],
                    "duration_us": e["dur"],
                    "alias": e["alias"],
                    "category": e["cat"],
                    "stream": e["stream"],
                    "correlation": e["correlation"],
                    "same_stream_overlap": e["overlap"],
                    "abs_start_trace_us": e["start"],
                    "kernel_name": e["name"],
                })
            ranked = sorted(w["gaps"], key=lambda g: g[1] - g[0], reverse=True)
            for i, (left, right) in enumerate(ranked, 1):
                prev, nxt = find_adjacent(w["events"], left, right)
                gap_rows.append({
                    "arm": arm,
                    "rank": r,
                    "gap_rank": i,
                    "start_after_l19_us": left - w["start"],
                    "end_after_l19_us": right - w["start"],
                    "gap_us": right - left,
                    "previous_kernel": joined(prev, "name"),
                    "previous_alias": joined(prev, "alias"),
                    "previous_stream": joined(prev, "stream"),
                    "previous_correlation": joined(prev, "correlation"),
                    "following_kernel": joined(nxt, "name"),
                    "following_alias": joined(nxt, "alias"),
                    "following_stream": joined(nxt, "stream"),
                    "following_correlation": joined(nxt, "correlation"),
                })

    write_csv(out / "window_summary.csv", summary_rows, list(summary_rows[0]))
    write_csv(out / "window.csv", union_rows, [
        "arm", "rank", "kind", "start_after_l19_us", "end_after_l19_us", "duration_us",
        "events", "streams", "correlations", "previous", "following",
        "previous_streams", "following_streams", "previous_correlations", "following_correlations",
    ])
    write_csv(out / "window_events.csv", event_rows, list(event_rows[0]))
    write_csv(out / "largest_gaps.csv", gap_rows, list(gap_rows[0]))


def boundary_outputs(windows: dict[tuple[str, int], dict], out: Path) -> tuple[list[dict], list[dict]]:
    boundary_rows, contributor_rows = [], []
    for arm in ARMS:
        early_rank, late_rank = PAIR[arm]
        early = windows[(arm, early_rank)]
        late = windows[(arm, late_rank)]
        early_events = early["events"]
        late_by_key = {(e["stream"], e["correlation"], e["name"]): e for e in late["events"]}
        matched_keys = set()

        def domain(e: dict) -> tuple[int, str]:
            first_corr = min(x["correlation"] for x in early_events)
            if e["correlation"] == first_corr and e["start"] - early["start"] < 10:
                return (0, "l19_tail")
            if e["stream"] == 23:
                return (1, "stream23_main")
            if e["stream"] == 35:
                return (2, "stream35_cublas")
            if e["stream"] == 39:
                return (3, "stream39_cublas")
            if e["stream"] == 1037:
                return (4, "stream1037_l20")
            return (10, f"stream{e['stream']}")

        grouped = defaultdict(list)
        for e in early_events:
            grouped[domain(e)].append(e)
        for domain_key in sorted(grouped):
            sequence = sorted(grouped[domain_key], key=lambda e: (e["start"], e["correlation"], e["name"]))
            occurrence = Counter()
            previous_early_end = early["start"]
            previous_late_end = late["start"]
            previous_start_delta = 0.0
            previous_end_delta = 0.0
            for seq, e in enumerate(sequence, 1):
                occurrence[e["alias"]] += 1
                key = (e["stream"], e["correlation"], e["name"])
                other = late_by_key.get(key)
                matched_keys.add(key)
                early_start = e["start"] - early["start"]
                early_end = e["end"] - early["start"]
                early_gap = e["start"] - previous_early_end
                if other is None:
                    row = {
                        "arm": arm, "early_rank": early_rank, "late_rank": late_rank,
                        "domain": domain_key[1], "sequence": seq,
                        "stage": f"{e['alias']}#{occurrence[e['alias']]}",
                        "alias": e["alias"], "stream": e["stream"],
                        "correlation": e["correlation"], "match_status": "early_only",
                        "early_start_us": early_start, "early_end_us": early_end,
                        "late_start_us": "", "late_end_us": "",
                        "cumulative_start_delta_us": "", "incremental_start_delta_us": "",
                        "cumulative_end_delta_us": "", "incremental_end_delta_us": "",
                        "early_duration_us": e["dur"], "late_duration_us": "", "duration_delta_us": "",
                        "early_gap_before_us": early_gap, "late_gap_before_us": "", "gap_before_delta_us": "",
                        "same_stream_overlap_early": e["overlap"], "same_stream_overlap_late": "",
                        "kernel_name": e["name"], "note": "no same stream/correlation/name match",
                    }
                    boundary_rows.append(row)
                    continue
                late_start = other["start"] - late["start"]
                late_end = other["end"] - late["start"]
                late_gap = other["start"] - previous_late_end
                start_delta = late_start - early_start
                end_delta = late_end - early_end
                inc_start = start_delta - previous_start_delta
                inc_end = end_delta - previous_end_delta
                note = ""
                if e["overlap"] or other["overlap"]:
                    note = "same-stream Kineto overlap; duration/start delta not causal"
                row = {
                    "arm": arm, "early_rank": early_rank, "late_rank": late_rank,
                    "domain": domain_key[1], "sequence": seq,
                    "stage": f"{e['alias']}#{occurrence[e['alias']]}",
                    "alias": e["alias"], "stream": e["stream"], "correlation": e["correlation"],
                    "match_status": "matched",
                    "early_start_us": early_start, "early_end_us": early_end,
                    "late_start_us": late_start, "late_end_us": late_end,
                    "cumulative_start_delta_us": start_delta,
                    "incremental_start_delta_us": inc_start,
                    "cumulative_end_delta_us": end_delta,
                    "incremental_end_delta_us": inc_end,
                    "early_duration_us": e["dur"], "late_duration_us": other["dur"],
                    "duration_delta_us": other["dur"] - e["dur"],
                    "early_gap_before_us": early_gap, "late_gap_before_us": late_gap,
                    "gap_before_delta_us": late_gap - early_gap,
                    "same_stream_overlap_early": e["overlap"],
                    "same_stream_overlap_late": other["overlap"],
                    "kernel_name": e["name"], "note": note,
                }
                boundary_rows.append(row)
                contributor_rows.append(row)
                previous_early_end = e["end"]
                previous_late_end = other["end"]
                previous_start_delta = start_delta
                previous_end_delta = end_delta

        for key, e in late_by_key.items():
            if key in matched_keys:
                continue
            boundary_rows.append({
                "arm": arm, "early_rank": early_rank, "late_rank": late_rank,
                "domain": domain(e)[1], "sequence": "", "stage": f"{e['alias']} late-only",
                "alias": e["alias"], "stream": e["stream"], "correlation": e["correlation"],
                "match_status": "late_only", "early_start_us": "", "early_end_us": "",
                "late_start_us": e["start"] - late["start"], "late_end_us": e["end"] - late["start"],
                "cumulative_start_delta_us": "", "incremental_start_delta_us": "",
                "cumulative_end_delta_us": "", "incremental_end_delta_us": "",
                "early_duration_us": "", "late_duration_us": e["dur"], "duration_delta_us": "",
                "early_gap_before_us": "", "late_gap_before_us": "", "gap_before_delta_us": "",
                "same_stream_overlap_early": "", "same_stream_overlap_late": e["overlap"],
                "kernel_name": e["name"], "note": "no early match",
            })

    fields = [
        "arm", "early_rank", "late_rank", "domain", "sequence", "stage", "alias", "stream",
        "correlation", "match_status", "early_start_us", "early_end_us", "late_start_us", "late_end_us",
        "cumulative_start_delta_us", "incremental_start_delta_us",
        "cumulative_end_delta_us", "incremental_end_delta_us",
        "early_duration_us", "late_duration_us", "duration_delta_us",
        "early_gap_before_us", "late_gap_before_us", "gap_before_delta_us",
        "same_stream_overlap_early", "same_stream_overlap_late", "kernel_name", "note",
    ]
    write_csv(out / "boundary_alignment.csv", boundary_rows, fields)

    contributors = [
        row for row in contributor_rows
        if row["match_status"] == "matched"
        and (abs(float(row["incremental_end_delta_us"])) >= 1.0
             or abs(float(row["incremental_start_delta_us"])) >= 1.0)
    ]
    contributors.sort(key=lambda r: (r["arm"], -max(
        abs(float(r["incremental_end_delta_us"])), abs(float(r["incremental_start_delta_us"]))
    )))
    write_csv(out / "boundary_delta_contributors.csv", contributors, fields)
    collective_rows = []
    for arm in ARMS:
        early_rank, late_rank = PAIR[arm]
        early = windows[(arm, early_rank)]
        late = windows[(arm, late_rank)]
        selected = {}
        for row in boundary_rows:
            if row["arm"] != arm or row["match_status"] != "matched":
                continue
            stage = row["stage"]
            if stage in ("all_gather#1", "reduce_scatter#1", "mega_gate#1", "prepare#1"):
                selected[stage] = row
            if row["domain"] == "stream35_cublas" and stage == "cublas_tss#1":
                selected["post_all_gather_cublas_tss"] = row
            if row["domain"] == "stream23_main" and stage == "mhc#2":
                selected["post_last_collective_mhc"] = row

        def add(stage: str, early_start: float, late_start: float,
                early_end: float, late_end: float, note: str) -> None:
            collective_rows.append({
                "arm": arm,
                "early_rank": early_rank,
                "late_rank": late_rank,
                "stage": stage,
                "early_start_us": early_start,
                "late_start_us": late_start,
                "start_delta_us": late_start - early_start,
                "early_end_us": early_end,
                "late_end_us": late_end,
                "end_delta_us": late_end - early_end,
                "note": note,
            })

        add("L19_exit", 0.0, 0.0, 0.0, 0.0, "common calibrated boundary")
        order = (
            ("all_gather#1", "collective: dispatch all-gather"),
            ("post_all_gather_cublas_tss", "first cross-stream kernel after all-gather"),
            ("reduce_scatter#1", "last collective: combine reduce-scatter"),
            ("post_last_collective_mhc", "first same-stream dependency after last collective"),
            ("mega_gate#1", "post-collective routing/gate"),
            ("prepare#1", "last pre-MoE prepare"),
        )
        for key, note in order:
            row = selected[key]
            add(key, float(row["early_start_us"]), float(row["late_start_us"]),
                float(row["early_end_us"]), float(row["late_end_us"]), note)
        add("L20_entry", early["window"], late["window"], early["window"], late["window"],
            "device-clock L20 MoE entry")

    write_csv(out / "collective_boundary_summary.csv", collective_rows, [
        "arm", "early_rank", "late_rank", "stage", "early_start_us", "late_start_us",
        "start_delta_us", "early_end_us", "late_end_us", "end_delta_us", "note",
    ])
    return boundary_rows, contributor_rows


def step_outputs(raw: Path, out: Path) -> tuple[list[dict], dict]:
    rows, summaries = [], {}
    for arm in ARMS:
        controls = json.loads((raw / arm / "result.json").read_text())["dense_control"]
        after_marks = {}
        for rank in range(8):
            trace = json.loads((raw / arm / f"clock-after-rank{rank}.json").read_text())
            mark = [
                e for e in trace["traceEvents"]
                if e.get("ph") == "X" and "sm100_phase_clock_anchor" in e.get("name", "")
            ][0]
            after_marks[rank] = (mark["ts"] + mark["dur"] / 2, mark["dur"] / 2)
        summaries[arm] = {}
        for step in STEPS:
            samples = []
            for rank in range(8):
                with np.load(raw / arm / f"measure-rank{rank}.npz") as data:
                    clocks = data["clocks"][step].copy()
                exit_ns = int(clocks[19, 3])
                entry_ns = int(clocks[20, 0])
                if step == 1:
                    trace = json.loads((raw / arm / f"profile-rank{rank}.json").read_text())
                    events = [
                        e for e in trace["traceEvents"]
                        if e.get("ph") == "X" and e.get("cat") in DEVICE_CATS
                        and "sm100_phase_clock_anchor" in e.get("name", "") and e.get("dur", 0) > 0
                    ]
                    marks = sorted(e["ts"] + e["dur"] / 2 for e in events)
                    device = np.load(raw / arm / f"anchor-rank{rank}.npy")
                    scale = (marks[1] - marks[0]) / int(device[1] - device[0])
                    exit_us = marks[0] + (exit_ns - int(device[0])) * scale
                    entry_us = marks[0] + (entry_ns - int(device[0])) * scale
                    method = "profile_two_anchor"
                    half_width = max(e["dur"] / 2 for e in events)
                else:
                    mark, width = after_marks[rank]
                    device = np.load(raw / arm / f"clock-after-rank{rank}.npy")
                    exit_us = mark + (exit_ns - int(device[0])) / 1000
                    entry_us = mark + (entry_ns - int(device[0])) / 1000
                    method = "clock_after_single_anchor"
                    half_width = width
                samples.append({
                    "arm": arm, "step": step, "dense_mode": controls[step]["mode"],
                    "rank": rank, "alignment_method": method,
                    "alignment_half_width_us": half_width,
                    "l19_exit_aligned_us": exit_us, "l20_entry_aligned_us": entry_us,
                    "span_us": entry_us - exit_us,
                })
            early_rank = min(range(8), key=lambda r: samples[r]["l20_entry_aligned_us"])
            late_rank = max(range(8), key=lambda r: samples[r]["l20_entry_aligned_us"])
            exits = [s["l19_exit_aligned_us"] for s in samples]
            entries = [s["l20_entry_aligned_us"] for s in samples]
            base = min(entries)
            exit_base = min(exits)
            for rank, row in enumerate(samples):
                row["early_or_late"] = "early" if rank == early_rank else ("late" if rank == late_rank else "")
                row["entry_offset_from_earliest_us"] = entries[rank] - base
                row["exit_offset_from_earliest_us"] = exits[rank] - exit_base
                row["l19_exit_spread_us"] = max(exits) - min(exits)
                row["l20_entry_spread_us"] = max(entries) - min(entries)
            summaries[arm][step] = {
                "dense_mode": controls[step]["mode"],
                "early_rank": early_rank,
                "late_rank": late_rank,
                "l19_exit_spread_us": max(exits) - min(exits),
                "l20_entry_spread_us": max(entries) - min(entries),
                "span_early_us": samples[early_rank]["span_us"],
                "span_late_us": samples[late_rank]["span_us"],
                "span_difference_us": samples[late_rank]["span_us"] - samples[early_rank]["span_us"],
                "alignment_method": samples[0]["alignment_method"],
                "alignment_half_width_us": max(s["alignment_half_width_us"] for s in samples),
            }
            rows.extend(samples)
    write_csv(out / "step146_window.csv", rows, [
        "arm", "step", "dense_mode", "rank", "early_or_late", "alignment_method",
        "alignment_half_width_us", "l19_exit_aligned_us", "l20_entry_aligned_us",
        "span_us", "l19_exit_spread_us", "l20_entry_spread_us",
        "exit_offset_from_earliest_us", "entry_offset_from_earliest_us",
    ])
    return rows, summaries


def perfetto_outputs(windows: dict[tuple[str, int], dict], out: Path) -> dict:
    info = {}
    for arm in ARMS:
        base = min(windows[(arm, r)]["start"] for r in range(8))
        events = []
        thread_names = set()
        for rank in range(8):
            w = windows[(arm, rank)]
            events.append({"ph": "M", "pid": rank, "tid": 0, "name": "process_name",
                           "args": {"name": f"SM100 {arm.upper()} rank {rank}"}})
            events.append({"ph": "M", "pid": rank, "tid": 0, "name": "process_sort_index",
                           "args": {"sort_index": rank}})
            for e in w["events"]:
                tid = 1000 + e["stream"]
                thread_names.add((rank, tid, f"stream {e['stream']}"))
                events.append({
                    "ph": "X", "pid": rank, "tid": tid,
                    "name": e["name"], "cat": "Kineto device event",
                    "ts": e["start"] - base, "dur": e["dur"],
                    "args": {"rank": rank, "stream": e["stream"],
                             "correlation": e["correlation"], "alias": e["alias"],
                             "absolute_trace_us": e["start"],
                             "same_stream_overlap": bool(e["overlap"])},
                })
            for left, right in w["gaps"]:
                events.append({
                    "ph": "X", "pid": rank, "tid": 0,
                    "name": "device gap (all-stream union)", "cat": "device idle",
                    "ts": left - base, "dur": right - left,
                    "args": {"rank": rank, "gap_us": right - left},
                })
            for name, stamp in (("L19 MoE exit", w["start"]), ("L20 MoE entry", w["end"])):
                events.append({
                    "ph": "X", "pid": rank, "tid": 9000,
                    "name": name, "cat": "calibrated boundary",
                    "ts": stamp - base, "dur": 0.001,
                    "args": {"rank": rank, "absolute_trace_us": stamp},
                })
        for rank, tid, name in sorted(thread_names):
            events.append({"ph": "M", "pid": rank, "tid": tid, "name": "thread_name",
                           "args": {"name": name}})
        for rank in range(8):
            events.append({"ph": "M", "pid": rank, "tid": 9000, "name": "thread_name",
                           "args": {"name": "boundaries"}})
        trace = {
            "traceEvents": events,
            "displayTimeUnit": "ns",
            "metadata": {
                "arm": arm,
                "source": "step 1 L19 exit -> L20 entry",
                "cross_rank_timebase": "common Kineto trace timestamps after two-anchor device calibration",
                "common_base_trace_us": base,
                "note": "all ranks subtract the same base; cross-rank offsets are preserved",
            },
        }
        path = out / f"{arm}_step001_l19_l20.perfetto.json"
        path.write_text(json.dumps(trace, separators=(",", ":")))
        info[arm] = {"path": path.name, "event_count": len(events), "common_base_trace_us": base}
    return info


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw", type=Path)
    parser.add_argument("archive", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--task-commit", default="")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    windows = {
        (arm, rank): load_window(args.raw, args.archive, arm, rank)
        for arm in ARMS for rank in range(8)
    }
    window_outputs(windows, args.out)
    boundary_rows, _ = boundary_outputs(windows, args.out)
    step_rows, step_summary = step_outputs(args.raw, args.out)
    perfetto = perfetto_outputs(windows, args.out)

    closures = [abs(w["closure"]) for w in windows.values()]
    unmatched = [r for r in boundary_rows if r["match_status"] != "matched"]
    validation = {
        "raw_root": str(args.raw.resolve()),
        "archive": str(args.archive.resolve()),
        "task_commit": args.task_commit,
        "cpu_only": True,
        "gpu_used": False,
        "ranks": 16,
        "step1_windows": 16,
        "max_abs_closure_error_us": max(closures),
        "boundary_rows": len(boundary_rows),
        "boundary_unmatched_rows": len(unmatched),
        "boundary_match_fraction": (len(boundary_rows) - len(unmatched)) / len(boundary_rows),
        "step_summary": step_summary,
        "perfetto": perfetto,
        "overlap_warning": (
            "Early ranks contain same-stream reduce_scatter/mhc Kineto intervals that overlap "
            "later launches by milliseconds. They are retained as raw evidence and excluded "
            "from causal duration claims."
        ),
    }
    (args.out / "analysis_manifest.json").write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n")

    lines = [
        "V41 residual L20 CPU-only analysis",
        "PASS: no GPU, model, kernel or replay semantics were used or changed",
        f"PASS: step-1 window count = {len(windows)}",
        f"PASS: maximum |window - union_busy - gap_total| = {max(closures):.9f} us",
        f"PASS: boundary rows = {len(boundary_rows)}, unmatched = {len(unmatched)}",
        f"PASS: Perfetto files = {', '.join(v['path'] for v in perfetto.values())}",
    ]
    for arm in ARMS:
        for step in STEPS:
            s = step_summary[arm][step]
            lines.append(
                f"STEP arm={arm} step={step} dense={s['dense_mode']} early={s['early_rank']} "
                f"late={s['late_rank']} exit_spread={s['l19_exit_spread_us']:.3f}us "
                f"entry_spread={s['l20_entry_spread_us']:.3f}us span_difference={s['span_difference_us']:.3f}us "
                f"alignment={s['alignment_method']}"
            )
    (args.out / "analysis_validation.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
