#!/usr/bin/env python3
"""CPU/GPU timeline attribution for the V41 L19 -> L20 skew window.

Only reads the archived step-1 Kineto trace, the external device-clock anchors,
the anchor-only step 4/6 captures, and the optional perf mono-clock file.  No
model, kernel, input, or timing path is modified by this analysis.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ARMS = ("gc", "mega")
DEVICE_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}
CPU_CATS = {"cpu_op", "cuda_runtime", "cuda_driver"}
API_CATS = {"cuda_runtime", "cuda_driver"}
BIG_GAP_US = 10.0
EPS = 1e-6


def require(value, message: str) -> None:
    if not value:
        raise ValueError(message)


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
        ("qnormrope", "fusedDeepseekV4QNormRopeKV"),
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
        if needle.lower() in name.lower():
            return short
    return "other"


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    if not rows:
        path.write_text("")
        return
    fields = fields or list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
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
    previous = [e for e in events if abs(e["end"] - left) <= 1e-3]
    following = [e for e in events if abs(e["start"] - right) <= 1e-3]
    if not previous and events:
        upper = max(e["end"] for e in events if e["end"] <= left + 1e-3)
        previous = [e for e in events if abs(e["end"] - upper) <= 1e-3]
    if not following and events:
        lower = min(e["start"] for e in events if e["start"] >= right - 1e-3)
        following = [e for e in events if abs(e["start"] - lower) <= 1e-3]
    return previous, following


def joined(events: list[dict], field: str) -> str:
    return " | ".join(str(e.get(field, "")) for e in events)


def read_trace(folder: Path, rank: int) -> dict:
    return json.loads((folder / f"profile-rank{rank}.json").read_text())


def anchor_events(events: list[dict]) -> list[dict]:
    return sorted(
        (e for e in events if e.get("cat") == "kernel" and e.get("ph") == "X"
         and "sm100_phase_clock_anchor" in e.get("name", "") and e.get("dur", 0) > 0),
        key=lambda e: e["ts"],
    )


def device_mapping(folder: Path, rank: int, step: int) -> dict:
    """Return trace_us(device_globaltimer_ns) and its calibration metadata."""
    if step == 1:
        raw = read_trace(folder, rank)
        marks = anchor_events(raw["traceEvents"])
        require(len(marks) == 2, f"{folder} rank {rank}: expected two live anchors")
        anchors = np.load(folder / f"anchor-rank{rank}.npy").astype(np.int64)
        require(anchors[1] > anchors[0], f"{folder} rank {rank}: unordered anchors")
        x0, x1 = marks[0]["ts"] + marks[0]["dur"] / 2, marks[1]["ts"] + marks[1]["dur"] / 2
        scale = (x1 - x0) / int(anchors[1] - anchors[0])
        width = max(marks[0]["dur"], marks[1]["dur"]) / 2
        method = "two_live_anchors"
        before = after = None
    else:
        raw = json.loads((folder / f"clock-after-rank{rank}.json").read_text())
        marks = anchor_events(raw["traceEvents"])
        require(len(marks) == 1, f"{folder} rank {rank}: expected one after anchor")
        after = np.load(folder / f"clock-after-rank{rank}.npy").astype(np.int64)
        before = np.load(folder / f"clock-before-rank{rank}.npy").astype(np.int64)
        require(after.size == 1, f"{folder} rank {rank}: malformed after anchor")
        x0 = marks[0]["ts"] + marks[0]["dur"] / 2
        anchors = np.array([after[0], after[0] + 1], dtype=np.int64)
        scale = 1.0 / 1000.0
        width = marks[0]["dur"] / 2
        method = "single_after_anchor_fallback"
    with np.load(folder / f"measure-rank{rank}.npz") as data:
        clocks = data["clocks"].astype(np.int64)
    if step == 1:
        device_ref = anchors[0]
        def map_trace(t: np.ndarray | float) -> np.ndarray:
            return x0 + (np.asarray(t) - device_ref) * scale
    else:
        # Device globaltimer is ns; Kineto is us around the sole valid anchor.
        def map_trace(t: np.ndarray | float) -> np.ndarray:
            return x0 + (np.asarray(t) - anchors[0]) / 1000.0
    return dict(raw=raw, clocks=clocks, map=map_trace, scale=scale,
                marks=marks, anchors=anchors, width=width, method=method,
                before=before, after=after)


def fit_host_to_trace(mapping: dict, rank: int, step: int = 1) -> dict:
    """Fit trace_ns = slope * monotonic_ns + offset on MoE graph launches."""
    raw = mapping["raw"]
    events = raw["traceEvents"]
    clocks = mapping["clocks"][step]
    device = [e for e in events if e.get("ph") == "X" and e.get("cat") in DEVICE_CATS]
    api = sorted(
        (e for e in events if e.get("ph") == "X" and e.get("cat") in API_CATS
         and e.get("name", "").startswith("cudaGraphLaunch")),
        key=lambda e: e["ts"],
    )
    with np.load(Path(raw["_folder"]) / f"measure-rank{rank}.npz") as data:
        host = data["host_submit_mono_ns"].astype(np.int64)
    require(host.shape == (7, 40), f"{raw.get('_folder')} rank {rank}: host shape")
    require(not np.any(host[step] <= 0), f"{raw.get('_folder')} rank {rank}: missing host stamps")
    launches, launch_indices = [], []
    mapped_entries = mapping["map"](clocks[:, 0])
    mapped_ends = mapping["map"](clocks[:, 3])
    for layer in range(40):
        members = [e for e in device if mapped_entries[layer] - 5 <= e["ts"] <= mapped_ends[layer] + 5
                   and alias(e.get("name", "")) in ("mega_moe", "gc_moe")]
        if not members:
            raise ValueError(f"rank {rank}: missing MoE kernel at layer {layer}")
        correlation = members[0].get("args", {}).get("correlation")
        candidates = [e for e in api
                      if e.get("args", {}).get("correlation") == correlation
                      and e["ts"] <= mapped_entries[layer] + 5]
        if not candidates:
            candidates = [e for e in api if mapped_entries[layer] - 15000 <= e["ts"] <= mapped_entries[layer] + 5]
        require(candidates, f"rank {rank}: missing graph launch for layer {layer}")
        launch = max(candidates, key=lambda e: e["ts"])
        launches.append(launch)
        launch_indices.append(api.index(launch))
    x = np.array([host[step, i] for i in range(40)], dtype=np.float64)
    y = np.array([e["ts"] * 1000.0 for e in launches], dtype=np.float64)
    slope, offset = np.polyfit(x, y, 1)
    predicted = slope * x + offset
    residual_us = (y - predicted) / 1000.0
    require(slope > 0.99 and slope < 1.01, f"rank {rank}: implausible host slope {slope}")
    return dict(slope=float(slope), offset=float(offset), residual_us=residual_us,
                launches=launches, launch_indices=launch_indices,
                host=host[step], trace_ns=y,
                mono_to_trace=lambda t: slope * np.asarray(t) + offset,
                trace_to_mono=lambda t: (np.asarray(t) * 1000.0 - offset) / slope)


def device_events_in_window(mapping: dict, step: int, start_layer: int = 19,
                            end_layer: int = 20) -> tuple[float, float, list[dict]]:
    clocks = mapping["clocks"][step]
    start = float(mapping["map"](clocks[start_layer, 3]))
    end = float(mapping["map"](clocks[end_layer, 0]))
    events = []
    for e in mapping["raw"]["traceEvents"]:
        if e.get("ph") != "X" or e.get("cat") not in DEVICE_CATS:
            continue
        left, right = max(start, e["ts"]), min(end, e["ts"] + e["dur"])
        if right <= left:
            continue
        args = e.get("args", {})
        events.append(dict(
            start=left, end=right, raw_start=e["ts"], raw_end=e["ts"] + e["dur"],
            dur=right - left, name=e.get("name", ""), alias=alias(e.get("name", "")),
            cat=e.get("cat", ""), stream=int(args.get("stream", -1)),
            correlation=args.get("correlation", ""),
            graph_id=args.get("graph id", 0), graph_node=args.get("graph node id", 0),
            overlap=0,
        ))
    events.sort(key=lambda e: (e["start"], e["stream"], str(e["correlation"])))
    by_stream = defaultdict(list)
    for event in events:
        by_stream[event["stream"]].append(event)
    for stream_events in by_stream.values():
        active_end = None
        for event in stream_events:
            if active_end is not None and event["start"] < active_end - 0.5:
                event["overlap"] = 1
            active_end = event["end"] if active_end is None else max(active_end, event["end"])
    return start, end, events


def enclosing_cpu(event: dict | None, cpu_ops: list[dict]) -> tuple[str, float, str]:
    if event is None:
        return "", 0.0, ""
    pid, tid = event.get("pid"), event.get("tid")
    mid = event["ts"] + event.get("dur", 0) / 2
    candidates = [op for op in cpu_ops if op.get("pid") == pid and op.get("tid") == tid
                  and op["ts"] <= mid <= op["ts"] + op.get("dur", 0)]
    if not candidates:
        return "", 0.0, ""
    op = min(candidates, key=lambda item: item.get("dur", 0))
    return op.get("name", ""), float(op.get("dur", 0)), op.get("cat", "")


def nearest_cpu(event: dict | None, cpu_ops: list[dict]) -> tuple[str, str]:
    if event is None:
        return "", ""
    pid, tid = event.get("pid"), event.get("tid")
    same = [op for op in cpu_ops if op.get("pid") == pid and op.get("tid") == tid]
    before = [op for op in same if op["ts"] + op.get("dur", 0) <= event["ts"] + EPS]
    after = [op for op in same if op["ts"] >= event["ts"] + event.get("dur", 0) - EPS]
    return (max(before, key=lambda op: op["ts"] + op.get("dur", 0)).get("name", "") if before else "",
            min(after, key=lambda op: op["ts"]).get("name", "") if after else "")


def api_for_device(next_event: dict, api: list[dict]) -> dict | None:
    corr = next_event.get("correlation", "")
    if corr == "":
        return None
    matches = [e for e in api if e.get("args", {}).get("correlation") == corr
               and e["ts"] <= next_event["raw_start"] + 1.0]
    return max(matches, key=lambda e: e["ts"]) if matches else None


def host_gap_evidence(gap: tuple[float, float], next_event: dict | None,
                      all_cpu: list[dict], mapping: dict, fit: dict) -> dict:
    g0, g1 = gap
    cpu_api = [e for e in all_cpu if e.get("cat") in API_CATS]
    cpu_ops = [e for e in all_cpu if e.get("cat") == "cpu_op"]
    api = api_for_device(next_event, cpu_api) if next_event else None
    pairs = Counter((e.get("pid"), e.get("tid")) for e in all_cpu)
    pair = (api.get("pid"), api.get("tid")) if api else (
        next(iter(pairs)) if len(pairs) == 1 else None)
    cpu_intervals = []
    if pair is not None:
        for item in all_cpu:
            if (item.get("pid"), item.get("tid")) != pair:
                continue
            left, right = max(g0, item["ts"]), min(g1, item["ts"] + item.get("dur", 0))
            if right > left:
                cpu_intervals.append((left, right))
    cpu_union = merge_intervals(cpu_intervals)
    cpu_busy = sum(right - left for left, right in cpu_union)
    cpu_gap = max(0.0, (g1 - g0) - cpu_busy)
    parent, parent_dur, parent_cat = enclosing_cpu(api, cpu_ops)
    previous_op, following_op = nearest_cpu(api or (next_event and {
        "pid": next_event.get("pid"), "tid": next_event.get("tid"),
        "ts": next_event.get("raw_start", g1), "dur": 0,
    }), cpu_ops)
    if api is None:
        return dict(api_name="", api_start_us="", api_dur_us="", api_end_us="",
                    host_not_entered_us="", api_exec_us="", api_to_device_us="",
                    api_relation="unmatched", enclosing_cpu_op=parent,
                    enclosing_cpu_duration_us=parent_dur, previous_cpu_op=previous_op,
                    following_cpu_op=following_op, api_pid="", api_tid="",
                    api_correlation="", cpu_busy_us=cpu_busy, cpu_gap_us=cpu_gap)
    api_start = api["ts"]
    api_end = api["ts"] + api.get("dur", 0)
    not_entered = max(0.0, api_start - g0)
    exec_us = max(0.0, min(g1, api_end) - max(g0, api_start))
    returned_wait = max(0.0, g1 - max(g0, api_end))
    if api_end <= g0 + 1.0:
        relation = "submitted_before_gap"
    elif api_start >= g1 - 1.0:
        relation = "api_after_gap"
    elif api_start >= g0 - 1.0 and api_end <= g1 + 1.0:
        relation = "inside_gap"
    else:
        relation = "overlaps_gap"
    return dict(api_name=api.get("name", ""), api_start_us=api_start - g0,
                api_dur_us=api.get("dur", 0), api_end_us=api_end - g0,
                host_not_entered_us=not_entered, api_exec_us=exec_us,
                api_to_device_us=returned_wait, api_relation=relation,
                enclosing_cpu_op=parent, enclosing_cpu_duration_us=parent_dur,
                previous_cpu_op=previous_op, following_cpu_op=following_op,
                api_pid=api.get("pid", ""), api_tid=api.get("tid", ""),
                api_correlation=api.get("args", {}).get("correlation", ""),
                cpu_busy_us=cpu_busy, cpu_gap_us=cpu_gap)


def build_rank(folder: Path, arm: str, rank: int, step: int = 1) -> dict:
    mapping = device_mapping(folder, rank, step)
    mapping["raw"]["_folder"] = str(folder)
    clocks = mapping["clocks"][step]
    start, end, events = device_events_in_window(mapping, step)
    union = merge_intervals([(e["start"], e["end"]) for e in events])
    gaps = complement(start, end, union)
    busy = sum(right - left for left, right in union)
    gap_total = sum(right - left for left, right in gaps)
    fit = fit_host_to_trace(mapping, rank, step) if step == 1 else None
    return dict(arm=arm, rank=rank, step=step, folder=folder, mapping=mapping,
                clocks=clocks, start=start, end=end, events=events, union=union,
                gaps=gaps, busy=busy, gap_total=gap_total,
                closure=(end - start) - busy - gap_total, fit=fit)


def boundary_rows_for_arm(arm: str, ranks: dict[int, dict], out: list[dict]) -> None:
    entries = np.array([ranks[r]["end"] for r in range(8)])
    exits = np.array([ranks[r]["start"] for r in range(8)])
    early = int(entries.argmin())
    late = int(entries.argmax())
    for rank in range(8):
        data = ranks[rank]
        clocks = data["clocks"]
        exchange_start = float(data["mapping"]["map"](clocks[20, 4]))
        exchange_end = float(data["mapping"]["map"](clocks[20, 5]))
        out.append(dict(
            arm=arm, rank=rank, early=rank == early, late=rank == late,
            l19_exit_us=data["start"], l20_entry_us=data["end"],
            l19_exit_spread_us=float(np.ptp(exits)),
            l20_entry_spread_us=float(np.ptp(entries)),
            entry_after_min_us=data["end"] - entries.min(),
            exit_after_min_us=data["start"] - exits.min(),
            window_us=data["end"] - data["start"], union_busy_us=data["busy"],
            gap_total_us=data["gap_total"], closure_error_us=data["closure"],
            count_exchange_start_us=exchange_start,
            count_exchange_end_us=exchange_end,
            count_exchange_duration_us=exchange_end - exchange_start,
            count_exchange_after_entry_us=exchange_start - data["end"],
        ))


def event_match_key(event: dict, occurrence: dict) -> tuple:
    occurrence[event["alias"]] += 1
    return (event["alias"], event["stream"], event["cat"], occurrence[event["alias"]])


def match_boundaries(early: dict, late: dict) -> tuple[list[dict], list[dict]]:
    early_occ, late_occ = defaultdict(int), defaultdict(int)
    early_keys = {}
    for event in early["events"]:
        key = event_match_key(event, early_occ)
        early_keys[key] = event
    rows, unmatched = [], []
    previous_early_end = early["start"]
    previous_late_end = late["start"]
    previous_end_delta = 0.0
    for event in late["events"]:
        key = event_match_key(event, late_occ)
        other = early_keys.get(key)
        if other is None:
            unmatched.append(dict(arm=late["arm"], rank=late["rank"], side="late_only",
                                  alias=event["alias"], stream=event["stream"],
                                  correlation=event["correlation"], name=event["name"]))
        else:
            early_start = other["start"] - early["start"]
            early_end = other["end"] - early["start"]
            late_start = event["start"] - late["start"]
            late_end = event["end"] - late["start"]
            start_delta = late_start - early_start
            end_delta = late_end - early_end
            rows.append(dict(
                arm=late["arm"], early_rank=early["rank"], late_rank=late["rank"],
                sequence=len(rows) + 1, alias=event["alias"], stream=event["stream"],
                correlation=event["correlation"], match_status="matched",
                early_start_us=early_start, early_end_us=early_end,
                late_start_us=late_start, late_end_us=late_end,
                cumulative_start_delta_us=start_delta, cumulative_end_delta_us=end_delta,
                incremental_end_delta_us=end_delta - previous_end_delta,
                early_duration_us=other["dur"], late_duration_us=event["dur"],
                duration_delta_us=event["dur"] - other["dur"],
                early_gap_before_us=other["start"] - previous_early_end,
                late_gap_before_us=event["start"] - previous_late_end,
                gap_before_delta_us=(event["start"] - previous_late_end) -
                                    (other["start"] - previous_early_end),
                same_stream_overlap_early=other["overlap"],
                same_stream_overlap_late=event["overlap"],
                kernel_name=event["name"], note="",
            ))
            previous_early_end, previous_late_end = other["end"], event["end"]
            previous_end_delta = end_delta
    return rows, unmatched


def perf_gap_summary(perf: dict | None, data: dict, gap: tuple[float, float],
                     api_pid: int | None) -> dict:
    blank = dict(perf_samples=0, perf_running_us=0.0, perf_runnable_us=0.0,
                 perf_blocked_us=0.0, perf_covered=False)
    fit = data.get("fit")
    if not perf or not perf.get("present") or fit is None or not api_pid:
        return blank
    mono = fit["trace_to_mono"](np.array([gap[0], gap[1]]))
    intervals = schedule_intervals(perf, int(api_pid), float(mono[0]), float(mono[1]))
    totals = defaultdict(float)
    for left, right, state in intervals:
        totals[state] += (right - left) / 1000.0
    samples = sum(1 for stamp, _ in perf["samples"].get(int(api_pid), [])
                  if mono[0] <= stamp <= mono[1])
    return dict(perf_samples=samples,
                perf_running_us=totals["running"],
                perf_runnable_us=totals["runnable"],
                perf_blocked_us=totals["blocked"],
                perf_covered=bool(intervals or samples))


def gap_rows_for_arm(arm: str, ranks: dict[int, dict], perf: dict | None = None) -> list[dict]:
    entries = {r: ranks[r]["end"] for r in range(8)}
    early_rank = min(entries, key=entries.get)
    late_rank = max(entries, key=entries.get)
    early, late = ranks[early_rank], ranks[late_rank]
    def event_table(data: dict) -> tuple[dict, list[tuple[tuple, dict]]]:
        table, occurrence, sequence = {}, defaultdict(int), []
        for event in data["events"]:
            occurrence[event["alias"]] += 1
            key = (event["alias"], event["stream"], event["cat"],
                   occurrence[event["alias"]])
            sequence.append((key, event))
            table[key] = event
        return table, sequence

    early_events, early_sequence = event_table(early)
    late_events, _ = event_table(late)

    def preceding_gap(data: dict, event: dict) -> dict:
        candidates = [gap for gap in data["gaps"] if gap[1] <= event["start"] + 1e-3]
        if not candidates:
            return dict(gap=(event["start"], event["start"]), prev=[],
                        next=[event])
        gap = max(candidates, key=lambda item: item[1])
        if event["start"] - gap[1] > 0.5:
            # The event starts inside an already-active union interval; there
            # is no separate idle interval immediately before this event.
            return dict(gap=(event["start"], event["start"]), prev=[],
                        next=[event])
        prev, nxt = find_adjacent(data["events"], *gap)
        return dict(gap=gap, prev=prev, next=nxt or [event])

    early_gaps = {key: preceding_gap(early, event) for key, event in early_sequence}
    rows = []
    used = set()
    for index, gap in enumerate(late["gaps"]):
        prev, nxt = find_adjacent(late["events"], *gap)
        occurrence = defaultdict(int)
        for item in late["events"]:
            if item["start"] > gap[1] + 1e-3:
                break
            occurrence[item["alias"]] += 1
        key = None
        if nxt:
            key = (nxt[0]["alias"], nxt[0]["stream"], nxt[0]["cat"],
                   occurrence[nxt[0]["alias"]])
        matched = early_gaps.get(key) if key is not None else None
        next_event = nxt[0] if nxt else None
        early_next = matched["next"][0] if matched and matched["next"] else None
        ev = host_gap_evidence(gap, next_event, late["folder"] and
                               [e for e in late["mapping"]["raw"]["traceEvents"]
                                if e.get("ph") == "X" and e.get("cat") in CPU_CATS],
                               late["mapping"], late["fit"])
        early_ev = host_gap_evidence(matched["gap"], early_next,
                                     [e for e in early["mapping"]["raw"]["traceEvents"]
                                      if e.get("ph") == "X" and e.get("cat") in CPU_CATS],
                                     early["mapping"], early["fit"]) if matched else {}
        late_us = gap[1] - gap[0]
        early_us = (matched["gap"][1] - matched["gap"][0]) if matched else None
        late_sched = perf_gap_summary(perf, late, gap,
                                      int(ev.get("api_pid")) if ev.get("api_pid") else None)
        early_sched = perf_gap_summary(perf, early, matched["gap"],
                                       int(early_ev.get("api_pid")) if early_ev.get("api_pid") else None) if matched else {}
        rows.append(dict(
            arm=arm, late_rank=late_rank, early_rank=early_rank,
            late_gap_index=index, late_gap_start_after_exit_us=gap[0] - late["start"],
            late_gap_us=late_us, early_gap_us=early_us,
            gap_delta_us=late_us - early_us if early_us is not None else "",
            previous_alias=joined(prev, "alias"), previous_stream=joined(prev, "stream"),
            previous_correlation=joined(prev, "correlation"),
            next_alias=joined(nxt, "alias"), next_stream=joined(nxt, "stream"),
            next_correlation=joined(nxt, "correlation"),
            next_graph_id=next_event.get("graph_id", "") if next_event else "",
            next_graph_node=next_event.get("graph_node", "") if next_event else "",
            next_is_graph_node=bool(next_event and next_event.get("graph_id", 0)),
            match_status="matched" if matched else "unmatched",
            late_api=ev.get("api_name", ""), late_api_relation=ev.get("api_relation", ""),
            late_api_pid=ev.get("api_pid", ""), late_api_tid=ev.get("api_tid", ""),
            late_api_correlation=ev.get("api_correlation", ""),
            late_api_duration_us=ev.get("api_dur_us", ""),
            late_enclosing_cpu_duration_us=ev.get("enclosing_cpu_duration_us", ""),
            late_cpu_busy_us=ev.get("cpu_busy_us", ""),
            late_cpu_gap_us=ev.get("cpu_gap_us", ""),
            late_perf_samples=late_sched["perf_samples"],
            late_perf_running_us=late_sched["perf_running_us"],
            late_perf_runnable_us=late_sched["perf_runnable_us"],
            late_perf_blocked_us=late_sched["perf_blocked_us"],
            late_perf_covered=late_sched["perf_covered"],
            late_host_not_entered_us=ev.get("host_not_entered_us", ""),
            late_api_exec_us=ev.get("api_exec_us", ""),
            late_api_to_device_us=ev.get("api_to_device_us", ""),
            late_enclosing_cpu_op=ev.get("enclosing_cpu_op", ""),
            late_previous_cpu_op=ev.get("previous_cpu_op", ""),
            late_following_cpu_op=ev.get("following_cpu_op", ""),
            early_api=early_ev.get("api_name", ""), early_api_relation=early_ev.get("api_relation", ""),
            early_api_pid=early_ev.get("api_pid", ""), early_api_tid=early_ev.get("api_tid", ""),
            early_api_correlation=early_ev.get("api_correlation", ""),
            early_api_duration_us=early_ev.get("api_dur_us", ""),
            early_enclosing_cpu_duration_us=early_ev.get("enclosing_cpu_duration_us", ""),
            early_cpu_busy_us=early_ev.get("cpu_busy_us", ""),
            early_cpu_gap_us=early_ev.get("cpu_gap_us", ""),
            early_perf_samples=early_sched.get("perf_samples", ""),
            early_perf_running_us=early_sched.get("perf_running_us", ""),
            early_perf_runnable_us=early_sched.get("perf_runnable_us", ""),
            early_perf_blocked_us=early_sched.get("perf_blocked_us", ""),
            early_perf_covered=early_sched.get("perf_covered", ""),
            early_host_not_entered_us=early_ev.get("host_not_entered_us", ""),
            early_api_exec_us=early_ev.get("api_exec_us", ""),
            early_api_to_device_us=early_ev.get("api_to_device_us", ""),
            early_enclosing_cpu_op=early_ev.get("enclosing_cpu_op", ""),
            sort_key=gap[0] - late["start"],
        ))
    return sorted(rows, key=lambda row: row["sort_key"])


def exchange_answer(arm: str, ranks: dict[int, dict]) -> dict:
    entries = {r: ranks[r]["end"] for r in range(8)}
    early_rank = min(entries, key=entries.get)
    late_rank = max(entries, key=entries.get)
    early = ranks[early_rank]
    c = early["clocks"][20]
    ex_start = float(early["mapping"]["map"](c[4]))
    ex_end = float(early["mapping"]["map"](c[5]))
    late_entry = ranks[late_rank]["end"]
    overlap = max(0.0, min(ex_end, late_entry) - ex_start)
    return dict(arm=arm, early_rank=early_rank, late_rank=late_rank,
                early_entry_us=entries[early_rank], late_entry_us=late_entry,
                early_entry_spread_us=late_entry - entries[early_rank],
                exchange_start_us=ex_start, exchange_end_us=ex_end,
                exchange_duration_us=ex_end - ex_start,
                exchange_before_late_entry_us=overlap,
                exchange_after_late_entry_us=max(0.0, ex_end - max(ex_start, late_entry)))


def step146_rows(folder: Path) -> list[dict]:
    rows = []
    for step in (1, 4, 6):
        mapping_by_rank = {r: device_mapping(folder, r, step) for r in range(8)}
        entries = np.array([float(mapping_by_rank[r]["map"](mapping_by_rank[r]["clocks"][step, 20, 0]))
                            for r in range(8)])
        exits = np.array([float(mapping_by_rank[r]["map"](mapping_by_rank[r]["clocks"][step, 19, 3]))
                          for r in range(8)])
        exchanges = np.array([
            float(mapping_by_rank[r]["map"](mapping_by_rank[r]["clocks"][step, 20, 5]) -
                  mapping_by_rank[r]["map"](mapping_by_rank[r]["clocks"][step, 20, 4]))
            for r in range(8)])
        early = int(entries.argmin())
        late = int(entries.argmax())
        rows.append(dict(
            arm=folder.name, step=step, early_rank=early, late_rank=late,
            l19_exit_spread_us=float(np.ptp(exits)), l20_entry_spread_us=float(np.ptp(entries)),
            early_entry_us=float(entries[early]), late_entry_us=float(entries[late]),
            early_exchange_us=float(exchanges[early]), late_exchange_us=float(exchanges[late]),
            max_exchange_us=float(exchanges.max()),
            clock_method="two_live_anchors" if step == 1 else "single_after_anchor_fallback",
            alignment_half_width_us=max(float(mapping_by_rank[r]["width"]) for r in range(8)),
        ))
    return rows


def key_call_rows(arm: str, ranks: dict[int, dict], gaps: list[dict]) -> list[dict]:
    entries = {r: ranks[r]["end"] for r in range(8)}
    early_rank = min(entries, key=entries.get)
    late_rank = max(entries, key=entries.get)
    late = ranks[late_rank]
    rows = []
    candidates = [g for g in gaps if g["gap_delta_us"] != "" and float(g["gap_delta_us"]) >= BIG_GAP_US]
    if candidates:
        gap = candidates[0]
        rows.append(dict(
            arm=arm, question="first_extra_wait_after_L19", early_rank=early_rank,
            late_rank=late_rank, gap_start_after_exit_us=gap["late_gap_start_after_exit_us"],
            gap_us=gap["late_gap_us"], early_gap_us=gap["early_gap_us"],
            delta_us=gap["gap_delta_us"], previous=gap["previous_alias"],
            following=gap["next_alias"], api=gap["late_api"],
            api_relation=gap["late_api_relation"],
            enclosing_cpu_op=gap["late_enclosing_cpu_op"],
            host_not_entered_us=gap["late_host_not_entered_us"],
            api_to_device_us=gap["late_api_to_device_us"],
            perf_covered=gap["late_perf_covered"],
            perf_running_us=gap["late_perf_running_us"],
            perf_runnable_us=gap["late_perf_runnable_us"],
            perf_blocked_us=gap["late_perf_blocked_us"],
        ))
    rs = [e for e in late["events"] if e["alias"] == "reduce_scatter"]
    rs_end = max((e["end"] for e in rs), default=late["start"])
    after = [g for g in gaps if g["gap_delta_us"] != "" and
             g["late_gap_start_after_exit_us"] + late["start"] >= rs_end - 1.0 and
             float(g["gap_delta_us"]) > 0.0]
    for gap in sorted(after, key=lambda g: float(g["gap_delta_us"]), reverse=True)[:8]:
        rows.append(dict(
            arm=arm, question="after_last_reduce_scatter", early_rank=early_rank,
            late_rank=late_rank, gap_start_after_exit_us=gap["late_gap_start_after_exit_us"],
            gap_us=gap["late_gap_us"], early_gap_us=gap["early_gap_us"],
            delta_us=gap["gap_delta_us"], previous=gap["previous_alias"],
            following=gap["next_alias"], api=gap["late_api"],
            api_relation=gap["late_api_relation"],
            enclosing_cpu_op=gap["late_enclosing_cpu_op"],
            host_not_entered_us=gap["late_host_not_entered_us"],
            api_to_device_us=gap["late_api_to_device_us"],
            perf_covered=gap["late_perf_covered"],
            perf_running_us=gap["late_perf_running_us"],
            perf_runnable_us=gap["late_perf_runnable_us"],
            perf_blocked_us=gap["late_perf_blocked_us"],
        ))
    return rows


def closure_rows(arm: str, ranks: dict[int, dict], gaps: list[dict]) -> list[dict]:
    entries = {r: ranks[r]["end"] for r in range(8)}
    early_rank = min(entries, key=entries.get)
    late_rank = max(entries, key=entries.get)
    early, late = ranks[early_rank], ranks[late_rank]
    window_delta = (late["end"] - late["start"]) - (early["end"] - early["start"])
    busy_delta = late["busy"] - early["busy"]
    gap_delta = late["gap_total"] - early["gap_total"]
    matched = [g for g in gaps if g["gap_delta_us"] != ""]
    pos = sum(max(0.0, float(g["gap_delta_us"])) for g in matched)
    neg = sum(min(0.0, float(g["gap_delta_us"])) for g in matched)
    return [dict(
        arm=arm, early_rank=early_rank, late_rank=late_rank,
        l19_exit_spread_us=float(np.ptp([ranks[r]["start"] for r in range(8)])),
        l20_entry_spread_us=float(np.ptp(list(entries.values()))), window_delta_us=window_delta,
        union_busy_delta_us=busy_delta, gap_total_delta_us=gap_delta,
        positive_gap_call_delta_us=pos, negative_gap_call_delta_us=neg,
        call_gap_delta_sum_us=pos + neg,
        busy_plus_gap_delta_us=busy_delta + gap_delta,
        closure_error_us=window_delta - busy_delta - gap_delta,
        matched_gap_count=len(matched), unmatched_gap_count=len(gaps) - len(matched),
    )]


def compare_step146(new_rows: list[dict], baseline: Path) -> list[dict]:
    if not baseline.is_file():
        return []
    old: dict[tuple[str, int], dict] = {}
    with baseline.open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["arm"], int(row["step"]))
            item = old.setdefault(key, dict(
                arm=row["arm"], step=int(row["step"]),
                entry_spread_us=float(row["l20_entry_spread_us"]),
                exit_spread_us=float(row["l19_exit_spread_us"]),
                early_rank="", late_rank="",
            ))
            if row.get("early_or_late") == "early":
                item["early_rank"] = int(row["rank"])
            if row.get("early_or_late") == "late":
                item["late_rank"] = int(row["rank"])
    result = []
    for row in new_rows:
        key = (row["arm"], int(row["step"]))
        prior = old.get(key)
        if not prior:
            continue
        result.append(dict(
            arm=row["arm"], step=row["step"],
            current_entry_spread_us=row["l20_entry_spread_us"],
            baseline_entry_spread_us=prior["entry_spread_us"],
            delta_entry_spread_us=row["l20_entry_spread_us"] - prior["entry_spread_us"],
            current_exit_spread_us=row["l19_exit_spread_us"],
            baseline_exit_spread_us=prior["exit_spread_us"],
            delta_exit_spread_us=row["l19_exit_spread_us"] - prior["exit_spread_us"],
            current_early_late=f"{row['early_rank']}/{row['late_rank']}",
            baseline_early_late=f"{prior['early_rank']}/{prior['late_rank']}",
        ))
    return result


def read_perf_for_ranks(path: Path, pids: set[int]) -> dict:
    """Parse ``perf script`` text in either pid/tid or single-task format.

    Default output on this host is ``comm pid timestamp: event`` plus an
    indented user stack for cpu-clock samples.  A second supported form is
    ``comm pid/tid [cpu] timestamp: event``.  With ``--show-switch-events`` the
    switch records are per task: SWITCH IN means the task starts running;
    SWITCH OUT means it leaves the CPU, and the ``preempt`` suffix means the
    task stayed runnable.  This is enough to reconstruct running, runnable and
    blocked intervals without inventing prev/next PIDs.
    """
    samples = defaultdict(list)
    switches = defaultdict(list)
    tids = defaultdict(set)
    if not path.is_file():
        return dict(samples=samples, switches=switches, tids=tids,
                    present=False, lines=0)
    pair_re = re.compile(r"^\s*\S.*?\s+(\d+)/(\d+)\s+(?:\[\d+\]\s+)?([0-9.]+):\s+(.*)$")
    single_re = re.compile(r"^\s*\S.*?\s+(\d+)\s+(?:\[\d+\]\s+)?([0-9.]+):\s+(.*)$")
    last_sample = None
    with path.open(errors="replace") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            if not line.strip():
                last_sample = None
                continue
            match = pair_re.match(line)
            if match:
                pid_s, tid_s, stamp, rest = match.groups()
                pid, tid = int(pid_s), int(tid_s)
                if pid in pids:
                    tids[pid].add(tid)
            else:
                match = single_re.match(line)
                if not match:
                    if last_sample is not None and (line.startswith("\t") or line.startswith("    ")):
                        pid, index = last_sample
                        ns, stack = samples[pid][index]
                        samples[pid][index] = (ns, (stack + " <- " + line.strip())[:1200])
                    continue
                pid_s, stamp, rest = match.groups()
                pid, tid = int(pid_s), int(pid_s)
                if pid in pids:
                    tids[pid].add(tid)
            ns = int(round(float(stamp) * 1e9))
            if "PERF_RECORD_SWITCH" in rest:
                last_sample = None
                if pid not in pids:
                    continue
                if " OUT" in rest:
                    state = "R" if "preempt" in rest else "S"
                    switches[pid].append((ns, pid, state, 0))
                else:
                    switches[pid].append((ns, 0, "", pid))
            elif "cpu-clock" in rest or "cycles" in rest:
                if pid not in pids:
                    last_sample = None
                    continue
                samples[pid].append((ns, rest.strip()))
                last_sample = (pid, len(samples[pid]) - 1)
            else:
                last_sample = None
    return dict(samples=samples, switches=switches, tids=tids, present=True,
                lines=sum(len(v) for v in samples.values()))


def read_tid_coverage(path: Path | None, pids: set[int]) -> dict:
    """Count samples and switch records by (PID, TID) in perf field output."""
    coverage = defaultdict(lambda: defaultdict(lambda: dict(samples=0, switches=0)))
    if not path or not path.is_file():
        return coverage
    pattern = re.compile(r"^\s*\S.*?\s+(\d+)/(\d+)\s+(?:\[\d+\]\s+)?([0-9.]+):\s+(.*)$")
    with path.open(errors="replace") as handle:
        for line in handle:
            match = pattern.match(line)
            if not match:
                continue
            pid, tid = int(match.group(1)), int(match.group(2))
            if pid not in pids:
                continue
            rest = match.group(4)
            if "PERF_RECORD_SWITCH" in rest:
                coverage[pid][tid]["switches"] += 1
            elif "cpu-clock" in rest or "cycles" in rest:
                coverage[pid][tid]["samples"] += 1
    return coverage

def schedule_summary(perf: dict, pid: int, start_ns: float, end_ns: float) -> dict:
    samples = [s for s in perf["samples"].get(pid, []) if start_ns <= s[0] <= end_ns]
    switches = [s for s in perf["switches"].get(pid, []) if start_ns <= s[0] <= end_ns]
    runnable = sum(1 for _, _, state, _ in switches if state.upper() == "R")
    blocked = sum(1 for _, _, state, _ in switches if state.upper() in ("S", "D"))
    return dict(samples=len(samples), switches=len(switches),
                runnable_switches=runnable, blocked_switches=blocked,
                first_sample_ns=samples[0][0] if samples else "",
                last_sample_ns=samples[-1][0] if samples else "",
                top_stack=samples[-1][1] if samples else "",
                covered=bool(samples))


def schedule_intervals(perf: dict, pid: int, start_ns: float, end_ns: float) -> list[tuple[int, int, str]]:
    """Reconstruct coarse running/runnable/blocked intervals for one PID."""
    switches = []
    for items in perf["switches"].values():
        switches.extend(item for item in items if item[1] == pid or item[3] == pid)
    switches.sort()
    intervals, cursor, state = [], start_ns, None
    for stamp, previous, prev_state, following in switches:
        if stamp <= start_ns:
            state = "running" if following == pid else (
                "runnable" if previous == pid and prev_state.upper() == "R" else
                "blocked" if previous == pid else state)
            continue
        if stamp >= end_ns:
            break
        if state is not None and stamp > cursor:
            intervals.append((cursor, stamp, state))
        if following == pid:
            state = "running"
        elif previous == pid:
            state = "runnable" if prev_state.upper() == "R" else "blocked"
        cursor = stamp
    if state is not None and end_ns > cursor:
        intervals.append((cursor, end_ns, state))
    return intervals


def perfetto_for_arm(arm: str, ranks: dict[int, dict], perf: dict, dest: Path,
                     step: int = 1) -> None:
    origin = min(data["start"] for data in ranks.values())
    entries = {rank: data["end"] for rank, data in ranks.items()}
    early_rank = min(entries, key=entries.get)
    late_rank = max(entries, key=entries.get)
    events = []
    for rank, data in ranks.items():
        fit = data["fit"]
        require(fit is not None, "Perfetto requires step-1 host mapping")
        role = "early" if rank == early_rank else "late" if rank == late_rank else ""
        events.append(dict(ph="M", pid=rank, tid=0, name="process_name",
                           args={"name": f"{arm} rank {rank}" + (f" ({role})" if role else "")}))
        events.append(dict(ph="M", pid=rank, tid=2000, name="thread_name",
                           args={"name": "host submitter / CPU op"}))
        events.append(dict(ph="M", pid=rank, tid=3000, name="thread_name",
                           args={"name": "device milestones"}))
        events.append(dict(ph="M", pid=rank, tid=5000, name="thread_name",
                           args={"name": "perf scheduler (mono)"}))
        events.append(dict(ph="M", pid=rank, tid=5001, name="thread_name",
                           args={"name": "perf stack samples (mono)"}))
        raw = data["mapping"]["raw"]["traceEvents"]
        submit_pid = fit["launches"][0].get("pid")
        submit_tid = fit["launches"][0].get("tid")
        for e in raw:
            if e.get("ph") != "X" or e.get("ts") is None:
                continue
            trace_start, trace_end = e["ts"], e["ts"] + e.get("dur", 0)
            if trace_end < data["start"] - 2000 or trace_start > data["end"] + 2000:
                continue
            if e.get("cat") in DEVICE_CATS:
                if trace_end < data["start"] - 2000 or trace_start > data["end"] + 2000:
                    continue
                stream = int(e.get("args", {}).get("stream", -1))
                events.append(dict(ph="X", pid=rank, tid=1000 + stream,
                                   name=e.get("name", "")[:180], cat="device",
                                   ts=trace_start - origin, dur=max(e.get("dur", 0), 0.001),
                                   args={"stream": stream,
                                         "correlation": e.get("args", {}).get("correlation", ""),
                                         "graph_id": e.get("args", {}).get("graph id", 0)}))
            elif e.get("cat") in CPU_CATS and e.get("pid") == submit_pid and e.get("tid") == submit_tid:
                if trace_end < data["start"] - 2000 or trace_start > data["end"] + 2000:
                    continue
                events.append(dict(ph="X", pid=rank, tid=2000,
                                   name=e.get("name", "")[:180], cat=e.get("cat", "cpu"),
                                   ts=trace_start - origin,
                                   dur=max(e.get("dur", 0), 0.001),
                                   args={"correlation": e.get("args", {}).get("correlation", "")}))
        for label, value, color in (
            ("L19 exit", data["start"], "yellow"),
            ("L20 entry", data["end"], "green"),
        ):
            events.append(dict(ph="i", pid=rank, tid=3000, name=label,
                               ts=value - origin,
                               args={"color": color, "rank": rank}))
        clocks = data["clocks"]
        exchange = [float(data["mapping"]["map"](clocks[20, 4])),
                    float(data["mapping"]["map"](clocks[20, 5]))]
        events.append(dict(ph="X", pid=rank, tid=3000, name="count exchange",
                           cat="phase", ts=exchange[0] - origin,
                           dur=max(exchange[1] - exchange[0], 0.001)))
        for e in data["events"]:
            if e["alias"] in ("all_gather", "reduce_scatter", "mhc", "mega_gate", "prepare",
                              "mega_moe", "gc_moe"):
                start_mono = float(e["raw_start"])
                end_mono = float(e["raw_end"])
                events.append(dict(ph="X", pid=rank, tid=3000, name=e["alias"],
                                   cat="boundary", ts=start_mono - origin,
                                   dur=max(end_mono - start_mono, 0.001),
                                   args={"stream": e["stream"], "correlation": e["correlation"]}))
        for label, launch in (("L19 graph launch", fit["launches"][19]),
                              ("L20 graph launch", fit["launches"][20])):
            launch_mono = float(launch["ts"])
            events.append(dict(ph="i", pid=rank, tid=3000, name=label,
                               ts=launch_mono - origin,
                               args={"correlation": launch.get("args", {}).get("correlation", "")}))
        pid = int(submit_pid)
        start_mono = float(fit["trace_to_mono"](np.array([data["start"]]))[0])
        end_mono = float(fit["trace_to_mono"](np.array([data["end"]]))[0])
        summary = schedule_summary(perf, pid, start_mono, end_mono)
        def mono_to_trace_us(value: float) -> float:
            return float(fit["mono_to_trace"](np.array([value], dtype=np.float64))[0]) / 1000.0
        for left, right, state in schedule_intervals(perf, pid, start_mono, end_mono):
            trace_left, trace_right = mono_to_trace_us(left), mono_to_trace_us(right)
            events.append(dict(ph="X", pid=rank, tid=5000, name=state,
                               cat="scheduler", ts=trace_left - origin,
                               dur=max(trace_right - trace_left, 0.001)))
        for stamp, stack in perf["samples"].get(pid, []):
            if start_mono <= stamp <= end_mono:
                events.append(dict(ph="i", pid=rank, tid=5001, name="perf sample",
                                   ts=mono_to_trace_us(stamp) - origin,
                                   args={"stack": stack[:240]}))
        events.append(dict(ph="i", pid=rank, tid=5000, name="perf coverage",
                           ts=data["start"] - origin,
                           args={key: value for key, value in summary.items()
                                 if key != "top_stack"}))
    dest.write_text(json.dumps({"traceEvents": events, "displayTimeUnit": "us"}, indent=None))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="run/bs512 directory with gc/ and mega/")
    parser.add_argument("out", type=Path, help="result directory")
    parser.add_argument("--perf", type=Path, default=None, help="worker-sched.txt (perf script text)")
    parser.add_argument("--perf-pids", default="", help="comma-separated worker PIDs")
    parser.add_argument("--perf-tids", type=Path, default=None,
                        help="field-form perf output used for PID/TID coverage")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    pids = {int(x) for x in args.perf_pids.split(",") if x.strip()}
    perf = read_perf_for_ranks(args.perf, pids) if args.perf else dict(
        samples=defaultdict(list), switches=defaultdict(list), tids=defaultdict(set),
        present=False, lines=0)
    tid_coverage = read_tid_coverage(args.perf_tids, pids)

    rank_rows, gap_rows, summary_rows, exchange_rows, perf_rows = [], [], [], [], []
    boundary_rows, key_rows, close_rows = [], [], []
    step_rows = []
    for arm in ARMS:
        folder = args.root / arm
        require(folder.is_dir(), f"missing arm directory {folder}")
        ranks = {rank: build_rank(folder, arm, rank, 1) for rank in range(8)}
        boundary_rows_for_arm(arm, ranks, rank_rows)
        gaps = gap_rows_for_arm(arm, ranks, perf)
        gap_rows.extend(gaps)
        early_rank = int(min(range(8), key=lambda r: ranks[r]["end"]))
        late_rank = int(max(range(8), key=lambda r: ranks[r]["end"]))
        matched, unmatched = match_boundaries(ranks[early_rank], ranks[late_rank])
        boundary_rows.extend(matched)
        boundary_rows.extend(dict(row, note="late-only") for row in unmatched)
        key_rows.extend(key_call_rows(arm, ranks, gaps))
        close_rows.extend(closure_rows(arm, ranks, gaps))
        answer = exchange_answer(arm, ranks)
        exchange_rows.append(answer)
        entries = np.array([ranks[r]["end"] for r in range(8)])
        early_rank, late_rank = int(entries.argmin()), int(entries.argmax())
        positive = sum(float(g["gap_delta_us"]) for g in gaps
                       if g["gap_delta_us"] != "" and float(g["gap_delta_us"]) > 0)
        negative = sum(float(g["gap_delta_us"]) for g in gaps
                       if g["gap_delta_us"] != "" and float(g["gap_delta_us"]) < 0)
        summary_rows.append(dict(
            arm=arm, early_rank=early_rank, late_rank=late_rank,
            l19_exit_spread_us=float(np.ptp([ranks[r]["start"] for r in range(8)])),
            l20_entry_spread_us=float(np.ptp(entries)),
            positive_gap_delta_us=positive, negative_gap_delta_us=negative,
            gap_delta_sum_us=positive + negative,
            unmatched_gaps=sum(g["match_status"] != "matched" for g in gaps),
            exchange_before_late_entry_us=answer["exchange_before_late_entry_us"],
            max_host_fit_residual_us=float(max(np.max(np.abs(ranks[r]["fit"]["residual_us"])) for r in range(8))),
            max_alignment_half_width_us=float(max(ranks[r]["mapping"]["width"] for r in range(8))),
        ))
        for rank in range(8):
            data = ranks[rank]
            submit_pid = int(data["fit"]["launches"][0].get("pid"))
            start_mono = float(data["fit"]["trace_to_mono"](np.array([data["start"]]))[0])
            end_mono = float(data["fit"]["trace_to_mono"](np.array([data["end"]]))[0])
            schedule = schedule_summary(perf, submit_pid, start_mono, end_mono)
            per_tid = tid_coverage.get(submit_pid, {})
            tids = ";".join(f"{tid}:{v['samples']}/{v['switches']}"
                            for tid, v in sorted(per_tid.items()))
            total_samples = (sum(v["samples"] for v in per_tid.values())
                             if per_tid else len(perf["samples"].get(submit_pid, [])))
            total_switches = (sum(v["switches"] for v in per_tid.values())
                              if per_tid else len(perf["switches"].get(submit_pid, [])))
            perf_rows.append(dict(arm=arm, rank=rank, pid=submit_pid,
                                  tids_sample_switch=tids,
                                  total_samples=total_samples,
                                  total_switches=total_switches,
                                  window_samples=int(schedule["samples"]),
                                  window_switches=int(schedule["switches"]),
                                  window_runnable_switches=int(schedule["runnable_switches"]),
                                  window_blocked_switches=int(schedule["blocked_switches"]),
                                  window_covered=schedule["covered"]))
        step_rows.extend(step146_rows(folder))
        perfetto_for_arm(arm, ranks, perf, args.out / f"{arm}_step001_l19_l20.perfetto.json", 1)

    write_csv(args.out / "l20_arrival_and_exchange.csv", rank_rows)
    write_csv(args.out / "gap_attribution.csv", gap_rows)
    write_csv(args.out / "boundary_alignment.csv", boundary_rows)
    write_csv(args.out / "call_chain.csv", boundary_rows)
    write_csv(args.out / "key_call_answers.csv", key_rows)
    write_csv(args.out / "closure.csv", close_rows)
    write_csv(args.out / "arm_summary.csv", summary_rows)
    write_csv(args.out / "exchange_intersection.csv", exchange_rows)
    write_csv(args.out / "perf_coverage.csv", perf_rows)
    write_csv(args.out / "step146.csv", step_rows)
    write_csv(args.out / "step146_compare.csv",
              compare_step146(step_rows, args.out.parent / "v41_residual_l20_results" / "step146_window.csv"))
    lines = ["status=OK", f"perf_present={perf['present']}",
             f"perf_lines={perf['lines']}", f"arms={len(ARMS)}"]
    for row in close_rows:
        lines.append(
            f"{row['arm']}: entry_spread={row['l20_entry_spread_us']:.6f}us "
            f"window_delta={row['window_delta_us']:.6f}us "
            f"busy_delta={row['union_busy_delta_us']:.6f}us "
            f"gap_delta={row['gap_total_delta_us']:.6f}us "
            f"closure_error={row['closure_error_us']:.6f}us "
            f"matched_gaps={row['matched_gap_count']} unmatched_gaps={row['unmatched_gap_count']}")
    for row in summary_rows:
        lines.append(
            f"{row['arm']}: host_fit_max_residual={row['max_host_fit_residual_us']:.6f}us "
            f"alignment_half_width={row['max_alignment_half_width_us']:.6f}us")
    (args.out / "analysis_validation.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
