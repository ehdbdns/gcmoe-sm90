"""Build structured evidence tables from the host-submission segment timeline."""
import argparse
import csv
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path


TARGET_LAYERS = (8, 20, 32, 39)
SIGNIFICANT_US = 100.0


def load_segments(path):
    windows = defaultdict(list)
    launches = {}
    with gzip.open(path, "rt", newline="") as handle:
        for raw in csv.DictReader(handle):
            key = (raw["arm"], int(raw["rank"]), int(raw["layer"]))
            row = {
                "arm": key[0],
                "rank": key[1],
                "layer": key[2],
                "pid": int(raw["pid"]),
                "tid": int(raw["tid"]),
                "launch_host_ns": int(raw["launch_host_ns"]),
                "offset_us": float(raw["offset_us"]),
                "exclusive_us": float(raw["exclusive_us"]),
                "category": raw["category"],
                "name": raw["name"],
                "enclosing_cpu_op": raw["enclosing_cpu_op"],
                "correlation": raw["correlation"],
                "gpu_activity": raw["gpu_activity"],
            }
            windows[key].append(row)
            previous = launches.setdefault(key, row["launch_host_ns"])
            if previous != row["launch_host_ns"]:
                raise ValueError(f"inconsistent launch timestamp for {key}")
    return windows, launches


def event_key(row):
    return (row["category"], row["name"], row["enclosing_cpu_op"])


def event_label(row):
    suffix = f" [{row['enclosing_cpu_op']}]" if row["enclosing_cpu_op"] else ""
    return f"{row['category']}:{row['name']}{suffix}"


def anchors(profile_path):
    trace = json.loads(profile_path.read_text())
    spans = [event for event in trace["traceEvents"]
             if event.get("ph") == "X" and "dur" in event]
    values = sorted(event["ts"] + event["dur"] / 2 for event in spans
                    if event.get("cat") == "kernel"
                    and "sm100_phase_clock_anchor" in event["name"])
    if len(values) != 2:
        raise ValueError(f"{profile_path}: expected two clock anchors")
    return values


def load_profile(profile_path):
    trace = json.loads(profile_path.read_text())
    spans = [event for event in trace["traceEvents"]
             if event.get("ph") == "X" and "dur" in event]
    kernels = defaultdict(list)
    for event in spans:
        correlation = event.get("args", {}).get("correlation")
        if correlation is not None and event.get("cat") in (
                "kernel", "gpu_memcpy", "gpu_memset"):
            kernels[correlation].append(event["name"])
    return {
        "base_ns": int(trace["baseTimeNanoseconds"]),
        "spans": spans,
        "kernels": kernels,
    }


def load_pairs(intervals_path):
    groups = defaultdict(list)
    with intervals_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            layer = int(row["layer"])
            if layer in TARGET_LAYERS:
                groups[(row["arm"], layer)].append(
                    (int(row["entry_host_ns"]), int(row["rank"])))
    pairs = {}
    for arm in ("gc", "mega"):
        for layer in TARGET_LAYERS:
            ordered = sorted(groups[(arm, layer)])
            pairs[arm, layer] = (ordered[0][1], ordered[-1][1])
    expected = {
        ("mega", 8): (5, 4), ("mega", 20): (6, 4),
        ("mega", 32): (0, 4), ("mega", 39): (0, 4),
        ("gc", 8): (7, 2), ("gc", 20): (0, 2),
        ("gc", 32): (4, 2), ("gc", 39): (5, 2),
    }
    if pairs != expected:
        raise ValueError(f"extreme-rank selection changed: {pairs}")
    return pairs


def category_totals(rows):
    totals = defaultdict(float)
    for row in rows:
        totals[row["category"]] += row["exclusive_us"]
    return totals


def actual_calls(profile, start_ns, end_ns):
    base_ns = profile["base_ns"]
    start = (start_ns - base_ns) / 1000.0
    end = (end_ns - base_ns) / 1000.0
    spans = profile["spans"]
    cpu_pool = [event for event in spans
                if event.get("cat") == "cpu_op"
                and event["ts"] < end and event["ts"] + event["dur"] > start]
    events = [event for event in spans
              if event.get("cat") in ("cpu_op", "cuda_runtime", "cuda_driver")
              and start - 0.01 <= event["ts"] < end - 0.01]
    events.sort(key=lambda event: (event["ts"], -event["dur"]))
    occurrences = Counter()
    calls = []
    for index, event in enumerate(events, 1):
        parents = [parent for parent in cpu_pool
                   if parent is not event
                   and parent.get("pid") == event.get("pid")
                   and parent.get("tid") == event.get("tid")
                   and parent["ts"] <= event["ts"] + 0.01
                   and parent["ts"] + parent["dur"]
                   >= event["ts"] + event["dur"] - 0.01]
        parent = min(parents, key=lambda item: (item["dur"], item["ts"]), default=None)
        enclosing = parent["name"] if parent else ""
        key = (event["cat"], event["name"], enclosing)
        occurrences[key] += 1
        calls.append({
            "index": index,
            "occurrence": occurrences[key],
            "category": event["cat"],
            "name": event["name"],
            "enclosing_cpu_op": enclosing,
            "event": event,
            "start_us": event["ts"],
            "duration_us": event["dur"],
            "end_us": event["ts"] + event["dur"],
            "exclusive_us": 0.0,
            "key": key,
        })
    return calls


def calls_with_exclusive_time(profile, rows, start_ns, end_ns):
    calls = actual_calls(profile, start_ns, end_ns)
    by_name = defaultdict(list)
    for call in calls:
        by_name[call["category"], call["name"]].append(call)
    for row in rows:
        if row["category"] == "unobserved":
            continue
        segment_start = ((start_ns - profile["base_ns"]) / 1000.0
                         + row["offset_us"])
        segment_end = segment_start + row["exclusive_us"]
        candidates = [call for call in by_name[row["category"], row["name"]]
                      if call["start_us"] <= segment_start + 0.01
                      and call["end_us"] >= segment_end - 0.01]
        if not candidates:
            raise ValueError(f"segment has no containing call: {event_key(row)}")
        call = min(candidates, key=lambda item: (item["duration_us"], item["start_us"]))
        call["exclusive_us"] += row["exclusive_us"]
    return calls


def matched_contributions(early_rows, late_rows, early_profile, late_profile,
                          early_start_ns, early_end_ns, late_start_ns, late_end_ns):
    early_calls = calls_with_exclusive_time(
        early_profile, early_rows, early_start_ns, early_end_ns)
    late_calls = calls_with_exclusive_time(
        late_profile, late_rows, late_start_ns, late_end_ns)
    if [call["key"] for call in early_calls] != [call["key"] for call in late_calls]:
        raise ValueError("actual-call sequences do not align")

    early_by_category = category_totals(early_rows)
    late_by_category = category_totals(late_rows)
    category_diffs = {
        category: late_by_category.get(category, 0.0)
        - early_by_category.get(category, 0.0)
        for category in sorted(set(early_by_category) | set(late_by_category))
    }

    matched_calls = []
    for early, late in zip(early_calls, late_calls):
        matched_calls.append({
            "index": early["index"],
            "occurrence": early["occurrence"],
            "category": early["category"],
            "name": early["name"],
            "enclosing_cpu_op": early["enclosing_cpu_op"],
            "early": early,
            "late": late,
            "early_us": early["exclusive_us"],
            "late_us": late["exclusive_us"],
            "delta_us": late["exclusive_us"] - early["exclusive_us"],
        })

    early_segment_keys = [event_key(row) for row in early_rows]
    late_segment_keys = [event_key(row) for row in late_rows]
    if early_segment_keys != late_segment_keys:
        raise ValueError("segment sequences do not align")
    occurrences = Counter()
    unobserved = []
    for index, (early, late) in enumerate(zip(early_rows, late_rows), 1):
        key = event_key(early)
        occurrences[key] += 1
        if early["category"] == "unobserved":
            unobserved.append({
                "index": index,
                "occurrence": occurrences[key],
                "category": early["category"],
                "name": early["name"],
                "enclosing_cpu_op": "",
                "early": early,
                "late": late,
                "early_us": early["exclusive_us"],
                "late_us": late["exclusive_us"],
                "delta_us": late["exclusive_us"] - early["exclusive_us"],
            })
    return category_diffs, matched_calls, unobserved


def top_positive(entries, count=5):
    return sorted(entries, key=lambda entry: entry["delta_us"], reverse=True)[:count]


def add_contribution_rows(writer, arm, target, role, early_rank, late_rank,
                          category_diffs, calls, unobserved):
    for category, delta in category_diffs.items():
        writer.writerow({
            "arm": arm, "target_layer": target, "window_role": role,
            "record_kind": "category_total", "early_rank": early_rank,
            "late_rank": late_rank, "call_index": "", "segment_index": "",
            "occurrence": "",
            "category": category, "name": "", "enclosing_cpu_op": "",
            "early_us": "", "late_us": "", "delta_us": f"{delta:.6f}",
            "early_parent_duration_us": "", "late_parent_duration_us": "",
        })

    for record_kind, entries in (("top_call", top_positive(calls)),
                                 ("top_unobserved", top_positive(unobserved))):
        for entry in entries:
            row = {
                "arm": arm, "target_layer": target, "window_role": role,
                "record_kind": record_kind, "early_rank": early_rank,
                "late_rank": late_rank,
                "call_index": entry["index"] if record_kind == "top_call" else "",
                "segment_index": entry["index"] if record_kind == "top_unobserved" else "",
                "occurrence": entry["occurrence"], "category": entry["category"],
                "name": entry["name"], "enclosing_cpu_op": entry["enclosing_cpu_op"],
                "early_us": f"{entry['early_us']:.6f}",
                "late_us": f"{entry['late_us']:.6f}",
                "delta_us": f"{entry['delta_us']:.6f}",
            }
            if record_kind == "top_call":
                row["early_parent_duration_us"] = f"{entry['early']['duration_us']:.6f}"
                row["late_parent_duration_us"] = f"{entry['late']['duration_us']:.6f}"
            else:
                row["early_parent_duration_us"] = ""
                row["late_parent_duration_us"] = ""
            writer.writerow(row)


def neighbor(entries, position):
    if position < 0 or position >= len(entries):
        return None
    return entries[position]


def event_nested_activity(profile, parent):
    children = [event for event in profile["spans"]
                if event.get("pid") == parent.get("pid")
                and event.get("tid") == parent.get("tid")
                and event.get("cat") in ("cuda_runtime", "cuda_driver")
                and event.get("args", {}).get("correlation") is not None
                and event["ts"] >= parent["ts"] - 0.01
                and event["ts"] + event["dur"] <= parent["ts"] + parent["dur"] + 0.01]
    correlations = sorted({str(event["args"]["correlation"]) for event in children},
                          key=int)
    gpu_names = sorted({name for event in children
                        for name in profile["kernels"].get(
                            event["args"]["correlation"], [])})
    return "|".join(correlations), "|".join(gpu_names)


def add_call_evidence(writer, arm, target, role, early_rank, late_rank,
                      early_rows, late_rows, calls, unobserved, early_calls,
                      early_profile, late_profile,
                      early_window_ns, late_window_ns):
    fieldnames = writer.fieldnames

    for entry in top_positive(calls):
        early = entry["early"]
        late = entry["late"]
        early_event = early["event"]
        late_event = late["event"]
        previous = neighbor(early_calls, early["index"] - 2)
        following = neighbor(early_calls, early["index"])
        early_nested_corr, early_nested_gpu = event_nested_activity(
            early_profile, early_event)
        late_nested_corr, late_nested_gpu = event_nested_activity(
            late_profile, late_event)
        early_offset = ((early_event["ts"] - (early_window_ns - early_profile["base_ns"]) / 1000.0))
        late_offset = ((late_event["ts"] - (late_window_ns - late_profile["base_ns"]) / 1000.0))
        row = {
            "arm": arm, "target_layer": target, "window_role": role,
            "evidence_kind": "call", "early_rank": early_rank,
            "late_rank": late_rank, "call_index": early["index"],
            "segment_index": "", "occurrence": entry["occurrence"],
            "category": early["category"], "name": early["name"],
            "enclosing_cpu_op": early["enclosing_cpu_op"],
            "early_pid": early_event.get("pid"), "early_tid": early_event.get("tid"),
            "late_pid": late_event.get("pid"), "late_tid": late_event.get("tid"),
            "early_offset_us": f"{early_offset:.6f}",
            "late_offset_us": f"{late_offset:.6f}",
            "early_us": f"{entry['early_us']:.6f}",
            "late_us": f"{entry['late_us']:.6f}",
            "delta_us": f"{entry['delta_us']:.6f}",
            "early_parent_duration_us": f"{early['duration_us']:.6f}",
            "late_parent_duration_us": f"{late['duration_us']:.6f}",
            "early_correlation": early_event.get("args", {}).get("correlation", ""),
            "late_correlation": late_event.get("args", {}).get("correlation", ""),
            "early_nested_api_correlations": early_nested_corr,
            "late_nested_api_correlations": late_nested_corr,
            "early_nested_gpu_activity": early_nested_gpu,
            "late_nested_gpu_activity": late_nested_gpu,
            "previous_index": previous["index"] if previous else "",
            "previous_event": event_label(previous) if previous else "",
            "next_index": following["index"] if following else "",
            "next_event": event_label(following) if following else "",
        }
        writer.writerow({field: row.get(field, "") for field in fieldnames})

    for entry in top_positive(unobserved):
        index = entry["index"]
        early = early_rows[index - 1]
        late = late_rows[index - 1]
        previous = neighbor(early_rows, index - 2)
        following = neighbor(early_rows, index)
        row = {
            "arm": arm, "target_layer": target, "window_role": role,
            "evidence_kind": "unobserved_gap", "early_rank": early_rank,
            "late_rank": late_rank, "call_index": "", "segment_index": index,
            "occurrence": entry["occurrence"], "category": early["category"],
            "name": early["name"], "enclosing_cpu_op": "",
            "early_pid": early["pid"], "early_tid": early["tid"],
            "late_pid": late["pid"], "late_tid": late["tid"],
            "early_offset_us": f"{early['offset_us']:.6f}",
            "late_offset_us": f"{late['offset_us']:.6f}",
            "early_us": f"{entry['early_us']:.6f}",
            "late_us": f"{entry['late_us']:.6f}",
            "delta_us": f"{entry['delta_us']:.6f}",
            "early_parent_duration_us": "", "late_parent_duration_us": "",
            "early_correlation": "", "late_correlation": "",
            "early_nested_api_correlations": "", "late_nested_api_correlations": "",
            "early_nested_gpu_activity": "", "late_nested_gpu_activity": "",
            "previous_index": index - 1 if previous else "",
            "previous_event": event_label(previous) if previous else "",
            "next_index": index + 1 if following else "",
            "next_event": event_label(following) if following else "",
        }
        writer.writerow({field: row.get(field, "") for field in fieldnames})


def selected_window_entries(early_rows, late_rows, early_profile, late_profile,
                            early_start_ns, early_end_ns, late_start_ns, late_end_ns):
    category_diffs, calls, unobserved = matched_contributions(
        early_rows, late_rows, early_profile, late_profile,
        early_start_ns, early_end_ns, late_start_ns, late_end_ns)
    early_calls = calls_with_exclusive_time(
        early_profile, early_rows, early_start_ns, early_end_ns)
    return category_diffs, calls, unobserved, early_calls


def find_layer_window(profile_path, layer):
    trace = json.loads(profile_path.read_text())
    spans = [event for event in trace["traceEvents"]
             if event.get("ph") == "X" and "dur" in event]
    clock = anchors(profile_path)
    launches = sorted((event for event in spans
                       if event.get("cat") in ("cuda_runtime", "cuda_driver")
                       and event["name"].startswith("cudaGraphLaunch")
                       and clock[0] < event["ts"] < clock[1]),
                      key=lambda event: event["ts"])
    if len(launches) != 40:
        raise ValueError(f"{profile_path}: expected 40 graph launches")
    start = launches[layer - 1]["ts"]
    end = launches[layer]["ts"]
    gpu_events = sorted((event for event in spans
                         if event.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")
                         and event["ts"] < end and event["ts"] + event["dur"] > start),
                        key=lambda event: event["ts"])
    return trace, spans, start, end, gpu_events


def largest_gap(gpu_events, start, end):
    gaps = []
    cursor = start
    previous = None
    for event in gpu_events:
        if event["ts"] > cursor:
            gaps.append((event["ts"] - cursor, cursor, event["ts"], previous, event))
        cursor = max(cursor, event["ts"] + event["dur"])
        previous = event
    if end > cursor:
        gaps.append((end - cursor, cursor, end, previous, None))
    return max(gaps, key=lambda gap: gap[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("segments", type=Path)
    parser.add_argument("intervals", type=Path)
    parser.add_argument("profiles", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    windows, launches = load_segments(args.segments)
    expected_windows = {(arm, rank, layer)
                        for arm in ("gc", "mega") for rank in range(8)
                        for layer in range(40)}
    if set(windows) != expected_windows:
        raise ValueError("incomplete arm/rank/layer window geometry")
    pairs = load_pairs(args.intervals)

    launch_sets = defaultdict(list)
    with args.intervals.open(newline="") as handle:
        for row in csv.DictReader(handle):
            launch_sets[row["arm"], int(row["rank"])].append(int(row["launches_seen"]))
    if any(values != [40] * 39 for values in launch_sets.values()):
        raise ValueError("not every inter-layer interval contains 40 graph launches")

    anchor_cache = {}
    for arm, target in pairs:
        for rank in pairs[arm, target]:
            key = arm, rank
            if key not in anchor_cache:
                anchor_cache[key] = anchors(
                    args.profiles / arm / f"profile-rank{rank}.json")

    identity_errors = []
    for (arm, rank, layer), rows in windows.items():
        if layer == 0:
            continue
        expected = (launches[arm, rank, layer]
                    - launches[arm, rank, layer - 1]) / 1000.0
        actual = sum(row["exclusive_us"] for row in rows)
        identity_errors.append((abs(actual - expected), arm, rank, layer, actual, expected))
    identity_errors.sort(reverse=True)

    cumulative_rows = []
    cumulative_by_pair = {}
    for arm, target in pairs:
        early_rank, late_rank = pairs[arm, target]
        early_anchor = anchor_cache[arm, early_rank]
        late_anchor = anchor_cache[arm, late_rank]
        previous = None
        for layer in range(target + 1):
            early_launch = launches[arm, early_rank, layer]
            late_launch = launches[arm, late_rank, layer]
            cumulative = (late_launch - early_launch) / 1000.0
            incremental = None if previous is None else cumulative - previous
            cumulative_by_pair[arm, target, layer] = cumulative
            cumulative_rows.append({
                "arm": arm, "target_layer": target,
                "early_rank": early_rank, "late_rank": late_rank,
                "layer": layer,
                "early_launch_host_ns": early_launch,
                "late_launch_host_ns": late_launch,
                "cumulative_delta_us": f"{cumulative:.6f}",
                "incremental_delta_us": "" if incremental is None
                else f"{incremental:.6f}",
                "first_anchor_delta_us": f"{late_anchor[0] - early_anchor[0]:.6f}",
                "second_anchor_delta_us": f"{late_anchor[1] - early_anchor[1]:.6f}",
            })
            previous = cumulative

    with (args.output / "cumulative_submission_diffs.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cumulative_rows[0]))
        writer.writeheader()
        writer.writerows(cumulative_rows)

    expansion_rows = []
    windows_to_expand = {}
    for arm, target in pairs:
        early_rank, late_rank = pairs[arm, target]
        series = [(layer, cumulative_by_pair[arm, target, layer]
                   - cumulative_by_pair[arm, target, layer - 1])
                  for layer in range(1, target + 1)]
        significant = [entry for entry in series if entry[1] >= SIGNIFICANT_US]
        if not significant:
            raise ValueError(f"no {SIGNIFICANT_US} us expansion for {arm} layer {target}")
        first = significant[0]
        later = [entry for entry in series if entry[0] != first[0]]
        maximum = max(later, key=lambda entry: entry[1])
        target_delta = series[-1][1]
        expansion_rows.append({
            "arm": arm, "target_layer": target,
            "early_rank": early_rank, "late_rank": late_rank,
            "first_significant_layer": first[0],
            "first_significant_delta_us": f"{first[1]:.6f}",
            "max_later_layer": maximum[0],
            "max_later_delta_us": f"{maximum[1]:.6f}",
            "target_layer_delta_us": f"{target_delta:.6f}",
            "target_cumulative_delta_us": f"{cumulative_by_pair[arm, target, target]:.6f}",
            "inherited_layer0_us": f"{cumulative_by_pair[arm, target, 0]:.6f}",
            "sum_new_layers_us": f"{cumulative_by_pair[arm, target, target] - cumulative_by_pair[arm, target, 0]:.6f}",
        })
        windows_to_expand[arm, target] = (first[0], maximum[0], early_rank, late_rank)

    with (args.output / "expansion_windows.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(expansion_rows[0]))
        writer.writeheader()
        writer.writerows(expansion_rows)

    contributions_file = (args.output / "contributions.csv").open("w", newline="")
    contribution_writer = csv.DictWriter(
        contributions_file,
        fieldnames=["arm", "target_layer", "window_role", "record_kind",
                    "early_rank", "late_rank", "call_index", "segment_index",
                    "occurrence", "category", "name", "enclosing_cpu_op",
                    "early_us", "late_us", "delta_us",
                    "early_parent_duration_us", "late_parent_duration_us"])
    contribution_writer.writeheader()

    evidence_file = (args.output / "call_evidence.csv").open("w", newline="")
    evidence_writer = csv.DictWriter(
        evidence_file,
        fieldnames=["arm", "target_layer", "window_role", "evidence_kind",
                    "early_rank", "late_rank", "call_index", "segment_index",
                    "occurrence", "category", "name",
                    "enclosing_cpu_op", "early_pid", "early_tid", "late_pid", "late_tid",
                    "early_offset_us", "late_offset_us", "early_us", "late_us",
                    "delta_us", "early_parent_duration_us", "late_parent_duration_us",
                    "early_correlation", "late_correlation",
                    "early_nested_api_correlations", "late_nested_api_correlations",
                    "early_nested_gpu_activity", "late_nested_gpu_activity",
                    "previous_index", "previous_event", "next_index", "next_event"])
    evidence_writer.writeheader()

    window_data = {}
    profile_cache = {}
    for arm, target in pairs:
        for rank in pairs[arm, target]:
            profile_cache[arm, rank] = load_profile(
                args.profiles / arm / f"profile-rank{rank}.json")
    alignment_rows = []
    for arm, target in pairs:
        early_rank, late_rank = pairs[arm, target]
        aligned_layers = 0
        startup_mismatch = []
        for layer in range(target + 1):
            early_rows = windows[arm, early_rank, layer]
            late_rows = windows[arm, late_rank, layer]
            early_keys = [event_key(row) for row in early_rows]
            late_keys = [event_key(row) for row in late_rows]
            if early_keys == late_keys:
                aligned_layers += 1
            else:
                startup_mismatch.append((layer, len(early_rows), len(late_rows)))
        alignment_rows.append({
            "arm": arm, "target_layer": target, "early_rank": early_rank,
            "late_rank": late_rank, "layers_checked": target + 1,
            "layers_aligned": aligned_layers,
            "mismatches": ";".join(f"L{layer}:{left}/{right}"
                                   for layer, left, right in startup_mismatch) or "none",
        })

        first_layer, maximum_layer, _, _ = windows_to_expand[arm, target]
        for role, layer in (("first_significant", first_layer),
                            ("max_later", maximum_layer)):
            early_rows = windows[arm, early_rank, layer]
            late_rows = windows[arm, late_rank, layer]
            categories, calls, unobserved, early_calls = selected_window_entries(
                early_rows, late_rows,
                profile_cache[arm, early_rank], profile_cache[arm, late_rank],
                launches[arm, early_rank, layer - 1], launches[arm, early_rank, layer],
                launches[arm, late_rank, layer - 1], launches[arm, late_rank, layer])
            add_contribution_rows(contribution_writer, arm, target, role,
                                  early_rank, late_rank, categories, calls, unobserved)
            add_call_evidence(evidence_writer, arm, target, role,
                              early_rank, late_rank, early_rows, late_rows,
                              calls, unobserved, early_calls,
                              profile_cache[arm, early_rank],
                              profile_cache[arm, late_rank],
                              launches[arm, early_rank, layer - 1],
                              launches[arm, late_rank, layer - 1])
            window_data[arm, target, role] = {
                "layer": layer,
                "categories": categories,
                "calls": calls,
                "unobserved": unobserved,
            }

    contributions_file.close()
    evidence_file.close()

    key_rows = []
    for arm in ("gc", "mega"):
        target = 20
        early_rank, late_rank = pairs[arm, target]
        for role in ("first_significant", "max_later"):
            data = window_data[arm, target, role]
            layer = data["layer"]
            after = cumulative_by_pair[arm, target, layer]
            before = cumulative_by_pair[arm, target, layer - 1]
            delta = after - before
            categories = data["categories"]
            category_sum = sum(categories.values())
            key_rows.append({
                "arm": arm, "target_layer": target, "window_role": role,
                "early_rank": early_rank, "late_rank": late_rank, "layer": layer,
                "cumulative_before_us": f"{before:.6f}",
                "new_delta_us": f"{delta:.6f}",
                "cumulative_after_us": f"{after:.6f}",
                "unobserved_us": f"{categories.get('unobserved', 0.0):.6f}",
                "cpu_op_us": f"{categories.get('cpu_op', 0.0):.6f}",
                "cuda_driver_us": f"{categories.get('cuda_driver', 0.0):.6f}",
                "cuda_runtime_us": f"{categories.get('cuda_runtime', 0.0):.6f}",
                "category_sum_us": f"{category_sum:.6f}",
                "category_identity_error_us": f"{abs(category_sum - delta):.6f}",
            })

    with (args.output / "key_intervals.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(key_rows[0]))
        writer.writeheader()
        writer.writerows(key_rows)

    gap_file = (args.output / "launch_gap_evidence.csv").open("w", newline="")
    gap_writer = csv.DictWriter(
        gap_file,
        fieldnames=["arm", "target_layer", "early_rank", "late_rank", "rank",
                    "rank_role", "window_start_host_ns", "gap_start_offset_us",
                    "gap_end_offset_us", "gap_us", "previous_gpu_activity",
                    "previous_correlation", "next_gpu_activity", "correlation",
                    "dense_kernel_duration_us", "cpu_api", "cpu_api_duration_us",
                    "cpu_launch_after_gap_start_us", "gpu_start_after_cpu_launch_us",
                    "not_yet_submitted_us", "pending_tail_us", "gap_identity_error_us"])
    gap_writer.writeheader()

    for arm in ("gc", "mega"):
        target = 20
        early_rank, late_rank = pairs[arm, target]
        for rank, rank_role in ((early_rank, "early_entry"), (late_rank, "late_entry")):
            profile_path = args.profiles / arm / f"profile-rank{rank}.json"
            trace, spans, start, end, gpu_events = find_layer_window(profile_path, target)
            gap_us, gap_start, gap_end, previous, following = largest_gap(
                gpu_events, start, end)
            if following is None or "dense_blockscaled_gemm" not in following["name"]:
                raise ValueError(f"{arm} rank {rank}: largest gap does not end at dense GEMM")
            correlation = following["args"].get("correlation")
            api_events = sorted((event for event in spans
                                 if event.get("cat") in ("cuda_runtime", "cuda_driver")
                                 and event.get("args", {}).get("correlation") == correlation),
                                key=lambda event: event["ts"])
            cpu_event = max((event for event in api_events if event["ts"] <= following["ts"]),
                            key=lambda event: event["ts"])
            launch_delay = cpu_event["ts"] - gap_start
            pending_tail = following["ts"] - cpu_event["ts"]
            gap_writer.writerow({
                "arm": arm, "target_layer": target,
                "early_rank": early_rank, "late_rank": late_rank,
                "rank": rank, "rank_role": rank_role,
                "window_start_host_ns": int(trace["baseTimeNanoseconds"])
                + round(start * 1000),
                "gap_start_offset_us": f"{gap_start - start:.6f}",
                "gap_end_offset_us": f"{gap_end - start:.6f}",
                "gap_us": f"{gap_us:.6f}",
                "previous_gpu_activity": previous["name"] if previous else "",
                "previous_correlation": previous["args"].get("correlation", "")
                if previous else "",
                "next_gpu_activity": following["name"],
                "correlation": correlation,
                "dense_kernel_duration_us": f"{following['dur']:.6f}",
                "cpu_api": cpu_event["name"],
                "cpu_api_duration_us": f"{cpu_event['dur']:.6f}",
                "cpu_launch_after_gap_start_us": f"{launch_delay:.6f}",
                "gpu_start_after_cpu_launch_us": f"{pending_tail:.6f}",
                "not_yet_submitted_us": f"{launch_delay:.6f}",
                "pending_tail_us": f"{pending_tail:.6f}",
                "gap_identity_error_us": f"{abs(launch_delay + pending_tail - gap_us):.6f}",
            })
    gap_file.close()

    with (args.output / "sequence_alignment.txt").open("w") as handle:
        handle.write("V41 host-submit call-sequence alignment\n")
        handle.write("Keys are (category,name,enclosing_cpu_op) and include unobserved gaps.\n")
        handle.write("A layer-0 startup mismatch is retained rather than silently aligned.\n")
        for row in alignment_rows:
            handle.write(
                f"{row['arm']} target=L{row['target_layer']:02d} "
                f"early=rank{row['early_rank']} late=rank{row['late_rank']} "
                f"layers={row['layers_aligned']}/{row['layers_checked']} aligned "
                f"mismatches={row['mismatches']}\n")
        handle.write(
            "Selected first/max windows: actual-call key sequences also align; "
            "duplicates are matched by per-key occurrence.\n")

    for arm, target in pairs:
        for role in ("first_significant", "max_later"):
            data = window_data[arm, target, role]
            categories = data["categories"]
            layer = data["layer"]
            delta = cumulative_by_pair[arm, target, layer] - cumulative_by_pair[
                arm, target, layer - 1]
            if abs(sum(categories.values()) - delta) > 0.01:
                raise ValueError(f"category identity failed for {arm} L{layer} {role}")

    print(f"segment_rows={sum(len(rows) for rows in windows.values())}")
    print(f"windows={len(windows)}")
    print(f"identity_max_error_us={identity_errors[0][0]:.6f} "
          f"at={identity_errors[0][1]} rank{identity_errors[0][2]} "
          f"L{identity_errors[0][3]}")
    print(f"selected_pairs={pairs}")


if __name__ == "__main__":
    main()
