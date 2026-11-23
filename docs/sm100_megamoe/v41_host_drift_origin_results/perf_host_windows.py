#!/usr/bin/env python3
"""Summarize archived perf samples in the L1-L19 host submission windows."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/sm100"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "v41_l20_host_device_results"))

from analyze_l20_host_device import device_mapping, fit_host_to_trace, write_csv  # noqa: E402


HEADER = re.compile(r"^\s*(\S+)\s+(\d+)\s+([0-9]+\.[0-9]+):\s+(.*)$")
FIELD_HEADER = re.compile(r"^\s*(\S+)\s+(\d+)/(\d+)\s+([0-9]+\.[0-9]+):\s+(.*)$")
FRAME = re.compile(r"^\s*[0-9a-f]+\s+(.+?)\s+\((.+)\)\s*$")


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def normalize_frame(frame: str, module: str) -> str:
    frame = frame.strip()
    if frame == "[unknown]":
        return "unknown"
    if frame.startswith(("_Py", "Py", "pymalloc", "GC_", "gen_", "builtin_")):
        return "python_interpreter"
    if any(token in frame for token in ("malloc", "free", "calloc", "realloc")):
        return "allocator"
    if any(token in frame for token in ("__tls_get_addr", "_dl_", "dl_lookup")):
        return "loader_tls"
    if any(token in frame for token in ("memcpy", "memmove", "memset")):
        return "libc_memory"
    if any(token in frame.lower() for token in ("futex", "pthread", "sched", "cond_", "mutex")):
        return "sync_scheduler"
    if "syscall" in frame.lower() or "clone" in frame.lower():
        return "syscall"
    if any(token in frame for token in ("cuda", "cuLaunch", "CUDA", "c10d", "NCCL", "nccl")):
        return "cuda_collective"
    if any(token in module for token in ("libtorch", "libc10", "libpython", "libc.so", "ld-linux")):
        return module
    return frame


def build_targets(raw: Path, windows: list[dict], coverage: list[dict]) -> tuple[dict, dict]:
    target = {}
    roles = {}
    by_arm = {arm: [row for row in windows if row["arm"] == arm] for arm in ("gc", "mega")}
    for row in coverage:
        arm, rank = row["arm"], int(row["rank"])
        pid = int(row["main_tid"])
        with np.load(raw / arm / f"measure-rank{rank}.npz") as data:
            stamps = data["host_submit_mono_ns"].astype(np.int64)
        if stamps.shape != (7, 40) or np.any(stamps[1] <= 0):
            raise ValueError(f"{arm} rank {rank}: missing step-1 host stamps")
        early = int(by_arm[arm][0]["early"])
        late = int(by_arm[arm][0]["late"])
        role = "early" if rank == early else "late" if rank == late else "other"
        intervals = [(int(stamps[1, layer]), int(stamps[1, layer + 1])) for layer in range(19)]
        target[pid] = dict(arm=arm, rank=rank, role=role, pid=pid, intervals=intervals,
                           main_samples=int(row["main_samples"]))
        roles[(arm, rank)] = role
    return target, roles


def state_before(events: list[dict], stamp: float) -> tuple[str, str]:
    previous = [event for event in events if event["time"] <= stamp]
    if not previous:
        return "unknown", ""
    return previous[-1]["state"], previous[-1]["reason"]


def interval_accounting(intervals: list[tuple[int, int]], events: list[dict]) -> dict:
    result = []
    for start_ns, end_ns in intervals:
        start, end = start_ns / 1e9, end_ns / 1e9
        state, _ = state_before(events, start)
        if state == "unknown" and events:
            state = "out" if events[0]["time"] >= start else "unknown"
        cursor = start
        running = offcpu = 0.0
        switches = preempt = blocked = 0
        for event in events:
            if event["time"] <= start:
                continue
            if event["time"] > end:
                break
            delta = event["time"] - cursor
            if state == "in":
                running += delta
            elif state == "out":
                offcpu += delta
            if event["kind"] == "switch":
                switches += 1
                if event["state"] == "out":
                    if "preempt" in event["reason"]:
                        preempt += 1
                    else:
                        blocked += 1
            state = event["state"] if event["kind"] == "switch" else state
            cursor = event["time"]
        delta = end - cursor
        if state == "in":
            running += delta
        elif state == "out":
            offcpu += delta
        result.append(dict(start=start, end=end, running_us=running * 1e6,
                           offcpu_us=offcpu * 1e6, state_at_start=state,
                           switch_events=switches, preempt_out=preempt, blocked_out=blocked))
    return result


def parse_switches(path: Path, targets: dict) -> dict:
    min_time = min(entry["intervals"][0][0] for entry in targets.values()) / 1e9
    max_time = max(entry["intervals"][-1][1] for entry in targets.values()) / 1e9
    events = defaultdict(list)
    initial = {}
    with path.open(errors="replace") as handle:
        for line in handle:
            match = FIELD_HEADER.match(line)
            if not match:
                continue
            tid = int(match.group(3))
            when = float(match.group(4))
            event = match.group(5).strip()
            if tid not in targets:
                continue
            if when > max_time + 1.0:
                break
            if not (event.startswith("PERF_RECORD_SWITCH IN")
                    or event.startswith("PERF_RECORD_SWITCH OUT")):
                continue
            if event.startswith("PERF_RECORD_SWITCH IN"):
                record = dict(kind="switch", state="in", time=when, reason="")
            else:
                reason = event.removeprefix("PERF_RECORD_SWITCH OUT").strip()
                record = dict(kind="switch", state="out", time=when, reason=reason)
            if when < min_time:
                initial[tid] = record
            else:
                events[tid].append(record)
    for tid, record in initial.items():
        events[tid].insert(0, record)
    return events


def parse_samples(path: Path, targets: dict) -> dict:
    min_time = min(entry["intervals"][0][0] for entry in targets.values()) / 1e9
    max_time = max(entry["intervals"][-1][1] for entry in targets.values()) / 1e9
    samples = defaultdict(list)
    sample_pid = None
    sample_time = 0.0
    first_frame = None

    def finish_sample() -> None:
        nonlocal sample_pid, first_frame
        if sample_pid is None:
            return
        entry = targets[sample_pid]
        for index, (start_ns, end_ns) in enumerate(entry["intervals"]):
            if start_ns / 1e9 <= sample_time < end_ns / 1e9:
                samples[sample_pid].append(dict(interval=index, time=sample_time,
                                                frame=first_frame or "unknown"))
                break
        sample_pid = None
        first_frame = None

    with path.open(errors="replace") as handle:
        for line in handle:
            match = HEADER.match(line)
            if match:
                finish_sample()
                pid = int(match.group(2))
                when = float(match.group(3))
                event = match.group(4).strip()
                if when > max_time + 1.0:
                    break
                if pid not in targets:
                    continue
                if min_time <= when <= max_time and "cpu-clock:u:" in event:
                    sample_pid, sample_time, first_frame = pid, when, None
                continue
            if sample_pid is None:
                continue
            frame_match = FRAME.match(line)
            if frame_match and first_frame is None:
                first_frame = normalize_frame(frame_match.group(1), frame_match.group(2))
            elif not line.strip():
                finish_sample()
        finish_sample()
    return samples


def write_summaries(raw: Path, targets: dict, events: dict, samples: dict, out: Path) -> None:

    rows = []
    hotspots = []
    window_rows = []
    for pid, entry in sorted(targets.items(), key=lambda item: (item[1]["arm"], item[1]["rank"])):
        accounting = interval_accounting(entry["intervals"], events[pid])
        window_samples = samples[pid]
        counts = Counter(sample["frame"] for sample in window_samples)
        wall_us = sum((end - start) / 1000.0 for start, end in entry["intervals"])
        running_us = sum(item["running_us"] for item in accounting)
        offcpu_us = sum(item["offcpu_us"] for item in accounting)
        switches = sum(item["switch_events"] for item in accounting)
        preempt = sum(item["preempt_out"] for item in accounting)
        blocked = sum(item["blocked_out"] for item in accounting)
        top_frame, top_count = counts.most_common(1)[0] if counts else ("", 0)
        category_counts = Counter()
        category_map = {
            "python_interpreter": "python_interpreter", "allocator": "allocator",
            "loader_tls": "loader_tls", "libc_memory": "libc_memory",
            "sync_scheduler": "sync_scheduler", "syscall": "syscall",
            "cuda_collective": "cuda_collective", "unknown": "unknown",
        }
        for frame, count in counts.items():
            category_counts[category_map.get(frame, "other")] += count
        sample_total = len(window_samples)
        note = "no_window_samples" if sample_total == 0 else (
            "single_sample" if sample_total == 1 else "")
        rows.append(dict(
            arm=entry["arm"], rank=entry["rank"], role=entry["role"], pid=pid,
            window_count=len(accounting), window_wall_us=wall_us,
            running_us=running_us, offcpu_us=offcpu_us,
            running_pct=100.0 * running_us / wall_us if wall_us else 0.0,
            offcpu_pct=100.0 * offcpu_us / wall_us if wall_us else 0.0,
            window_samples=sample_total, main_samples_total=entry["main_samples"],
            perf_cpu_clock_hz=199.0, sampling_period_us=5025.125,
            sample_coverage_fraction=(sample_total * 5025.125 / wall_us) if wall_us else 0.0,
            wall_us_per_instance=wall_us / len(accounting),
            running_us_per_instance=running_us / len(accounting),
            offcpu_us_per_instance=offcpu_us / len(accounting),
            samples_per_instance=sample_total / len(accounting),
            switch_events=switches, preempt_out=preempt, blocked_out=blocked,
            python_interpreter_pct=100.0 * category_counts["python_interpreter"] / sample_total if sample_total else 0.0,
            allocator_pct=100.0 * category_counts["allocator"] / sample_total if sample_total else 0.0,
            loader_tls_pct=100.0 * category_counts["loader_tls"] / sample_total if sample_total else 0.0,
            libc_memory_pct=100.0 * category_counts["libc_memory"] / sample_total if sample_total else 0.0,
            sync_scheduler_pct=100.0 * category_counts["sync_scheduler"] / sample_total if sample_total else 0.0,
            syscall_pct=100.0 * category_counts["syscall"] / sample_total if sample_total else 0.0,
            cuda_collective_pct=100.0 * category_counts["cuda_collective"] / sample_total if sample_total else 0.0,
            unknown_pct=100.0 * category_counts["unknown"] / sample_total if sample_total else 0.0,
            top_first_frame=top_frame, top_first_frame_samples=top_count,
            top_first_frame_pct=100.0 * top_count / sample_total if sample_total else 0.0,
            coverage_note=note,
        ))
        for frame, count in counts.most_common(12):
            hotspots.append(dict(arm=entry["arm"], rank=entry["rank"], role=entry["role"],
                                 frame=frame, samples=count,
                                 pct=100.0 * count / sample_total if sample_total else 0.0))
        for index, item in enumerate(accounting):
            window_rows.append(dict(
                arm=entry["arm"], rank=entry["rank"], role=entry["role"], interval=index,
                start_mono_s=f"{item['start']:.9f}", end_mono_s=f"{item['end']:.9f}",
                wall_us=(item["end"] - item["start"]) * 1e6,
                running_us=item["running_us"], offcpu_us=item["offcpu_us"],
                state_at_start=item["state_at_start"], switch_events=item["switch_events"],
                preempt_out=item["preempt_out"], blocked_out=item["blocked_out"],
                samples=sum(1 for sample in window_samples if sample["interval"] == index),
            ))
    write_csv(out / "perf_stack_summary.csv", rows)
    write_csv(out / "perf_stack_hotspots.csv", hotspots)
    write_csv(out / "perf_window_windows.csv", window_rows)

    fit_rows = []
    for entry in targets.values():
        arm, rank = entry["arm"], entry["rank"]
        mapping = device_mapping(raw / arm, rank, 1)
        mapping["raw"]["_folder"] = str(raw / arm)
        fit = fit_host_to_trace(mapping, rank, 1)
        fit_rows.append(dict(arm=arm, rank=rank, role=entry["role"],
                             monotonic_to_kineto_slope=fit["slope"],
                             max_abs_residual_us=float(np.max(np.abs(fit["residual_us"]))),
                             mean_abs_residual_us=float(np.mean(np.abs(fit["residual_us"])))))
    write_csv(out / "perf_clock_fit.csv", fit_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--perf", type=Path, required=True)
    parser.add_argument("--fields", type=Path, required=True)
    parser.add_argument("--coverage", type=Path, required=True)
    args = parser.parse_args()
    windows = read_csv(args.out / "windows.csv")
    coverage = read_csv(args.coverage)
    targets, _ = build_targets(args.raw, windows, coverage)
    events = parse_switches(args.fields, targets)
    samples = parse_samples(args.perf, targets)
    write_summaries(args.raw, targets, events, samples, args.out)


if __name__ == "__main__":
    main()
