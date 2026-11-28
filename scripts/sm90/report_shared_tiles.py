"""Compare SM90 native GC shared M32, shared M64 and vLLM shared on one replay."""
import json
import sys
from pathlib import Path

import numpy as np

from report_eight import check, read_arm
from main20_report import write_rows
from shared_pair_cpu import NAMES, intervals


def timing(folder, fused):
    ranks, tails = [], []
    for rank in range(8):
        with np.load(folder / f"rank{rank}.npz") as data:
            bounds = data["ffn_ns"]
            clocks = data["clocks"]
            ranks.append(intervals(bounds, clocks, data["counts"]))
            if fused:
                # Unwrap the existing 32-bit microsecond stamps against the same
                # rank's FFN start. Slot 17 is an observed L2 CTA exit.
                marks = data["shared_stamps"].astype(np.uint32).astype(np.int64)
                check(marks.shape == (240, 3) and np.array_equal(marks[:, 0], clocks[-240:, 4]),
                      "shared ring epoch mismatch")
                begin, end = bounds.reshape(-1, 2)[-240:].T // 1000
                stamps = (marks[:, 1:] - begin[:, None]) % (1 << 32)
                valid = (stamps[:, 0] <= stamps[:, 1]) & (stamps[:, 1] <= end - begin)
                first = clocks[-240:, 1] // 1000 - begin
                tails.extend((stamps[valid, 1] - first[valid]).tolist())
    calls = np.stack(ranks, axis=1)
    slow = calls[np.arange(len(calls)), calls[:, :, 0].argmax(1)]
    means = slow.mean(0) / 1000
    check(np.isclose(means[0], means[1:].sum()), "FFN phase sum")
    return dict(zip(NAMES, means)), tails


def report(root):
    sha = (root / "measurement.sha").read_text().strip()
    old = "47e9b06328336e10584ef942a710aa332538327b"
    audits, native, controls, fingerprints = {}, {}, {}, {}
    summary, results = [], {}
    for key, arm, width in (("0", 0, 32), ("2", 2, 32), ("2-m64", 2, 64), ("3", 3, 0)):
        folder = root / f"arm{key}"
        expected_sha = old if key == "0" and (root / "reference-path").is_file() else sha
        check((folder / "complete").is_file() and
              (folder / "execution.sha").read_text().strip() == expected_sha, f"{key}: source")
        row, audit, tokens = read_arm(root, arm, steps=1000, kv_gib=10, tuned=True, folder=folder)
        owner = "_ZeroShared" if width else "DeepseekV4MLP"
        check(audit["shared"] == [[owner] * 40] * 8, f"{key}: shared owner")
        run = json.loads((folder / "run.json").read_text())
        for phase in ("stats", "measured"):
            check(all(r["ordinal_calls_verified"] == 40000 for r in run[phase]["audit"]),
                  f"{key}: route ordinal coverage")
        audits[key], native[key] = audit, tokens
        if key == "0":
            continue
        for name in ("trajectory", "order", "stats", "measured", "prefill_kv", "kv", "weights",
                     "placement", "installed_layout", "counts"):
            check(audit[name] == audits["0"][name], f"{key}: {name} differs from publisher")
        env = run["controls"].copy()
        check(env.pop("GC_MOE_SM90_SPLIT_SHARED_M") == str(width or 32) and
              env.pop("GC_MOE_SHARED_IN_GC") == str(int(bool(width))), f"{key}: shared controls")
        controls[key] = env
        check(env == controls["2"], f"{key}: other runtime controls differ")
        roles = (1, 4, 8, 2, 3) if width else (1, 2, 3)
        l1_grid = {32: 144, 64: 72, 0: 0}[width]
        l2_grid = {32: 18, 64: 9, 0: 0}[width]
        marker = (f"[shared-decode] tokens=256 routed_m=64 shared_m={width} "
                  f"l1_grid={l1_grid} l2_grid={l2_grid} roles={roles}")
        check(marker in (folder / "runner.log").read_text(errors="replace"), f"{key}: missing {marker}")
        if width:
            workers = sorted(run["measured"]["audit"], key=lambda r: r["rank"])
            fingerprints[key] = [r["final_shared_sha256"] for r in workers]
            check(all(set(d) == {"shared_l2_acts", "shared_l2_acts_sf", "shared_y"} and
                      all(len(h) == 64 for h in d.values()) for d in fingerprints[key]),
                  f"{key}: missing final-layer tensors")
        phases, tails = timing(folder, bool(width))
        check(np.isclose(phases["ffn"], 25 * row["moe_ms_per_step"]), f"{key}: FFN reduction")
        summary.append(dict(arm=key, step_ms=row["step_ms"], **{k + "_us": v for k, v in phases.items()},
                            ring_valid_rank_calls=len(tails),
                            l2_exit_after_first_routed_pct=100 * np.mean(np.asarray(tails) > 0) if tails else None,
                            median_l2_exit_minus_first_routed_us=np.median(tails) if tails else None))
        results[key] = (phases["ffn"], row["step_ms"])
    write_rows(root / "shared-timings.csv", summary)
    pairs = []
    for a, b in (("2", "2-m64"), ("3", "2-m64"), ("3", "2")):
        diff = int(np.count_nonzero(native[a] != native[b]))
        warm_equal = all(audits[a]["warm_native"]) and audits[a]["warm_native"] == audits[b]["warm_native"]
        tensors_equal = fingerprints[a] == fingerprints[b] if a in fingerprints and b in fingerprints else None
        pairs.append(dict(reference=a, candidate=b, ffn_change_pct=100*(results[b][0]/results[a][0]-1),
                          step_change_pct=100*(results[b][1]/results[a][1]-1), native_top1_differences=diff,
                          native_samples=native[a].size, warm_native_equal=warm_equal,
                          last_layer_shared_tensors_equal=tensors_equal))
    write_rows(root / "shared-pairs.csv", pairs)
    fused = pairs[0]
    check(fused["warm_native_equal"] and fused["native_top1_differences"] == 0 and
          fused["last_layer_shared_tensors_equal"], "M64 differs from M32 on the frozen workload")


if __name__ == "__main__":
    report(Path(sys.argv[1]))
