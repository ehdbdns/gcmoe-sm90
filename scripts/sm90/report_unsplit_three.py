"""Compare exactly three H20 unsplit consumers to the archived BS256 GC pairs."""
import json
import sys
from pathlib import Path

import numpy as np

from report_eight import check, read_arm
from main20_report import write_rows
from report_shared_tiles import timing

ARCHIVE = (Path(__file__).resolve().parents[2] / "docs/trace_executor_20261119"
           / "sm90_h20_token_ir_eight_results/kv-10")
OLD_SHA = "47e9b06328336e10584ef942a710aa332538327b"
# Only these paths must move when reusing the banks from a new checkout.
PATH_CONTROLS = {"GC_MOE_EXEC_ROOT", "GC_MOE_SRC", "GC_MOE_FIXED_DIR", "GC_MOE_STRICT_STATE_DIR"}


def measure(root, arm, variant, sha):
    folder = root / f"arm{arm}"
    check((folder / "complete").is_file() and (folder / "execution.sha").read_text().strip() == sha,
          f"{folder}: incomplete or wrong revision")
    row, audit, native = read_arm(root, arm, kv_gib=10, extra=4, tuned=True)
    run = json.loads((folder / "run.json").read_text())
    fused = variant != "unfused"
    owner = "_ZeroShared" if fused else "DeepseekV4MLP"
    check(audit.pop("shared") == [[owner] * 40] * 8, f"{folder}: wrong shared owner")
    warm = audit.pop("warm_native")
    check(all(warm), f"{folder}: missing warm outputs")
    controls = {k: v for k, v in run["controls"].items() if k not in PATH_CONTROLS}
    check(controls.pop("GC_MOE_SM90_SPLIT_SHARED", "1") == ("0" if variant == "unsplit" else "1")
          and controls.pop("GC_MOE_SHARED_IN_GC") == str(int(fused)), f"{folder}: invalid shared mode")
    for phase in ("stats", "measured"):
        audits = sorted(run[phase]["audit"], key=lambda r: r["rank"])
        check(all(r["ordinal_calls_verified"] == 40000 for r in audits), f"{folder}: routing order")
        audit[f"{phase}_first_epochs"] = [r["first_epoch"] for r in audits]
    prefix = (5, 7) if arm in (7, 8) else ()
    middle = {"unsplit": (1, 4), "split": (1, 4, 8), "unfused": (1,)}[variant]
    roles = prefix + middle + (2, 3)
    marker = f"[six] GC M=64 roles={roles}"
    if variant == "unsplit":
        marker = ("[shared-unsplit] tokens=256 shared_m=64 grid=72 threads=256 smem=199680 "
                  f"phases=(0, 1) routed_wait=(4,) roles={roles}")
    check(marker in (folder / "runner.log").read_text(errors="replace"), f"{folder}: missing {marker}")
    phases, _ = timing(folder, False)  # archived clocks; no extra GPU instrumentation
    check(np.isclose(phases["ffn"], 25 * row["moe_ms_per_step"]), "FFN phase reduction")
    numbers = dict(arm=arm, shared=variant, measurement_sha=sha, window_steps=1000, kv_gib_per_rank=10,
                   extra_per_rank=4 if arm >= 4 else 0, step_ms=row["step_ms"],
                   ffn_ms_per_step=row["moe_ms_per_step"], ffn_us_per_layer=phases["ffn"],
                   **{k: row[k] for k in ("block_m", "token_ir", "total_m_blocks", "peak_rank_m_blocks",
                                          "m_block_ir", "active_physical_slots", "accepted")},
                   routed_mae_pp=row.get("routed_mae_pp"),
                   **{k + "_us": v for k, v in phases.items() if k != "ffn"})
    return numbers, audit, native, warm, controls


def report(root):
    sha = (root / "measurement.sha").read_text().strip()
    for file in ("gpus.csv", "input.sha256"):
        check((root / file).read_bytes() == (ARCHIVE / file).read_bytes(), f"archived {file} differs")
    check((root / "reference-path").read_text().strip(), "missing bank provenance")
    rows, pairs = [], []
    for arm, external in ((2, 3), (4, 6), (7, 8)):
        new, audit, tokens, warm, controls = measure(root, arm, "unsplit", sha)
        rows.append(new)
        for old_arm, variant in ((arm, "split"), (external, "unfused")):
            old, prior, old_tokens, old_warm, old_controls = measure(ARCHIVE, old_arm, variant, OLD_SHA)
            changed = [k for k in audit if audit[k] != prior[k]]
            check(not changed, f"arm {arm} vs {old_arm}: workload mismatch {changed}")
            changed = [k for k in controls.keys() | old_controls.keys() if controls.get(k) != old_controls.get(k)]
            check(not changed, f"arm {arm} vs {old_arm}: runtime control mismatch {changed}")
            for name in ("runtime.log", "installed-layout.json"):
                check((root / f"arm{arm}" / name).read_bytes() == (ARCHIVE / f"arm{old_arm}" / name).read_bytes(),
                      f"arm {arm} vs {old_arm}: {name} mismatch")
            rows.append(old)
            diff = int(np.count_nonzero(tokens != old_tokens))
            pairs.append(dict(unsplit_arm=arm, reference_arm=old_arm, reference_shared=variant,
                              step_delta_ms=new["step_ms"] - old["step_ms"],
                              step_delta_pct=100 * (new["step_ms"] / old["step_ms"] - 1),
                              ffn_delta_us=new["ffn_us_per_layer"] - old["ffn_us_per_layer"],
                              ffn_delta_pct=100 * (new["ffn_us_per_layer"] / old["ffn_us_per_layer"] - 1),
                              native_diff=diff, samples=tokens.size, warm_native_equal=warm == old_warm,
                              strict_pair_comparable=(warm == old_warm and diff == 0)))
    write_rows(root / "unsplit-summary.csv", rows)
    write_rows(root / "unsplit-pairs.csv", pairs)
    check(all(p["strict_pair_comparable"] for p in pairs if p["reference_shared"] == "split"),
          "fused output mismatch; see unsplit-pairs.csv")


if __name__ == "__main__":
    report(Path(sys.argv[1]))
