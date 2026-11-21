#!/usr/bin/env python3
import json
import pathlib
import sys


def load_json_lines(path):
    rows = []
    for line in pathlib.Path(path).read_text(errors="replace").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def arm_result(work):
    rows = load_json_lines(pathlib.Path(work) / "model-full.log")
    records = {
        "placement": [row["placement"] for row in rows if "placement" in row],
        "rr_calls": [row["rr_calls"] for row in rows if "rr_calls" in row],
        "replica_layers": [row["replica_layers"] for row in rows if "replica_layers" in row],
        "generated_ids": [row["generated_ids"] for row in rows if "generated_ids" in row],
    }
    if any(len(value) != 1 for value in records.values()):
        raise RuntimeError(f"{work}: incomplete records {[(k, len(v)) for k, v in records.items()]}")
    return {key: value[0] for key, value in records.items()}


def main():
    base = pathlib.Path(__file__).resolve().parent
    previous = base.parent / "v41_deepinfer_rr_results"
    works = {
        "deep_gemm_mega_moe": pathlib.Path(sys.argv[1]),
        "gc_moe": pathlib.Path(sys.argv[2]),
    }
    arms = {name: arm_result(work) for name, work in works.items()}
    previous_ids = json.loads((previous / "decode-ids.json").read_text())
    previous_layers = {
        "deep_gemm_mega_moe": json.loads((previous / "deep_gemm_mega_moe" / "replica-layers.json").read_text())[0]["replica_layers"],
        "gc_moe": json.loads((previous / "gc_moe" / "replica-layers.json").read_text())[0]["replica_layers"],
    }
    for name, arm in arms.items():
        if len(arm["rr_calls"]) != 8 or min(arm["rr_calls"]) < 40:
            raise RuntimeError(f"{name}: bad rr_calls {arm['rr_calls']}")
        if len(arm["replica_layers"]) != 40:
            raise RuntimeError(f"{name}: expected 40 layers")
        if len(arm["placement"]["changed_slots"]) != 40 or min(arm["placement"]["changed_slots"]) <= 0:
            raise RuntimeError(f"{name}: placement did not move every layer")
        if arm["placement"]["sampled_slot_mismatches"] != 0:
            raise RuntimeError(f"{name}: sampled weight mismatch is nonzero")

    mega = arms["deep_gemm_mega_moe"]
    gc = arms["gc_moe"]
    plan_same = mega["placement"]["plan_sha256"] == gc["placement"]["plan_sha256"]
    layers_equal = mega["replica_layers"] == gc["replica_layers"]
    ids_equal = mega["generated_ids"] == gc["generated_ids"]
    baseline_equal = all(arm["generated_ids"] == previous_ids[name] for name, arm in arms.items())
    layer_diffs = [i for i, (a, b) in enumerate(zip(mega["replica_layers"], gc["replica_layers"]), 1) if a != b]
    previous_diffs = [i for i, (a, b) in enumerate(zip(mega["replica_layers"], previous_layers["deep_gemm_mega_moe"]), 1) if a == b]

    summary = {
        "source_commit": pathlib.Path(works["deep_gemm_mega_moe"] / "source.sha").read_text().strip(),
        "task_blob": "179c3945708551e41332369fbc778689b8f1c46e",
        "launches": 2,
        "arms": list(arms),
        "plan_sha256": mega["placement"]["plan_sha256"],
        "same_plan_sha": plan_same,
        "changed_slots_per_layer": mega["placement"]["changed_slots"],
        "changed_slots_total": sum(mega["placement"]["changed_slots"]),
        "sampled_slot_mismatches": {
            name: arm["placement"]["sampled_slot_mismatches"] for name, arm in arms.items()
        },
        "sample_bytes_per_component_per_slot": 32,
        "rr_calls": {name: arm["rr_calls"] for name, arm in arms.items()},
        "rr_calls_min": {name: min(arm["rr_calls"]) for name, arm in arms.items()},
        "generated_ids": {name: arm["generated_ids"] for name, arm in arms.items()},
        "previous_ids": previous_ids,
        "baseline_ids_equal": baseline_equal,
        "two_arms_ids_equal": ids_equal,
        "layers_total": 40,
        "two_arms_layers_equal": layers_equal,
        "first_layer_difference": layer_diffs[0] if layer_diffs else None,
        "layers_equal_to_previous": previous_diffs,
        "final_layer_hashes": mega["replica_layers"],
        "rr_preflight_exit_codes": {
            name: int((works[name] / "rr-preflight-exit-code.txt").read_text().strip())
            for name in works
        },
        "model_exit_codes": {
            name: int((works[name] / "exit-code.txt").read_text().strip()) for name in works
        },
        "launcher_exit_codes": {
            name: int((works[name] / "launcher-exit-code.txt").read_text().strip()) for name in works
        },
    }
    (base / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (base / "decode-ids.json").write_text(json.dumps(summary["generated_ids"], indent=2) + "\n")
    (base / "rr-calls.json").write_text(json.dumps(summary["rr_calls"], indent=2) + "\n")
    (base / "changed-slots.json").write_text(json.dumps(summary["changed_slots_per_layer"], indent=2) + "\n")
    comparison = {
        "summary": summary,
        "mega_layers": mega["replica_layers"],
        "gc_layers": gc["replica_layers"],
        "previous_mega_layers": previous_layers["deep_gemm_mega_moe"],
    }
    (base / "placement-comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    table = ["# Layer-specific placement comparison", "", "| layer | moved slots | hash |", "|---:|---:|---|"]
    for i, (moved, row) in enumerate(zip(summary["changed_slots_per_layer"], mega["replica_layers"]), 1):
        table.append(f"| {i} | {moved} | `{row['placement_sha256']}` |")
    (base / "placement-comparison.txt").write_text("\n".join(table) + "\n")
    check = [
        "SM100 V41 DEEPINFER PLACEMENT ASSERTIONS",
        f"source_commit={summary['source_commit']}",
        f"plan_sha256={summary['plan_sha256']}",
        f"same_plan_sha={summary['same_plan_sha']}",
        f"changed_slots_total={summary['changed_slots_total']}",
        f"sampled_slot_mismatches={summary['sampled_slot_mismatches']}",
        f"rr_calls_mega={summary['rr_calls']['deep_gemm_mega_moe']}",
        f"rr_calls_gc={summary['rr_calls']['gc_moe']}",
        f"generated_mega={summary['generated_ids']['deep_gemm_mega_moe']}",
        f"generated_gc={summary['generated_ids']['gc_moe']}",
        f"baseline_ids_equal={summary['baseline_ids_equal']}",
        f"two_arms_ids_equal={summary['two_arms_ids_equal']}",
        f"two_arms_layers_equal={summary['two_arms_layers_equal']}",
        f"first_layer_difference={summary['first_layer_difference']}",
        f"layers_equal_to_previous={summary['layers_equal_to_previous']}",
    ]
    (base / "layout-check.txt").write_text("\n".join(check) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
