#!/usr/bin/env python3
import hashlib
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
    rr = [row["rr_calls"] for row in rows if "rr_calls" in row]
    layouts = [row for row in rows if "replica_layers" in row]
    generated = [row for row in rows if "generated_ids" in row]
    if len(rr) != 1 or len(layouts) != 1 or len(generated) != 1:
        raise RuntimeError(f"{work}: incomplete JSON records rr={len(rr)} layouts={len(layouts)} generated={len(generated)}")
    return {
        "rr_calls": rr[0],
        "replica_layers": layouts[0]["replica_layers"],
        "generated_ids": generated[0]["generated_ids"],
        "backend": layouts[0]["backend"],
    }


def main():
    base = pathlib.Path(__file__).resolve().parent
    previous = base.parent / "v41_eplb_decode_results"
    works = {
        "deep_gemm_mega_moe": pathlib.Path(sys.argv[1]),
        "gc_moe": pathlib.Path(sys.argv[2]),
    }
    expected = {
        "deep_gemm_mega_moe": json.loads((previous / "replica-layers-deep_gemm_mega_moe.json").read_text()),
        "gc_moe": json.loads((previous / "replica-layers-gc_moe.json").read_text()),
    }
    reference_ids = json.loads((previous / "decode-ids.json").read_text())
    arms = {name: arm_result(work) for name, work in works.items()}

    for name, arm in arms.items():
        if len(arm["replica_layers"]) != 40:
            raise RuntimeError(f"{name}: expected 40 layers, got {len(arm['replica_layers'])}")
        if len(arm["rr_calls"]) != 8 or min(arm["rr_calls"]) < 40:
            raise RuntimeError(f"{name}: bad rr_calls {arm['rr_calls']}")

    mega = arms["deep_gemm_mega_moe"]["replica_layers"]
    gc = arms["gc_moe"]["replica_layers"]
    prev_mega = expected["deep_gemm_mega_moe"]
    prev_gc = expected["gc_moe"]

    differences = []
    for i, (a, b) in enumerate(zip(mega, gc), 1):
        if a != b:
            differences.append({"layer": i, "mega": a, "gc": b})
    prev_differences = []
    for i, row in enumerate(mega, 1):
        if row != prev_mega[i - 1] or row != prev_gc[i - 1]:
            prev_differences.append({"layer": i, "current": row,
                                     "previous_mega": prev_mega[i - 1],
                                     "previous_gc": prev_gc[i - 1]})

    id_differences = []
    for name, arm in arms.items():
        if arm["generated_ids"] != reference_ids[name]:
            id_differences.append({"arm": name, "current": arm["generated_ids"],
                                   "previous": reference_ids[name]})
    cross_id_difference = arms["deep_gemm_mega_moe"]["generated_ids"] != arms["gc_moe"]["generated_ids"]

    shared = hashlib.sha256(str([(row["layer"], row["placement_sha256"]) for row in mega]).encode()).hexdigest()
    summary = {
        "source_commit": pathlib.Path(works["deep_gemm_mega_moe"] / "source.sha").read_text().strip(),
        "task_blob": "9ce26ee4b06b248a60a79694a11c59268a0aa77c",
        "launches": 2,
        "arms": list(arms),
        "mode": "eplb-decode+deepinfer-rr",
        "rr_calls": {name: arm["rr_calls"] for name, arm in arms.items()},
        "rr_calls_min": {name: min(arm["rr_calls"]) for name, arm in arms.items()},
        "rr_calls_all_ranks": {name: len(arm["rr_calls"]) == 8 for name, arm in arms.items()},
        "generated_ids": {name: arm["generated_ids"] for name, arm in arms.items()},
        "reference_ids": reference_ids,
        "layers_total": 40,
        "current_arms_equal_layers": len(differences) == 0,
        "previous_arms_equal_layers": len(prev_differences) == 0,
        "first_current_hash_difference": None if not differences else differences[0]["layer"],
        "first_previous_hash_difference": None if not prev_differences else prev_differences[0]["layer"],
        "first_id_difference": None if not id_differences and not cross_id_difference else id_differences,
        "all_current_layers": mega,
        "previous_mega_layers": prev_mega,
        "previous_gc_layers": prev_gc,
        "shared_layer_record_sha256": shared,
        "rr_preflight_exit_codes": {
            name: int((works[name] / "rr-preflight-exit-code.txt").read_text().strip())
            for name in works
        },
        "model_exit_codes": {
            name: int((works[name] / "exit-code.txt").read_text().strip())
            for name in works
        },
        "launcher_exit_codes": {
            name: int((works[name] / "launcher-exit-code.txt").read_text().strip())
            for name in works
        },
    }

    comparison = {
        "summary": summary,
        "current_differences": differences,
        "previous_differences": prev_differences,
        "id_differences": id_differences,
    }
    (base / "replica-comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    (base / "decode-ids.json").write_text(json.dumps(summary["generated_ids"], indent=2) + "\n")
    (base / "rr-calls.json").write_text(json.dumps(summary["rr_calls"], indent=2) + "\n")
    (base / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    lines = ["# DeepInfer RR layer comparison", ""]
    lines += ["| layer | current hash | previous Mega hash | previous GC hash | equal |",
              "|---:|---|---|---|:---:|"]
    for i, (row, pm, pg) in enumerate(zip(mega, prev_mega, prev_gc), 1):
        h = row["placement_sha256"]
        lines.append(f"| {i} | `{h}` | `{pm['placement_sha256']}` | `{pg['placement_sha256']}` | {row == pm == pg} |")
    (base / "replica-comparison.txt").write_text("\n".join(lines) + "\n")

    check = [
        "SM100 V41 DEEPINFER RR ASSERTIONS",
        f"source_commit={summary['source_commit']}",
        f"rr_calls_mega={summary['rr_calls']['deep_gemm_mega_moe']}",
        f"rr_calls_gc={summary['rr_calls']['gc_moe']}",
        f"generated_mega={summary['generated_ids']['deep_gemm_mega_moe']}",
        f"generated_gc={summary['generated_ids']['gc_moe']}",
        f"current_layers_equal={summary['current_arms_equal_layers']}",
        f"previous_layers_equal={summary['previous_arms_equal_layers']}",
        f"first_current_hash_difference={summary['first_current_hash_difference']}",
        f"first_previous_hash_difference={summary['first_previous_hash_difference']}",
        f"first_id_difference={summary['first_id_difference']}",
        f"shared_layer_record_sha256={shared}",
    ]
    (base / "layout-check.txt").write_text("\n".join(check) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
