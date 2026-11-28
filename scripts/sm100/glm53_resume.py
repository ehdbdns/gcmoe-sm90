"""Mount completed GLM arms and their inputs in a separate writable run directory."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

ARCHIVE = Path(__file__).resolve().parents[2] / "docs/sm100_megamoe/glm53_formal_results/formal.20261128200535"
PLACEMENT_SHA = "87feac206f590f780a571791d37a2e49183d9dd09ee567ced206fb830d005cb4"
RR_ARCHIVE = ARCHIVE.parent / "formal.VHnr9l8N"


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def json_hash(obj):
    return hashlib.sha256(json.dumps(obj, separators=(",", ":"), sort_keys=True).encode()).hexdigest()


def check_separate(source, root):
    if source == root or source in root.parents or root in source.parents:
        raise ValueError("resume needs a separate private output directory")


def link_measurements(source, root, receipt, arm):
    files = [r for r in receipt["files"] if r["path"].startswith(f"{arm}/")]
    required = {f"{arm}/result.json", *(f"{arm}/{phase}-rank{r}.npz"
                                        for phase in ("stats", "measure") for r in range(8))}
    if not required.issubset(row["path"] for row in files):
        raise ValueError(f"incomplete {arm} inventory")
    linked = 0
    for row in files:
        path = source / row["path"]
        if not path.exists() and row["path"].startswith(f"{arm}/perfetto/"):
            # Diagnostic traces are already inventoried by the original receipt.
            continue
        if path.stat().st_size != row["bytes"] or file_hash(path) != row["sha256"]:
            raise ValueError(f"measurement differs from the published receipt: {path}")
        dest = root / row["path"]
        dest.parent.mkdir(exist_ok=True, parents=True)
        dest.symlink_to(path)
        linked += 1
    return linked


def stage(source, root):
    source, root = source.resolve(strict=True), root.resolve(strict=True)
    check_separate(source, root)
    receipt = json.loads((ARCHIVE / "receipt.json").read_text())
    revision = (source / "revision").read_text().strip()
    if revision != receipt["revision"] or receipt["complete_arms"] != ["vanilla"]:
        raise ValueError("wrong baseline revision or receipt")
    primary = json.loads((source / "vanilla/result.json").read_text())
    if primary["arm"] != "vanilla" or file_hash(source / "prompts.json") != primary["prompt_sha256"]:
        raise ValueError("baseline prompt/arm mismatch")
    if file_hash(source / "placement.npy") != PLACEMENT_SHA:
        raise ValueError("placement differs from the original eight-extra-slot cohort")
    verified = link_measurements(source, root, receipt, "vanilla")
    tokens = json.loads((source / "tokens.json").read_text())["tokens"]
    if set(tokens) != {f"req{i}" for i in range(256)}:
        raise ValueError("incomplete native feedback")
    # glm_end saves native_history[request, decode_step] in both windows.
    trajectory = np.asarray([tokens[f"req{i}"][-2000:] for i in range(256)])
    if trajectory.shape != (256, 2000):
        raise ValueError("incomplete full-batch reference trajectory")
    refs = ["prompts.json", "tokens.json", "model"]
    for phase in ("stats", "measure"):
        refs.append(f"kv-{phase}")
        if len(primary[phase]) != 8:
            raise ValueError("incomplete original phase")
        for r, worker in enumerate(primary[phase]):
            name = f"{phase}-reference-{r}.pt"
            if (source / name).stat().st_size <= 0 or worker["full_replays"] != 1000:
                raise ValueError(f"incomplete reference for {name}")
            refs.append(name)
            prefix = source / f"kv-{phase}/rank{r}"
            manifest = json.loads(prefix.with_suffix(".json").read_text())
            before = worker["boundary"]
            if (manifest["rank"] != r or manifest["sha256"] != before["source_sha256"]
                    or before["source_sha256"] != before["readback_sha256"]
                    or json_hash(manifest["layout"]) != before["layout_sha256"]
                    or json_hash(manifest["controls"]) != before["controls_sha256"]
                    or sum(manifest["layout"]["sizes"]) != before["kv_bytes"]
                    or prefix.with_suffix(".bin").stat().st_size != before["kv_bytes"]):
                raise ValueError(f"incomplete or mismatched KV reference: {prefix}")
            with np.load(source / f"vanilla/{phase}-rank{r}.npz") as data:
                expected = trajectory[:, :1000] if phase == "stats" else trajectory[:, 1000:]
                np.testing.assert_array_equal(data["native"], expected)
    for name in refs:
        (root / name).symlink_to((source / name).resolve(strict=True))
    for name in ("placement.npy", "arch-proof.json", "deepgemm-proof.json"):
        shutil.copyfile(source / name, root / name)
    (root / "resume.json").write_text(json.dumps(dict(
        vanilla_source=str(source), vanilla_revision=revision, consumer_revision=(root / "revision").read_text().strip(),
        source_receipt="a766f3a", vanilla_files_verified=verified, original_placement_sha256=PLACEMENT_SHA,
        original_gpus_in_rank_order=[s["gpu"] for s in primary["setup"]]), indent=2) + "\n")
    print(f"Verified {verified} vanilla files, 16 KV manifests and original feedback.")


def stage_eco_only(source, root):
    from glm53_matrix import reduce

    source, root = source.resolve(strict=True), root.resolve(strict=True)
    check_separate(source, root)
    receipt = json.loads((RR_ARCHIVE / "receipt.json").read_text())
    origin = json.loads((source / "resume.json").read_text())
    revision = (source / "revision").read_text().strip()
    if (receipt["complete_arms"] != ["vanilla", "rr"] or revision != receipt["revision"]
            or revision != origin["consumer_revision"] or file_hash(source / "placement.npy") != PLACEMENT_SHA):
        raise ValueError("wrong RR revision or eight-extra-slot placement")
    stage(Path(origin["vanilla_source"]), root)
    rr_files = link_measurements(source, root, receipt, "rr")
    resume = json.loads((root / "resume.json").read_text())
    setup = json.loads((root / "rr/result.json").read_text())["setup"]
    if ([r["gpu"] for r in setup] != resume["original_gpus_in_rank_order"]
            or any(r["trace_level"] != 2 or r["slots"] != 40 or r["block_m"] != 16 for r in setup)):
        raise ValueError("RR hardware, trace or slot geometry differs")
    rows = reduce(root, ("vanilla", "rr"), "reference-main.csv")
    if not all(row["full_native_match"] for row in rows):
        raise ValueError("completed RR inputs or native outputs differ from vanilla")
    resume.update(rr_source=str(source), rr_revision=revision, rr_files_verified=rr_files,
                  rr_receipt="36f8f19", scheduled_arms=["eco"])
    (root / "resume.json").write_text(json.dumps(resume, indent=2) + "\n")
    print(f"Verified {rr_files} RR files and reduced the completed arms. Only ECO will run.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("root", type=Path)
    p.add_argument("--eco-only", action="store_true")
    args = p.parse_args()
    (stage_eco_only if args.eco_only else stage)(args.source, args.root)
