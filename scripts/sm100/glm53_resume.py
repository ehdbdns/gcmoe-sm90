"""Mount the verified vanilla cohort in a new, writable RR/ECO run directory."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

ARCHIVE = Path(__file__).resolve().parents[2] / "docs/sm100_megamoe/glm53_formal_results/formal.20261128200535"
PLACEMENT_SHA = "87feac206f590f780a571791d37a2e49183d9dd09ee567ced206fb830d005cb4"


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def json_hash(obj):
    return hashlib.sha256(json.dumps(obj, separators=(",", ":"), sort_keys=True).encode()).hexdigest()


def stage(source, root):
    source, root = source.resolve(strict=True), root.resolve(strict=True)
    if source == root or source in root.parents or root in source.parents:
        raise ValueError("resume needs a separate private output directory")
    receipt = json.loads((ARCHIVE / "receipt.json").read_text())
    revision = (source / "revision").read_text().strip()
    if revision != receipt["revision"] or receipt["complete_arms"] != ["vanilla"]:
        raise ValueError("wrong baseline revision or receipt")
    files = [r for r in receipt["files"] if r["path"].startswith("vanilla/")]
    primary = json.loads((source / "vanilla/result.json").read_text())
    if primary["arm"] != "vanilla" or file_hash(source / "prompts.json") != primary["prompt_sha256"]:
        raise ValueError("baseline prompt/arm mismatch")
    if file_hash(source / "placement.npy") != PLACEMENT_SHA:
        raise ValueError("placement differs from the original eight-extra-slot cohort")
    available = []
    for row in files:
        path = source / row["path"]
        if not path.exists() and row["path"].startswith("vanilla/perfetto/"):
            # Diagnostic traces are already inventoried by the original receipt.
            continue
        if path.stat().st_size != row["bytes"] or file_hash(path) != row["sha256"]:
            raise ValueError(f"original measurement differs from the published receipt: {path}")
        available.append(row)
    tokens = json.loads((source / "tokens.json").read_text())["tokens"]
    if set(tokens) != {f"req{i}" for i in range(256)}:
        raise ValueError("incomplete native feedback")
    suffix = np.asarray([tokens[f"req{i}"][-1000:] for i in range(256)]).T
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
            if phase == "measure":
                with np.load(source / f"vanilla/measure-rank{r}.npz") as data:
                    np.testing.assert_array_equal(data["native"], suffix)
    for name in refs:
        (root / name).symlink_to((source / name).resolve(strict=True))
    for row in available:
        dest = root / row["path"]
        dest.parent.mkdir(exist_ok=True, parents=True)
        dest.symlink_to(source / row["path"])
    for name in ("placement.npy", "arch-proof.json", "deepgemm-proof.json"):
        shutil.copyfile(source / name, root / name)
    (root / "resume.json").write_text(json.dumps(dict(
        vanilla_source=str(source), vanilla_revision=revision, consumer_revision=(root / "revision").read_text().strip(),
        source_receipt="a766f3a", vanilla_files_verified=len(available), original_placement_sha256=PLACEMENT_SHA,
        original_gpus_in_rank_order=[s["gpu"] for s in primary["setup"]]), indent=2) + "\n")
    print(f"Verified {len(available)} vanilla files, 16 KV manifests and original feedback; running RR/ECO only.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("root", type=Path)
    args = p.parse_args()
    stage(args.source, args.root)
