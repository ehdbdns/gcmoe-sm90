"""Locate or prepare a private GLM runtime and verify real expert payloads on CPU."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
from tempfile import mkdtemp


MODEL = Path("/mnt/hf3fs/3fs/models/GLM-5.3-FP4Moe-FP8ue8m0")
CONFIG_SHA = "82bd28e0ac294868b134211748cbe27a166621928231607a5510fced5a86cb2b"
INDEX_SHA = "fbf6cbce2835c7002fcf62a0e9bcf57016323d25083f4f9d011ac30ccc1aac0c"
SCRIPTS = Path(__file__).resolve().parent


def candidates():
    paths = [os.environ.get("GLM_PYTHON"), shutil.which("python"), sys.executable]
    roots = [Path.home() / ".conda", Path("/opt/conda"), Path("/opt/miniconda3"),
             Path("/data0/home/admin/workspace/megamoe"), Path("/home/admin/workspace/megamoe")]
    conda_file = Path.home() / ".conda/environments.txt"
    if conda_file.is_file():
        paths.extend(str(Path(p) / "bin/python") for p in conda_file.read_text().splitlines() if p)
    for base in roots:
        for pat in ("bin/python", "venv/bin/python", "envs/*/bin/python", "*/venv/bin/python"):
            paths.extend(map(str, sorted(base.glob(pat))))
    return [p for p in dict.fromkeys(paths) if p and os.access(p, os.X_OK)]


def identify():
    # Called in the candidate interpreter. Preserve its own sys.path and torch build.
    from glm53_checkpoint import loader
    info = loader()
    try:
        import torch
        import safetensors
        import vllm
        info["torch_cuda_build"] = torch.version.cuda
        info["usable"] = bool(torch.version.cuda and hasattr(torch, "float8_e8m0fnu"))
        info["vllm_source"] = vllm.__file__
        # The generic fp8 parser and a site fork can interpret expert_dtype
        # differently. Exercise the installed parser with the actual config.
        try:
            from vllm.model_executor.layers.quantization.fp8 import Fp8Config
            actual = json.loads(Path(os.environ["GLM_CHECK_CONFIG"]).read_text())["quantization_config"]
            parsed = Fp8Config.from_config(dict(actual))
            info["parsed_fp8_config"] = {key: getattr(parsed, key, None) for key in
                                        ("store_dtype", "expert_dtype", "is_scale_e8m0", "weight_block_size")}
        except Exception as exc:
            info["quant_parser_error"] = f"{type(exc).__name__}: {exc}"
    except (ImportError, OSError, RuntimeError) as exc:
        info.update(usable=False, error=type(exc).__name__ + ": " + str(exc))
    return info


def explain(log):
    lines = log.read_text(errors="replace").splitlines()
    lines = [l for l in lines if re.search(r"error|exception|failed|not found|no matching|no space|permission", l, re.I)]
    return [re.sub(r"\S+://\S+", "[URL]", l)[:350] for l in lines[-5:]]


def run(args, log, env):
    with log.open("w") as out:
        return subprocess.run(list(map(str, args)), env=env, stdout=out, stderr=subprocess.STDOUT).returncode


def work(model, storage, dest, report):
    for name, expected in (("config.json", CONFIG_SHA), ("model.safetensors.index.json", INDEX_SHA)):
        if hashlib.sha256((model / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"{name} differs from receipt 68d019e")
    # Attention/indexer scale names matter to vLLM's full-model loader as well.
    # Read two sparse layers to include both full and shared indexer variants.
    index = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
    selected = {}
    for key, shard in index.items():
        parts = key.split(".")
        if "layers" in parts:
            layer = int(parts[parts.index("layers") + 1])
            if layer not in (0, 3, 6) or ".experts." in key:
                continue
        selected.setdefault(shard, []).append(key)
    samples = {}
    for shard, keys in selected.items():
        with (model / shard).open("rb") as f:
            length, = struct.unpack("<Q", f.read(8))
            if not 0 < length <= 100_000_000:
                raise ValueError(f"invalid header: {shard}")
            header = json.loads(f.read(length))
        samples.update({key: [header[key]["dtype"], header[key]["shape"]] for key in keys})
    report["full_model_loader_tensors"] = samples
    private = Path(mkdtemp(prefix="glm53-runtime-", dir=storage))
    report["private_workspace"] = str(private)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "GC_MOE_SM100_TRACE": "2",
           "PIP_DISABLE_PIP_VERSION_CHECK": "1", "GLM_CHECK_CONFIG": str(model / "config.json")}
    def check(py):
        idx = len(report["interpreters"])
        path = private / f"python-{idx}.json"
        log = private / f"python-{idx}.log"
        rc = run([py, __file__, "--identify", path], log, env)
        info = json.loads(path.read_text()) if rc == 0 else {"usable": False, "error": explain(log)}
        report["interpreters"].append({"path": str(py), "usable": info["usable"], "packages": info.get("packages"),
                                       "error": info.get("error")})
        if info["usable"]:
            report["runtime"] = info
        return info["usable"]
    py = next((p for p in candidates() if check(p)), None)
    if not py:
        log = private / "install.log"
        rc = run([sys.executable, "-m", "venv", private / "venv"], log, env)
        py = private / "venv/bin/python"
        if rc == 0:
            rc = run([py, "-m", "pip", "install", "--no-cache-dir", "--only-binary=vllm,torch",
                      "vllm==0.31.0", "safetensors>=0.7.0"], log, env)
        if rc or not check(py):
            report["install_diagnostics"] = explain(log)
            raise RuntimeError(f"private runtime setup failed; see {log}")
    report["python"] = str(py)
    subprocess_args = [py, SCRIPTS / "glm53_weights.py", model, "--out", dest / "weight-checks.json"]
    log = private / "weights.log"
    if run(subprocess_args, log, env):
        report["weight_diagnostics"] = explain(log)
        raise RuntimeError(f"weight layout validation failed; see {log}")
    checks = json.loads((dest / "weight-checks.json").read_text())
    if len(checks) != 54 or len({c["key"] for c in checks}) != 54:
        raise ValueError("incomplete layer/projection coverage")
    report.update(status="cpu_verified", checked_fields=len(checks))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--identify", type=Path)
    p.add_argument("--model", type=Path, default=MODEL)
    p.add_argument("--storage", type=Path, default=Path(os.environ.get("GLM_WORK_ROOT", os.environ.get("TMPDIR", "/tmp"))))
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    if a.identify:
        a.identify.write_text(json.dumps(identify()))
        return 0
    if not a.out:
        p.error("provide --out")
    dest = a.out / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    dest.mkdir(parents=True)
    report = dict(status="blocked", interpreters=[], model=str(a.model), gpu_arms=0,
                  revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip())
    try:
        a.storage.mkdir(parents=True, exist_ok=True)
        work(a.model, a.storage, dest, report)
    except (ValueError, KeyError, OSError, struct.error, RuntimeError, subprocess.SubprocessError) as e:
        report["error"] = str(e)
    (dest / "runtime.json").write_text(json.dumps(report, indent=2) + "\n")
    print(report["status"], report.get("error", ""), dest)
    return int(report["status"] != "cpu_verified")


if __name__ == "__main__":
    sys.exit(main())
