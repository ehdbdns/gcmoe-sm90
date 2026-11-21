"""Read-only inventory. Run with the Python interpreter intended for inference."""
import argparse
import importlib.metadata as metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys


def command(args):
    try:
        run = subprocess.run(args, capture_output=True, text=True, timeout=45)
        return {"exit": run.returncode, "output": (run.stdout + run.stderr).strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": str(exc)}


def repository(path):
    git = ["git", "-C", str(path)]
    files = command(git + ["ls-files"])
    candidates = [f for f in files.get("output", "").splitlines()
                  if any(k in f.lower() for k in ("megamoe", "mega_moe", "sm100", "eco_plan"))]
    return dict(path=str(path), head=command(git + ["rev-parse", "HEAD"]),
                tracked_changes=command(git + ["diff", "--name-only", "HEAD"]),
                candidate_files=candidates[:100], candidate_count=len(candidates))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, action="append", default=[])
    parser.add_argument("--model", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--scratch", type=Path, default=Path("/tmp"))
    args = parser.parse_args()
    result = dict(os=platform.platform(), machine=platform.machine(),
                  python=sys.version, interpreter=sys.executable,
                  repositories=[repository(p.resolve()) for p in args.repo])
    commands = {
        "gpus": ["nvidia-smi", "--query-gpu=index,name,compute_cap,memory.total,memory.free,driver_version", "--format=csv"],
        "gpu_list": ["nvidia-smi", "-L"], "topology": ["nvidia-smi", "topo", "-m"],
        "nvcc": ["nvcc", "--version"], "ptxas": ["ptxas", "--version"],
        "cxx": ["c++", "--version"], "cmake": ["cmake", "--version"], "ninja": ["ninja", "--version"],
        "rdma": ["ibv_devices"],
    }
    result["commands"] = {k: command(v) for k, v in commands.items()}
    result["packages"] = {}
    for name in ("torch", "vllm", "sglang", "triton", "nvidia-cutlass-dsl", "deep-gemm",
                 "deep-ep", "flashinfer-python", "nvidia-nccl-cu12", "nvidia-nccl-cu13"):
        try:
            result["packages"][name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result["packages"][name] = None
    result["module_paths"] = {}
    for name in ("deep_gemm", "deep_ep", "mega_moe", "gc_moe", "vllm", "sglang", "flashinfer"):
        spec = importlib.util.find_spec(name)
        result["module_paths"][name] = spec.origin if spec else None
    try:
        import torch
        n = torch.cuda.device_count()
        result["torch"] = dict(version=torch.__version__, cuda=torch.version.cuda,
                               cxx11_abi=torch._C._GLIBCXX_USE_CXX11_ABI,
                               arch_list=torch.cuda.get_arch_list(),
                               capability=[torch.cuda.get_device_capability(i) for i in range(n)],
                               peer=[[i == j or torch.cuda.can_device_access_peer(i, j)
                                      for j in range(n)] for i in range(n)])
        if n:
            result["torch"]["nccl"] = torch.cuda.nccl.version()
    except Exception as exc:
        result["torch_error"] = repr(exc)
    result["scratch"] = dict(path=str(args.scratch), writable=os.access(args.scratch, os.W_OK),
                             free_bytes=shutil.disk_usage(args.scratch).free if args.scratch.is_dir() else None)
    if args.model:
        cfg = args.model / "config.json"
        result["model"] = {"path": str(args.model), "readable_config": os.access(cfg, os.R_OK)}
        if os.access(cfg, os.R_OK):
            obj = json.loads(cfg.read_text())
            keys = ("model_type", "architectures", "hidden_size", "num_hidden_layers", "n_routed_experts",
                    "num_experts_per_tok", "n_shared_experts", "moe_intermediate_size", "torch_dtype")
            result["model"]["config"] = {k: obj[k] for k in keys if k in obj}
            result["model"]["quant_method"] = (obj.get("quantization_config") or {}).get("quant_method")
    if args.dataset:
        result["dataset"] = dict(path=str(args.dataset), readable=os.access(args.dataset, os.R_OK))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
