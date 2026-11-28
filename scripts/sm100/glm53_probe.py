"""Verify that each granted device executes the same native SM103f cubin before loading GLM."""
import ctypes as c
import json
import os
import sys
from pathlib import Path

import torch


def probe(root):
    driver = c.CDLL("libcuda.so.1")

    def call(name, *args):
        status = getattr(driver, name)(*args)
        if status:
            error = c.c_char_p()
            driver.cuGetErrorName(status, c.byref(error))
            raise RuntimeError(f"{name}: {error.value!r} ({status})")

    ids = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    report = {"target": "sm_103f", "devices": []}
    try:
        if len(set(ids)) != 8 or torch.cuda.device_count() != 8:
            raise RuntimeError("expected eight distinct granted devices")
        for rank, uuid in enumerate(ids):
            with torch.cuda.device(rank):
                cap = torch.cuda.get_device_capability(rank)
                row = {"rank": rank, "uuid": uuid, "cuda_capability": cap}
                report["devices"].append(row)
                if cap != (10, 3):
                    raise RuntimeError(f"{uuid}: expected the allocated node's sm_103, found {cap}")
                value = torch.zeros(1, dtype=torch.int32, device=f"cuda:{rank}")
                module, fn = c.c_void_p(), c.c_void_p()
                call("cuModuleLoad", c.byref(module), os.fsencode(root / "sm103f.cubin"))
                try:
                    call("cuModuleGetFunction", c.byref(fn), module, b"glm_arch_probe")
                    ptr = c.c_uint64(value.data_ptr())
                    params = (c.c_void_p * 1)(c.addressof(ptr))
                    stream = c.c_void_p(torch.cuda.current_stream(rank).cuda_stream)
                    call("cuLaunchKernel", fn, 1, 1, 1, 1, 1, 1, 0, stream, params, None)
                    call("cuStreamSynchronize", stream)
                    row["observed"] = value.item()
                    if row["observed"] != 0x103:
                        raise RuntimeError(f"{uuid}: cubin write-back mismatch")
                finally:
                    call("cuModuleUnload", module)
        print("[arch] sm_103f cubin executed and verified on all eight devices", flush=True)
    except Exception as exc:
        report["error"] = str(exc)
        raise
    finally:
        (root / "arch-proof.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    probe(Path(sys.argv[1]))
