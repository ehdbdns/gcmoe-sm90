"""Load the pinned routed-only wheel without adding its site to Python's search path."""
from functools import lru_cache
from importlib.util import module_from_spec, spec_from_file_location
import os
from pathlib import Path
import sys


@lru_cache(maxsize=1)
def mega_backend():
    package = Path(os.environ["SM90_DG_SITE"]).resolve() / "deep_gemm"
    name = "_gcmoe_sm90_official_deep_gemm"
    spec = spec_from_file_location(name, package / "__init__.py",
                                   submodule_search_locations=[str(package)])
    module = module_from_spec(spec)
    sys.modules[name] = module  # Relative imports, including ._C, resolve inside this wheel.
    try:
        spec.loader.exec_module(module)
        for entry in ("fp8_mega_moe", "get_symm_buffer_for_sm90_mega_moe",
                      "transform_weights_for_mega_moe_sm90"):
            if not callable(getattr(module, entry, None)):
                raise ImportError(f"{package} lacks {entry}")
    except Exception:
        for key in tuple(sys.modules):
            if key == name or key.startswith(name + "."):
                del sys.modules[key]
        raise
    return module


def attention_backend():
    from vllm.utils.deep_gemm import _import_deep_gemm
    backend = _import_deep_gemm()
    if getattr(backend, "__name__", None) != "vllm.third_party.deep_gemm":
        raise ImportError(f"unexpected vLLM attention backend {backend!r}")
    return backend


if __name__ == "__main__":
    attention = attention_backend()
    print(f"vLLM attention: {attention.__file__}", flush=True)
    if sys.argv[1:] == ["mega"]:
        from importlib import import_module
        mega = mega_backend()
        print(f"SM90 Mega: {mega.__file__}, extension: {mega._C.__file__}", flush=True)
        if attention_backend() is not attention or import_module(attention.__name__) is not attention:
            raise RuntimeError("loading the private wheel changed vLLM's attention backend")
