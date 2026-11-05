"""Make Triton 3.7.1 tolerate JIT sources that start at a decorator.

``JITFunction.__init__`` slices the source it got from ``inspect`` at the first
``^def``. When the outermost decorator is a *call* whose body has no ``def``
(``@triton.heuristics({...})``), inspect returns only the decorator lines and
the regex misses, so importing the module dies with
``AttributeError: 'NoneType' object has no attribute 'start'``.

vLLM 0.31 runs into this in
``vllm/models/minimax_m3/common/ops/index_topk.py``; that module is imported
(never executed for DSV4.1) by the kernel-warmup path, so it blocks engine
bring-up on every GPU.

Fix: on that specific failure, re-read the function's file from disk and slice
from its own ``def`` line.
"""
import re

import triton.runtime.jit as _jit

_DEF_RE = re.compile(r"^def\s+\w+\s*\(", re.MULTILINE)


def _source_from_disk(fn):
    code = getattr(fn, '__code__', None)
    if code is None:
        return None
    try:
        with open(code.co_filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except OSError:
        return None
    text = ''.join(lines[code.co_firstlineno - 1:])
    m = _DEF_RE.search(text)
    return None if m is None else text[m.start():]


def install() -> None:
    if getattr(_jit.JITFunction, '_src_compat_patched', False):
        return
    orig_init = _jit.JITFunction.__init__

    def _patched_init(self, fn, *args, **kwargs):
        try:
            return orig_init(self, fn, *args, **kwargs)
        except AttributeError:
            src = _source_from_disk(fn)
            if src is None:
                raise
            self._src = src
            self.hash = None
            if getattr(self, 'raw_src', None) is None:
                self.raw_src = src.splitlines(keepends=True)
            return None

    _jit.JITFunction.__init__ = _patched_init
    _jit.JITFunction._src_compat_patched = True
