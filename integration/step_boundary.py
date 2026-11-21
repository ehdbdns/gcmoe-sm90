"""Sparse rank-local GPU timestamps for the SM90 decode boundary audit."""
from functools import wraps

import torch
import triton
import triton.language as tl


@triton.jit
def _stamp(buffer):
    now = tl.inline_asm_elementwise("mov.u64 $0, %globaltimer;", constraints="=l",
                                    args=[], dtype=tl.int64, is_pure=False, pack=1)
    tl.store(buffer, now)


class BoundaryClocks:
    """Mark adjacent steps in every 50-step block on the runner's current stream."""

    def __init__(self, runner, steps):
        self.runner = runner
        self.steps = steps
        self.step = -1
        self.clocks = torch.zeros((steps, 4, 2), device="cuda", dtype=torch.int64)
        _stamp[(1,)](self.clocks[0, 0])
        self.clocks.zero_()
        torch.cuda.synchronize()
        self.originals = {}
        for name, col in (("execute_model", 0), ("sample_tokens", 2)):
            original = getattr(runner, name)
            self.originals[name] = runner.__dict__.get(name)
            setattr(runner, name, self.wrap(original, col))

    def wrap(self, original, col):
        @wraps(original)
        def timed(*args, **kwargs):
            if col == 0:
                self.step += 1
            step = self.step
            if not (0 <= step < self.steps and step % 50 in (24, 25)):
                return original(*args, **kwargs)
            _stamp[(1,)](self.clocks[step, col])
            result = original(*args, **kwargs)
            _stamp[(1,)](self.clocks[step, col + 1])
            return result
        return timed

    def finish(self):
        for name, original in self.originals.items():
            if original is None:
                delattr(self.runner, name)
            else:
                setattr(self.runner, name, original)
        if self.step + 1 != self.steps:
            raise ValueError("Unexpected execute_model count in boundary audit")
        return self.clocks[:, :, 0].cpu().numpy()
