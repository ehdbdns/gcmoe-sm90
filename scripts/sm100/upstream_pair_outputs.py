"""Reuse each expert's output allocation for eager warmup and graph capture."""

import torch


class OutputAllocation:
    def __init__(self):
        super().__init__()
        self.output = None
        self.input = None
        self.spec = None

    def begin(self):
        self.seen = False
        self.reused = False

    def __enter__(self):
        self.original_empty_like = torch.empty_like
        torch.empty_like = self.empty_like
        return self

    def __exit__(self, *exc):
        torch.empty_like = self.original_empty_like

    def empty_like(self, source, *args, **kwargs):
        candidate = (
            source.shape,
            kwargs.get("dtype", source.dtype),
            kwargs.get("device", source.device),
            kwargs.get("layout", source.layout),
        )
        if self.output is not None and not self.reused and candidate == self.spec:
            self.reused = True
            return self.output
        tensor = self.original_empty_like(source, *args, **kwargs)
        if self.output is None:
            self.input = source
            self.spec = (tensor.shape, tensor.dtype, tensor.device, tensor.layout)
        return tensor

    def observe(self, y):
        assert not self.seen, "multiple backend launches in one expert forward"
        self.seen = True
        if self.output is None:
            self.output = y
        assert y is self.output, "expert output allocation changed"
