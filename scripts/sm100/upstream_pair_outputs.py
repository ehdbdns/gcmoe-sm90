"""Reuse each expert's output allocation for eager warmup and graph capture."""

from torch.utils._python_dispatch import TorchDispatchMode


class OutputAllocation(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.output = None
        self.factory = None
        self.index = None
        self.calls = {}
        self.candidates = {}

    def begin(self):
        self.seen = False
        self.calls.clear()
        self.candidates.clear()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func._schema.name not in {
            "aten::empty", "aten::empty_like", "aten::empty_strided",
            "aten::new_empty", "aten::new_empty_strided",
        }:
            return func(*args, **kwargs)
        i = self.calls.get(func, 0)
        self.calls[func] = i + 1
        if func == self.factory and i == self.index:
            return self.output
        tensor = func(*args, **kwargs)
        if self.output is None:
            self.candidates[id(tensor)] = (func, i)
        return tensor

    def observe(self, y):
        assert not self.seen, "multiple backend launches in one expert forward"
        self.seen = True
        if self.output is None:
            self.factory, self.index = self.candidates[id(y)]
            self.output = y
        assert y is self.output, "expert output allocation changed"
