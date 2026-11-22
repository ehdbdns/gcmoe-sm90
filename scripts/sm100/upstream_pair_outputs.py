"""Reuse each expert's output allocation for eager warmup and graph capture."""

from torch.utils._python_dispatch import TorchDispatchMode


class OutputAllocation(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.output = None
        self.input = None
        self.spec = None

    def begin(self):
        self.seen = False
        self.reused = False

    @staticmethod
    def _spec(func, args, kwargs):
        if func._schema.name != "aten::empty_like":
            return None
        source = args[0]
        return (
            source.shape,
            kwargs.get("dtype", source.dtype),
            kwargs.get("device", source.device),
            kwargs.get("layout", source.layout),
        )

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func._schema.name not in {
            "aten::empty", "aten::empty_like", "aten::empty_strided",
            "aten::new_empty", "aten::new_empty_strided",
        }:
            return func(*args, **kwargs)
        spec = self._spec(func, args, kwargs)
        if (
            self.output is not None
            and not self.reused
            and spec == self.spec
            and args[0] is self.input
        ):
            self.reused = True
            return self.output
        tensor = func(*args, **kwargs)
        if self.output is None and spec is not None:
            self.input = args[0]
            self.spec = spec
        return tensor

    def observe(self, y):
        assert not self.seen, "multiple backend launches in one expert forward"
        self.seen = True
        if self.output is None:
            self.output = y
        assert y is self.output, "expert output allocation changed"
