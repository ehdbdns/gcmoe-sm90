"""Replay the installed FlashInfer BF16 -> MXFP8 -> dense projection on its caller stream."""
from functools import wraps

import torch


class DenseReplay:
    def __init__(self):
        from vllm.model_executor.kernels.linear.mxfp8.flashinfer import FlashInferCutedslMxfp8LinearKernel
        self.mode = "off"
        self.graphs = {}
        self.validated = set()
        self.calls = 0
        cls = FlashInferCutedslMxfp8LinearKernel
        eager = cls.apply_weights

        @wraps(eager)
        def apply(kernel, layer, x, bias=None):
            if self.mode == "off" or not isinstance(x, torch.Tensor) or x.dtype != torch.bfloat16:
                return eager(kernel, layer, x, bias)
            stream = torch.cuda.current_stream(x.device)
            key = (layer, x.shape, x.stride(), stream.cuda_stream, layer.weight.data_ptr(),
                   layer.weight_scale.data_ptr(), bias.data_ptr() if bias is not None else None)
            if self.mode == "capture" and key not in self.graphs:
                reference = eager(kernel, layer, x, bias)
                source = torch.empty_like(x)
                source.copy_(x)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    target = eager(kernel, layer, source, bias)
                graph.replay()
                self._verify(target, reference)
                self.graphs[key] = graph, source, target
                return reference
            if key not in self.graphs:
                raise ValueError(f"unprimed MXFP8 projection at {self.mode}: {tuple(x.shape)}")
            graph, source, target = self.graphs[key]
            source.copy_(x)
            graph.replay()
            if self.mode in ("capture", "validate"):
                reference = eager(kernel, layer, x, bias)
                self._verify(target, reference)
                if self.mode == "validate":
                    self.validated.add(key)
                return reference
            self.calls += 1
            # The caller owns its output; a later replay overwrites target.
            return target.clone()

        cls.apply_weights = apply

    @staticmethod
    def _verify(replayed, reference):
        if not torch.equal(replayed, reference) or not torch.isfinite(replayed).all().item():
            raise ValueError("MXFP8 projection eager/replay output mismatch")

    def select(self, mode):
        if mode not in ("capture", "validate", "on", "off"):
            raise ValueError(mode)
        if mode == "on" and (not self.graphs or self.validated != self.graphs.keys()):
            raise ValueError("dense replay requires validation on the following real decode step")
        previous, self.calls, self.mode = self.calls, 0, mode
        return dict(previous_replays=previous, graphs=len(self.graphs), validated=len(self.validated))
