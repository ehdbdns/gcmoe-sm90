"""Same-input EP8 expert comparison. All validation and I/O are outside events."""

from pathlib import Path
import hashlib
import inspect

import torch
import torch.distributed as dist

from v41_gc_worker import MoeAuditWorker


class NativePairWorker(MoeAuditWorker):
    def pair_setup(self, folder, arm):
        from vllm.utils.deep_gemm import _import_deep_gemm

        self.pair_dir = Path(folder)
        self.pair_rank = dist.get_rank()
        self.pair_arm = arm
        self.pair_layers = [
            m.experts
            for n, m in self.get_model().named_modules()
            if n.endswith(".ffn") and hasattr(m, "experts")
        ]
        assert len(self.pair_layers) == 40
        assert all(
            m.num_local_experts == 48 and m.has_fused_shared_experts
            for m in self.pair_layers
        )
        backend = _import_deep_gemm().fp8_fp4_mega_moe.__module__
        assert ("gc_moe" in backend) == (arm == "gc"), backend
        return backend

    def pair_record(self):
        self.pair_rows = []
        self.pair_hooks = []

        def register(layer, module):
            signature = inspect.signature(module.forward)

            def capture(mod, args, kwargs, out):
                assert layer == len(self.pair_rows)
                bound = dict(signature.bind(*args, **kwargs).arguments)
                assert bound["topk_ids"].shape == (128, 6)
                self.pair_rows.append(
                    (
                        {
                            k: v.detach().clone() if isinstance(v, torch.Tensor) else v
                            for k, v in bound.items()
                        },
                        out.detach().clone(),
                    )
                )

            return module.register_forward_hook(capture, with_kwargs=True)

        for layer, module in enumerate(self.pair_layers):
            self.pair_hooks.append(register(layer, module))
        return True

    def pair_save(self, step):
        for hook in self.pair_hooks:
            hook.remove()
        assert len(self.pair_rows) == 40
        path = self.pair_dir / f"step{step}-rank{self.pair_rank}.pt"
        with path.open("xb") as f:
            torch.save(self.pair_rows, f)
        del self.pair_rows
        return True

    @torch.inference_mode()
    def pair_time(self, step):
        path = self.pair_dir / f"step{step}-rank{self.pair_rank}.pt"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows = torch.load(
            path,
            map_location=torch.device("cuda", torch.cuda.current_device()),
            weights_only=True,
        )

        def forward():
            return [
                module(**args)
                for module, (args, _) in zip(self.pair_layers, rows, strict=True)
            ]

        for _ in range(3):
            eager = forward()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outputs = forward()
        graph.replay()
        mismatches = []
        for y, original, (_, ref) in zip(outputs, eager, rows, strict=True):
            assert torch.isfinite(y).all() and torch.isfinite(ref).all()
            assert torch.equal(y, original), (
                "same backend graph differs from its eager output"
            )
            mismatches.append(
                dict(
                    elements=y.numel(),
                    different=int((y != ref).sum()),
                    max_abs=float((y.float() - ref.float()).abs().max()),
                )
            )
        for _ in range(10):
            graph.replay()
        torch.cuda.synchronize()
        times = []
        for _ in range(15):
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            dist.barrier()
            start.record()
            for _ in range(20):
                graph.replay()
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end) / 20)
        return dict(rank=self.pair_rank, fixture=digest, ms=times, outputs=mismatches)
