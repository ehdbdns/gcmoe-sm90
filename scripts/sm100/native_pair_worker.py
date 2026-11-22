"""Same-input EP8 expert comparison. All validation and I/O are outside events."""

from pathlib import Path
import hashlib
import inspect
import os

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
        if os.environ.get("SM100_UPSTREAM_PAIR"):
            from upstream_pair_outputs import OutputAllocation
            from functools import wraps
            import gc_moe
            from gc_moe.runtime.gc_moe import get_context

            expected = Path(os.environ["SM100_UPSTREAM_PAIR"]).resolve()
            assert expected in Path(gc_moe.__file__).resolve().parents
            dg = _import_deep_gemm()
            fn = dg.fp8_fp4_mega_moe
            info = dict(rank=self.pair_rank, backend=backend, file=inspect.getfile(fn),
                        gc_version=gc_moe.__version__, capability=torch.cuda.get_device_capability())
            if arm == "gc":
                assert fn is gc_moe.fp8_fp4_gc_moe
                ctx = get_context(num_ranks=8)
                info.update(dispatch_sms=ctx.dispatch_sms, compute_sms=ctx.compute_sms,
                            primary=ctx.primary_ctx_roles)
            self.output_mode = None

            @wraps(fn)
            def checked(y, *args, **kwargs):
                if self.output_mode is not None:
                    self.output_mode.observe(y)
                return fn(y, *args, **kwargs)

            dg.fp8_fp4_mega_moe = checked
            self.output_modes = [OutputAllocation() for _ in self.pair_layers]
            return info
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
            if hasattr(self, "output_modes"):
                outputs = []
                for mode, module, (args, _) in zip(self.output_modes, self.pair_layers, rows, strict=True):
                    mode.begin()
                    self.output_mode = mode
                    try:
                        with mode:
                            outputs.append(module(**args))
                        assert mode.seen, "expert bypassed selected backend"
                    finally:
                        self.output_mode = None
                return outputs
            return [
                module(**args)
                for module, (args, _) in zip(self.pair_layers, rows, strict=True)
            ]

        for _ in range(3):
            eager = forward()
        if hasattr(self, "output_modes"):
            eager = [y.clone() for y in eager]
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
