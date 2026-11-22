"""Worker telemetry and fixed-route replay for eager, full-batch SM100 decode."""

from functools import wraps
import inspect
import hashlib
import os
from pathlib import Path

import numpy as np
import torch

if os.getenv("SM100_FORMAL_512"):
    from stream_checkpoint import install_cache_names
    install_cache_names()

STEPS, LAYERS = 1000, 40


def tensor_digest(tensor):
    digest = hashlib.sha256()
    for chunk in tensor.split(256):
        digest.update(chunk.contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def first_gate_difference(native_ids, replay_ids, native_weights, replay_weights):
    for start in range(0, len(native_ids), 256):
        part = slice(start, start + 256)
        different = ((native_ids[part] != replay_ids[part])
                     | (native_weights[part] != replay_weights[part]))
        if different.any().item():
            offset, token, choice = different.nonzero()[0].tolist()
            call = start + offset
            index = call, token, choice
            return dict(step=call // LAYERS, layer=call % LAYERS, local_token=token,
                        choice=choice, native_id=native_ids[index].item(),
                        replay_id=replay_ids[index].item(),
                        native_weight=native_weights[index].item(),
                        replay_weight=replay_weights[index].item())
    return None


class FiveWorker:
    def five_sync(self):
        torch.cuda.synchronize()
        return True

    def five_setup(self, root, arm, bs):
        from vllm.utils.deep_gemm import _import_deep_gemm
        from fixed_tokens import install

        self.arm, self.bs, self.root = arm, bs, Path(root)
        self.publisher = arm == "mega"
        self.rank = torch.distributed.get_rank()
        self.folder = self.root / arm
        self.tokens = install(self.model_runner, self.root, capacity=2128)
        self.experts = [
            m.experts
            for n, m in self.get_model().named_modules()
            if n.endswith(".ffn") and hasattr(m, "experts")
        ]
        assert len(self.experts) == LAYERS
        self.phase = None
        self.eco_tables = None
        self.current = None
        self.prime_inputs = {}
        self.layer_graphs = {}
        self.graph_checks = []
        dg = _import_deep_gemm()
        self.backend_file = inspect.getfile(dg.fp8_fp4_mega_moe) if os.getenv("SM100_FORMAL_512") else None
        if os.getenv("SM100_FORMAL_512") and arm.startswith("gc"):
            upstream = Path(os.environ["SM100_CLOCKED_UPSTREAM"]).resolve()
            assert upstream in Path(self.backend_file).resolve().parents
            from gc_moe.runtime.gc_moe import get_context
            ctx = get_context()
            self.sm_split = dict(dispatch=ctx.dispatch_sms, compute=ctx.compute_sms)
        self.select_m = dg.get_block_m_for_mega_moe
        if (
            Path(os.environ["SM100_MATRIX_OVERLAY"]).resolve()
            not in Path(dg.__file__).resolve().parents
        ):
            raise RuntimeError(
                f"private DeepGEMM overlay was not loaded: {dg.__file__}"
            )
        original_launch = dg.fp8_fp4_mega_moe
        assert original_launch.__module__.startswith(
            "gc_moe" if arm.startswith("gc") else "deep_gemm"
        )

        @wraps(original_launch)
        def launch(y, l1, l2, buf, *args, **kwargs):
            if self.phase is None:
                return original_launch(y, l1, l2, buf, *args, **kwargs)
            layer, call = self.current
            scratch = self.kernel_scratch
            scratch.copy_(self.recv[call])
            counters = scratch[: self.local_slots]
            counters._matrix_clock_scratch = True
            kwargs["cumulative_local_expert_recv_stats"] = counters
            if self.eco_tables is not None:
                meta, state = self.eco_tables[layer]
                kwargs.update(eco_meta=meta, eco_state=state, eco_logical_experts=384)
                buf.eco_auction_audit = self.audit_scratch
            if self.phase == "measure":
                self.audit_scratch.zero_()
                if layer not in self.layer_graphs:
                    # The same layer always reads the same symmetric input, packed
                    # weights and plan state. Only their contents change per step.
                    target = torch.empty_like(y)
                    if self.eco_tables is not None:
                        old_state = state.clone()
                    original_launch(target, l1, l2, buf, *args, **kwargs)
                    reference = target.clone()
                    eager_counts = counters.clone()
                    eager_accepted = self.audit_scratch[:1].clone()
                    if self.eco_tables is not None:
                        state.copy_(old_state)
                    scratch.copy_(self.recv[call])
                    self.audit_scratch.zero_()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        original_launch(target, l1, l2, buf, *args, **kwargs)
                    self.layer_graphs[layer] = graph, target
                    graph.replay()
                    self.graph_checks.append(torch.stack((
                        (target != reference).sum(), (counters != eager_counts).sum(),
                        (self.audit_scratch[:1] != eager_accepted).sum())))
                else:
                    graph, target = self.layer_graphs[layer]
                    graph.replay()
                y.copy_(target)
                if self.eco_tables is not None and os.getenv("SM100_FORMAL_512"):
                    plan_start = self.audit_scratch[2:4].view(torch.int64)
                    start = scratch[self.local_slots:self.local_slots + 2].view(torch.int64)
                    start.copy_(torch.minimum(start, plan_start))
                    self.plan_counts[call].copy_(self.quota_view)
                self.auction[call].copy_(self.audit_scratch[0])
                self.recv[call].copy_(scratch)
                return None
            result = original_launch(y, l1, l2, buf, *args, **kwargs)
            self.recv[call].copy_(scratch)
            return result

        dg.fp8_fp4_mega_moe = launch
        for layer, expert in enumerate(self.experts):
            expert.forward = self._five_forward(layer, expert.forward)
        return True

    def _five_forward(self, layer, forward):
        @wraps(forward)
        def route(hidden_states, topk_weights, topk_ids, **kwargs):
            if self.phase is None:
                return forward(hidden_states, topk_weights, topk_ids, **kwargs)
            call = self.calls
            if (
                call >= STEPS * LAYERS
                or layer != call % LAYERS
                or topk_ids.shape != (self.bs // 8, 6)
            ):
                raise ValueError(
                    f"unexpected decode workload at call {call}, layer {layer}, shape {topk_ids.shape}"
                )
            self.calls += 1
            self.current = layer, call
            if self.phase == "stats" and call // LAYERS == STEPS - 1:
                self.prime_inputs[layer] = (
                    hidden_states.detach().clone(),
                    topk_weights.detach().clone(),
                    topk_ids.detach().clone(),
                    dict(kwargs),
                )
            self.raw_ids[call].copy_(topk_ids)
            self.raw_weights[call].copy_(topk_weights)
            ids = self.bank_ids[call].to(topk_ids.dtype)
            weights = self.bank_weights[call].to(topk_weights.dtype)
            out = forward(hidden_states, weights, ids, **kwargs)
            return out

        return route

    def five_begin(self, phase):
        from vllm.distributed.parallel_state import get_ep_group

        self.calls = 0
        self.phase = phase
        self.phase_start = self.tokens.positions.clone()
        if os.getenv("SM100_FORMAL_512"):
            from stream_checkpoint import checkpoint, named_caches, runner_boundary

            folder = self.root / f"kv-{phase}"
            folder.mkdir(exist_ok=True)
            self.boundary = checkpoint(
                folder, self.rank, self.publisher, named_caches(self.model_runner),
                runner_boundary(self.model_runner, self.tokens))
            if phase == "stats":
                valid = torch.arange(self.tokens.native.shape[1], device="cuda")[None, :] < self.phase_start[:, None]
                self.prefill_native = tensor_digest(self.tokens.native.masked_fill(~valid, -1))
                self.prefill_diff = {}
                if not self.publisher:
                    diff = (valid & (self.tokens.native != self.tokens.table)).nonzero()
                    self.prefill_diff["count"] = len(diff)
                    if len(diff):
                        req, pos = diff[diff[:, 1].argmin()].tolist()
                        self.prefill_diff["first"] = dict(req=req, position=pos,
                            native=int(self.tokens.native[req, pos]), reference=int(self.tokens.table[req, pos]))

        n = STEPS * LAYERS
        shape = (n, self.bs // 8, 6)
        self.raw_ids = torch.empty(shape, device="cuda", dtype=torch.int32)
        self.raw_weights = torch.empty(shape, device="cuda", dtype=torch.float32)
        path = self.root / f"{phase}-routes-{self.rank}.pt"
        if self.publisher:
            self.bank_ids, self.bank_weights = self.raw_ids, self.raw_weights
        else:
            self.bank_ids, self.bank_weights = torch.load(
                path, map_location=self.raw_ids.device, weights_only=True
            )
            assert self.bank_ids.shape == self.bank_weights.shape == shape
        per_rank = self.experts[0].num_local_experts
        self.local_slots = per_rank
        if self.arm.startswith("gc"):
            buf = self.experts[0].get_symm_buffer()
            self.kernel_scratch = buf.buffer.narrow(
                0, buf.clock_offset - per_rank * 4, per_rank * 4 + 32
            ).view(torch.int32)
        else:
            self.kernel_scratch = torch.empty(
                per_rank + 8, device="cuda", dtype=torch.int32
            )
        self.recv = torch.zeros((n, per_rank + 8), dtype=torch.int32, device="cuda")
        self.recv[:, per_rank:].view(torch.int64)[:, :2].fill_(-1)
        self.bm = self.select_m(
            8,
            384 if self.eco_tables is not None else 8 * per_rank,
            self.experts[0].get_symm_buffer().num_max_tokens_per_rank,
            self.bs // 8,
            6,
            "fp8xfp4",
        )
        assert get_ep_group().rank_in_group == self.rank
        if phase == "measure":
            self.audit_scratch = torch.zeros(per_rank + 32, dtype=torch.int32, device="cuda")
            if self.eco_tables is not None and os.getenv("SM100_FORMAL_512"):
                self.plan_counts = torch.empty((n, per_rank), dtype=torch.int32, device="cuda")
            self.auction = torch.zeros(n, dtype=torch.int32, device="cuda")
        if phase == "measure":
            # Warm each real packed weight address, shape, instrumentation and JIT before
            # the clock starts. Standalone expert calls do not step the model or its KV.
            for layer, expert in enumerate(self.experts):
                h, w, i, kw = self.prime_inputs.pop(layer)
                expert(h, w, i, **kw)
            torch.cuda.synchronize()
            if self.eco_tables is not None:
                for _, state in self.eco_tables:
                    state.zero_()
                self.auction.zero_()
            assert len(self.layer_graphs) == LAYERS
            mismatches = torch.stack(self.graph_checks).sum(dim=0)
            torch.distributed.all_reduce(mismatches)
            if mismatches.any().item():
                raise RuntimeError(f"MoE eager/replay mismatch (outputs, counts, accepts): {mismatches.tolist()}")
            self.calls = 0
            self.recv.zero_()
            self.recv[:, per_rank:].view(torch.int64)[:, :2].fill_(-1)
        torch.cuda.synchronize()
        return True

    def five_eco(self, filename):
        from gc_moe import get_symm_buffer_for_gc_moe
        from gc_moe.eco.meta_sm100 import build_eco_meta_sm100
        from vllm.models.deepseek_v4.nvidia import model
        import torch.distributed as dist

        old = self.experts[0].get_symm_buffer()
        buf = get_symm_buffer_for_gc_moe(
            old.group,
            448,
            old.num_max_tokens_per_rank,
            6,
            old.hidden,
            old.intermediate_hidden,
            num_shared_experts=1,
            eco=True,
        )
        if os.getenv("SM100_FORMAL_512"):
            from gc_moe.layout.gc_moe import ECO_SIGNAL_OFFSETS
            local_slots = buf.num_experts // buf.group.size()
            offset = buf.layout.eco_base + ECO_SIGNAL_OFFSETS["eco_quota"]
            quotas = buf.buffer.narrow(0, offset, buf.num_experts * 4).view(torch.int32)
            self.quota_view = quotas.narrow(0, self.rank * local_slots, local_slots)
        with np.load(filename) as f:
            self.eco_tables = [
                (
                    torch.from_numpy(build_eco_meta_sm100(row, 8, 384)).cuda(),
                    torch.zeros(448 + 384, device="cuda", dtype=torch.int16),
                )
                for row in f["layout"]
            ]
        from vllm.utils.deep_gemm import _import_deep_gemm

        def shared_block_m(ranks, physical, capacity, tokens, topk, dtype):
            return self.select_m(ranks, 384, capacity, tokens, topk, dtype)

        _import_deep_gemm().get_block_m_for_mega_moe = shared_block_m
        for expert in self.experts:
            expert.get_symm_buffer = lambda: buf
        # Role 5 consumes logical IDs; replay already verified full decode rows.
        model.eplb_map_to_physical_and_record = lambda *, topk_ids, **kw: topk_ids
        assert dist.get_world_size(old.group) == 8
        return True

    def five_finish(self):
        self.five_sync()
        phase = self.phase
        self.phase = None
        assert self.calls == STEPS * LAYERS
        assert torch.all(self.tokens.positions - self.phase_start == STEPS).item()
        path = self.root / f"{phase}-routes-{self.rank}.pt"
        if self.publisher:
            torch.save((self.bank_ids.cpu(), self.bank_weights.cpu()), path)
        evidence = dict(
            logical_ids=tensor_digest(self.raw_ids),
            gate_weights=tensor_digest(self.raw_weights),
            replay_ids=tensor_digest(self.bank_ids),
            replay_weights=tensor_digest(self.bank_weights),
            batch_order=self.tokens.batch_hash.hexdigest(),
            block_m=int(self.bm),
        )
        if os.getenv("SM100_FORMAL_512"):
            offsets = self.phase_start[:, None] + torch.arange(STEPS, device="cuda")
            window = self.tokens.native.gather(1, offsets)
            assert window.shape == (self.bs, STEPS) and (window >= 0).all()
            evidence.update(native_window=tensor_digest(window), kv_boundary=self.boundary,
                            prefill_native=self.prefill_native, backend_file=self.backend_file,
                            prefill_difference=self.prefill_diff, sm_split=getattr(self, "sm_split", None))
            if not self.publisher:
                reference = self.tokens.table.gather(1, offsets)
                diff = (window != reference).nonzero()
                evidence["native_diff"] = len(diff)
                if len(diff):
                    req, step = diff[diff[:, 1].argmin()].tolist()
                    evidence["first_native_diff"] = dict(req=req, step=step,
                        native=int(window[req, step]), reference=int(reference[req, step]))
        if phase == "measure":
            evidence["validated_layer_graphs"] = len(self.layer_graphs)
            evidence["native"] = tensor_digest(self.tokens.native)
        evidence["first_gate_difference"] = None if self.publisher else first_gate_difference(
            self.raw_ids, self.bank_ids, self.raw_weights, self.bank_weights)
        del self.bank_ids, self.bank_weights, self.raw_ids, self.raw_weights
        result = dict(
            counts=self.recv[:, : self.local_slots]
            .cpu()
            .numpy()
            .reshape(STEPS, LAYERS, -1)
        )
        if phase == "measure":
            result["clocks"] = (
                self.recv[:, self.local_slots :]
                .view(torch.int64)
                .cpu()
                .numpy()
                .reshape(STEPS, LAYERS, 4)
            )
        if phase == "measure":
            if self.eco_tables is not None:
                if os.getenv("SM100_FORMAL_512"):
                    if not torch.equal(self.plan_counts, self.recv[:, :self.local_slots]):
                        raise ValueError("physical receive counts differ from auction quotas")
                    from gc_moe.eco.plan import eco_unit_config
                    evidence["eco_unit_config"] = eco_unit_config(8, self.bm, 64)
                    evidence["verified_local_quota_rows"] = STEPS * LAYERS
                result["accepted"] = (
                    self.auction.cpu().numpy().reshape(STEPS, LAYERS)
                )
                del self.auction
        np.savez_compressed(self.folder / f"{phase}-rank{self.rank}.npz", **result)
        del self.recv
        return evidence


from v41_gc_worker import MoeAuditWorker


class MatrixWorker(MoeAuditWorker):
    def matrix_setup(self, root, arm, bs):
        self._matrix = FiveWorker()
        self._matrix.model_runner = self.model_runner
        self._matrix.get_model = self.get_model
        return self._matrix.five_setup(root, arm, bs)

    def matrix_begin(self, phase):
        return self._matrix.five_begin(phase)

    def matrix_finish(self):
        return self._matrix.five_finish()

    def matrix_eco(self, filename):
        return self._matrix.five_eco(filename)

    def matrix_sync(self):
        return self._matrix.five_sync()
