"""Fixed-address logical route banks and telemetry for the V4.1 full decode graph on B300."""
from functools import partial, wraps
import inspect
import os
from pathlib import Path

import numpy as np
import torch
import triton
import triton.language as tl

from five_worker import FiveWorker
from full_decode_worker import begin
from v41_gc_worker import MoeAuditWorker

CALLS = 40_000
STATE = None


@triton.jit
def stage_routes(ids, weights, out_i, out_w, raw_i, raw_w, bank_i, bank_w, cursor, active,
                 size: tl.constexpr, block: tl.constexpr, publisher: tl.constexpr):
    lane = tl.arange(0, block)
    mask = lane < size
    i = tl.load(ids + lane, mask, 0)
    w = tl.load(weights + lane, mask, 0.)
    n = tl.load(cursor)
    if (tl.load(active) != 0) & (n < 40000):
        tl.store(raw_i + n * size + lane, i, mask)
        tl.store(raw_w + n * size + lane, w, mask)
        if not publisher:
            i = tl.load(bank_i + n * size + lane, mask, 0)
            w = tl.load(bank_w + n * size + lane, mask, 0.).to(w.dtype)
    tl.store(out_i + lane, i, mask)
    tl.store(out_w + lane, w, mask)


@triton.jit
def collect(scratch, result, audit, quota, details, cursor, active, tags, layer: tl.constexpr,
            slots: tl.constexpr, eco: tl.constexpr):
    if tl.load(active) != 0:
        n = tl.load(cursor)
        if n < 40000:
            lane = tl.arange(0, 128)
            width: tl.constexpr = slots + 8
            data = tl.load(scratch + lane, lane < width, 0)
            tl.store(result + n * width + lane, data, lane < width)
            if eco:
                # Four 64-bit clock fields follow the slot counters. Include plan entry
                # in graph start, even if dispatch or compute arrives later.
                # Preserve the raw plan timestamp for adjustment outside the graph.
                tl.store(details + n * (slots + 5) + lane,
                         tl.load(audit + lane, lane < 5, 0), lane < 5)
                tl.store(details + n * (slots + 5) + 5 + lane,
                         tl.load(quota + lane, lane < slots, 0), lane < slots)
            tl.store(tags + n, layer)
        tl.store(cursor, n + 1)


@triton.jit
def collect_upstream(header, counts, quota, result, details, cursor, active, tags, layer: tl.constexpr,
                     slots: tl.constexpr, eco: tl.constexpr):
    if tl.load(active) != 0:
        n = tl.load(cursor)
        if n < 40000:
            lane = tl.arange(0, 128)
            out = result + n * (slots + 8)
            tl.store(out + lane, tl.load(counts + lane, lane < slots, 0), lane < slots)
            start = tl.minimum(tl.load(header), tl.load(header + 1))
            if eco:
                plan = tl.load(header + 15)
                start = tl.minimum(start, plan)
                row = details + n * (slots + 5)
                tl.store(row, tl.load(header + 17))
                tl.store(row + 1, tl.load(header + 18))
                tl.store(row + 2, plan.to(tl.int32))
                tl.store(row + 3, (plan >> 32).to(tl.int32))
                tl.store(row + 4, tl.load(header + 19))
                tl.store(row + 5 + lane, tl.load(quota + lane, lane < slots, 0), lane < slots)
            times = (out + slots).to(tl.pointer_type(tl.int64))
            tl.store(times, start)
            tl.store(times + 1, tl.load(header + 4))
            tl.store(times + 2, tl.load(header + 7))
            tl.store(times + 3, tl.load(header + 13))
            tl.store(tags + n, layer)
        tl.store(cursor, n + 1)


class GraphState:
    clocks = 4

    def __init__(self):
        from gc_moe.eco.meta_sm100 import build_eco_meta_sm100

        self.arm = os.environ["SM100_EP8_ARM"]
        self.bs = int(os.environ["GC_MOE_MATRIX_BS"])
        self.publisher = self.arm == "vanilla"
        self.eco = self.arm == "eco"
        self.slots = 48 if self.publisher else 56
        self.layers = {}
        self.samples = {}
        self.rows = self.bs // 8
        self.cursor = torch.zeros(1, device="cuda", dtype=torch.int32)
        self.active = torch.zeros_like(self.cursor)
        self.tags = self.cursor.new_empty(CALLS)
        self.recv = self.cursor.new_empty((CALLS, self.slots + 8))
        self.template = torch.zeros_like(self.recv[0])
        self.template[self.slots:].view(torch.int64)[:2].fill_(-1)
        self.ids = self.cursor.new_empty((CALLS, self.rows, 6))
        self.weights = torch.empty_like(self.ids, dtype=torch.float32)
        self.bank_i = self.ids if self.publisher else torch.empty_like(self.ids)
        self.bank_w = self.weights if self.publisher else torch.empty_like(self.weights)
        self.audit = self.cursor.new_zeros(8)
        self.details = self.cursor.new_empty((CALLS if self.eco else 1, self.slots + 5))
        if self.eco:
            root = Path(os.environ["SM100_EP8_GROUP"])
            with np.load(root / "eco-initial.npz") as start, np.load(root / "plan.npz") as end:
                initial = build_eco_meta_sm100(start["layout"][0], 8, 384)
                self.new_meta = [build_eco_meta_sm100(row, 8, 384) for row in end["layout"]]
            width = max(len(initial), max(map(len, self.new_meta)))
            self.meta = self.cursor.new_zeros((40, width))
            self.meta[:, :len(initial)].copy_(torch.from_numpy(initial))
            self.warm = torch.zeros((40, 832), device="cuda", dtype=torch.int16)

    def routes(self, forward):
        @wraps(forward)
        def wrapper(expert, hidden_states, topk_weights, topk_ids, **kw):
            if topk_ids.shape == (self.rows, 6):
                ids, weights = topk_ids.contiguous(), topk_weights.contiguous()
                out_i, out_w = torch.empty_like(ids), torch.empty_like(weights)
                stage_routes[(1,)](ids, weights, out_i, out_w, self.ids, self.weights, self.bank_i,
                                    self.bank_w, self.cursor, self.active, ids.numel(),
                                    triton.next_power_of_2(ids.numel()), self.publisher)
                topk_ids, topk_weights = out_i, out_w
            return forward(expert, hidden_states, topk_weights, topk_ids, **kw)
        return wrapper

    def instrument(self, kernel):
        @wraps(kernel)
        def launch(y, l1, l2, buf, *args, **kw):
            layer = self.layers[l1[0].data_ptr()]
            if self.eco:
                kw.update(eco_meta=self.meta[layer], eco_state=self.warm[layer], eco_logical_experts=384)
            if y.shape[0] != self.rows:
                return kernel(y, l1, l2, buf, *args, **kw)
            if hasattr(buf, "trace_level"):
                from gc_moe.eco.measurement import HEADER
                from gc_moe.layout.gc_moe import ECO_SIGNAL_OFFSETS
                if not buf.trace_level:
                    # The formal table requires counts. A bare replay timing
                    # uses the ordinary, uninstrumented GC entrypoint.
                    raise ValueError("table collection requires GC_MOE_SM100_TRACE=1 or 2")
                output = kernel(y, l1, l2, buf, *args, **kw)
                if buf.trace_level == 2 and layer in (8, 20, 32) and torch.cuda.is_current_stream_capturing():
                    from gc_moe.eco.measurement import snapshot
                    self.samples[layer] = snapshot(buf)
                clocks = buf.trace_header[:HEADER * 8].view(torch.int64)
                counts = buf.trace_header[HEADER * 8:].view(torch.int32)
                quota = self.audit
                if self.eco:
                    offset = buf.layout.eco_base + ECO_SIGNAL_OFFSETS["eco_quota"] + 4 * buf.rank_idx * self.slots
                    quota = buf.buffer.narrow(0, offset, self.slots * 4).view(torch.int32)
                collect_upstream[(1,)](clocks, counts, quota, self.recv, self.details, self.cursor,
                                       self.active, self.tags, layer, self.slots, self.eco)
                return output
            scratch = buf.buffer.narrow(0, buf.clock_offset - self.slots * 4,
                                        self.template.numel() * 4).view(torch.int32)
            scratch.copy_(self.template)
            counters = scratch[:self.slots]
            counters._matrix_clock_scratch = True
            kw["cumulative_local_expert_recv_stats"] = counters
            quota = self.audit
            if self.eco:
                from gc_moe.layout.gc_moe import ECO_SIGNAL_OFFSETS
                offset = buf.layout.eco_base + ECO_SIGNAL_OFFSETS["eco_quota"] + 4 * buf.rank_idx * self.slots
                quota = buf.buffer.narrow(0, offset, self.slots * 4).view(torch.int32)
                buf.eco_auction_audit = self.audit
                self.audit.zero_()
            output = kernel(y, l1, l2, buf, *args, **kw)
            collect[(1,)](scratch, self.recv, self.audit, quota, self.details, self.cursor, self.active,
                           self.tags, layer, self.slots, self.eco)
            return output
        return launch


def install():
    from vllm.models.deepseek_v4.nvidia import model
    from vllm.utils.deep_gemm import _import_deep_gemm
    from gc_moe import fp8_fp4_gc_moe, get_symm_buffer_for_gc_moe, get_block_m_for_gc_moe

    expert_class = model.DeepseekV4MegaMoEExperts
    original = expert_class.finalize_weights

    @wraps(original)
    def finalize(expert, *a, **kw):
        global STATE
        result = original(expert, *a, **kw)
        if STATE is None:
            STATE = GraphState()
            expert_class.forward = STATE.routes(expert_class.forward)
        STATE.layers[expert._transformed_l1_weights[0].data_ptr()] = expert.layer_id
        dg = _import_deep_gemm()
        dg.fp8_fp4_mega_moe = STATE.instrument(fp8_fp4_gc_moe)
        if STATE.eco:
            dg.get_symm_buffer_for_mega_moe = partial(get_symm_buffer_for_gc_moe, eco=True)
            dg.get_block_m_for_mega_moe = lambda r, p, cap, rows, k, dtype: get_block_m_for_gc_moe(
                r, 384, cap, rows, k, dtype)
            model.eplb_map_to_physical_and_record = lambda *, topk_ids, **kwargs: topk_ids
        return result
    expert_class.finalize_weights = finalize


install()


class EP8Worker(MoeAuditWorker):
    def ep8_setup(self, directory, arm, bs, capacity):
        from fixed_tokens import install as install_tokens
        from gc_moe import fp8_fp4_gc_moe
        from gc_moe.runtime.gc_moe import get_context
        from vllm.utils.deep_gemm import _import_deep_gemm
        s = STATE
        if (s.bs, s.arm) != (bs, arm) or sorted(s.layers.values()) != list(range(40)):
            raise ValueError("missing native expert capture")
        m = self._matrix = FiveWorker()
        m.root, m.arm, m.bs = Path(directory), arm, bs
        m.folder, m.rank = m.root / arm, torch.distributed.get_rank()
        m.publisher, m.eco_tables, m.host_submit, m.layer_graphs = s.publisher, None, None, {}
        m.model_runner = self.model_runner
        m.tokens = install_tokens(m.model_runner, m.root, capacity=capacity)
        m.backend_file = inspect.getfile(fp8_fp4_gc_moe)
        assert Path(os.environ["SM100_CLOCKED_UPSTREAM"]).resolve() in Path(m.backend_file).resolve().parents
        port = os.getenv("SM100_BRUCE_ECO") == "1"
        ctx = get_context(num_ranks=8) if port else get_context()
        m.sm_split = dict(dispatch=ctx.dispatch_sms, compute=ctx.compute_sms,
                          dispatch_ctas=ctx.dispatch_sms - int(s.eco), plan_ctas=int(s.eco),
                          roles=ctx.ep8_roles)
        exp = next(v.experts for name, v in self.get_model().named_modules() if name.endswith(".ffn"))
        buf = exp.get_symm_buffer()
        assert (buf.trace_level if port else buf.clock_offset is not None)
        assert buf.num_experts == s.slots * 8 and buf.layout.eco == s.eco
        m.bm = _import_deep_gemm().get_block_m_for_mega_moe(8, buf.num_experts,
                                                             buf.num_max_tokens_per_rank, s.rows, 6, "fp8xfp4")
        m.phase = None
        mgr = self.model_runner.cudagraph_manager
        run = mgr.run_fullgraph

        @wraps(run)
        def replay(desc):
            result = run(desc)
            if m.phase is not None:
                m.full_replays += 1
            return result
        mgr.run_fullgraph = replay
        return m.sm_split

    def ep8_profile_begin(self):
        from eco_trace import ProfileWindow
        self.ep8_window = ProfileWindow()
        return True

    def ep8_profile_end(self, step):
        from eco_trace import save_sample
        m = self._matrix
        folder = m.folder / "perfetto"
        folder.mkdir(exist_ok=True)
        ticks = self.ep8_window.finish(folder / f"step{step:04d}-rank{m.rank}.kineto.json")
        if set(STATE.samples) != {8, 20, 32}:
            raise ValueError("missing sampled layers in the decode graph")
        for layer, sample in STATE.samples.items():
            save_sample(sample, ticks, folder, rank=m.rank, step=step, layer=layer, block_m=m.bm)
        del self.ep8_window
        return True

    def ep8_begin(self, phase):
        s, m = STATE, self._matrix
        if s.eco:
            if phase == "measure":
                for dst, src in zip(s.meta, s.new_meta, strict=True):
                    dst[:len(src)].copy_(torch.from_numpy(src))
            s.warm.zero_()
        m.full_replays = 0
        if m.arm != "vanilla" and m.rank == 0:
            state, = self.model_runner.eplb_state.model_states.values()
            np.savez_compressed(m.folder / f"{phase}-layout.npz", **{
                key: tensor.cpu().numpy() for key, tensor in (
                    ("layout", state.physical_to_logical_map), ("inverse", state.logical_to_physical_map),
                    ("counts", state.logical_replica_count))})
        return begin(m, phase, s)

    def ep8_finish(self):
        s, m = STATE, self._matrix
        s.active.zero_()
        m.calls = int(s.cursor.item())
        if m.calls != CALLS or m.full_replays != 1000 or not np.array_equal(
                s.tags.cpu().numpy(), np.tile(np.arange(40), 1000)):
            raise ValueError(f"incomplete full graph: {m.full_replays} replays, {m.calls} layer calls")
        if s.eco:
            from gc_moe.eco.plan import eco_unit_config
            d = s.details.cpu().numpy()
            if not np.array_equal(d[:, 5:], s.recv[:, :s.slots].cpu().numpy()):
                raise ValueError("device auction quotas differ from physical receive counts")
            plan_start = np.ascontiguousarray(d[:, 2:4]).view(np.int64).ravel()
            if np.any(plan_start <= 0):
                raise ValueError("missing plan clock")
            clocks = s.recv[:, s.slots:].view(torch.int64)
            clocks[:, 0].copy_(torch.minimum(clocks[:, 0], torch.from_numpy(plan_start).to(clocks.device)))
            if m.phase == "measure":
                np.savez_compressed(m.folder / f"auction-rank{m.rank}.npz",
                                    accepted=d[:, 0], offered=d[:, 1], score=d[:, 4])
        positions = m.phase_start[:, None] + torch.arange(1000, device="cuda")
        native = m.tokens.native.gather(1, positions)
        np.save(m.folder / f"{m.phase}-native-rank{m.rank}.npy", native.cpu().numpy().astype(np.int32))
        if (m.publisher and m.phase == "measure" and os.getenv("SM100_BRUCE_ECO") == "1"):
            # Keep the three post-placement replay fixtures after large banks are released.
            for layer in (8, 20, 32):
                call = 40 + layer
                torch.save((s.bank_i[call].cpu(), s.bank_w[call].cpu()),
                           m.root / f"step0002-layer{layer:02d}-routes-{m.rank}.pt")
        result = m.five_finish()
        result.pop("validated_layer_graphs", None)
        result.update(full_graph_replays=m.full_replays, full_graph_observed_calls=m.calls)
        if s.eco:
            result["eco_unit_config"] = eco_unit_config(8, m.bm, 64)
        return result
