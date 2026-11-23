"""Fixed-address record/replay at the native BS512 expert boundary inside vLLM's graph."""
from functools import wraps
import os
from pathlib import Path

import numpy as np
import torch
import triton
import triton.language as tl

LAYERS, STEPS, ROWS, K, SLOTS, CLOCKS = 40, 1000, 64, 6, 48, 19
FIRST_DIFF = os.getenv("SM100_FIRST_DIFF") == "1"
if FIRST_DIFF:
    STEPS = 1
FAIR = os.getenv("SM100_FAIR_BOUNDARY") == "1"
if FAIR:
    STEPS = 3
OBSERVE = os.getenv("SM100_NATIVE_GREEDY") == "1"
STATE = None


@triton.jit
def full_route_stage(ids, weights, native_i, native_w, bank_i, bank_w, cursor, active,
                     scratch, template, limit: tl.constexpr, replay: tl.constexpr = True):
    lane = tl.arange(0, 512)
    n = tl.load(cursor)
    on = (tl.load(active) != 0) & (n >= 0) & (n < limit)
    mask = (lane < 384) & on
    addr = n * 384 + lane
    i = tl.load(ids + lane, mask, 0)
    w = tl.load(weights + lane, mask, 0.)
    tl.store(native_i + addr, i, mask)
    tl.store(native_w + addr, w, mask)
    # Publisher aliases the banks to its newly recorded rows. Both arms run the same instructions.
    if replay:
        tl.store(ids + lane, tl.load(bank_i + addr, mask, 0), mask)
        tl.store(weights + lane, tl.load(bank_w + addr, mask, 0.), mask)
    tl.store(scratch + lane, tl.load(template + lane, lane < 86, 0), lane < 86)


@triton.jit
def full_collect_layer(scratch, results, cursor, active, tags, layer: tl.constexpr, limit: tl.constexpr):
    lane = tl.arange(0, 128)
    n = tl.load(cursor)
    if tl.load(active) != 0:
        if n < limit:
            tl.store(results + n * 86 + lane, tl.load(scratch + lane, lane < 86, 0), lane < 86)
            tl.store(tags + n, layer)
        tl.store(cursor, n + 1)


class CaptureState:
    def __init__(self, consumer):
        self.consumer = consumer
        self.layers = {}
        self.snapshots = {}
        self.cursor = torch.zeros(1, dtype=torch.int32, device="cuda")
        self.active = torch.zeros_like(self.cursor)
        self.tags = torch.empty(STEPS * LAYERS, dtype=torch.int32, device="cuda")
        self.recv = torch.empty((STEPS * LAYERS, SLOTS + 2*CLOCKS), dtype=torch.int32, device="cuda")
        self.scratch = torch.empty_like(self.recv[0])
        self.template = torch.zeros_like(self.scratch)
        clocks = self.template[SLOTS:].view(torch.int64)
        clocks[:2].fill_(-1)
        clocks[4::2].fill_(-1)
        shape = (STEPS * LAYERS, ROWS, K)
        self.ids = torch.empty(shape, device="cuda", dtype=torch.int32)
        self.weights = torch.empty(shape, device="cuda", dtype=torch.float32)
        self.bank_i = torch.empty_like(self.ids) if consumer and not OBSERVE else self.ids
        self.bank_w = torch.empty_like(self.weights) if consumer and not OBSERVE else self.weights

    def wrap(self, fn):
        @wraps(fn)
        def launch(y, l1, l2, buf, *args, **kw):
            if y.shape[0] != ROWS:
                return fn(y, l1, l2, buf, *args, **kw)
            key = l1[0].data_ptr()
            layer = self.layers[key]
            ids, weights = buf.topk_idx[:ROWS], buf.topk_weights[:ROWS]
            if ids.shape != (ROWS, K) or weights.shape != ids.shape or not (ids.is_contiguous() and weights.is_contiguous()):
                raise ValueError("unexpected symmetric route buffer layout")
            scratch = (buf.buffer.narrow(0, buf.clock_offset - SLOTS*4, self.scratch.numel()*4)
                       .view(torch.int32) if self.consumer else self.scratch)
            full_route_stage[(1,)](ids, weights, self.ids, self.weights, self.bank_i, self.bank_w,
                                   self.cursor, self.active, scratch, self.template, STEPS * LAYERS, not OBSERVE)
            counters = scratch[:SLOTS]
            counters._matrix_clock_scratch = True
            kw["cumulative_local_expert_recv_stats"] = counters
            if FIRST_DIFF and layer < 4:
                # Fixed storage reused by warmup, capture, and decode replay.
                inputs = dict(x=buf.x[:ROWS].view(torch.uint8), sf=buf.x_sf[:ROWS],
                              ids=ids, weights=weights)
                if layer not in self.snapshots:
                    self.snapshots[layer] = {k: torch.empty_like(v) for k, v in inputs.items()}
                    self.snapshots[layer]["y"] = torch.empty_like(y)
                for name, value in inputs.items():
                    self.snapshots[layer][name].copy_(value)
            result = fn(y, l1, l2, buf, *args, **kw)
            if FIRST_DIFF and layer < 4:
                self.snapshots[layer]["y"].copy_(y)
            full_collect_layer[(1,)](scratch, self.recv, self.cursor, self.active, self.tags, layer, STEPS * LAYERS)
            return result
        launch._full_decode_telemetry = True
        return launch


def install_before_capture():
    if FAIR:
        import fair_boundary
        fair_boundary.install()
    from vllm.models.deepseek_v4.nvidia.model import DeepseekV4MegaMoEExperts
    from vllm.utils.deep_gemm import _import_deep_gemm
    original = DeepseekV4MegaMoEExperts.finalize_weights

    @wraps(original)
    def finalize(expert, *args, **kwargs):
        global STATE
        result = original(expert, *args, **kwargs)
        dg = _import_deep_gemm()
        gc = os.environ["GC_MOE_V41_ENGINE"] == "gc"
        if STATE is None:
            STATE = CaptureState(gc)
        key = expert._transformed_l1_weights[0].data_ptr()
        STATE.layers[key] = expert.layer_id
        if not getattr(dg.fp8_fp4_mega_moe, "_full_decode_telemetry", False):
            if dg.fp8_fp4_mega_moe.__module__.startswith("gc_moe") != gc:
                raise ValueError("full decode expert backend mismatch")
            dg.fp8_fp4_mega_moe = STATE.wrap(dg.fp8_fp4_mega_moe)
        return result
    DeepseekV4MegaMoEExperts.finalize_weights = finalize


def setup(worker, root, arm, bs):
    from fixed_tokens import install
    from vllm.utils.deep_gemm import _import_deep_gemm
    from five_worker import FiveWorker
    import inspect
    if bs != 512 or arm not in ("gc", "mega") or STATE is None or sorted(STATE.layers.values()) != list(range(LAYERS)):
        raise ValueError("full-decode instrumentation did not see all native layers before replay")
    m = FiveWorker()
    m.model_runner = worker.model_runner
    m.root, m.arm, m.bs = Path(root), arm, bs
    m.folder, m.rank = m.root / arm, torch.distributed.get_rank()
    m.publisher, m.eco_tables = arm == "mega", None
    if not OBSERVE:
        m.tokens = install(m.model_runner, m.root, capacity=2128)
    m.layer_graphs = {}
    m.host_submit = None
    dg = _import_deep_gemm()
    m.backend_file = inspect.getfile(dg.fp8_fp4_mega_moe.__wrapped__)
    if arm == "gc" and Path(os.environ["SM100_CLOCKED_UPSTREAM"]).resolve() not in Path(m.backend_file).resolve().parents:
        raise ValueError("wrong GC implementation")
    if Path(os.environ["SM100_MATRIX_OVERLAY"]).resolve() not in Path(dg.__file__).resolve().parents:
        raise ValueError("private DeepGEMM overlay is missing")
    if arm == "gc":
        from gc_moe.runtime.gc_moe import get_context
        ctx = get_context()
        m.sm_split = dict(dispatch=ctx.dispatch_sms, compute=ctx.compute_sms)
    experts = [v.experts for name, v in worker.get_model().named_modules()
               if name.endswith(".ffn") and hasattr(v, "experts")]
    buf = experts[0].get_symm_buffer()
    m.bm = dg.get_block_m_for_mega_moe(8,384,buf.num_max_tokens_per_rank,ROWS,K,"fp8xfp4")
    if FAIR:
        from vllm.models.deepseek_v41.nvidia.model import DeepseekV4DecoderLayer
        m.fair_layers = sorted(
            (v for v in worker.get_model().modules() if isinstance(v, DeepseekV4DecoderLayer)),
            key=lambda v: v.ffn.experts.layer_id,
        )
        assert [v.ffn.experts.layer_id for v in m.fair_layers] == list(range(LAYERS))
    worker._matrix = m
    return True


def begin(m, phase):
    from stream_checkpoint import checkpoint, named_caches, runner_boundary
    from five_worker import tensor_digest
    m.phase, m.steps, m.local_slots, m.nclocks = phase, STEPS, SLOTS, CLOCKS
    if not OBSERVE:
        m.phase_start = m.tokens.positions.clone()
        folder = m.root / f"kv-{phase}"
        folder.mkdir(exist_ok=True)
        m.boundary = checkpoint(folder, m.rank, m.publisher, named_caches(m.model_runner),
                                runner_boundary(m.model_runner, m.tokens))
        if phase == "stats":
            valid = torch.arange(m.tokens.native.shape[1], device="cuda")[None, :] < m.phase_start[:, None]
            m.prefill_native = tensor_digest(m.tokens.native.masked_fill(~valid, -1))
            m.prefill_diff = {} if m.publisher else {"count": int((valid & (m.tokens.native != m.tokens.table)).sum())}
        m.raw_ids, m.raw_weights = STATE.ids, STATE.weights
        m.bank_ids, m.bank_weights = STATE.bank_i, STATE.bank_w
        if not m.publisher:
            ids, weights = torch.load(m.root / f"{phase}-routes-{m.rank}.pt", map_location="cpu", weights_only=True)
            for target, source in ((m.bank_ids, ids), (m.bank_weights, weights)):
                if target.shape != source.shape or target.dtype != source.dtype:
                    raise ValueError("route reference shape/dtype mismatch")
                target.copy_(source)
    if FAIR:
        import fair_boundary
        fair_boundary.boundary(m)
    m.recv = STATE.recv
    STATE.cursor.zero_()
    STATE.tags.fill_(-1)
    STATE.active.fill_(1)
    torch.cuda.synchronize()
    return True


def finish(m):
    STATE.active.zero_()
    m.calls = int(STATE.cursor.item())
    if not np.array_equal(STATE.tags.cpu().numpy(), np.tile(np.arange(LAYERS), STEPS)):
        raise ValueError("missing, overwritten or out-of-order full graph layer")
    if OBSERVE:
        if m.calls != STEPS * LAYERS:
            raise ValueError("incomplete native decode")
        if FIRST_DIFF:
            for layer in range(4):
                tensors = STATE.snapshots[layer]
                np.savez(m.folder / f"layer{layer}-rank{m.rank}.npz",
                         **{k: (v.view(torch.int16) if k == "y" and v.dtype == torch.bfloat16 else v)
                            .cpu().numpy() for k, v in tensors.items()})
        phase = m.phase
        counts = STATE.recv[:, :SLOTS].cpu().numpy().reshape(STEPS, LAYERS, SLOTS)
        clocks = STATE.recv[:, SLOTS:].view(torch.int64).cpu().numpy().reshape(STEPS, LAYERS, CLOCKS)
        np.savez_compressed(m.folder / f"{phase}-rank{m.rank}.npz", counts=counts, clocks=clocks)
        for key, tensor in (("ids", STATE.ids), ("weights", STATE.weights)):
            np.save(m.folder / f"{phase}-{key}-rank{m.rank}.npy", tensor.cpu().numpy())
        m.phase = None
        extra = {}
        if FAIR:
            import fair_boundary
            extra["kv"] = fair_boundary.replay(m)
        return dict(**extra, block_m=int(m.bm), backend_file=m.backend_file, sm_split=getattr(m, "sm_split", None),
                    full_graph_observed_calls=m.calls)
    result = m.five_finish()
    result.pop("validated_layer_graphs", None)
    result["full_graph_observed_calls"] = m.calls
    return result
