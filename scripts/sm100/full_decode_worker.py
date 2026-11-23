"""Fixed-address record/replay at the native BS512 expert boundary inside vLLM's graph."""
from functools import wraps
import os
from pathlib import Path

import numpy as np
import torch
import triton
import triton.language as tl

LAYERS, STEPS, ROWS, K, SLOTS, CLOCKS = 40, 1000, 64, 6, 48, 19
STATE = None


@triton.jit
def full_route_stage(ids, weights, native_i, native_w, bank_i, bank_w, cursor, active,
                     scratch, template):
    lane = tl.arange(0, 512)
    n = tl.load(cursor)
    on = (tl.load(active) != 0) & (n >= 0) & (n < 40000)
    mask = (lane < 384) & on
    addr = n * 384 + lane
    i = tl.load(ids + lane, mask, 0)
    w = tl.load(weights + lane, mask, 0.)
    tl.store(native_i + addr, i, mask)
    tl.store(native_w + addr, w, mask)
    # Publisher aliases the banks to its newly recorded rows. Both arms run the same instructions.
    tl.store(ids + lane, tl.load(bank_i + addr, mask, 0), mask)
    tl.store(weights + lane, tl.load(bank_w + addr, mask, 0.), mask)
    tl.store(scratch + lane, tl.load(template + lane, lane < 86, 0), lane < 86)


@triton.jit
def full_collect_layer(scratch, results, cursor, active, tags, layer: tl.constexpr):
    lane = tl.arange(0, 128)
    n = tl.load(cursor)
    if tl.load(active) != 0:
        if n < 40000:
            tl.store(results + n * 86 + lane, tl.load(scratch + lane, lane < 86, 0), lane < 86)
            tl.store(tags + n, layer)
        tl.store(cursor, n + 1)


class CaptureState:
    def __init__(self, consumer):
        self.consumer = consumer
        self.layers = {}
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
        self.bank_i = torch.empty_like(self.ids) if consumer else self.ids
        self.bank_w = torch.empty_like(self.weights) if consumer else self.weights

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
                                   self.cursor, self.active, scratch, self.template)
            counters = scratch[:SLOTS]
            counters._matrix_clock_scratch = True
            kw["cumulative_local_expert_recv_stats"] = counters
            result = fn(y, l1, l2, buf, *args, **kw)
            full_collect_layer[(1,)](scratch, self.recv, self.cursor, self.active, self.tags, layer)
            return result
        launch._full_decode_telemetry = True
        return launch


def install_before_capture():
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
    worker._matrix = m
    return True


def begin(m, phase):
    from stream_checkpoint import checkpoint, named_caches, runner_boundary
    from five_worker import tensor_digest
    m.phase, m.steps, m.local_slots, m.nclocks = phase, STEPS, SLOTS, CLOCKS
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
    result = m.five_finish()
    result.pop("validated_layer_graphs", None)
    result["full_graph_observed_calls"] = m.calls
    return result
