"""GLM-DSA TP8 decoder adapter and fixed-address telemetry for the three GC arms."""

import os
import time
from functools import wraps
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from glm53_weights import GlmWeights

ROOT = Path(os.environ["GLM_RUN"])
ARM = os.environ["GLM_ARM"]
MODEL = Path(os.environ["GLM_MODEL"])
LAYERS = tuple(range(3, 78))
SAMPLE = (8, 20, 32)
STEPS = 1000
STATE = None


def translate(weights, observed=None):
    for name, tensor in weights:
        routed = ".mlp.experts." in name
        if routed and observed is not None:
            observed.add(int(name.split(".mlp.experts.", 1)[1].split(".", 1)[0]))
        if name.endswith(".scale"):
            if tensor.dtype != torch.float8_e8m0fnu:
                raise ValueError(f"expected UE8M0 for {name}")
            if routed:
                name = name[:-6] + ".weight_scale"
                tensor = tensor.view(torch.uint8)
            else:
                name = name[:-6] + ".weight_scale_inv"
                tensor = tensor.float()
        elif routed and name.endswith(".weight"):
            if tensor.dtype != torch.int8:
                raise ValueError(f"unexpected routed payload {name}")
            tensor = tensor.view(torch.uint8)
        yield name, tensor


def slot_tables(row):
    from gc_moe.eco.meta_sm90 import reorder_single_copy_first

    row, _ = reorder_single_copy_first(row, 8)
    ids = [np.flatnonzero(row == e) for e in range(256)]
    counts = np.array(list(map(len, ids)), dtype=np.int32)
    inverse = np.zeros((256, 65), dtype=np.int32)
    for i, slots in enumerate(ids):
        inverse[i, : len(slots)] = slots
    return row, inverse, counts


class Decode:
    def __init__(self, model):
        from gc_moe import get_block_m_for_gc_moe, get_symm_buffer_for_gc_moe
        from gc_moe.eco.meta_sm100 import build_eco_meta_sm100
        from vllm.distributed import get_tp_group
        from vllm.model_executor.models.deepseek_v2 import DeepseekV2MoE

        self.group = get_tp_group().device_group
        self.rank = dist.get_rank(self.group)
        self.plan = None if ARM == "vanilla" else np.load(ROOT / "placement.npy", allow_pickle=False)
        self.slots = 32 if self.plan is None else self.plan.shape[1] // 8
        if self.plan is not None and self.plan.shape not in ((75, 320), (75, 288)):
            raise ValueError("invalid GLM EPLB capacity")
        p = self.slots * 8
        if dist.get_world_size(self.group) != 8 or torch.cuda.get_device_capability()[0] != 10:
            raise ValueError("expected eight Blackwell ranks")
        self.buf = get_symm_buffer_for_gc_moe(self.group, p, 32, 8, 6144, 2048, 1, eco=ARM == "eco")
        if self.buf.trace_level != 2:
            raise ValueError("formal GLM cohort requires trace 2")
        self.bm = get_block_m_for_gc_moe(
            8, 256 if ARM == "eco" else p, self.buf.num_max_tokens_per_rank, 32, 8
        )
        self.reader = GlmWeights(MODEL)
        self.bank, self.maps, self.samples = {}, {}, {}
        row, inv, ct = slot_tables(np.arange(p) % 256)
        self.initial_layout = row.copy()
        load_start = time.monotonic()
        for layer in LAYERS:
            mlp = model.model.layers[layer].mlp
            if not isinstance(mlp, DeepseekV2MoE):
                raise TypeError(f"GLM layer {layer} did not resolve to a routed layer")
            if not any(
                type(getattr(module, "quant_method", None)).__name__ == "Mxfp4MoEMethod"
                for module in mlp.modules()
            ):
                raise ValueError(f"prefill layer {layer} did not select the MXFP4 loader")
            self.bank[layer] = self.reader.load_gc(layer, row.reshape(8, -1)[self.rank])
            if ARM != "vanilla":
                self.maps[layer] = [torch.as_tensor(a, device="cuda") for a in (inv, ct)]
            if ARM == "eco":
                initial = build_eco_meta_sm100(row, 8, 256)
                final = build_eco_meta_sm100(self.plan[layer - 3], 8, 256)
                meta = torch.zeros(max(len(initial), len(final)), dtype=torch.int32, device="cuda")
                meta[: len(initial)].copy_(torch.from_numpy(initial))
                self.maps[layer] += [meta, torch.zeros(p + 256, dtype=torch.int16, device="cuda")]
            mlp.forward = self.wrapper(mlp, layer)
            if self.rank == 0 and (layer % 10 == 0 or layer == 77):
                print(f"[GLM GC weights] layer {layer} / 77", flush=True)
        torch.cuda.synchronize()
        self.gc_load_seconds = time.monotonic() - load_start
        self.reader.close()
        self.pos = torch.zeros(1, device="cuda", dtype=torch.long)
        self.on = torch.zeros_like(self.pos)
        self.tags = torch.full((STEPS + 1, 75), -1, device="cuda", dtype=torch.int32)
        shape = (STEPS + 1, 75, 32, 8)
        self.ids = torch.empty(shape, device="cuda", dtype=torch.int32)
        self.gates = torch.empty(shape, device="cuda", dtype=torch.float32)
        self.ref_i, self.ref_w = (
            (self.ids, self.gates)
            if ARM == "vanilla"
            else (torch.zeros_like(self.ids), torch.zeros_like(self.gates))
        )
        self.recv = torch.empty((STEPS + 1, 75, self.slots), dtype=torch.int32, device="cuda")
        self.clocks = torch.empty((STEPS + 1, 75, 21), dtype=torch.int64, device="cuda")
        self.quota = torch.empty_like(self.recv) if ARM == "eco" else None
        self.rows = torch.arange(32, device="cuda")
        self.shared_rows = self.rows // self.bm * 128 + self.rows.remainder(self.bm).remainder(32) * 4
        self.full_replays = 0
        self.setup_complete = False

    def place(self):
        from glm53_relocate import relocate

        from gc_moe.eco.meta_sm100 import build_eco_meta_sm100

        for layer in LAYERS:
            row, inv, ct = slot_tables(self.plan[layer - 3])
            if not np.array_equal(row, self.plan[layer - 3]):
                raise ValueError("placement lost single-copy prefix")
            relocate(self.bank[layer][0], self.initial_layout, row, self.group)
            for dst, src in zip(self.maps[layer][:2], (inv, ct)):
                dst.copy_(torch.from_numpy(src))
            if ARM == "eco":
                meta = build_eco_meta_sm100(row, 8, 256)
                self.maps[layer][2][: len(meta)].copy_(torch.from_numpy(meta))
                self.maps[layer][3].zero_()

    def wrapper(self, mlp, layer):
        from gc_moe import fp8_fp4_gc_moe
        from gc_moe.eco.measurement import HEADER, snapshot
        from gc_moe.layout.gc_moe import ECO_SIGNAL_OFFSETS
        from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import (
            grouped_topk,
        )

        original = mlp.forward

        @wraps(original)
        def invoke(hidden_states, already_sequence_parallel=False):
            # In vLLM DSA TP8/DP1 the next layer performs a fused TP sum.
            # Each source returns its own rows once; the sum reconstructs the batch.
            local = already_sequence_parallel
            if self.setup_complete or hidden_states.shape[0] != (32 if local else 256):
                return original(hidden_states, already_sequence_parallel=local)
            if bool(mlp.is_sequence_parallel) != local:
                raise ValueError("unexpected DSA sequence-parallel contract")
            x = hidden_states if local else hidden_states.narrow(0, self.rank * 32, 32)
            logits = mlp.gate(x)
            if isinstance(logits, tuple):
                logits = logits[0]
            w, ids = grouped_topk(
                x,
                logits,
                8,
                True,
                1,
                1,
                scoring_func="sigmoid",
                routed_scaling_factor=2.5,
                e_score_correction_bias=mlp.gate.e_score_correction_bias,
            )
            col = layer - 3
            idx = torch.where(self.on.bool(), self.pos.clamp(max=STEPS), STEPS)
            self.ids[:, col].index_copy_(0, idx, ids.int().unsqueeze(0))
            self.gates[:, col].index_copy_(0, idx, w.float().unsqueeze(0))
            if ARM != "vanilla":
                ids = torch.where(self.on.bool(), self.ref_i[:, col].index_select(0, idx)[0], ids)
                w = torch.where(self.on.bool(), self.ref_w[:, col].index_select(0, idx)[0], w)
            buf = self.buf
            if ARM == "rr":
                inv, ct = self.maps[layer]
                copy = (self.rows[:, None] * 23333 + self.rank) % ct[ids.long()]
                ids = inv[ids.long(), copy.long()]
            blocks = x.view(32, -1, 32)
            raw = blocks.abs().float().amax(-1).clamp_min(1e-4) / 448.0
            bits = raw.view(torch.int32)
            exp = ((bits >> 23) + (bits.bitwise_and(0x7FFFFF) != 0)).clamp(1, 254)
            sf = (exp << 23).view(torch.float32)
            quant = (blocks * sf.reciprocal().unsqueeze(-1)).to(torch.float8_e4m3fn).reshape_as(x)
            packed = exp.to(torch.uint8).contiguous().view(torch.int32)
            buf.x[:32].copy_(quant)
            buf.x_sf[:32].copy_(packed)
            buf.shared_l1_acts_sf.index_copy_(0, self.shared_rows, packed)
            buf.topk_idx[:32].copy_(ids)
            buf.topk_weights[:32].copy_(w)
            routed, shared = self.bank[layer]
            opts = {}
            if ARM == "eco":
                opts.update(
                    eco_meta=self.maps[layer][2], eco_state=self.maps[layer][3], eco_logical_experts=256
                )
            y = torch.empty_like(x)
            fp8_fp4_gc_moe(
                y,
                *routed,
                buf,
                shared_l1_weights=shared[0],
                shared_l2_weights=shared[1],
                shared_recipe=(1, 128, 128),
                **opts,
            )
            head = buf.trace_header[: HEADER * 8].view(torch.int64)
            counts = buf.trace_header[HEADER * 8 :].view(torch.int32)[: self.slots]
            self.clocks[:, col].index_copy_(0, idx, head[:21].unsqueeze(0))
            self.recv[:, col].index_copy_(0, idx, counts.unsqueeze(0))
            self.tags[:, col].index_fill_(0, idx, layer)
            if ARM == "eco":
                off = buf.layout.eco_base + ECO_SIGNAL_OFFSETS["eco_quota"] + 4 * self.rank * self.slots
                quotas = buf.buffer.narrow(0, off, self.slots * 4).view(torch.int32)
                self.quota[:, col].index_copy_(0, idx, quotas.unsqueeze(0))
            if layer in SAMPLE and torch.cuda.is_current_stream_capturing():
                self.samples[layer] = snapshot(buf)
            if layer == 77:
                self.pos.add_(self.on)
            if local:
                return y
            out = torch.zeros_like(hidden_states)
            out.narrow(0, self.rank * 32, 32).copy_(y)
            return out

        return invoke


def install():
    from stream_checkpoint import install_cache_names
    from vllm.models.deepseek_v32.nvidia.model import DeepseekV32ForCausalLM

    install_cache_names()
    original = DeepseekV32ForCausalLM.load_weights

    @wraps(original)
    def load(model, weights):
        global STATE
        start = time.monotonic()
        observed = set()
        result = original(model, translate(weights, observed))
        torch.cuda.synchronize()
        stock_seconds = time.monotonic() - start
        STATE = Decode(model)
        STATE.stock_load_seconds = stock_seconds
        STATE.stock_expert_ids = sorted(observed)
        print(f"[GLM load] rank={STATE.rank} vLLM read {len(observed)}/256 routed expert IDs, "
              f"vLLM={stock_seconds:.1f}s GC={STATE.gc_load_seconds:.1f}s", flush=True)
        return result

    DeepseekV32ForCausalLM.load_weights = load


install()


class GlmWorker:
    def glm_setup(self):
        from fixed_tokens import install as install_tokens

        s = STATE
        if s is None or set(s.samples) != set(SAMPLE):
            raise RuntimeError(
                "GLM hooks did not reach all sampled decoder layers in the captured graph"
            )
        s.setup_complete = True
        self.reference = install_tokens(self.model_runner, ROOT, capacity=2400)
        self.phase = None
        mgr = self.model_runner.cudagraph_manager
        original = mgr.run_fullgraph

        @wraps(original)
        def replay(*a, **kw):
            result = original(*a, **kw)
            if self.phase:
                s.full_replays += 1
            return result

        mgr.run_fullgraph = replay
        from gc_moe.eco.plan import eco_unit_config
        from gc_moe.runtime.gc_moe import get_context

        ctx = get_context(num_ranks=8)
        return {
            "rank": s.rank,
            "slots": s.slots,
            "block_m": s.bm,
            "gpu": str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid),
            "dispatch_sms": ctx.dispatch_sms,
            "compute_sms": ctx.compute_sms,
            "roles": ctx.ep8_roles,
            "trace_level": s.buf.trace_level,
            "stock_load_seconds": s.stock_load_seconds,
            "stock_expert_ids": s.stock_expert_ids,
            "gc_load_seconds": s.gc_load_seconds,
            "eco_unit": eco_unit_config(8, s.bm, (s.slots - 32) * 8) if ARM == "eco" else None,
        }

    def glm_begin(self, phase):
        from stream_checkpoint import checkpoint, named_caches, runner_boundary

        s = STATE
        if ARM != "vanilla":
            if phase == "measure":
                s.place()
            path = ROOT / f"{phase}-reference-{s.rank}.pt"
            ri, rw = torch.load(path, weights_only=True, map_location="cpu")
            for dst, src in ((s.ref_i[:-1], ri), (s.ref_w[:-1], rw)):
                if dst.shape != src.shape or dst.dtype != src.dtype:
                    raise ValueError("reference bank incompatible")
                dst.copy_(src)
        if ARM == "eco":
            for state in s.maps.values():
                state[3].zero_()
        if os.getenv("GLM_REFIT4") == "1":
            self.reference.source_batch = 32
            self.reference.source_rank = s.rank
            self.reference.source_steps = 0
            self.reference.audit_source = True
        self.phase = phase
        self.start_positions = self.reference.positions.clone()
        self.prefill_diff = 0
        if phase == "stats" and ARM != "vanilla":
            mask = (
                torch.arange(self.reference.native.shape[1], device="cuda")[None, :]
                < self.start_positions[:, None]
            )
            self.prefill_diff = int(
                ((self.reference.native != self.reference.table) & mask).sum().item()
            )
        self.boundary = checkpoint(
            ROOT / f"kv-{phase}",
            s.rank,
            ARM == "vanilla",
            named_caches(self.model_runner),
            runner_boundary(self.model_runner, self.reference),
            sparse_zeros=True,
        )
        s.full_replays = 0
        s.tags.fill_(-1)
        s.pos.zero_()
        s.on.fill_(1)
        torch.cuda.synchronize()
        return True

    def glm_profile(self, start):
        from eco_trace import ProfileWindow, save_sample

        s = STATE
        if start:
            self.window = ProfileWindow()
        else:
            folder = ROOT / ARM / "perfetto"
            ticks = self.window.finish(folder / f"step0002-rank{s.rank}.kineto.json")
            for layer, sample in s.samples.items():
                if not torch.equal(sample["counts"], s.recv[1, layer - 3]) or not torch.equal(
                    sample["clocks"], s.clocks[1, layer - 3]
                ):
                    raise ValueError("captured sample differs from replay 2 archive")
                save_sample(sample, ticks, folder, rank=s.rank, step=2, layer=layer, block_m=s.bm)
            del self.window
        return True

    def glm_end(self):
        s = STATE
        s.on.zero_()
        if (
            int(s.pos.item()) != STEPS
            or s.full_replays != STEPS
            or not torch.equal(s.tags[:-1], torch.arange(3, 78, device="cuda").expand(STEPS, -1))
        ):
            raise ValueError(f"missing calls: steps={s.pos.item()} full_replays={s.full_replays}")
        phase = self.phase
        self.phase = None
        folder = ROOT / ARM
        ids, gates = s.ids[:-1].cpu(), s.gates[:-1].cpu()
        if ARM == "vanilla":
            torch.save((ids, gates), ROOT / f"{phase}-reference-{s.rank}.pt")
        differs = (
            (0, 0)
            if ARM == "vanilla"
            else (
                int((s.ids[:-1] != s.ref_i[:-1]).sum().item()),
                int((s.gates[:-1] != s.ref_w[:-1]).sum().item()),
            )
        )
        actual = s.recv[:-1].cpu().numpy()
        if s.quota is not None and not torch.equal(s.recv[:-1], s.quota[:-1]):
            raise ValueError("ECO pull counts differ from plan quotas")
        if not torch.equal(
            self.reference.positions - self.start_positions, torch.full_like(self.start_positions, STEPS)
        ):
            raise ValueError("incomplete per-request feedback")
        ix = self.start_positions[:, None] + torch.arange(STEPS, device="cuda")
        native = self.reference.native.gather(1, ix)
        native_diff = (
            0 if ARM == "vanilla" else int((native != self.reference.table.gather(1, ix)).sum().item())
        )
        np.savez_compressed(
            folder / f"{phase}-rank{s.rank}.npz",
            counts=actual,
            clocks=s.clocks[:-1].cpu().numpy(),
            bm=s.bm,
            native=native.cpu().numpy(),
        )
        ownership = {}
        if os.getenv("GLM_REFIT4") == "1":
            self.reference.audit_source = False
            if self.reference.source_steps != STEPS or len(self.reference.source_members or ()) != 32:
                raise ValueError("incomplete independent source requests")
            ownership.update(source_requests=list(self.reference.source_members), source_steps=self.reference.source_steps)
        return {
            **ownership,
            "boundary": self.boundary,
            "batch_order": self.reference.batch_hash.hexdigest(),
            "full_replays": s.full_replays,
            "raw_gate_id_diff": differs[0],
            "raw_gate_weight_diff": differs[1],
            "native_top1_diff": native_diff,
            "prefill_native_diff": self.prefill_diff,
        }
