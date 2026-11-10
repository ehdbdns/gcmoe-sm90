"""Public allocation and checkpoint conversion for the independent SM90 GC path."""
from __future__ import annotations
import os as _os
import types
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from ..layout.sm90_gc_moe import SM90GCBufferLayout, align


class SM90GCSymmBuffer:
    """Own symmetric SM90 workspace. Serialize calls and retire graphs before destroy."""
    def __init__(self, group, num_experts, num_max_tokens_per_rank, num_topk,
                 hidden, intermediate_hidden, activation='swiglu', num_shared_experts=0,
                 physical=0, eco=False, trace=False):
        ranks = group.size()
        if torch.cuda.get_device_capability()[0] != 9:
            raise ValueError('SM90 GC requires Hopper')
        if not (1 <= ranks <= 32 and num_experts > 0 and num_experts % ranks == 0
                and 1 <= num_topk <= min(32, num_experts)
                and 512 <= hidden <= 8192 and hidden % 512 == 0
                and 256 <= intermediate_hidden <= 4096 and intermediate_hidden % 256 == 0
                and num_max_tokens_per_rank > 0 and activation == 'swiglu'):
            raise ValueError('invalid SM90 GC geometry or activation')
        self.group, self.rank_idx = group, dist.get_rank(group)
        self.num_experts, self.num_topk = num_experts, num_topk
        self.num_max_tokens_per_rank = align(num_max_tokens_per_rank)
        self.hidden, self.intermediate_hidden = hidden, intermediate_hidden
        if not isinstance(num_shared_experts, int) or num_shared_experts < 0:
            raise ValueError("num_shared_experts must be a nonnegative integer")
        self.num_shared_experts = num_shared_experts
        self.activation = activation
        self.physical = int(physical) or num_experts
        # [ECO-SM90] `eco` reserves the plan/quota/counts region.
        self.eco = bool(eco)
        # Logical expert count for the plan (equals num_experts: EPLB replicates
        # physical slots, it does not add logical experts).
        self.eco_logical = int(num_experts) if eco else 0
        # [ECO-TRACE] `trace` reserves the per-step timeline even when ECO is
        # off, so the RR arm can be measured with the same columns.
        self.trace = bool(trace)
        self.layout = SM90GCBufferLayout(ranks, num_experts, self.num_max_tokens_per_rank,
                                         num_topk, hidden, intermediate_hidden, num_shared_experts,
                                         self.physical, self.eco, self.trace)
        allocator = torch if ranks == 1 else symm_mem
        self.buffer = allocator.empty(self.layout.nbytes, dtype=torch.int8, device='cuda')
        self.handle = (types.SimpleNamespace(buffer_ptrs=[self.buffer.data_ptr()]) if ranks == 1
                       else symm_mem.rendezvous(self.buffer, group=group))
        self.peer_addresses = tuple(int(p) for p in self.handle.buffer_ptrs)
        if (len(self.peer_addresses) != ranks or any(p <= 0 or p % 16 for p in self.peer_addresses)
                or self.peer_addresses[self.rank_idx] != self.buffer.data_ptr()):
            raise RuntimeError('invalid SM90 symmetric peer mappings')
        self.buffer.zero_()
        for name, spec in self.layout.views.items():
            setattr(self, name, self.buffer.narrow(0, spec.offset, spec.nbytes)
                    .view(getattr(torch, spec.dtype)).view(spec.shape))
        self.peer_table.copy_(torch.tensor(self.peer_addresses, dtype=torch.int64, device=self.buffer.device))
        self.header[0] = 1
        # [ECO-FINE-FLOOR] Arm the fine event log BEFORE the first forward.
        #
        # `fine_ctr` is monotonic and `fine_tl` holds 65536 records, so if the
        # log is left ungated the prefill calls alone consume it: at BS=256 one
        # layer emits ~39k records, i.e. the whole buffer is gone after the
        # first two prefill layers and the measured window records NOTHING.
        # Starting the floor at the buffer's own epoch (`header[0] == 1`) plus
        # `GC_MOE_FINE_EPOCHS=1` means prefill records exactly one layer, and
        # `irstep_begin()` then resets both the floor and the counter so the
        # measured window gets a clean, complete layer.
        #
        # Opt-in: standalone trace scripts (GC_MOE_FINE_MOD / _MIN_EPOCH) keep
        # the old "record everything" behaviour when this is unset.
        if _os.environ.get('GC_MOE_FINE_FLOOR') == '1' and 'fine_floor' in self.layout.views:
            self.fine_floor[0] = self.header[0].to(self.fine_floor.dtype)
        group.barrier().wait()
        torch.cuda.synchronize()

    def destroy(self):
        for name in self.layout.views:
            setattr(self, name, None)
        self.peer_addresses = self.handle = self.buffer = self.group = None


def install_eplb_plan(buffer, physical_to_logical):
    """Install one EPLB plan into the buffer and return the per-rank views.

    Args:
        buffer: live SM90GCSymmBuffer created with ``physical=len(plan)``.
        physical_to_logical: ``[physical]`` logical expert id for every slot.

    Returns:
        ``(slot_logical, expert_slots)`` as *this rank's* tensors, plus the
        global ``plan`` tensor, so the caller can remap its router output and
        build the per-slot weight view.
    """
    import torch as _t

    plan = _t.tensor([int(x) for x in physical_to_logical], dtype=_t.int32,
                     device=buffer.buffer.device)
    physical = plan.numel()
    if physical != buffer.physical:
        raise ValueError(f'plan has {physical} slots, buffer has {buffer.physical}')
    experts = buffer.num_experts
    # slot -> logical (global) and logical -> its slots (max 4 copies).
    slots_of = _t.full((experts, 4), -1, dtype=_t.int32, device=buffer.buffer.device)
    cnt = _t.zeros(experts, dtype=_t.int32, device=buffer.buffer.device)
    for g in range(physical):
        e = int(plan[g])
        k = int(cnt[e])
        if k < 4:
            slots_of[e, k] = g
        cnt[e] += 1
    buffer.slot_logical.copy_(plan)
    buffer.expert_slots.copy_(slots_of)
    return plan, slots_of


def alloc_slot_weights(pair, buffer):
    """Allocate this rank's per-slot weight rows once.

    They must keep a fixed address: a CUDA graph bakes the weight pointers into
    the kernels it records, so a rearrangement rewrites *these* tensors in
    place instead of swapping in new ones.
    """
    import torch as _t

    per_rank = buffer.physical // buffer.group.size()
    out = []
    for triple in pair:
        rows = []
        for tensor in triple:
            rows.append(_t.empty((per_rank,) + tuple(tensor.shape[1:]),
                                 dtype=tensor.dtype, device=tensor.device))
        out.append(tuple(rows))
    return tuple(out)


def _reshuffle(pair, src_rank, src_row, group, here, outs):
    """Collect ``per_rank`` rows from peers, writing into ``outs`` in place.

    ``pair``/``outs`` are ``(w1, w2)`` where each member is the
    ``(data, offsets, secondary)`` triple the kernel consumes. When ``outs`` is
    given (a rearrangement), the rows are rewritten in place: a captured CUDA
    graph has these addresses baked in.
    """
    import torch as _t

    ranks = group.size()
    want = {}
    for s, peer in enumerate(src_rank):
        want.setdefault(peer, []).append(s)
    produced_outer = []
    for gi, triple in enumerate(pair):
        produced = []
        for ti, tensor in enumerate(triple):
            rows = outs[gi][ti]
            snapshot = (tensor.clone() if rows.data_ptr() == tensor.data_ptr()
                        else tensor)
            staging = _t.empty_like(tensor.contiguous())
            for peer in range(ranks):
                if peer == here:
                    staging.copy_(snapshot)
                dist.broadcast(staging, src=peer, group=group)
                for s in want.get(peer, []):
                    rows[s].copy_(staging[src_row[s]])
            produced.append(rows)
            del staging
            if snapshot is not tensor:
                del snapshot
        produced_outer.append(tuple(produced))
    return tuple(produced_outer)


def gather_slot_weights(pair, buffer, plan, group, out=None):
    """Build this rank's weight rows, one per physical slot.

    Slot g must hold the weights of logical expert ``plan[g]``; those live on
    rank ``(plan[g] // logical_local)`` at local row
    ``(plan[g] % logical_local)``. This gathers them once (not per step).

    ``out`` rewrites an existing per-slot view in place, which is what a live
    CUDA graph requires (its kernels hold these addresses).
    """
    ranks = group.size()
    logical_local = buffer.num_experts // ranks
    per_rank = buffer.physical // ranks
    here = buffer.rank_idx
    mine = plan[here * per_rank:(here + 1) * per_rank]
    src_rank = (mine // logical_local).tolist()
    src_row = (mine % logical_local).tolist()
    outs = alloc_slot_weights(pair, buffer) if out is None else out
    return _reshuffle(pair, src_rank, src_row, group, here, outs)


def gather_slot_weights_from_slots(pair, buffer, source_slot, plan, group, out=None):
    """Rebuild per-slot weight rows from the *already gathered* slot view.

    Rearrangement must not keep a second, logical-row copy of the weights
    alive: at 384 experts x 40 layers that is another ~36 GiB per rank, which
    does not fit next to the live slot view. Every logical expert is still
    reachable through ``source_slot[e]`` -- one global slot id that currently
    holds it -- so a rearrangement is another shuffle of the same bytes.

    Args:
        pair: the current ``(w1, w2)`` per-slot view for this rank.
        source_slot: ``[experts]`` global slot id holding each logical expert
            under the *current* plan (e.g. ``old_slots_of[:, 0]``).
        plan: the new ``[physical]`` slot -> logical map.
        out: existing per-slot view to overwrite (required once captured).
    """
    ranks = group.size()
    per_rank = buffer.physical // ranks
    here = buffer.rank_idx
    mine = plan[here * per_rank:(here + 1) * per_rank]
    src_slot = source_slot.to(torch.int64)[mine.to(torch.int64)]
    src_rank = (src_slot // per_rank).tolist()
    src_row = (src_slot % per_rank).tolist()
    outs = alloc_slot_weights(pair, buffer) if out is None else out
    return _reshuffle(pair, src_rank, src_row, group, here, outs)


def get_symm_buffer_for_gc_moe_sm90(group, num_experts, num_max_tokens_per_rank,
                                  num_topk, hidden, intermediate_hidden,
                                  activation='swiglu', num_shared_experts=0, physical=0,
                                  eco=False, trace=False):
    # [ECO-SM90] `eco=True` reserves the ECO plan/quota/counts region.
    return SM90GCSymmBuffer(group, num_experts, num_max_tokens_per_rank, num_topk,
                           hidden, intermediate_hidden, activation, num_shared_experts,
                           physical, eco, trace)


# Exact Humming code-space round-downscale mapping, including .25 -> .5.
_DOWNSCALE = (
    (0,1,2,3,4,5,6,7,0,9,10,11,12,13,14,15),
    (0,1,1,2,2,3,4,5,0,9,9,10,10,11,12,13),
    (0,0,1,1,1,2,2,3,0,8,9,9,9,10,10,11),
    (0,0,0,0,1,1,1,2,0,8,8,8,9,9,9,10),
    (0,0,0,0,0,0,1,1,0,8,8,8,8,8,9,9),
    (0,0,0,0,0,0,0,0,0,8,8,8,8,8,8,8),
)


def _scale_codes(scales):
    if scales.dtype in (torch.uint8, torch.float8_e8m0fnu):
        codes = scales.view(torch.uint8).to(torch.int32)
        if bool((codes == 255).any()):
            raise ValueError('NaN UE8M0 weight scale')
        return codes
    if scales.dtype != torch.float32:
        raise ValueError('scales must be UE8M0 bytes or FP32 powers of two')
    bits = scales.contiguous().view(torch.int32)
    valid = ((bits > 0) & (bits < 0x7f800000) & ((bits & 0x7fffff) == 0)) | (bits == 0x00400000)
    if not bool(valid.all()):
        raise ValueError('weight scales must be finite positive powers of two')
    return ((bits >> 23) & 255).to(torch.int32)


def _transform(raw, scale_2=None, paired=False):
    packed, scales = raw
    if packed.ndim != 3 or packed.dtype not in (torch.int8, torch.uint8):
        raise ValueError('expected [E,N,K/2] packed MXFP4')
    e, n, half_k = packed.shape
    k = 2 * half_k
    if n % 256 or k % 128 or tuple(scales.shape) != (e, n, k // 32) or scales.device != packed.device:
        raise ValueError('invalid SM90 weight or scale shape')
    codes = _scale_codes(scales)
    if scale_2 is not None and (scale_2.shape != (e,) or scale_2.dtype != torch.float32
                               or scale_2.device != packed.device or not bool((torch.isfinite(scale_2) & (scale_2 > 0)).all())):
        raise ValueError('secondary checkpoint scale must be positive finite FP32 [E]')
    data = torch.empty((e, n // 256, k // 128, 4, 256, 16), device=packed.device, dtype=torch.int8)
    offsets = torch.empty((e, n // 256, k // 128, 256, 4), device=packed.device, dtype=torch.uint8)
    secondary = torch.empty(e, device=packed.device, dtype=torch.float32)
    lut = torch.tensor(_DOWNSCALE, dtype=torch.uint8, device=packed.device)
    for expert in range(e):
        sf = codes[expert]
        base = torch.maximum(sf.min(), sf.max() - 11)
        target = torch.maximum(sf, base)
        delta = target - sf
        if bool((delta >= 128).any()):
            raise ValueError('weight exponent span exceeds Humming finite range')
        source = packed[expert].contiguous().view(torch.uint8).reshape(n, k // 32, 16)
        d = delta.clamp_max(5).long().unsqueeze(-1)
        converted = (lut[d, (source & 15).long()] | (lut[d, (source >> 4).long()] << 4)).reshape(n, half_k)
        # Each lane owns four consecutive K values and four at K+16.
        # Move bytes, then interleave only the sign bits for two PRMT outputs.
        order = torch.tensor([0,1,8,9,2,3,10,11,4,5,12,13,6,7,14,15],
                             device=packed.device)
        relative = (target - base + 1).to(torch.uint8)
        if paired:
            converted = converted.reshape(2, n // 256, 128, half_k).permute(1, 0, 2, 3).contiguous().reshape(n, half_k)
            relative = relative.reshape(2, n // 256, 128, k // 32).permute(1, 0, 2, 3).contiguous().reshape(n, k // 32)
        units = converted.reshape(n // 256, 256, k // 128, 4, 16)
        units = units.permute(0, 2, 3, 1, 4).contiguous()[..., order]
        words = units.reshape(-1, 4)
        moved = words & 0x77
        moved[:, 0] |= (words[:, 0] & 0x08) | ((words[:, 2] & 0x08) << 4)
        moved[:, 1] |= ((words[:, 0] & 0x80) >> 4) | (words[:, 2] & 0x80)
        moved[:, 2] |= (words[:, 1] & 0x08) | ((words[:, 3] & 0x08) << 4)
        moved[:, 3] |= ((words[:, 1] & 0x80) >> 4) | (words[:, 3] & 0x80)
        # Per M64: two 512B halves per warp, each lane owns a 16B
        # vector of {g0/s0,g0/s1,g1/s0,g1/s1} packed words. This is
        # an exact permutation of the same FP4 bytes, with no expansion.
        vectors = moved.reshape(n // 256, k // 128, 2, 2, 4, 4, 2, 8, 4, 4)
        vectors = vectors.permute(0, 1, 4, 5, 2, 7, 8, 3, 6, 9).contiguous()
        data[expert].copy_(vectors.reshape(data[expert].shape).view(torch.int8))
        sf_vectors = relative.reshape(n // 256, 4, 4, 2, 8, k // 128, 4)
        sf_vectors = sf_vectors.permute(0, 5, 1, 2, 4, 3, 6).contiguous()
        offsets[expert].copy_(sf_vectors.reshape(offsets[expert].shape))
        secondary[expert] = torch.exp2(base.float() - 128) * (1 if scale_2 is None else scale_2[expert])
    if not bool((torch.isfinite(secondary) & (secondary > 0)).all()):
        raise ValueError('secondary scale overflow or underflow')
    return data, offsets, secondary


def transform_weights_for_gc_moe_sm90(l1_weights, l2_weights,
                                      l1_weight_scale_2=None, l2_weight_scale_2=None,
                                      shared_recipe=(1, 1, 32)):
    """Convert raw MXFP4 to independent register-A tiles, retaining natural gate/up rows."""
    if l1_weights[0].dtype == torch.float8_e4m3fn:
        if l1_weight_scale_2 is not None or l2_weight_scale_2 is not None:
            raise ValueError('dense shared weights do not have secondary scales')
        return (_transform_shared(l1_weights, shared_recipe, True),
                _transform_shared(l2_weights, shared_recipe, False))
    if (l1_weight_scale_2 is None) != (l2_weight_scale_2 is None):
        raise ValueError('both checkpoint secondary scales must be supplied together')
    if (l1_weights[0].ndim != 3 or l2_weights[0].ndim != 3
            or l1_weights[0].shape[0] != l2_weights[0].shape[0]
            or l1_weights[0].shape[1] != 4 * l2_weights[0].shape[2]
            or l2_weights[0].shape[1] != 2 * l1_weights[0].shape[2]):
        raise ValueError('W1/W2 geometry mismatch')
    return _transform(l1_weights, l1_weight_scale_2, paired=True), _transform(l2_weights, l2_weight_scale_2)


def _transform_shared(pair, recipe, paired):
    """Lossless FP8 byte permutation and exact FP32 scale expansion to row/K32.

    GLM block N128/K128 scales are repeated over rows and four K32 groups.
    Flash per-row K32 scales are retained without rescaling any FP8 value.
    """
    data, sf = pair
    n, k = data.shape
    if recipe not in ((1, 128, 128), (1, 1, 32)) or n % 256 or k % 128:
        raise ValueError('invalid dense shared geometry or recipe')
    if data.dtype != torch.float8_e4m3fn or sf.dtype != torch.float32:
        raise ValueError('shared weights require E4M3 data and FP32 scales')
    if tuple(sf.shape) != (n // recipe[1], k // recipe[2]):
        raise ValueError('invalid dense shared scale shape')
    scales = sf.repeat_interleave(recipe[1], 0).repeat_interleave(recipe[2] // 32, 1)
    values = data.view(torch.uint8)
    if paired:
        values = values.reshape(2, n // 256, 128, k).permute(1, 0, 2, 3).reshape(n, k)
        scales = scales.reshape(2, n // 256, 128, k // 32).permute(1, 0, 2, 3).reshape(n, k // 32)
    lanes = torch.arange(128, device=data.device)
    rows = (lanes // 32) * 16 + (lanes % 32) // 4
    ks = (lanes % 4) * 4
    row_idx = (torch.arange(n // 256, device=data.device)[:, None, None, None, None, None] * 256
               + torch.arange(4, device=data.device)[None, None, :, None, None, None] * 64
               + rows[None, None, None, :, None, None])
    k_idx = (torch.arange(k // 128, device=data.device)[None, :, None, None, None, None] * 128
             + torch.arange(4, device=data.device)[None, None, None, None, :, None] * 32
             + ks[None, None, None, :, None, None])
    words = [values[row_idx + rr, k_idx + kk + torch.arange(4, device=data.device)]
             for rr, kk in ((0, 0), (8, 0), (0, 16), (8, 16))]
    # At fixed K32, adjacent lanes read contiguous LDS.128 vectors.
    tiled = torch.cat(words, dim=-1).permute(0, 1, 2, 4, 3, 5).contiguous()
    tiled_sf = scales.reshape(n // 256, 4, 64, k // 128, 4).permute(0, 3, 1, 2, 4).contiguous()
    return tiled, tiled_sf


def sm90_fp8_fp4_gc_moe(y, l1_weights, l2_weights, sym_buffer,
                        cumulative_local_expert_recv_stats=None, recipe=(1,1,32),
                        activation='swiglu', activation_clamp=None, fast_math=True,
                        num_tokens_bound=None, shared_l1_weights=None,
                        shared_l2_weights=None, shared_recipe=None):
    from ..apis.sm90_gc_moe import sm90_fp8_fp4_gc_moe as launch
    launch(y, l1_weights, l2_weights, sym_buffer, cumulative_local_expert_recv_stats,
           recipe, activation, activation_clamp, fast_math, num_tokens_bound,
           shared_l1_weights, shared_l2_weights, shared_recipe)
