"""First post-checkpoint forward, captured in the measured graph on all ranks."""
import hashlib
import json
from pathlib import Path

import torch
import triton
import triton.language as tl


FIELDS = ('quantized', 'scales', 'shared_input', 'shared_scales', 'routed_slots', 'shared_y', 'output')


@triton.jit
def _audit_call(USED, BANK, POSITION, ENABLED, SOURCES, DEST, MARKS, EPOCH,
                N: tl.constexpr, CALLS: tl.constexpr, LAYERS: tl.constexpr,
                SIZES: tl.constexpr, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    pos = tl.load(POSITION)
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    # This is the ordinal copy already required by RouteTape on every call.
    tl.store(BANK + (pos % CALLS) * N + i, tl.load(USED + i, i < N, 0), i < N)
    if tl.load(ENABLED) & (pos < LAYERS):
        dest = DEST + pos * WIDTH
        for field in tl.static_range(len(SIZES)):
            for off in range(tl.program_id(0) * BLOCK, SIZES[field], tl.num_programs(0) * BLOCK):
                idx = off + tl.arange(0, BLOCK)
                value = tl.load(SOURCES[field] + idx, idx < SIZES[field], 0)
                tl.store(dest + idx, value, idx < SIZES[field])
            dest += SIZES[field]
        if tl.program_id(0) == 0:
            tl.store(MARKS + pos, tl.load(EPOCH))


def byte_rows(tensor):
    return tensor.view(torch.uint8).reshape(len(tensor), -1)


def row_hashes(rows):
    return [hashlib.sha256(row).hexdigest() for row in rows]


class ReplayProbe:
    def __init__(self, buf, tape):
        self.batch = tape.batch
        self.topk = buf.num_topk
        self.shared_pairs = {}
        x = buf.x[:self.batch]
        self.output_shape = x.shape
        self.raw_input = torch.empty((tape.layers, *x.shape), dtype=torch.bfloat16, device=x.device)
        self.inputs = (x, buf.x_sf[:self.batch], buf.shared_l1_acts[:self.batch],
                       buf.shared_l1_acts_sf[:self.batch], buf.combine[:self.batch], buf.shared_y[:self.batch])
        self.sizes = tuple(t.numel() * t.element_size() for t in self.inputs) + (x.numel() * 2,)
        self.data = torch.empty((tape.layers, sum(self.sizes)), dtype=torch.uint8, device=x.device)
        self.marks = torch.full_like(tape.layer_ids[:tape.layers], -1)

    def record(self, buf, output):
        tape = buf.fixed_routes
        if output.dtype != torch.bfloat16 or output.shape != self.output_shape or not output.is_contiguous():
            raise ValueError('unexpected MoE output layout in first-step probe')
        sources = tuple(t.view(torch.uint8) for t in (*self.inputs, output))
        used = buf.used_route_ordinal[:self.batch]
        _audit_call[(triton.cdiv(used.numel(), 256),)](
            used, tape.used_ordinals, tape.position, tape.replaying, sources, self.data, self.marks,
            buf.eco_tl_step, used.numel(), len(tape.ids), tape.layers, self.sizes, sum(self.sizes), 256)

    def read(self, tape, directory, rank, publish, sampler):
        from stream_checkpoint import tensor_hash
        if set(self.shared_pairs) != set(range(tape.layers)):
            raise ValueError('missing live packed role-4 weight pair for a replay layer')
        epochs = self.marks.tolist()
        if (tape.position.item() != tape.layers or epochs != tape.epochs[:tape.layers].tolist()
                or tape.layer_ids[:tape.layers].tolist() != list(range(tape.layers))
                or epochs[0] < 1 or epochs != list(range(epochs[0], epochs[0] + tape.layers))):
            raise ValueError(f'rank {rank} did not capture each layer of the first replay step')
        ids = sampler['row_requests']
        result = []
        ref_path = Path(directory) / f'probe-rows-rank{rank}.json'
        refs = {} if publish else json.loads(ref_path.read_text())
        for layer, epoch in enumerate(epochs):
            row = dict(layer=layer, epoch=epoch, hashes={}, different_rows={}, shared_nonzero={}, nonfinite={})
            row['shared_weights'] = [tensor_hash(t) for pair in self.shared_pairs[layer] for t in pair]
            fields = ('hidden_input', *FIELDS)
            slabs = (self.raw_input[layer].view(torch.uint8).reshape(-1), *self.data[layer].split(self.sizes))
            for name, raw in zip(fields, slabs, strict=True):
                host = raw.cpu().numpy()
                row['hashes'][name] = hashlib.sha256(host).hexdigest()
                if name in ('quantized', 'shared_input'):
                    row['nonfinite'][name] = int(((host & 0x7f) == 0x7f).sum())
                elif name in ('scales', 'shared_scales'):
                    row['nonfinite'][name] = int(((host.view('<u4') & 0x7f800000) == 0x7f800000).sum())
                else:
                    row['nonfinite'][name] = int(((host.view('<u2') & 0x7f80) == 0x7f80).sum())
                if name in ('shared_input', 'shared_y'):
                    # Ignore the sign bit of FP8/BF16 zero.
                    words = host if name == 'shared_input' else host.view('<u2')
                    mask = 0x7f if name == 'shared_input' else 0x7fff
                    row['shared_nonzero'][name] = bool((words & mask).any())
                slots = self.topk if name == 'routed_slots' else 1
                hashes = row_hashes(host.reshape(self.batch * slots, -1))
                key = f'{layer}/{name}'
                if publish:
                    refs[key] = hashes
                else:
                    diff = [i for i, (a, b) in enumerate(zip(refs[key], hashes, strict=True)) if a != b]
                    row['different_rows'][name] = dict(count=len(diff), examples=[
                        dict(row=i//slots, request=ids[i//slots], **({'topk_slot': i%slots} if slots > 1 else {}))
                        for i in diff[:4]])
            result.append(row)
        if publish:
            with ref_path.open('x') as f:
                json.dump(refs, f, separators=(',', ':'))
        return dict(rank=rank, layers=result, sampler=sampler)


def sample_first_step(runner, original, hidden, batch, grammar):
    """Observe the actual GPU runner sampler after TP logits gathering/sharding."""
    indices = batch.logits_indices.long()
    positions = indices.tolist()
    requests = list(batch.req_ids)
    if sorted(positions) != list(range(len(requests))) or len(requests) != len(set(requests)):
        raise ValueError('first-step audit requires one decode row per request')
    row_requests = [None] * len(requests)
    for request, pos in zip(requests, positions, strict=True):
        row_requests[pos] = request
    hidden_rows = byte_rows(hidden.index_select(0, indices).contiguous()).cpu().numpy()
    snapshot = dict(row_requests=row_requests, hidden=dict(zip(requests, row_hashes(hidden_rows), strict=True)),
                    logits={})
    sampler = runner.sampler

    class ObserveLogits:
        def __getattr__(self, name):
            return getattr(sampler, name)

        def __call__(self, logits, local_batch):
            n = local_batch.num_reqs
            ids = [runner.req_states.index_to_req_id[int(i)] for i in local_batch.idx_mapping_np[:n]]
            if len(logits) != n or len(set(ids)) != n or snapshot['logits']:
                raise ValueError('unexpected first-step logits batch')
            cpu = logits.detach().contiguous().cpu()
            hashes = row_hashes(byte_rows(cpu).numpy())
            best = cpu.argmax(dim=1).tolist()
            snapshot['logits'] = {req: dict(sha256=h, argmax=arg)
                                  for req, h, arg in zip(ids, hashes, best, strict=True)}
            return sampler(logits, local_batch)

    runner.sampler = ObserveLogits()
    try:
        returned = original(hidden, batch, grammar)
    finally:
        runner.sampler = sampler
    return returned, snapshot
