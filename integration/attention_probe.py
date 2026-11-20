"""Observe the live H20 FlashMLA decode call in layer 2 of the two startup steps."""
from contextvars import ContextVar
from functools import wraps

import torch
import triton
import triton.language as tl

from replay_probe import row_hashes


@triton.jit
def _snapshot(X, Y, MARK, POSITION, ENABLED, MODE, LAYER: tl.constexpr, LAYERS: tl.constexpr,
              WIDTH: tl.constexpr, ROW_STRIDE: tl.constexpr, COL_STRIDE: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    pos = tl.load(POSITION)
    if tl.load(ENABLED) & ((pos == LAYER) | (pos == LAYER + LAYERS)):
        for start in range(0, WIDTH, BLOCK):
            col = start + tl.arange(0, BLOCK)
            val = tl.load(X + row * ROW_STRIDE + col * COL_STRIDE, col < WIDTH, 0)
            tl.store(Y + row * WIDTH + col, val, col < WIDTH)
        if row == 0:
            tl.store(MARK, pos)
            tl.store(MARK + 1, tl.load(MODE))


def snapshot(buf, field, value):
    tape = buf.fixed_routes
    banks = getattr(tape, 'attention_fields', None)
    if banks is None:
        banks = tape.attention_fields = {}
    rows = value[:tape.batch].view(tape.batch, -1)
    if field not in banks:
        banks[field] = (torch.empty(rows.shape, device=rows.device, dtype=rows.dtype),
                        torch.empty(2, device=rows.device, dtype=torch.int32))
    dest, mark = banks[field]
    if dest.shape != rows.shape or dest.dtype != rows.dtype:
        raise ValueError(f'layer 2 FlashMLA decode changed {field} layout')
    _snapshot[(tape.batch,)](rows, dest, mark, tape.position, tape.replaying, buf.replay_stream_mode,
                             2, tape.layers, rows.shape[1], *rows.stride(), 256)


def wrap_decode(module, buffer):
    cls = module.DeepseekV4FlashMLAAttention
    decode, flash = cls._forward_decode, module.flash_mla_with_kvcache
    if hasattr(decode, '_gc_attention_probe'):
        return
    active = ContextVar('gc_flashmla_layer2', default=None)

    @wraps(decode)
    def observe_decode(self, *args, **kwargs):
        if self.layer_id != 2:
            return decode(self, *args, **kwargs)
        token = active.set(self.n_local_heads)
        try:
            return decode(self, *args, **kwargs)
        finally:
            active.reset(token)

    @wraps(flash)
    def observe_flash(*args, **kwargs):
        heads = active.get()
        buf = buffer() if heads is not None else None
        tape = getattr(buf, 'fixed_routes', None)
        if tape is None or len(kwargs['q']) != tape.batch:
            return flash(*args, **kwargs)
        # Native tensors go through the original backend in their original order.
        # The output projection consumes the real local heads of the padded buffer.
        snapshot(buf, 'query', kwargs['q'][:, 0, :heads])
        for field, name in (('swa_indices', 'indices'), ('swa_lengths', 'topk_length'),
                            ('extra_indices', 'extra_indices_in_kvcache'), ('extra_lengths', 'extra_topk_length')):
            snapshot(buf, field, kwargs[name])
        result = flash(*args, **kwargs)
        snapshot(buf, 'mla_output', result[0][:, 0, :heads])
        return result

    observe_decode._gc_attention_probe = True
    cls._forward_decode = observe_decode
    module.flash_mla_with_kvcache = observe_flash


def install_flash_probe(buffer):
    from vllm.models.deepseek_v41.nvidia import flashmla
    wrap_decode(flashmla, buffer)


def flash_banks(tape):
    banks = getattr(tape, 'attention_fields', {})
    fields = {'query', 'swa_indices', 'swa_lengths', 'extra_indices', 'extra_lengths', 'mla_output'}
    if set(banks) != fields:
        raise ValueError('missing live layer 2 FlashMLA fields')
    return banks


def reset_flash_probe(tape):
    # A captured allocation must never replay an initializer that clears marks
    # after step one. Arm only once, outside capture, at the verified boundary.
    for _, mark in flash_banks(tape).values():
        mark.fill_(-1)


def read_flash_probe(tape, step):
    banks = flash_banks(tape)
    marks = {tuple(mark.tolist()) for _, mark in banks.values()}
    if len(marks) != 1 or next(iter(marks))[0] != step * tape.layers + 2:
        raise ValueError('layer 2 FlashMLA fields were not observed at the requested call')
    pos, mode = marks.pop()
    values = {key: x.cpu().contiguous() for key, (x, _) in banks.items()}
    hashes = {}
    for key in ('query', 'mla_output'):
        hashes[key] = row_hashes(values[key].view(torch.uint8).numpy())
    lengths = {}
    for kind in ('swa', 'extra'):
        indices = values[kind+'_indices']
        lens = values[kind+'_lengths']
        if lens.shape != (tape.batch, 1) or (lens < 0).any() or (lens > indices.shape[1]).any():
            raise ValueError(f'invalid {kind} FlashMLA lengths')
        lengths[kind] = lens[:, 0].tolist()
        rows = [row[:n] for row, n in zip(indices, lengths[kind], strict=True)]
        hashes[kind+'_order'] = row_hashes([r.numpy() for r in rows])
        hashes[kind+'_multiset'] = row_hashes([r.sort().values.numpy() for r in rows])
    return dict(position=pos, mode=mode, hashes=hashes, lengths=lengths)
