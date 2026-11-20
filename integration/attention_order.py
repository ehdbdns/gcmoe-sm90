"""Order the selected physical KV slots at the vLLM FlashMLA decode boundary."""
import hashlib

import numpy as np
import torch
import triton
import triton.language as tl


@triton.jit
def _order_prefix(X, LENGTH, Y, RAW, USED, SIZES, COUNTS, POSITION, ENABLED, MODE, layer,
                  LAYERS: tl.constexpr, STEPS: tl.constexpr, AUDIT: tl.constexpr,
                  WIDTH: tl.constexpr, STRIDE: tl.constexpr, COL: tl.constexpr,
                  LEN_STRIDE: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    n = tl.load(LENGTH + row * LEN_STRIDE)
    original = tl.load(X + row * STRIDE + col * COL, col < WIDTH, other=-1)
    prefix = (col < n) & (col < WIDTH)
    valid = prefix & (original >= 0)
    ordered = tl.sort(tl.where(valid, original, 2147483647), descending=False)
    # Keep any negative mask entries in their exact positions inside the prefix.
    if tl.sum((prefix & (original < 0)).to(tl.int32), 0) > 0:
        ordinal = tl.maximum(tl.cumsum(valid.to(tl.int32), 0) - 1, 0)
        ordered = tl.gather(ordered, ordinal, 0)
    used = tl.where(valid, ordered, original)
    tl.store(Y + row * WIDTH + col, used, col < WIDTH)
    if AUDIT:
        pos = tl.load(POSITION)
        if tl.load(ENABLED) & (pos >= 0) & (pos < STEPS * LAYERS) & (pos % LAYERS == layer):
            step = pos // LAYERS
            if row == 0:
                mode = tl.load(MODE)
                tl.atomic_add(COUNTS + step * 2 + (mode != 0).to(tl.int32), 1, sem='relaxed')
            if step < 2:
                tl.store(RAW + row * WIDTH + col, original, col < WIDTH)
                tl.store(USED + row * WIDTH + col, used, col < WIDTH)
                tl.store(SIZES + row, n)


def order_extra(indices, lengths, buf, layer):
    if (indices.ndim != 3 or indices.shape[1] != 1 or indices.dtype != torch.int32
            or lengths.shape != indices.shape[:1] or lengths.dtype != torch.int32 or lengths.device != indices.device):
        raise ValueError('unexpected sparse FlashMLA decode index layout')
    rows, _, width = indices.shape
    result = torch.empty(indices.shape, dtype=indices.dtype, device=indices.device)
    if rows == 0 or width == 0:
        return result
    tape = getattr(buf, 'fixed_routes', None)
    audit = tape is not None and rows == tape.batch
    if audit:
        banks = getattr(tape, 'order_banks', None)
        if banks is None:
            banks = tape.order_banks = {}
        if layer not in banks:
            sizes = torch.empty(rows, device=indices.device, dtype=torch.int32)
            counts = torch.empty((tape.steps, 2), device=indices.device, dtype=torch.int32)
            banks[layer] = (torch.empty_like(result), torch.empty_like(result), sizes, counts)
        raw, used, sizes, counts = banks[layer]
        if raw.shape != result.shape:
            raise ValueError(f'compressed index width changed at layer {layer}')
        pos, enabled, mode = tape.position, tape.replaying, buf.replay_stream_mode
    else:
        raw = used = sizes = counts = pos = enabled = mode = result
    _order_prefix[(rows,)](indices, lengths, result, raw, used, sizes, counts, pos, enabled, mode, layer,
                           tape.layers if audit else 1, tape.steps if audit else 1, audit, width,
                           indices.stride(0), indices.stride(-1), lengths.stride(0),
                           triton.next_power_of_2(width))
    return result


def reset_order(tape, modules, backend):
    ratios = {m.layer_id: m.compress_ratio for m in modules if isinstance(m, backend)}
    expected = {layer for layer, ratio in ratios.items() if ratio > 0}
    banks = getattr(tape, 'order_banks', {})
    if set(ratios) != set(range(tape.layers)) or 2 not in expected or expected != set(banks):
        raise ValueError(f'incomplete compressed decode coverage: model={ratios}, observed={sorted(banks)}')
    for _, _, _, counts in banks.values():
        counts.zero_()
    return sorted(expected)


def order_coverage(tape):
    modes = {}
    for layer, (_, _, _, counts) in tape.order_banks.items():
        counts = counts.cpu()
        if not torch.all((counts == 0) | (counts == 1)) or not torch.all(counts.sum(dim=1) == 1):
            raise ValueError(f'expected one ordered FlashMLA decode per step at layer {layer}')
        modes[str(layer)] = counts[:, 1].tolist()
    return modes


def row_digest(rows):
    # Hash the fixed-length SHA-256 fingerprints in request-row order.
    return hashlib.sha256(b''.join(bytes.fromhex(h) for h in rows)).hexdigest()


def read_order(tape, step, layer2):
    result = {}
    for layer, (raw, used, sizes, counts) in tape.order_banks.items():
        if counts[step].tolist() != [int(layer2['mode'] == 0), int(layer2['mode'] == 1)]:
            raise ValueError(f'missing ordered FlashMLA call at step {step}, layer {layer}')
        lengths = sizes.cpu().numpy()
        native_rows = raw[:, 0].cpu().numpy()
        used_rows = used[:, 0].cpu().numpy()
        original, actual = [], []
        for a, b, n in zip(native_rows, used_rows, lengths, strict=True):
            if not 0 <= n <= len(a):
                raise ValueError(f'invalid compressed prefix at layer {layer}')
            expected = a.copy()
            active = a[:n] >= 0
            prefix = expected[:n]
            prefix[active] = np.sort(prefix[active])
            if not np.array_equal(expected, b):
                raise ValueError(f'canonical order changed selection, mask, or padding at layer {layer}')
            original.append(hashlib.sha256(a[:n]).hexdigest())
            actual.append(hashlib.sha256(b[:n]).hexdigest())
        if layer == 2 and (actual != layer2['hashes']['extra_order'] or lengths.tolist() != layer2['lengths']['extra']):
            raise ValueError('ordered indices differ from the independent FlashMLA argument snapshot')
        reordered = sum(a != b for a, b in zip(original, actual, strict=True))
        result[str(layer)] = dict(rows=len(actual), permuted_rows=reordered, native=row_digest(original),
                                  consumed=row_digest(actual), lengths=hashlib.sha256(lengths).hexdigest())
    return result
