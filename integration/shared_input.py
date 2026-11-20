"""Stage the SM90 dense shared expert's K32 input directly at its graph address."""
import torch
import triton
import triton.language as tl


@triton.jit
def _quantize_shared(X, Q, SF, RAW, ENABLED, POS, PROBE: tl.constexpr, LAYERS: tl.constexpr, BATCH: tl.constexpr, H: tl.constexpr, ROW_STRIDE: tl.constexpr,
                     COL_STRIDE: tl.constexpr, GROUPS: tl.constexpr):
    row = tl.program_id(0)
    group = tl.program_id(1) * GROUPS + tl.arange(0, GROUPS)
    col = group[:, None] * 32 + tl.arange(0, 32)[None, :]
    source = tl.load(X + row * ROW_STRIDE + col * COL_STRIDE, mask=col < H, other=0)
    if PROBE:
        pos = tl.load(POS)
        if tl.load(ENABLED) & (pos < LAYERS):
            tl.store(RAW + (pos * BATCH + row) * H + col, source, mask=col < H)
    x = source.to(tl.float32)
    scale = tl.div_rn(tl.maximum(tl.max(tl.abs(x), 1), 1.e-4), 448.)
    bits = scale.to(tl.uint32, bitcast=True)
    exponent = ((bits >> 23) & 255) + ((bits & 0x7fffff) != 0).to(tl.uint32)
    scale = (tl.minimum(tl.maximum(exponent, 1), 254) << 23).to(tl.float32, bitcast=True)
    q = (x * (1. / scale[:, None])).to(tl.float8e4nv)
    tl.store(Q + row * H + col, q.to(tl.uint8, bitcast=True), mask=col < H)
    tl.store(SF + row * (H // 32) + group, scale, mask=group < H // 32)


def stage_shared_input(x, buf):
    rows, hidden = x.shape
    if hidden % 32 or x.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError('SM90 shared input requires floating-point rows with K divisible by 32')
    tape = getattr(buf, 'fixed_routes', None)
    probe = getattr(tape, 'first_step_probe', None) if tape is not None and rows == tape.batch else None
    if probe is not None and x.dtype != probe.raw_input.dtype:
        raise ValueError('first-step diagnostic requires the original BF16 activation')
    raw = probe.raw_input if probe is not None else x
    enabled = tape.replaying if probe is not None else x
    position = tape.position if probe is not None else x
    if rows:
        _quantize_shared[(rows, triton.cdiv(hidden // 32, 16))](
            x, buf.shared_l1_acts.view(torch.uint8), buf.shared_l1_acts_sf,
            raw, enabled, position, probe is not None, tape.layers if probe is not None else 0, rows if probe is not None else 0,
            hidden, x.stride(0), x.stride(1), 16)
