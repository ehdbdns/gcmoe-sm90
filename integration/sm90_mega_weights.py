"""Convert a checkpoint expert projection to DeepGEMM's SM90 E4M3 block format."""
import torch


def convert_projection(packed, scales):
    rows, packed_k = packed.shape
    k = packed_k * 2
    if (rows % 128 or k % 128 or scales.shape != (rows, k // 32)
            or packed.dtype not in (torch.int8, torch.uint8)
            or scales.dtype != torch.float8_e8m0fnu):
        raise ValueError("expected aligned checkpoint E2M1 bytes and row-by-K32 E8M0 scales")
    values = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.,
                           -0., -.5, -1., -1.5, -2., -3., -4., -6.], device=packed.device)
    weights = torch.empty((rows, k), device=packed.device, dtype=torch.float8_e4m3fn)
    block_scales = torch.empty((rows // 128, k // 128), device=packed.device, dtype=torch.float32)
    max_error = torch.zeros((), device=packed.device)
    for row in range(0, rows, 128):
        payload = packed[row:row + 128].view(torch.uint8)
        codes = torch.stack((payload & 15, payload >> 4), dim=-1).long()
        decoded = values[codes].reshape(128, k // 32, 32) * scales[row:row + 128].float().unsqueeze(-1)
        blocks = decoded.reshape(128, k // 128, 128)
        sf = blocks.abs().amax(dim=(0, 2)).clamp_min(1e-4) / 448.
        q = (blocks / sf[None, :, None]).to(torch.float8_e4m3fn)
        weights[row:row + 128].copy_(q.reshape(128, k))
        block_scales[row // 128].copy_(sf)
        max_error = torch.maximum(max_error, (q.float() * sf[None, :, None] - blocks).abs().max())
    if not max_error.isfinite().item():
        raise ValueError("nonfinite routed projection")
    return weights, block_scales, max_error.item()
