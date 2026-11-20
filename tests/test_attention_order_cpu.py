"""Partial, masked, noncontiguous and reused compressed FlashMLA index rows."""
import os
import sys
from pathlib import Path

gpu = sys.argv[1:] == ['--cuda']
if not gpu:
    os.environ['TRITON_INTERPRET'] = '1'
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'integration'))
from attention_order import order_extra


def exercise(device):
    rows = 12
    # The interpreter exercises mask and stride cases on small rows. The device
    # preflight additionally covers the model width and a larger non-power-of-two.
    widths = (1, 7, 512, 1025) if gpu else (1, 7, 31)
    for width in widths:
        gen = torch.Generator(device=device).manual_seed(24)
        source = torch.randint(0, 23, (rows, width * 2), generator=gen, dtype=torch.int32, device=device)
        x = source[:, None, ::2]
        lengths = torch.linspace(0, width, rows, device=device).to(torch.int32).repeat_interleave(2)[::2]
        x[8:, :, :1] = 2147483647
        x[3:6, :, ::3] = -1
        x[5, :, 1::3] = -7
        orig = x.clone()
        actual = order_extra(x, lengths, None, 7)
        expected = orig.clone()
        for row, n in zip(expected, lengths.tolist(), strict=True):
            prefix = row[0, :n]
            valid = prefix >= 0
            prefix[valid] = prefix[valid].sort().values
        assert torch.equal(actual, expected)
        assert torch.equal(x, orig) and actual.data_ptr() != x.data_ptr()
        assert torch.equal(order_extra(actual, lengths, None, 7), actual)
        assert order_extra(x[:0], lengths[:0], None, 7).shape == (0, 1, width)


if __name__ == '__main__':
    exercise('cuda' if gpu else 'cpu')
    print('ATTENTION_ORDER_CUDA PASS' if gpu else 'ATTENTION_ORDER_CPU PASS')
