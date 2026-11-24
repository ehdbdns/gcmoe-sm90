"""Extract verified, ordered top-k rows for three consecutive measured decode steps."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

LAYERS = (8, 20, 32)
STEPS = (0, 1, 2)


def extract(group, bank, out, bs):
    hashes = {name: digest for digest, name in
              (line.split() for line in (group / 'workload.sha256').read_text().splitlines())}
    expected = [hashes[f'measure-rank{rank}.pt'] for rank in range(8)]
    out.mkdir()
    demand = np.zeros((3, 3, 384), dtype=np.int64)
    sources = []
    for digest in dict.fromkeys(expected):
        ranks = [rank for rank, value in enumerate(expected) if value == digest]
        candidates = [bank / f'measure-rank{rank}.pt' for rank in ranks]
        path = next((p for p in candidates if p.is_file()), None)
        if path is None:
            raise FileNotFoundError(f'missing measured bank for {digest} in {bank}')
        h = hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda: f.read(8 << 20), b''):
                h.update(block)
        if h.hexdigest() != digest:
            raise ValueError(f'bank checksum mismatch: {path}')
        data = torch.load(path, map_location='cpu', mmap=True, weights_only=True)
        for key, dtype in [('ids', torch.int64), ('weights', torch.float32)]:
            t = data[key]
            if t.shape != (40000, bs, 6) or t.dtype != dtype:
                raise ValueError(f'{path}: wrong {key} shape or dtype')
        for li, layer in enumerate(LAYERS):
            rows = [40 * step + layer for step in STEPS]
            ids = data['ids'][rows].numpy()
            weights = data['weights'][rows].numpy()
            if ids.min() < 0 or ids.max() >= 384 or not np.isfinite(weights).all():
                raise ValueError(f'invalid taped routes at L{layer}')
            for step, ids_step in enumerate(ids):
                demand[li, step] += len(ranks) * np.bincount(ids_step.ravel(), minlength=384)
            for rank in ranks:
                np.savez(out / f'layer{layer}-rank{rank}.npz', ids=ids, weights=weights)
        sources.append(dict(path=str(path), sha256=digest, source_ranks=ranks))

    # Vanilla has identity placement. Its 8 contiguous physical shards are
    # independent evidence for the logical demand at each measured step.
    for rank in range(8):
        with np.load(group / 'vanilla' / f'rank{rank}.npz') as archive:
            counts = archive['counts']
            for li, layer in enumerate(LAYERS):
                rows = [40 * step + layer for step in STEPS]
                if not np.array_equal(counts[rows], demand[li, :, rank*48:(rank+1)*48]):
                    raise ValueError(f'historical logical demand mismatch: L{layer} rank{rank}')
    (out / 'manifest.json').write_text(json.dumps(dict(
        bs=bs, tokens_per_source=bs, steps_zero_based=STEPS, layers=LAYERS, sources=sources,
        archive=str(group), routes_per_layer=bs*8*6), indent=2)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('group', type=Path)
    p.add_argument('bank', type=Path)
    p.add_argument('out', type=Path)
    p.add_argument('bs', type=int, choices=(512, 1024))
    args = p.parse_args()
    extract(args.group, args.bank, args.out, args.bs)
