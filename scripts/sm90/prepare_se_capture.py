"""Build three layer placements from the new vanilla statistics and verify route slices."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from gc_moe.eco.meta_sm90 import reorder_single_copy_first
from gc_moe.eplb.policy import rebalance_experts_global


def prepare(group, bs):
    run = json.loads((group/'capture.json').read_text())
    layers = [8, 20, 32]
    if (run['batch'], run['layers'], run['stats_steps'], run['capture_steps']) != (bs, layers, 1000, 3):
        raise ValueError('wrong collection window')
    gen = torch.Generator().manual_seed(0)
    random = torch.cat([torch.randperm(384, generator=gen), torch.randint(384, (64,), generator=gen)]).tolist()
    maps = {k: [None]*40 for k in ('vanilla','random','rr','eco','eco_smr')}
    with np.load(group/'stats-demand.npz') as archive:
        history = archive['logical']
    if (history.shape != (1000, 40, 384) or np.any(history < 0) or
            not np.all(history.sum(-1) == 8*bs*6)):
        raise ValueError('incomplete per-step EPLB demand')
    hashes = {}
    for layer in layers:
        plan, _, _ = rebalance_experts_global(torch.from_numpy(history[:,layer:layer+1].copy()), 448, 56)
        rr = plan[0].tolist()
        eco = reorder_single_copy_first(rr, 8)[0].tolist()
        for arm, layout in zip(maps, (list(range(384)), random, rr, eco, eco)):
            maps[arm][layer] = layout
        demand = np.zeros((3,384), dtype=np.int64)
        receipts = []
        for rank in range(8):
            path = group/'routes'/f'layer{layer}-rank{rank}.npz'
            hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
            with np.load(path) as data:
                ids, weights = data['ids'], data['weights']
                if (ids.shape != (3,bs,6) or weights.shape != ids.shape or
                        ids.dtype != np.int64 or weights.dtype != np.float32 or
                        ids.min() < 0 or ids.max() >= 384 or not np.isfinite(weights).all()):
                    raise ValueError(f'invalid route slice: {path}')
                for step in range(3):
                    demand[step] += np.bincount(ids[step].ravel(), minlength=384)
            with np.load(group/'vanilla'/f'rank{rank}.npz') as a:
                receipts.append(a['counts'][layer:120:40])
        if not np.array_equal(np.concatenate(receipts, axis=1), demand):
            raise ValueError(f'L{layer}: slice demand differs from collector slot counts')
    previous = group/'routes'/'manifest.json'
    manifest = dict(source='fresh_vanilla', batch=bs, layers=layers, steps_zero_based=[0,1,2],
                    files_sha256=hashes, demand_sha256=hashlib.sha256(history.tobytes()).hexdigest())
    if previous.exists() and json.loads(previous.read_text()) != manifest:
        raise ValueError('capture changed after preparation')
    for arm, layout in maps.items():
        (group/arm).mkdir(exist_ok=True)
        (group/arm/'installed-layout.json').write_text(json.dumps(layout))
    (group/'placement.json').write_text(json.dumps(maps['rr']))
    previous.write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('group', type=Path)
    p.add_argument('bs', type=int, choices=(512,1024))
    a = p.parse_args()
    prepare(a.group, a.bs)
