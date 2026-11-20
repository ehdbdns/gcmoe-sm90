"""Fork the fixed EP8 workload at its post-EPLB, pre-stream boundary."""
import hashlib
import json
import weakref
from functools import wraps
from pathlib import Path

import torch


CHUNK = 16 << 20


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def tensor_hash(tensor):
    """Hash logical elements in bounded row slices, including noncontiguous packed scales."""
    h = hashlib.sha256()
    if tensor.numel():
        rows = tensor.reshape(1) if tensor.ndim == 0 else tensor
        per_row = rows.numel() * rows.element_size() // len(rows)
        for slab in rows.split(max(1, CHUNK // per_row)):
            host = slab.detach().cpu().contiguous()
            h.update(host.as_strided((host.numel(),), (1,)).view(torch.uint8).numpy())
    return h.hexdigest()


def install_cache_names():
    # vLLM binds named, often aliased views, then stores just their values in
    # runner.kv_caches. Retain the names without retaining profiling allocations.
    from vllm.v1.worker.gpu import model_runner

    original = model_runner.init_kv_cache
    if hasattr(original, 'gc_cache_names'):
        return

    @wraps(original)
    def named(*args, **kwargs):
        caches = original(*args, **kwargs)
        named.gc_cache_names = {key: weakref.ref(t) for key, t in caches.items()}
        return caches

    named.gc_cache_names = {}
    model_runner.init_kv_cache = named


def named_caches(runner):
    from vllm.v1.worker.gpu import model_runner

    live = {id(t) for t in runner.kv_caches}
    views = {key: ref() for key, ref in model_runner.init_kv_cache.gc_cache_names.items()}
    if not live or any(t is None or t.device != runner.device for t in views.values()) or {
            id(t) for t in views.values()} != live:
        raise ValueError('cannot identify every live KV allocation by layer name')
    # Upstream V2 rebuilds Engram lookback from all_token_ids on each step.
    # Include the separate slot-keyed store if an installed model enables it.
    for name, module in runner.model.named_modules():
        if getattr(module, 'use_slot_cache', False) and hasattr(module, '_kv_cache_ref'):
            cache = module._cache
            if not isinstance(cache, torch.Tensor) or cache.device != runner.device:
                raise ValueError(f'missing live Engram slot cache for {name}')
            views[f'engram:{name}'] = cache
    return views


def cache_layout(views):
    schemas, blobs, aliases = {}, [], {}
    for name, tensor in sorted(views.items()):
        storage = tensor.untyped_storage()
        key = (tensor.device, storage.data_ptr())
        if key not in aliases:
            aliases[key] = len(blobs)
            blobs.append(torch.empty(0, device=tensor.device, dtype=torch.uint8).set_(
                storage, 0, (storage.nbytes(),), (1,)))
        schemas[name] = [aliases[key], str(tensor.dtype), list(tensor.shape),
                         list(tensor.stride()), tensor.storage_offset()]
    if not blobs or any(not blob.numel() for blob in blobs):
        raise ValueError('empty KV backing store')
    return dict(views=schemas, sizes=[b.numel() for b in blobs]), blobs


def checkpoint(directory, rank, publish, views, controls):
    """Both arms reload into existing graph addresses and independently read back."""
    prefix = Path(directory) / f'rank{rank}'
    manifest = prefix.with_suffix('.json')
    payload = prefix.with_suffix('.bin')
    layout, blobs = cache_layout(views)
    boundary = dict(version=1, rank=rank, layout=layout, controls=controls)
    if publish:
        # The directory is private to this pair. A stale baseline must never
        # overwrite the source under a candidate or silently resume a partial run.
        with payload.open('xb') as out:
            h = hashlib.sha256()
            for blob in blobs:
                for slab in blob.split(CHUNK):
                    host = slab.cpu().numpy()
                    h.update(host)
                    out.write(host)
        with manifest.open('x') as out:
            json.dump(dict(boundary, sha256=h.hexdigest()), out, sort_keys=True)
    expected = json.loads(manifest.read_text())
    changed = [key for key in controls if controls[key] != expected['controls'].get(key)]
    if changed or any(expected[k] != v for k, v in boundary.items()):
        raise ValueError(f'rank {rank}: incompatible checkpoint layout or controls {changed}')
    if payload.stat().st_size != sum(layout['sizes']):
        raise ValueError(f'rank {rank}: incorrect checkpoint byte count')
    readback = hashlib.sha256()
    with payload.open('rb') as src:
        for blob in blobs:
            for slab in blob.split(CHUNK):
                data = bytearray(slab.numel())
                if src.readinto(data) != len(data):
                    raise IOError('short read restoring KV checkpoint')
                slab.copy_(torch.frombuffer(data, dtype=torch.uint8))
                readback.update(slab.cpu().numpy())
    if readback.hexdigest() != expected['sha256']:
        raise ValueError(f'rank {rank}: live KV readback differs from published checkpoint')
    return dict(source_sha256=expected['sha256'], readback_sha256=readback.hexdigest(),
                controls_sha256=json_hash(controls), layout_sha256=json_hash(layout),
                kv_bytes=sum(layout['sizes']), kv_views=len(views))


def runner_boundary(runner, reference):
    states, tables = runner.req_states, runner.block_tables
    mapping = states.req_id_to_index
    if set(mapping) != set(reference.keys) or states.num_speculative_steps != 0:
        raise ValueError('checkpoint requires all reference requests and greedy decoding')
    # Only active prefixes of token histories and physical block-table rows
    # are meaningful; unused tails can retain data from earlier requests.
    slots = [mapping[name] for name in sorted(mapping)]
    requests = dict(mapping=dict(mapping), free_indices=list(states.free_indices),
                    reference_positions=tensor_hash(reference.positions),
                    batch_order=reference.batch_hash.hexdigest())
    for field in ('prompt_len', 'prefill_len', 'total_len', 'num_computed_tokens'):
        requests[field] = getattr(states, field).gpu.cpu()[slots].tolist()
    for field in ('max_seq_len', 'num_computed_tokens_np', 'num_computed_prefill_tokens'):
        requests[field] = getattr(states, field)[slots].tolist()
    requests['last_tokens'] = states.last_sampled_tokens.cpu()[slots].tolist()
    all_tokens = states.all_token_ids.gpu.cpu()
    requests['token_histories'] = [tensor_hash(all_tokens[i, :length])
                                   for i, length in zip(slots, requests['total_len'], strict=True)]
    groups = []
    block_counts = tables.num_blocks.gpu.cpu()
    for group, table in enumerate(tables.block_tables):
        blocks = table.gpu.cpu()
        counts = block_counts[group, slots].tolist()
        if counts != tables.num_blocks.np[group, slots].tolist() or any(
                not 0 <= count <= blocks.shape[1] for count in counts):
            raise ValueError(f'inconsistent block table lengths in group {group}')
        groups.append([tensor_hash(blocks[i, :count]) for i, count in zip(slots, counts, strict=True)])
    groups_config = [list(group.layer_names) for group in runner.kv_cache_config.kv_cache_groups]
    return dict(requests=json_hash(requests), physical_blocks=json_hash(groups),
                cache_groups=json_hash([groups_config, tables.block_sizes, tables.kernel_block_sizes]))


def prepare_worker(worker, buf, modules, reference, directory, publish):
    runner = worker.model_runner
    owner = next(m for m in modules if getattr(m, '_eplb_shuffler', None) is not None)
    layers = owner._eplb_shuffler.components
    if len(layers) != buf.fixed_routes.layers or any(len(row) != 6 for row in layers):
        raise ValueError('incomplete packed expert weights at the EPLB boundary')
    controls = runner_boundary(runner, reference)
    controls.update(next_epoch=int(buf.header[0].item()), eco_step=int(buf.eco_tl_step.item()),
                    placement=tensor_hash(owner._eplb['plan']), meta=tensor_hash(buf.eco_meta),
                    eco_state=tensor_hash(buf.eco_state),
                    packed_weights=json_hash([[tensor_hash(t) for t in row] for row in layers]))
    return checkpoint(directory, int(buf.rank_idx), publish, named_caches(runner), controls)
