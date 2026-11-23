"""Eight-rank transport oracle for the synthetic SM90 ECO CUDA Graph check."""
import hashlib

import torch
import torch.distributed as dist

from fixed_workload import route_ordinals


def check_equal(actual, expected, context):
    if torch.equal(actual, expected):
        return
    different = actual != expected
    index = tuple(different.nonzero()[0].tolist())
    raise AssertionError(f'{context}: mismatches={int(different.sum())} first={index} '
                         f'actual={actual[index].item()} expected={expected[index].item()}')


def verify_rows(buf, physical_to_logical, sources, rank):
    xs, scales, ids, gates = (torch.stack(values) for values in zip(*sources))
    ranks, tokens, choices = ids.shape
    counts = buf.expert_count.cpu().tolist()
    seen = torch.zeros(ranks * tokens * choices, dtype=torch.int32)
    starts = buf.eco_quota_start.cpu().tolist()
    quotas = buf.eco_quota.cpu().tolist()
    for local, count in enumerate(counts):
        physical = rank * len(counts) + local
        logical = physical_to_logical[physical]
        routes = (ids == logical).nonzero()  # source rank, token, top-k canonical order
        if list(physical_to_logical).count(logical) > 1:
            wanted = routes[starts[physical]:starts[physical] + quotas[physical]]
            meta = buf.route_meta[local, :count].cpu().tolist()
            assert sorted(meta) == wanted.tolist(), (rank, local, 'quota ownership')
        if not 0 <= count <= ranks * tokens:
            raise AssertionError(f'invalid slot count: rank={rank} slot={local} count={count}')
        if not count:
            continue
        src, tok, k = buf.route_meta[local, :count].cpu().long().unbind(1)
        assert ((src >= 0) & (src < ranks) & (tok >= 0) & (tok < tokens)
                & (k >= 0) & (k < choices)).all(), (rank, local, 'invalid return address')
        seen.scatter_add_(0, (src * tokens + tok) * choices + k, torch.ones_like(k, dtype=seen.dtype))
        assert (ids[src, tok, k] == physical_to_logical[rank * len(counts) + local]).all(), (rank, local, 'expert')
        for name, actual, expected in (
            ('activation', buf.l1_acts[local, :count].view(torch.uint8), xs[src, tok]),
            ('scale', buf.l1_acts_sf[local, :, :count].T, scales[src, tok] * 64),
            ('gate', buf.route_weight[local, :count], gates[src, tok, k]),
        ):
            check_equal(actual.cpu(), expected, (rank, local, name))
    dist.all_reduce(seen)  # CPU/Gloo, outside replay
    assert (seen == 1).all(), 'lost or duplicate source routing choices'


def verify_push_replays(buf, graph, out, placement, raw, scores, hot, topk, quantize):
    rank, ranks = dist.get_rank(), dist.get_world_size()
    tokens = len(raw)
    assert ranks == 8 and tokens == 17 and buf.layout.fixed_route_order and buf.layout.single_push
    # For each token choose distinct non-hot alternatives. Use different input
    # bytes, gate slots and scale magnitudes across ranks and generations.
    replica = max(set(placement), key=list(placement).count)
    assert list(placement).count(replica) > 1 and replica != hot
    alternatives = scores.clone()
    alternatives[:, replica] = -torch.inf
    keep = alternatives.topk(topk, dim=1).indices
    assert (keep != hot).all()
    order = ((rank + ranks - 1) % ranks) * tokens + torch.arange(tokens, device=raw.device)
    owner, local = divmod(list(placement).index(hot), len(placement) // ranks)
    # Switching to legacy mode for each reference would force all async calls
    # onto one parity. Run three consecutive push replays per input instead.
    for generation, target in enumerate((135, 0, 64, 65)):
        ids = keep.clone()
        ids[order < target, 0] = hot
        ids[:, 1] = replica
        ids = ids.roll((rank + generation) % topk, dims=1)
        weights = (scores.gather(1, ids).masked_fill(ids == hot, 1.0)
                   + torch.arange(topk, device=raw.device) * 0.125).softmax(-1)
        q, sf = quantize(raw.roll(generation, dims=0) * (1 + rank + generation),
                         use_ue8m0=False, gran_k=128)
        buf.x[:tokens].copy_(q)
        buf.x_sf[:tokens].copy_(sf)
        buf.topk_idx[:tokens].copy_(ids)
        buf.topk_weights[:tokens].copy_(weights)
        buf.route_ordinal[:tokens].copy_(route_ordinals(ids[None])[0])
        sources = [None] * ranks
        dist.all_gather_object(sources, (q.view(torch.uint8).cpu(), sf.cpu(), ids.cpu(), weights.cpu()))
        buf.replay_stream_mode.zero_()
        buf.combine[:tokens].fill_(float('nan'))
        out.fill_(float('nan'))
        torch.cuda.synchronize()
        dist.barrier()
        graph.replay()
        reference = out.clone()
        reference_routes = buf.combine[:tokens].clone()
        torch.cuda.synchronize()
        assert torch.isfinite(reference).all() and torch.isfinite(reference_routes).all()
        buf.replay_stream_mode.fill_(1)
        for replay in range(3):
            # Poison previous rows/return slots so a missing write cannot match
            # an earlier replay. All ranks retire poisoning before peers push.
            buf.route_meta.zero_()
            buf.l1_acts.view(torch.uint8).zero_()
            buf.l1_acts_sf.zero_()
            buf.route_weight.zero_()
            buf.combine[:tokens].fill_(float('nan'))
            out.fill_(float('nan'))
            torch.cuda.synchronize()
            dist.barrier()
            graph.replay()
            torch.cuda.synchronize()
            verify_rows(buf, placement, sources, rank)
            if rank == owner:
                assert int(buf.expert_count[local]) == target
            check_equal(buf.combine[:tokens], reference_routes, (rank, generation, replay, 'L2 scatter'))
            check_equal(out, reference, (rank, generation, replay, 'combine'))
        if rank == 0:
            digest = hashlib.sha256(reference.cpu().view(torch.int16).numpy().tobytes()).hexdigest()
            print(f'PUSH_PAYLOAD_OK hot_rows={target} replays=3 rank0_sha256={digest}', flush=True)
