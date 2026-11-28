"""H20 legacy upstream M selection for the six audited per-source workloads."""
SIZES = (8, 16, 32, 64, 128, 256)


def routed_m(local):
    if local not in SIZES:
        raise ValueError(f"unsupported local H20 batch {local}")
    if local < 32:
        return 8
    rows_per_expert = local * 8 * 6 / 384
    return 16 if rows_per_expert <= 12 else (32 if rows_per_expert <= 24 else 64)


def run_local_moe(group, rank, batch, output, hidden, weights, ids, run):
    if group.world_size != 8 or group.rank_in_group != rank:
        raise ValueError("source-sharded EP8 requires identical TP and EP rank order")
    if any(x.shape[0] != 8 * batch for x in (output, hidden, weights, ids)):
        raise ValueError("hidden, routes and output must share the full ordered TP batch")
    source = slice(rank * batch, (rank + 1) * batch)
    result = output.new_empty((batch, *output.shape[1:]))
    run(result, hidden[source], weights[source], ids[source])
    output.copy_(group.all_gather(result, dim=0))
