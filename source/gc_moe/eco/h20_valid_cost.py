"""EP8 H20, 5120x2304, M64. Fixed point in eighths of a CTA-microsecond.

Reproduce with scripts/trace_layer/fit_valid_cost.py and the layer 8 after/stream
capture. L2 includes its push exactly once.
"""
BUCKETS = (8, 16, 32, 40, 48, 64)
INTERCEPTS = (4642, 4841, 5588, 6727, 7940, 10381)
ROW_COST = 17
FULL = INTERCEPTS[-1] + 64 * ROW_COST


def block_cost(valid):
    if not 0 <= valid <= 64:
        raise ValueError(f'expected 0..64 valid rows, got {valid}')
    if valid == 0:
        return 0
    i = next(i for i, end in enumerate(BUCKETS) if valid <= end)
    return INTERCEPTS[i] + ROW_COST * valid


def slot_cost(owned):
    full, tail = divmod(owned, 64)
    return full * FULL + block_cost(tail)
