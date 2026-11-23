"""Exercise the inbox block-credit protocol with out-of-order DMA completion."""
from collections import Counter
import random


def check_batches(counts, seed):
    rng = random.Random(seed)
    remaining = counts.copy()
    reserved = 0
    inflight = []
    # Reserve in source order within a queue, interleave queues arbitrarily.
    while any(remaining):
        src = rng.choice([i for i, n in enumerate(remaining) if n])
        n = rng.randint(1, min(8, remaining[src]))
        inflight.append((src, reserved, n))
        reserved += n
        remaining[src] -= n

    # Each source warp retires its reservations in order. Other warps can
    # retire later reservations first, including ones spanning a block edge.
    queues = [[entry for entry in inflight if entry[0] == src] for src in range(len(counts))]
    open_sources = list(range(len(counts)))
    credits = Counter()
    written = set()
    published = {}
    while open_sources:
        src = rng.choice(open_sources)
        if queues[src]:
            _, row, n = queues[src].pop(0)
            end = row + n
            written.update(range(row, end))
            for block in range(row // 64, (end - 1) // 64 + 1):
                lo = block * 64
                credits[block] += min(end, lo + 64) - max(row, lo)
                if credits[block] == 64:
                    assert block not in published and set(range(lo, lo + 64)) <= written
                    published[block] = 64
        else:
            open_sources.remove(src)
    if reserved % 64:
        block, tail = divmod(reserved, 64)
        assert block not in published and credits[block] == tail
        published[block] = tail
    assert published == {row // 64: min(64, reserved - row) for row in range(0, reserved, 64)}


def test_inbox_batches():
    for seed in range(32):
        for counts in ([0] * 8, [8] * 8, [8] * 7 + [9], [0, 0, 0, 0, 1, 64, 4, 192]):
            check_batches(counts, seed)


if __name__ == '__main__':
    test_inbox_batches()
    print('SM90_INBOX_BATCH_CPU PASS')
