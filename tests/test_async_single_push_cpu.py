"""Exercise source announcements, mixed blocks and per-block publication under reordering."""
import random


def replay(counts, m, seed):
    rng = random.Random(seed)
    n = sum(counts)
    slots = [None] * n
    delivered = [0] * ((n + m - 1) // m)
    mailbox = [0] * len(counts)
    starts = {}
    remaining = [list(range(count)) for count in counts]
    published = {}
    reserved = 0
    done_bit = 1 << 31
    while True:
        known = sum(bool(w) for w in mailbox)
        expected = sum((w & (done_bit - 1)) - 1 for w in mailbox if w)
        finished = sum(bool(w & done_bit) for w in mailbox)
        for start in range(0, reserved, m):
            b = start // m
            valid = min(m, expected - start) if known == len(counts) else m
            if delivered[b] == valid:
                assert all(row is not None for row in slots[start:start + valid])
                assert b not in published
                published[b] = known, finished, valid
                delivered[b] = -1
        if finished == len(counts) and sum(v for _, _, v in published.values()) == expected:
            break
        active = [src for src, w in enumerate(mailbox) if not w & done_bit]
        src = rng.choice(active)
        if not mailbox[src]:
            mailbox[src] = counts[src] + 1
            if not counts[src]:
                mailbox[src] |= done_bit
        elif src not in starts:
            starts[src] = reserved
            reserved += counts[src]
        elif remaining[src]:
            ordinal = remaining[src].pop(rng.randrange(len(remaining[src])))
            row = starts[src] + ordinal
            assert slots[row] is None
            slots[row] = (src, ordinal)
            delivered[row // m] += 1
        else:
            mailbox[src] |= done_bit
    assert sorted(slots) == [(src, i) for src, count in enumerate(counts) for i in range(count)]
    assert [published[b][2] for b in sorted(published)] == [min(m, n-i) for i in range(0, n, m)]
    return published


if __name__ == '__main__':
    for counts in ([0] * 8, [8, 0, 0, 0, 0, 0, 0, 0], [4, 7, 0, 0, 0, 2, 1, 0]):
        for seed in range(100):
            replay(counts, 4, seed)
    assert any(k < 8 for seed in range(100) for k, _, _ in
               replay([4, 2, 1, 0, 0, 0, 0, 0], 2, seed).values())
    print('ASYNC_SINGLE_PUSH_CPU PASS')
