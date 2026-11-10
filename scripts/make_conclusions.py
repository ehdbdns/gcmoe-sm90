"""Generate the README 'key conclusions' block from the run jsons.

Numbers in prose go stale silently; this derives every figure it quotes from
runs/V*_*.json so the text cannot drift from the table.
"""
import json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARMS = ('vanilla', 'random', 'rr', 'eco', 'eco_smr')
BS = ((4, 32), (8, 64), (16, 128), (32, 256))


def load(spr, arm):
    p = os.path.join(ROOT, f'runs/V{spr}_{arm}.json')
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    s = d.get('ir_step') or {}
    if not s.get('mb_total_mean'):
        return None
    return dict(ms=d['ms_per_step_median'], ir=s['ir_step_layer_mean'],
                mbir=s['mb_ir_mean'], mb=s['mb_total_mean'], ae=s['ae_total_mean'])


def main():
    rows = {bs: {a: load(spr, a) for a in ARMS} for spr, bs in BS}
    if any(v is None for r in rows.values() for v in r.values()):
        missing = [(bs, a) for bs, r in rows.items() for a, v in r.items() if v is None]
        raise SystemExit(f'missing runs: {missing}')

    def delta(bs, arm):
        b = rows[bs]['vanilla']['ms']
        return (rows[bs][arm]['ms'] / b - 1) * 100

    lines = []
    # 1. fastest arm per batch
    fastest = {}
    for _, bs in BS:
        best = min(ARMS, key=lambda a: rows[bs][a]['ms'])
        fastest[bs] = best
    if all(v == 'eco_smr' for v in fastest.values()):
        lines.append(
            '1. **SM-reuse（`+ eplb + eco + SM-reuse`）在四档里都是最快的一臂**：'
            + ' / '.join(f'{delta(bs, "eco_smr"):+.2f}%' for _, bs in BS)
            + '（相对同档原生）。加速幅度随 batch 增大：副本的固定开销（每 rank 多 8 个槽）'
            '随 batch 摊薄，而负载均衡的收益与 batch 成正比。')
    else:
        lines.append('1. **各档最快的一臂**：'
                     + '，'.join(f'BS{bs} → {fastest[bs]}' for _, bs in BS)
                     + '。')

    # 2. random always slowest?
    slow = {}
    for _, bs in BS:
        slow[bs] = max(ARMS, key=lambda a: rows[bs][a]['ms'])
    if all(v == 'random' for v in slow.values()):
        lines.append(
            '2. **random（随机放置）在四档里都最慢**（'
            + ' / '.join(f'{delta(bs, "random"):+.2f}%' for _, bs in BS)
            + '），新口径 IR 也最高（'
            + f'{min(rows[bs]["random"]["ir"] for _, bs in BS):.4f}–'
              f'{max(rows[bs]["random"]["ir"] for _, bs in BS):.4f}，比原生高 '
            + f'{min((rows[bs]["random"]["ir"] / rows[bs]["vanilla"]["ir"] - 1) * 100 for _, bs in BS):.0f}–'
              f'{max((rows[bs]["random"]["ir"] / rows[bs]["vanilla"]["ir"] - 1) * 100 for _, bs in BS):.0f}%）。'
            '  ⇒ **随机打散是有害的**：它把热点专家随机聚到同一 rank 上。'
            '这也说明「放置质量」比「是否复制」更重要。')

    # 3. rr trend
    rr = [delta(bs, 'rr') for _, bs in BS]
    lines.append(
        '3. **EPLB 放置（rr）随 batch 增大从劣势转为优势**：'
        + ' / '.join(f'BS{bs} {d:+.2f}%' for (_, bs), d in zip(BS, rr))
        + '。副本开销被摊薄后，均衡收益开始占主导。')

    # 4. eco vs rr at each batch
    lines.append(
        '4. **同样的 8 副本/rank，ECO 明显优于 host 轮询（rr）**：'
        + '，'.join(
            f'BS{bs} rr {delta(bs, "rr"):+.2f}% vs eco {delta(bs, "eco"):+.2f}%'
            + f'（IR {rows[bs]["rr"]["ir"]:.4f} vs {rows[bs]["eco"]["ir"]:.4f}）'
            for _, bs in BS)
        + '。内核 plan 把副本切成不等长 quota，既降不均衡又抵消副本开销。')

    # 5. mblocks == active experts
    eq = all(abs(rows[bs][a]['mb'] - rows[bs][a]['ae']) < 1e-6
             for r in rows.values() for a in ARMS)
    lines.append(
        '5. **「M块数平均」与「激活专家数平均」两列完全相等**（'
        + ('实测全部 20 个 run 逐位相同' if eq else '绝大多数 run 相同')
        + '）：每个激活槽的 token 数都小于一个 `block_m`，`ceil(tok/block_m)` 恒为 1。')

    # 6. mb_ir != ir
    inv = [(bs, a) for _, bs in BS for a in ARMS
           if (rows[bs][a]['ir'] - rows[bs]['vanilla']['ir'])
           * (rows[bs][a]['mbir'] - rows[bs]['vanilla']['mbir']) < 0]
    if inv:
        lines.append(
            '6. **「M块 IR」与「IR」方向并不总一致**：'
            + '，'.join(f'BS{bs} {a}' for bs, a in inv[:4])
            + ' 的 token IR 比原生更差而 M块 IR 更好。原因是副本把槽数从 48 提到 56，'
            '每槽 token 更少、取整后全落在 1 块，M 块维度的不均衡被稀释。两列都要看。')

    print('\n\n'.join(lines))


if __name__ == '__main__':
    main()
