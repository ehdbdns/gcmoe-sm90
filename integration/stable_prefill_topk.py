"""Deterministic packed-row index selection for strict V4.1 prefill comparisons."""
import torch


def select(logits, starts, ends, indices, num_rows, stride0, stride1, topk):
    if (logits.ndim != 2 or indices.shape != (num_rows, topk)
            or logits.stride() != (stride0, stride1)
            or starts.numel() != num_rows or ends.numel() != num_rows):
        raise ValueError("unexpected packed prefill top-k layout")
    width = logits.shape[1]
    if not width:
        indices.fill_(-1)
        return
    # Pack each request's valid logits at the beginning of its row. Stable
    # sorting therefore ranks even valid -inf ahead of out-of-request padding.
    cols = torch.arange(width, device=logits.device)
    output_cols = torch.arange(topk, device=logits.device)
    tile = max(1, min(64, 262144 // width))
    for lo in range(0, num_rows, tile):
        hi = min(num_rows, lo + tile)
        start = starts[lo:hi].reshape(-1, 1).long()
        length = ends[lo:hi].reshape(-1, 1) - start
        packed = logits[lo:hi].gather(1, (start + cols).clamp(max=width - 1))
        packed.masked_fill_(cols >= length, -torch.inf)
        ranked = packed.argsort(dim=1, descending=True, stable=True)[:, :topk]
        count = ranked.shape[1]
        # The native short-row path returns all local positions in natural order.
        chosen = torch.where(length <= topk, output_cols[:count], ranked)
        indices[lo:hi, :count].copy_(torch.where(output_cols[:count] < length, chosen, -1))
        if count < topk:
            indices[lo:hi, count:].fill_(-1)


def install():
    from vllm.model_executor.layers import sparse_attn_indexer

    original = sparse_attn_indexer.ops

    class IndexerOps:
        top_k_per_row_prefill = staticmethod(select)

        def __getattr__(self, name):
            return getattr(original, name)

    sparse_attn_indexer.ops = IndexerOps()


def select_decode(logits, ends, indices, topk):
    """Break score ties by request-local position, including during graph replay."""
    rows, width = logits.shape
    if ends.numel() != rows or indices.shape != (rows, topk):
        raise ValueError("unexpected decode top-k layout")
    cols = torch.arange(width, device=logits.device)
    output_cols = torch.arange(topk, device=logits.device)
    tile = max(1, min(64, 262144 // max(1, width)))
    for lo in range(0, rows, tile):
        hi = min(rows, lo + tile)
        length = ends[lo:hi].view(-1, 1)
        scores = logits[lo:hi].masked_fill(cols >= length, -torch.inf)
        ranked = scores.argsort(dim=1, descending=True, stable=True)[:, :topk]
        count = ranked.size(1)
        selected = torch.where(length <= topk, output_cols[:count], ranked)
        indices[lo:hi, :count].copy_(torch.where(output_cols[:count] < length, selected, -1))
        if count < topk:
            indices[lo:hi, count:].fill_(-1)


def install_decode():
    from vllm.model_executor.layers import sparse_attn_indexer
    from vllm.model_executor.layers.indexer_topk import SparseIndexerTopk

    def dispatch(logits, seq_lens, next_n, topk_indices, topk_tokens, max_seq_len):
        ends = SparseIndexerTopk._row_ends(seq_lens, next_n, logits.shape[0])
        select_decode(logits, ends, topk_indices, topk_tokens)

    sparse_attn_indexer.get_indexer_topk = lambda backend: dispatch
