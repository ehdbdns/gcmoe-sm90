"""Teacher-forced decode for the vLLM 0.31 GPU model runner used on the H20 host."""
from array import array
import hashlib
import json
from pathlib import Path

import torch


class ReferenceTokens:
    def __init__(self, tokens, device):
        self.keys = {f'req{i}': i for i in range(len(tokens))}
        lengths = [len(tokens[key]) for key in self.keys]
        table = torch.zeros((len(tokens), max(lengths)), dtype=torch.long)
        for key, i in self.keys.items():
            table[i, :lengths[i]] = torch.tensor(tokens[key])
        self.table = table.to(device)
        self.lengths = torch.tensor(lengths, device=device)
        self.positions = torch.zeros_like(self.lengths)
        self.native = torch.full_like(self.table, -1)
        self.batch_hash = hashlib.sha256()
        self.device = device

    def feed(self, req_ids, output, num_sampled):
        sampled = output.sampled_token_ids
        if sampled.shape != (len(req_ids), 1) or num_sampled.shape != (len(req_ids),):
            raise ValueError('reference requires one greedy sample column per request')
        # Chunked-prefill rows can have num_sampled=0. They retain their
        # position until the first token is actually accepted.
        indices = [self.keys[key] for key in req_ids]
        self.batch_hash.update(array('i', [len(indices), *indices]).tobytes())
        rows = torch.tensor(indices, device=self.device)
        pos = self.positions.index_select(0, rows)
        self.native[rows, pos] = sampled[:, 0].long()
        expected = self.table[rows, pos]
        sampled[:, 0].copy_(torch.where(num_sampled.bool(), expected, sampled[:, 0]))
        self.positions.index_add_(0, rows, num_sampled.long())

    def mark_replay_start(self):
        self.replay_start = self.positions.clone()

    def native_history(self, steps):
        if not torch.equal(self.positions, self.lengths) or not torch.all(self.positions - self.replay_start == steps):
            raise ValueError('native samples do not cover the replay interval')
        # Padding and unaccepted prefill positions are excluded from warmup.
        pos = torch.arange(self.native.shape[1], device=self.device)
        warm = self.native.masked_fill(pos >= self.replay_start[:, None], -1)
        idx = self.replay_start[:, None] + pos[:steps]
        measured = self.native.gather(1, idx)
        if (measured < 0).any():
            raise ValueError('missing native sample in the replay interval')
        return dict(warm_top1_sha256=hashlib.sha256(warm.cpu().numpy().tobytes()).hexdigest(),
                    replay_top1=measured.T.cpu().tolist())

    def native_sample(self, remaining=103):
        if not torch.equal(self.positions, self.lengths):
            raise ValueError('worker consumed a different number of reference tokens')
        rows = torch.arange(len(self.lengths), device=self.device)
        values = self.native[rows, self.lengths - remaining].cpu().tolist()
        if min(values) < 0:
            raise ValueError('missing native greedy token at the profiled step')
        return values


def install(runner, directory):
    # V2 computes its native greedy sample and gathers TP shards inside
    # sample(). Replace the result before KV bookkeeping and frontend export.
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    if not isinstance(runner, GPUModelRunner):
        raise ValueError('the controlled workload requires the vLLM V2 GPUModelRunner')
    with (Path(directory) / 'tokens.json').open() as src:
        reference = ReferenceTokens(json.load(src)['tokens'], runner.device)
    original = runner.sample

    def sample(hidden_states, input_batch, grammar_output):
        output, num_sampled, num_rejected = original(hidden_states, input_batch, grammar_output)
        reference.batch_hash.update(input_batch.query_start_loc_np[:input_batch.num_reqs+1].tobytes())
        reference.feed(input_batch.req_ids, output, num_sampled)
        return output, num_sampled, num_rejected

    runner.sample = sample
    return reference
