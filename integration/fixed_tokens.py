"""Teacher-forced decode for the vLLM 0.31 GPU model runner used on the H20 host."""
from array import array
import os
import hashlib
import json
from pathlib import Path

import torch


class ReferenceTokens:
    def __init__(self, tokens, device, capacity=None, follow_native=None):
        self.keys = {f'req{i}': i for i in range(len(tokens))}
        lengths = [len(tokens[key]) for key in self.keys]
        width = max(lengths) if capacity is None else capacity
        if width < max(lengths):
            raise ValueError('token reference exceeds the decode budget')
        self.follow_native = (None if follow_native is None else torch.tensor(follow_native, device=device))
        table = torch.zeros((len(tokens), width), dtype=torch.long)
        for key, i in self.keys.items():
            table[i, :lengths[i]] = torch.tensor(tokens[key])
        self.table = table.to(device)
        self.lengths = torch.tensor(lengths, device=device)
        self.positions = torch.zeros_like(self.lengths)
        self.native = torch.full_like(self.table, -1)
        self.batch_hash = hashlib.sha256()
        self.device = device
        self.audit_source = False
        self.source_members = None

    def feed(self, req_ids, output, num_sampled):
        sampled = output.sampled_token_ids
        if sampled.shape != (len(req_ids), 1) or num_sampled.shape != (len(req_ids),):
            raise ValueError('reference requires one greedy sample column per request')
        # Chunked-prefill rows can have num_sampled=0. They retain their
        # position until the first token is actually accepted.
        indices = [self.keys[key] for key in req_ids]
        self.batch_hash.update(array('i', [len(indices), *indices]).tobytes())
        if self.audit_source:
            b = int(os.environ["GC_MOE_SOURCE_BATCH"])
            if len(indices) != 8 * b or len(set(indices)) != 8 * b:
                raise ValueError("source ownership requires eight disjoint full request slices")
            members = tuple(indices[self.source_rank * b:(self.source_rank + 1) * b])
            if self.source_members is None:
                self.source_members = members
            elif members != self.source_members:
                raise ValueError("decode request ownership/order changed during the experiment")
            self.source_steps += 1
        rows = torch.tensor(indices, device=self.device)
        pos = self.positions.index_select(0, rows)
        self.native[rows, pos] = sampled[:, 0].long()
        expected = self.table[rows, pos]
        if self.follow_native is not None:
            expected = torch.where(self.follow_native, sampled[:, 0], expected)
        sampled[:, 0].copy_(torch.where(num_sampled.bool(), expected, sampled[:, 0]))
        self.positions.index_add_(0, rows, num_sampled.long())

    def verify_native_prefix(self, phase, start):
        # Called outside timing. Compare only accepted samples, in request
        # coordinates, with the publisher's original greedy token stream.
        pos = torch.arange(self.native.shape[1], device=self.device)
        accepted = (pos >= start[:, None]) & (pos < self.positions[:, None])
        different = accepted & (self.native != self.table)
        if different.any().item():
            rows, cols = different.nonzero(as_tuple=True)
            offsets = cols - start[rows]
            i = offsets.argmin()
            row, col = rows[i].item(), cols[i].item()
            if os.getenv('SM90_EIGHT_ARM'):
                if not hasattr(self, 'differences'):
                    self.differences = {}
                self.differences[phase] = dict(count=len(rows), request=row, position=col,
                                               native=int(self.native[row, col]), reference=int(self.table[row, col]))
                return
            raise ValueError(f'{phase}: {len(rows)} native top-1 mismatches; '
                             f'first relative offset {offsets[i].item()}, req{row}, position {col}, '
                             f'native {self.native[row, col].item()}, reference {self.table[row, col].item()}')

    def mark_replay_start(self):
        self.replay_start = self.positions.clone()
        self.capture_first_sample = os.environ.get('GC_MOE_FIRST_STEP_AUDIT') == '1'

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


def install(runner, directory, capacity=None):
    # V2 computes its native greedy sample and gathers TP shards inside
    # sample(). Replace the result before KV bookkeeping and frontend export.
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    if not isinstance(runner, GPUModelRunner):
        raise ValueError('the controlled workload requires the vLLM V2 GPUModelRunner')
    follow_native = None
    if capacity is not None:
        from fixed_workload import publishes_checkpoint
        follow_native = publishes_checkpoint() and os.environ.get("GC_MOE_PREFIX_DIAG") != "1"
    if follow_native:
        tokens = {f'req{i}': [] for i in range(int(os.environ.get('GC_MOE_MATRIX_BS', '256')))}
    else:
        with (Path(directory) / 'tokens.json').open() as src:
            tokens = json.load(src)['tokens']
    reference = ReferenceTokens(tokens, runner.device, capacity, follow_native)
    trace_hidden = os.environ.get("GC_MOE_PREFIX_HIDDEN_AA") == "1"
    if trace_hidden:
        if os.environ.get("GC_MOE_PREFIX_DIAG") != "1":
            raise ValueError("prefix hidden audit requires prefill-only diagnostic")
        reference.hidden_samples = []
    original = runner.sample

    def sample(hidden_states, input_batch, grammar_output):
        if getattr(reference, 'capture_first_sample', False):
            from replay_probe import sample_first_step
            returned, reference.first_sample = sample_first_step(runner, original, hidden_states, input_batch, grammar_output)
            reference.capture_first_sample = False
        else:
            returned = original(hidden_states, input_batch, grammar_output)
        output, num_sampled, num_rejected = returned
        offsets = input_batch.query_start_loc_np[:input_batch.num_reqs + 1]
        if reference.audit_source and (offsets[0] != 0 or not ((offsets[1:] - offsets[:-1]) == 1).all()):
            raise ValueError("source batch contains padding or multiple input tokens for a request")
        reference.batch_hash.update(offsets.tobytes())
        if trace_hidden:
            # logits_indices picks the last hidden row for each mixed-prefill
            # request. Drop chunk rows which have no accepted sample.
            rows = hidden_states.index_select(0, input_batch.logits_indices.long()).contiguous()
            host = rows.view(torch.uint8).cpu().numpy()
            accepted = num_sampled.tolist()
            batch = []
            for req, row, take in zip(input_batch.req_ids, host, accepted, strict=True):
                if take:
                    batch.append([reference.keys[req], hashlib.sha256(row).hexdigest()])
            reference.hidden_samples.append(batch)
        reference.feed(input_batch.req_ids, output, num_sampled)
        return output, num_sampled, num_rejected

    runner.sample = sample
    return reference
