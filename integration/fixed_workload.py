"""Device-resident route replay for the controlled EP8 scheduling study."""
import hashlib
import os
from pathlib import Path

import torch


def recording_reference():
    return (os.environ['GC_MOE_SM90_STREAM_M_BLOCKS'] == '0'
            and os.environ.get('GC_MOE_FIXED_REUSE') != '1')


def digest(tensor):
    return hashlib.sha256(tensor.cpu().contiguous().numpy().tobytes()).hexdigest()


def route_ordinals(ids):
    """Stable row number within (call, source rank, logical expert)."""
    rows = ids.flatten(1)
    order = rows.argsort(dim=1, stable=True)
    sorted_ids = rows.gather(1, order)
    starts = torch.ones_like(sorted_ids, dtype=torch.bool)
    starts[:, 1:] = sorted_ids[:, 1:] != sorted_ids[:, :-1]
    positions = torch.arange(rows.shape[1], device=rows.device)
    group_start = torch.where(starts, positions, 0).cummax(dim=1).values
    ordinals = torch.empty_like(rows, dtype=torch.int32)
    return ordinals.scatter_(1, order, (positions - group_start).int()).reshape_as(ids)


class RouteTape:
    # Two flush steps, one profile, two post-profile steps and 100 timed steps.
    steps = 105
    layers = 40
    batch = 256

    def __init__(self, device, topk, slots, steps=steps, layers=layers, batch=batch):
        self.steps, self.layers, self.batch = steps, layers, batch
        calls = steps * layers
        shape = (calls, batch, topk)
        self.ids = torch.zeros(shape, dtype=torch.int64, device=device)
        self.weights = torch.zeros(shape, dtype=torch.float32, device=device)
        self.ordinals = torch.full(shape, -1, dtype=torch.int32, device=device)
        self.used_ordinals = torch.empty_like(self.ordinals)
        self.counts = torch.zeros((calls, slots), dtype=torch.int32, device=device)
        self.epochs = torch.zeros(calls, dtype=torch.int32, device=device)
        self.layer_ids = torch.full((calls,), -1, dtype=torch.int32, device=device)
        self.position = torch.zeros(1, dtype=torch.long, device=device)
        self.index = torch.zeros_like(self.position)
        self.replaying = torch.zeros((), dtype=torch.bool, device=device)
        self.selected_ids = torch.empty_like(self.ids[:1])
        self.selected_weights = torch.empty_like(self.weights[:1])

    def select(self, live_ids, live_weights, layer, ordinal_out):
        # The CUDA Graph uses the same reads, selection and writes for both
        # measured arms. In warmup this ring retains the latest real routes.
        torch.remainder(self.position, len(self.ids), out=self.index)
        self.layer_ids.index_fill_(0, self.index, layer)
        torch.index_select(self.ordinals, 0, self.index, out=ordinal_out.unsqueeze(0))
        for bank, scratch, live in ((self.ids, self.selected_ids, live_ids),
                                    (self.weights, self.selected_weights, live_weights)):
            torch.index_select(bank, 0, self.index, out=scratch)
            torch.where(self.replaying, scratch, live.reshape_as(scratch), out=scratch)
            bank.index_copy_(0, self.index, scratch)
        return self.selected_ids[0], self.selected_weights[0]

    def finish_call(self, buf, output=None):
        if hasattr(self, 'first_step_probe'):
            self.first_step_probe.record(buf, output)
        else:
            self.used_ordinals.index_copy_(0, self.index, buf.used_route_ordinal[:self.batch].unsqueeze(0))
        epoch = buf.eco_tl_step
        row = epoch.long().remainder(256)
        self.epochs.index_copy_(0, self.index, epoch)
        self.counts.index_copy_(0, self.index, buf.profile_slot_counts.index_select(0, row))
        self.position.add_(1)

    def freeze(self, directory, rank, reference):
        # Outside capture and timers. Both arms load the exact same files.
        path = Path(directory) / f'routes-rank{rank}.pt'
        count = self.position.item()
        calls = len(self.ids)
        if count < calls:
            raise ValueError(f'only {count} full-batch MoE calls available, need {calls}')
        order = (torch.arange(calls, device=self.ids.device) + count) % calls
        epochs = self.epochs.index_select(0, order).cpu().long()
        if epochs[0] < 1 or not torch.all(epochs[1:] - epochs[:-1] == 1):
            raise ValueError('reference warmup does not contain consecutive full-batch calls')
        layers = self.layer_ids.index_select(0, order).cpu()
        if (len(layers[:self.layers].unique()) != self.layers
                or not torch.equal(layers, layers[:self.layers].repeat(self.steps))):
            raise ValueError('warm route bank does not cover the same ordered layers every step')
        if reference:
            torch.save(dict(ids=self.ids.index_select(0, order).cpu(),
                            weights=self.weights.index_select(0, order).cpu(), layers=layers), path)
        bank = torch.load(path, map_location='cpu', weights_only=True)
        if not torch.equal(layers, bank['layers']):
            raise ValueError('reference layer order differs from this worker')
        for key, dst in (('ids', self.ids), ('weights', self.weights)):
            src = bank[key]
            if src.dtype != dst.dtype or src.shape != dst.shape:
                raise ValueError(f'invalid reference {key} shape or dtype')
            dst.copy_(src)
        # Build on CPU, before timing. Bound scratch space per worker.
        for pos in range(0, calls, 32):
            self.ordinals[pos:pos + 32].copy_(route_ordinals(bank['ids'][pos:pos + 32]))
        self.used_ordinals.fill_(-1)
        self.position.zero_()
        self.replaying.fill_(True)
        self.epochs.zero_()
        self.counts.fill_(-1)
        self.layer_ids.fill_(-1)
        self.expected_layers = layers
        self.reference_hashes = [digest(bank[key]) for key in ('ids', 'weights')]

    def audit(self, rank):
        if self.position.item() != len(self.ids):
            raise ValueError('incorrect number of full-batch MoE calls during replay')
        epochs = self.epochs.cpu().long()
        if (epochs[0] < 1 or not torch.all(epochs[1:] - epochs[:-1] == 1)
                or not torch.equal(self.layer_ids.cpu(), self.expected_layers)):
            raise ValueError('device epoch or executed layer order differs from replay')
        if not torch.equal((epochs - 1).remainder(self.layers), self.expected_layers.long()):
            raise ValueError('ECO metadata row does not match the packed weights of the executed layer')
        if (self.counts < 0).any().item():
            raise ValueError('missing slot counts')
        observed = [digest(self.ids), digest(self.weights)]
        if observed != self.reference_hashes:
            raise ValueError('routes supplied to dispatch differ from the reference')
        if not torch.equal(self.used_ordinals, self.ordinals):
            raise ValueError('dispatch placed a route at the wrong ordinal within its expert')
        rows = self.counts.cpu().reshape(self.steps, self.layers, -1)
        return dict(rank=rank, calls=len(epochs), first_epoch=int(epochs[0]),
                    route_ordinals=digest(self.used_ordinals), route_ids=observed[0], route_weights=observed[1],
                    layer_order=digest(self.expected_layers), owned_steps=[digest(step) for step in rows],
                    owned_totals=rows.sum(-1).tolist())
