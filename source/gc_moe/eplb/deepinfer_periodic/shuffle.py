"""DeepInfer's snapshot/IPC and NCCL P2P expert movement, without a weight loader."""

from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class Transfer:
    expert: int
    source_rank: int
    source_slot: int
    target_rank: int
    target_slot: int


def plan_transfers(origin: list[int], target: list[int], num_local: int):
    if num_local <= 0 or len(origin) != len(target) or len(origin) % num_local:
        raise ValueError('Invalid shuffle layout or local capacity')
    if set(origin) != set(target):
        raise ValueError('Shuffle cannot create or remove a logical expert')
    transfers = []
    for expert in sorted(set(target)):
        senders = {}
        receivers = {}
        for physical, logical in enumerate(origin):
            if logical == expert:
                senders.setdefault(physical // num_local, physical % num_local)
        for physical, logical in enumerate(target):
            rank, slot = divmod(physical, num_local)
            if logical == expert and rank not in senders:
                receivers.setdefault(rank, slot)
        sender_items = sorted(senders.items())
        receiver_items = sorted(receivers.items())
        per_sender = len(receiver_items) // len(sender_items)
        for receiver_index, (target_rank, target_slot) in enumerate(receiver_items):
            balanced_prefix = per_sender * len(sender_items)
            sender_index = (receiver_index // per_sender if receiver_index < balanced_prefix
                            else receiver_index - balanced_prefix)
            source_rank, source_slot = sender_items[sender_index]
            transfers.append(Transfer(expert, source_rank, source_slot, target_rank, target_slot))
    return transfers


class WeightShuffler:
    def __init__(self, group, components, intranode: bool, p2p_groups: int = 1):
        if not components or not components[0] or p2p_groups <= 0:
            raise ValueError('Missing expert components or invalid P2P bucket count')
        self.group = group
        self.rank = dist.get_rank(group)
        self.size = dist.get_world_size(group)
        self.components = components
        self.intranode = intranode
        self.p2p_groups = p2p_groups
        self.num_local = components[0][0].shape[0]
        self.initialized = False
        self.handles = []
        self.peers = []
        for layer in components:
            if len(layer) != len(components[0]):
                raise ValueError('Different number of components between layers')
            for tensor, reference in zip(layer, components[0], strict=True):
                if tensor.shape != reference.shape or tensor.dtype != reference.dtype:
                    raise ValueError('Shuffle requires homogeneous expert shapes and dtypes')
                if tensor.device != reference.device or tensor.device.type != 'cuda':
                    raise ValueError('Shuffle requires local CUDA tensors')

    def _barrier(self):
        torch.cuda.synchronize()
        dist.barrier(group=self.group, async_op=True).wait()
        torch.cuda.synchronize()

    def _initialize(self):
        self._barrier()
        if self.intranode:
            import torch.distributed._symmetric_memory as symm_mem

            self.buffers = [symm_mem.empty(tensor.shape, dtype=tensor.dtype, device=tensor.device)
                            for tensor in self.components[0]]
            self.handles = [symm_mem.rendezvous(buffer, self.group) for buffer in self.buffers]
            self.peers = [[handle.get_buffer(rank, buffer.shape, buffer.dtype)
                           for handle, buffer in zip(self.handles, self.buffers, strict=True)]
                          for rank in range(self.size)]
        else:
            self.buffers = [torch.empty(tensor.shape, dtype=tensor.dtype, device=tensor.device)
                            for tensor in self.components[0]]
        self.initialized = True

    def move(self, origin, target):
        if len(origin) != len(self.components) or len(target) != len(origin):
            raise ValueError('Layer count differs from the shuffle plan')
        for previous, following in zip(origin, target, strict=True):
            if len(previous) != self.size * self.num_local:
                raise ValueError('Physical expert count differs from the EP world')
            plan_transfers(previous, following, self.num_local)
        if not self.initialized:
            self._initialize()
        for previous, following, layer in zip(origin, target, self.components, strict=True):
            self._ipc_layer(previous, following, layer) if self.intranode else self._p2p_layer(previous, following, layer)
        self._barrier()

    def _ipc_layer(self, origin, target, weights):
        self._barrier()
        for buffer, weight in zip(self.buffers, weights, strict=True):
            buffer.copy_(weight)
        self._barrier()
        local_origin = origin[self.rank * self.num_local:(self.rank + 1) * self.num_local]
        source_by_expert = {}
        for slot, expert in enumerate(local_origin):
            source_by_expert.setdefault(expert, (self.rank, slot))
        for transfer in plan_transfers(origin, target, self.num_local):
            if transfer.target_rank == self.rank:
                source_by_expert[transfer.expert] = (transfer.source_rank, transfer.source_slot)
        local_target = target[self.rank * self.num_local:(self.rank + 1) * self.num_local]
        for slot, expert in enumerate(local_target):
            if local_origin[slot] == expert:
                continue
            source_rank, source_slot = source_by_expert[expert]
            if source_rank == self.rank:
                # [FIX] Same-rank source must be read from the *snapshot*
                # buffer, not from `weights`: the loop writes into `weights`
                # while later iterations still need untouched rows, so reading
                # a source slot that was already overwritten silently corrupts
                # the move (observed: 13/53 correct on one rank).
                for weight, buffer in zip(weights, self.buffers, strict=True):
                    weight[slot].copy_(buffer[source_slot])
            else:
                for weight, peer in zip(weights, self.peers[source_rank], strict=True):
                    weight[slot].copy_(peer[source_slot])

    def _p2p_layer(self, origin, target, weights):
        local_origin = origin[self.rank * self.num_local:(self.rank + 1) * self.num_local]
        local_target = target[self.rank * self.num_local:(self.rank + 1) * self.num_local]
        local_sources = {}
        for slot, expert in enumerate(local_origin):
            local_sources.setdefault(expert, slot)
        for slot, expert in enumerate(local_target):
            if local_origin[slot] != expert and expert in local_sources:
                for buffer, weight in zip(self.buffers, weights, strict=True):
                    buffer[slot].copy_(weight[local_sources[expert]])
        transfers = plan_transfers(origin, target, self.num_local)
        buckets = [[] for _ in range(self.p2p_groups)]
        for transfer in transfers:
            bucket = buckets[(transfer.source_rank - transfer.target_rank) % self.p2p_groups]
            if transfer.source_rank == self.rank:
                peer = dist.get_global_rank(self.group, transfer.target_rank)
                bucket.extend(dist.P2POp(dist.isend, weight[transfer.source_slot].contiguous().view(torch.uint8), peer, self.group)
                              for weight in weights)
            if transfer.target_rank == self.rank:
                peer = dist.get_global_rank(self.group, transfer.source_rank)
                bucket.extend(dist.P2POp(dist.irecv, buffer[transfer.target_slot].view(torch.uint8), peer, self.group)
                              for buffer in self.buffers)
        for operations in buckets:
            if operations:
                for request in dist.batch_isend_irecv(operations):
                    request.wait()
        received = {transfer.expert: transfer.target_slot for transfer in transfers
                    if transfer.target_rank == self.rank}
        for slot, expert in enumerate(local_target):
            if local_origin[slot] == expert:
                continue
            source_slot = slot if expert in local_sources else received[expert]
            for weight, buffer in zip(weights, self.buffers, strict=True):
                weight[slot].copy_(buffer[source_slot])
