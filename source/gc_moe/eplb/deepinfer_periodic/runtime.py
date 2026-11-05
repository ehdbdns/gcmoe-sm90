"""Periodic EPLB driven by completed real forwards and LL receive counts."""

from datetime import timedelta
import hashlib
import json
import logging
import os
from pathlib import Path
import socket
import time

import torch
import torch.distributed as dist

from .config import RearrangementClock, Settings, chunk_ranges, initial_layout, resolve_num_extra_experts
from .policy import rebalance_experts_global
from .shuffle import WeightShuffler

LOGGER = logging.getLogger(__name__)


def enabled():
    return os.environ.get('DEEPINFER_EPLB', '0') == '1'


def validate_environment(env):
    if env.get('EPLB_MODE') != 'STATS':
        raise ValueError('Private EPLB requires EPLB_MODE=STATS; native dynamic reload must stay disabled')
    if env.get('EPIC_STEPWISE', '0') != '0':
        raise ValueError('Periodic EPLB and EPIC swap cannot move the same weights together')
    if env.get('EPLB_DISPATCH_POLICY', 'rr') != 'rr':
        raise ValueError('This isolated comparison keeps the verified all-replica rr dispatcher')
    for name in ('ENABLE_CUDA_GRAPH', 'ENABLE_NATIVE_CUDA_GRAPH'):
        if env.get(name, '0').lower() not in ('0', 'false'):
            raise ValueError('This host-scheduled adapter requires CUDA graphs disabled: ' + name)


def logical_window(physical_window, local_map, logical_experts):
    output = torch.zeros((*physical_window.shape[:2], logical_experts),
                         dtype=torch.int64, device=physical_window.device)
    indices = local_map.long().unsqueeze(0).expand_as(physical_window)
    output.scatter_add_(-1, indices, physical_window.long())
    return output


def layout_maps(layout, logical_experts, device):
    physical = len(layout[0])
    if any(len(row) != physical or set(row) != set(range(logical_experts)) for row in layout):
        raise ValueError('Every layout must cover all logical experts with the same capacity')
    inverse = torch.full((len(layout), logical_experts, physical - logical_experts + 1),
                         -1, dtype=torch.int32)
    counts = torch.zeros((len(layout), logical_experts), dtype=torch.int32)
    for layer_index, row in enumerate(layout):
        for physical_index, logical_index in enumerate(row):
            replica_index = int(counts[layer_index, logical_index])
            inverse[layer_index, logical_index, replica_index] = physical_index
            counts[layer_index, logical_index] += 1
    return inverse.to(device), counts.to(device)


class PeriodicEplb:
    def __init__(self, layers, parallelism, settings):
        validate_environment(os.environ)
        if not dist.is_initialized():
            raise RuntimeError('EPLB requires initialized torch.distributed')
        self.layers = list(layers)
        self.settings = settings
        self.size = dist.get_world_size()
        self.rank = dist.get_rank()
        if int(parallelism.ep_size) != self.size or int(parallelism.ep_rank) != self.rank:
            raise ValueError('Only full-world EP with matching rank order is supported')
        if int(parallelism.tp_size) != 1:
            raise ValueError('This adapter supports the verified TP1 expert layout only')
        self.logical = int(self.layers[0]._logical_expert_num)
        self.physical = int(self.layers[0].num_experts)
        self.num_local = self.physical // self.size
        extra = resolve_num_extra_experts(self.logical, settings.minimum_extra, self.size)
        if self.physical != self.logical + extra:
            raise ValueError(f'Configure REDUNDANT_EXPERT={extra} before loading the model')
        if len({layer.layer_idx for layer in self.layers}) != len(self.layers):
            raise ValueError('Duplicate MoE layer IDs')
        self.components = []
        self.identities = []
        for layer in self.layers:
            executor = layer.fused_moe.fused_experts
            if type(executor).__name__ != 'DeepGemmMaskedExecutor':
                raise ValueError('Only the verified LL FP8 DeepGemmMaskedExecutor is supported')
            if type(layer.fused_moe.router).__name__ != 'DeepEpLowLatencyRouter':
                raise ValueError('Only the verified DeepEP low-latency router is supported')
            components = [executor._w1, executor._w1_scale, executor._w2, executor._w2_scale]
            if any(not isinstance(tensor, torch.Tensor) or tensor.ndim == 0 or
                   tensor.shape[0] != self.num_local for tensor in components):
                raise ValueError('Expected active per-expert FP8 weights and both scales')
            if layer._logical_expert_num != self.logical or layer.num_experts != self.physical:
                raise ValueError('Heterogeneous expert layouts require separate managers')
            self.components.append(components)
            self.identities.append([self._identity(tensor) for tensor in components])
            layer._skip_stats_recording = True
        self.device = self.components[0][0].device
        if self.device.type != 'cuda':
            raise ValueError('The serving adapter requires active CUDA expert weights')
        initial = initial_layout(self.logical, extra)
        self.layout = [initial.copy() for _ in self.layers]
        if os.environ.get('RTP_EPLB_USE_LOADED_LAYOUT', '0') == '1':
            loaded = json.loads(Path(os.environ['PHY2LOG_PATH']).read_text())
            self.layout = [loaded[int(layer.layer_idx)] for layer in self.layers]
        inverse, counts = layout_maps(self.layout, self.logical, self.device)
        for layer_index, layer in enumerate(self.layers):
            if layer.log2phy is None or layer.logic_expert_cnt is None:
                raise ValueError('Missing loaded EPLB routing maps')
            if not torch.equal(layer.log2phy, inverse[layer_index]) or not torch.equal(layer.logic_expert_cnt, counts[layer_index]):
                raise ValueError('Loaded layout does not match the private initial PHY2LOG_PATH')
            if layer.fused_moe.fused_experts.config.enable_cuda_graph:
                raise ValueError('CUDA graphs must be disabled in the actual executor config')
            layer.fused_moe.router._periodic_eplb = (self, layer_index)
        self.group = dist.new_group(ranks=list(range(self.size)), backend='nccl', timeout=timedelta(seconds=600))
        hosts = [None for _ in range(self.size)]
        dist.all_gather_object(hosts, socket.gethostname(), group=self.group)
        intranode = len(set(hosts)) == 1
        self.shuffler = WeightShuffler(self.group, self.components, intranode, settings.p2p_groups)
        self.physical_map = torch.tensor(self.layout, dtype=torch.int64, device=self.device)
        self.current_counts = torch.zeros((len(self.layers), self.num_local), dtype=torch.int32, device=self.device)
        self.window = torch.zeros((settings.window, len(self.layers), self.num_local), dtype=torch.int32, device=self.device)
        self.window_index = 0
        self.valid_window_steps = 0
        self.stat_tokens = torch.zeros((settings.stat_interval, len(self.layers)), dtype=torch.int64, device=self.device)
        self.stat_index = 0
        self.clock = RearrangementClock(settings.interval, settings.first_rearrangement)
        self.rearrangements = 0
        self.prefill_seen = False
        self.prefill_active = False
        self.prefill_global_tokens = 0
        self.active = False
        self.collecting = True
        self.calls = [0 for _ in self.layers]
        self.output = Path(os.environ['RTP_DEEPINFER_EPLB_OUTPUT'])
        self.output.mkdir(parents=True, exist_ok=True)
        self._event('initialized', phase=settings.phase, window=settings.window, interval=settings.interval,
                    first_at=(settings.first_rearrangement or max(1, settings.interval // 4)) if settings.interval else None,
                    logical=self.logical, physical=self.physical, local=self.num_local,
                    shuffle='symm_mem_ipc' if intranode else 'nccl_p2p', hosts=hosts,
                    active_gemm_components=['_w1', '_w1_scale', '_w2', '_w2_scale'])

    @staticmethod
    def _identity(tensor):
        return (tensor.data_ptr(), tuple(tensor.shape), tensor.stride(), tensor.dtype)

    def _event(self, kind, **fields):
        record = dict(event=kind, rank=self.rank, monotonic_ns=time.monotonic_ns(),
                      phase_steps=self.clock.completed_steps, rearrangements=self.rearrangements, **fields)
        with (self.output / f'events_rank{self.rank}.jsonl').open('a') as stream:
            stream.write(json.dumps(record) + '\n')
        LOGGER.warning('[Periodic DeepInfer EPLB] %s', json.dumps(record))

    def begin_forward(self, inputs):
        if self.active:
            raise RuntimeError('Nested model forwards are unsupported')
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('Periodic EPLB is not CUDA graph capture safe')
        attention = inputs.attention_inputs
        is_prefill = bool(attention.is_prefill)
        self.prefill_active = False
        batch = len(attention.input_lengths) if is_prefill else len(attention.sequence_lengths)
        tokens = inputs.input_ids.numel()
        if batch <= 1 and tokens <= 1:
            return
        if batch != self.settings.expected_batch:
            raise ValueError(f'Expected synchronized batch {self.settings.expected_batch}, received {batch}')
        if is_prefill:
            self.prefill_active = True
            global_tokens = torch.tensor(tokens, dtype=torch.int64, device=self.device)
            dist.all_reduce(global_tokens, op=dist.ReduceOp.MAX, group=self.group)
            self.prefill_global_tokens = int(global_tokens)
        elif not self.prefill_seen:
            raise ValueError('Refusing to count decode without a real prefill')
        elif tokens != batch or len(attention.prefix_lengths):
            raise ValueError('Mixed or non-one-token-per-sequence decode is unsupported')
        self.active = (self.settings.phase == 'prefill') == is_prefill
        self.collecting = self.settings.collect_steps is None or self.clock.completed_steps < self.settings.collect_steps
        if self.active:
            if self.collecting:
                self.current_counts.zero_()
            self.calls = [0 for _ in self.layers]

    def run_moe(self, fused_moe, hidden_states, topk_weights, topk_ids):
        if not self.prefill_active:
            return fused_moe(hidden_states=hidden_states, topk_weights=topk_weights,
                             topk_ids=topk_ids, activation='SiGLU')
        capacity = min(int(fused_moe.router._num_max_dispatch_tokens_per_rank),
                       int(os.environ.get('REAL_PREFILL_MOE_CHUNK_TOKENS', '256')))
        outputs = []
        for begin, end in chunk_ranges(hidden_states.shape[0], self.prefill_global_tokens, capacity):
            outputs.append(fused_moe(hidden_states=hidden_states[begin:end],
                                     topk_weights=topk_weights[begin:end], topk_ids=topk_ids[begin:end],
                                     activation='SiGLU').clone())
        return torch.cat(outputs, dim=0)

    def record(self, layer_index, counts):
        if not self.active:
            return
        if counts.shape != self.current_counts[layer_index].shape:
            raise ValueError('LL receive count shape does not match local expert slots')
        if self.collecting:
            self.current_counts[layer_index].add_(counts)
        self.calls[layer_index] += 1

    def end_forward(self, success):
        if self.prefill_active and success:
            self.prefill_seen = True
        if not self.active:
            return
        self.active = False
        if not success:
            return
        if any(count == 0 for count in self.calls):
            raise RuntimeError('Missing LL dispatch receive counts for a MoE layer')
        if self.settings.phase == 'decode' and any(count != 1 for count in self.calls):
            raise RuntimeError('Decode must execute every MoE layer exactly once')
        rearrange = self.clock.tick()
        if not self.collecting:
            return
        if self.settings.interval or self.settings.collect_steps is not None:
            self.window[self.window_index].copy_(self.current_counts)
            self.window_index = (self.window_index + 1) % self.settings.window
            self.valid_window_steps = min(self.valid_window_steps + 1, self.settings.window)
        self.stat_tokens[self.stat_index].copy_(self.current_counts.sum(-1))
        self.stat_index += 1
        if self.stat_index == self.settings.stat_interval:
            self._flush_stats()
        if rearrange and (self.settings.max_rearrangements is None or self.rearrangements < self.settings.max_rearrangements):
            self._rearrange()

    def _flush_stats(self):
        gathered = torch.empty((self.size, *self.stat_tokens.shape), dtype=torch.int64, device=self.device)
        dist.all_gather_into_tensor(gathered, self.stat_tokens, group=self.group)
        if self.rank == 0:
            values = gathered.float()
            mean = values.mean(0)
            peak = values.max(0).values
            valid = mean > 0
            self._event('load_balance', steps=self.stat_index,
                        moe_lb=float(peak.sum() / mean.sum()) if bool(valid.any()) else 1.0,
                        gpu_ir_mean=float((peak[valid] / mean[valid]).mean()) if bool(valid.any()) else 1.0)
        self.stat_index = 0

    def _rearrange(self):
        torch.cuda.synchronize()
        started = time.perf_counter()
        local_map = self.physical_map[:, self.rank * self.num_local:(self.rank + 1) * self.num_local]
        history = logical_window(self.window, local_map, self.logical)
        dist.all_reduce(history, group=self.group)
        torch.cuda.synchronize()
        reduced_at = time.perf_counter()
        if self.rank == 0:
            next_map, next_inverse, next_counts = rebalance_experts_global(history, self.physical, self.num_local)
            next_map = next_map.contiguous()
            next_inverse = next_inverse.contiguous()
            next_counts = next_counts.contiguous()
        else:
            next_map = torch.empty_like(self.physical_map, dtype=torch.int32)
            next_inverse = torch.empty((len(self.layers), self.logical, self.physical - self.logical + 1),
                                       dtype=torch.int32, device=self.device)
            next_counts = torch.empty((len(self.layers), self.logical), dtype=torch.int32, device=self.device)
        dist.broadcast(next_map, src=0, group=self.group)
        dist.broadcast(next_inverse, src=0, group=self.group)
        dist.broadcast(next_counts, src=0, group=self.group)
        next_layout = next_map.cpu().tolist()
        layout_maps(next_layout, self.logical, 'cpu')
        planned_at = time.perf_counter()
        for layer_index, layer in enumerate(self.layers):
            if next_inverse[layer_index].shape != layer.log2phy.shape:
                raise RuntimeError('Routing map shape changed across a rearrangement')
            executor = layer.fused_moe.fused_experts
            active = [executor._w1, executor._w1_scale, executor._w2, executor._w2_scale]
            if [self._identity(tensor) for tensor in active] != self.identities[layer_index]:
                raise RuntimeError('Active GEMM weights no longer alias the shuffle components')
        verify_weights = os.environ.get('RTP_EPLB_VERIFY_WEIGHTS', '0') == '1'
        before_samples = None
        if verify_weights:
            local_samples = self._weight_samples()
            gathered_samples = torch.empty((self.size, *local_samples.shape), dtype=torch.uint8, device=self.device)
            dist.all_gather_into_tensor(gathered_samples, local_samples, group=self.group)
            before_samples = gathered_samples.permute(1, 0, 2, 3, 4).reshape(len(self.layers), self.physical, 4, 16)
        torch.cuda.synchronize()
        moving_at = time.perf_counter()
        self.shuffler.move(self.layout, next_layout)
        torch.cuda.synchronize()
        moved_at = time.perf_counter()
        if verify_weights:
            source_indices = torch.tensor([[previous.index(expert) for expert in following[
                self.rank * self.num_local:(self.rank + 1) * self.num_local]]
                for previous, following in zip(self.layout, next_layout, strict=True)], device=self.device)
            expected = before_samples[torch.arange(len(self.layers), device=self.device).unsqueeze(1), source_indices]
            if not torch.equal(self._weight_samples(), expected):
                raise RuntimeError('Post-shuffle active GEMM weight/scale byte samples do not match logical experts')
        for layer_index, layer in enumerate(self.layers):
            layer.log2phy.copy_(next_inverse[layer_index])
            layer.logic_expert_cnt.copy_(next_counts[layer_index])
        self.physical_map.copy_(next_map)
        torch.cuda.synchronize()
        dist.barrier(group=self.group, async_op=True).wait()
        torch.cuda.synchronize()
        old_layout = self.layout
        self.layout = next_layout
        self.rearrangements += 1
        payload = json.dumps(next_layout, separators=(',', ':')) + '\n'
        layout_sha256 = hashlib.sha256(payload.encode()).hexdigest()
        if os.environ.get('RTP_EPLB_BOUNDARY_PROOF', '0') == '1':
            self._event('boundary_committed', weights_verified=verify_weights, layout_sha256=layout_sha256,
                        window_steps=self.valid_window_steps, elapsed_s=time.perf_counter() - started,
                        statistics_reduce_s=reduced_at - started, planning_and_broadcast_s=planned_at - reduced_at,
                        weight_shuffle_s=moved_at - moving_at,
                        sample_and_commit_s=(moving_at - planned_at) + (time.perf_counter() - moved_at))
        if self.rank == 0:
            path = self.output / f'layout_{self.rearrangements:04d}.json'
            path.write_text(payload)
            self._event('rearranged', elapsed_s=time.perf_counter() - started,
                        valid_window_steps=self.valid_window_steps,
                        changed_slots=sum(previous != following for previous_layer, following_layer
                                          in zip(old_layout, next_layout, strict=True)
                                          for previous, following in zip(previous_layer, following_layer, strict=True)),
                        layout_path=str(path), layout_sha256=hashlib.sha256(payload.encode()).hexdigest())

    def _weight_samples(self):
        samples = []
        for layer in self.components:
            component_samples = []
            for tensor in layer:
                raw = tensor.contiguous().view(torch.uint8).reshape(self.num_local, -1)
                indices = torch.linspace(0, raw.shape[1] - 1, 16, device=self.device).long()
                component_samples.append(raw.index_select(1, indices))
            samples.append(torch.stack(component_samples, dim=1))
        return torch.stack(samples, dim=0)


def build_manager(layers, parallelism):
    if not enabled():
        return None
    if not layers:
        raise ValueError('No MoE layers to balance')
    return PeriodicEplb(layers, parallelism, Settings.from_env(os.environ))
