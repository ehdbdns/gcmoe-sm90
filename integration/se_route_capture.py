"""Vanilla-only publisher for the SM90 three-layer operator replay."""
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist


def register_capture(worker, get_buffer, modules):
    @torch.inference_mode()
    def begin(self, phase):
        if os.environ.get('GC_MOE_MATRIX_ARM') != 'vanilla' or phase not in ('stats', 'measure'):
            raise ValueError('capture needs the live vanilla router')
        buf = get_buffer()
        tape = buf.fixed_routes
        tape.start_phase(phase, buf.rank_idx)
        if not tape.publish or tape.use_reference.item():
            raise ValueError('vanilla did not select live routes')
        if phase == 'stats':
            tape.replaying.fill_(False)
        else:
            from attention_order import reset_order
            from vllm.models.deepseek_v41.nvidia.flashmla import DeepseekV4FlashMLAAttention
            self._gc_reference_tokens.mark_replay_start()
            reset_order(tape, self.model_runner.model.modules(), DeepseekV4FlashMLAAttention)
        buf.replay_stream_mode.fill_(1)
        buf.long_clocks.zero_()
        buf.long_clocks[:, :2].fill_(-1)
        self._capture_epoch = int(buf.header[0])
        return self._capture_epoch

    @torch.inference_mode()
    def finish(self, phase):
        from matrix_workload import source_history
        buf = get_buffer()
        tape = buf.fixed_routes
        rank = int(buf.rank_idx)
        n = (1000 if phase == 'stats' else 3) * tape.layers
        epochs = tape.epochs[:n].long()
        first = self._capture_epoch
        if (int(tape.position) != n or int(buf.header[0]) != first+n or
                not torch.equal(epochs, torch.arange(first, first+n, device=epochs.device)) or
                not torch.equal(tape.layer_ids[:n].long(), torch.arange(n, device=epochs.device) % 40) or
                not tape.ordinal_match[:n].all()):
            raise ValueError(f'{phase} rank{rank}: incomplete or misaligned route tape')
        if phase == 'stats':
            return source_history(buf, modules, phase)
        folder = Path(os.environ['GC_MOE_FIXED_DIR'])
        counts = tape.counts[:n]
        local_demands = torch.zeros((n, 384), dtype=torch.int32, device=epochs.device)
        ids = tape.ids[:n].flatten(1)
        local_demands.scatter_add_(1, ids, torch.ones_like(ids, dtype=torch.int32))
        dist.all_reduce(local_demands, group=buf.group)
        if not torch.equal(counts, local_demands[:, 48*rank:48*(rank+1)]):
            raise ValueError('collected vanilla counts differ from source demand')
        clocks = buf.long_clocks.index_select(0, epochs % len(buf.long_clocks))
        if not torch.equal(clocks[:, 4], epochs) or not (clocks[:, 3] > clocks[:, 0]).all():
            raise ValueError('invalid collected graph timing')
        np.savez_compressed(folder/'vanilla'/f'rank{rank}.npz', counts=counts.cpu().numpy(),
                            epochs=epochs.cpu().numpy(), clocks=clocks.cpu().numpy())
        for layer in (8, 20, 32):
            selection = slice(layer, n, 40)
            weights = tape.weights[selection].cpu().numpy()
            if not np.isfinite(weights).all():
                raise ValueError('nonfinite routing weights')
            np.savez_compressed(folder/'routes'/f'layer{layer}-rank{rank}.npz',
                                ids=tape.ids[selection].cpu().numpy(), weights=weights)
        return rank

    worker._gc_se_capture_begin = begin
    worker._gc_se_capture_finish = finish
