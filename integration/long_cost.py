"""Device-resident controls and window export for the paired 1000-step H20 study."""

import hashlib
import os
from pathlib import Path

import numpy as np
import torch

from fixed_workload import RouteTape, publishes_checkpoint, route_ordinals


class LongRouteTape(RouteTape):
    def __init__(self, device, topk, slots, steps=1000, layers=40, batch=256):
        super().__init__(device, topk, slots, steps=steps, layers=layers, batch=batch)
        # Both arms execute the same captured operations. The linear arm records
        # its live router decisions; the fine arm selects the published decisions.
        self.use_reference = torch.zeros_like(self.replaying)

    def select_ordinals(self, ordinal_out):
        stable = route_ordinals(self.selected_ids)
        ordinal_out.copy_(stable[0])
        self.ordinals.index_copy_(0, self.index, stable)

    def start_phase(self, phase, rank):
        path = Path(os.environ["GC_MOE_FIXED_DIR"]) / f"{phase}-rank{rank}.pt"
        self.publish = publishes_checkpoint()
        if not self.publish:
            bank = torch.load(path, weights_only=True, map_location="cpu", mmap=True)
            for key in ("ids", "weights"):
                dst, src = getattr(self, key), bank[key]
                if dst.shape != src.shape or dst.dtype != src.dtype:
                    raise ValueError(f"invalid {phase} {key} reference")
                dst.copy_(src)
        self.bank_path = path
        self.position.zero_()
        self.counts.fill_(-1)
        self.epochs.zero_()
        self.layer_ids.fill_(-1)
        self.used_ordinals.fill_(-1)
        self.use_reference.fill_(not self.publish)
        self.replaying.fill_(True)

    def finish_phase(self, first_epoch):
        epochs = self.epochs.cpu().long()
        expected = torch.arange(first_epoch, first_epoch + len(epochs))
        layers = self.layer_ids.cpu()
        if (
            self.position.item() != len(epochs)
            or not torch.equal(epochs, expected)
            or not torch.equal(layers, (epochs - 1) % self.layers)
            or layers[0].item() != 0
            or (self.counts < 0).any().item()
            or not torch.equal(self.ordinals, self.used_ordinals)
        ):
            raise ValueError("incomplete 1000-step route, layer, epoch or ordinal history")
        bank = {k: getattr(self, k).cpu() for k in ("ids", "weights")}
        if self.publish:
            with self.bank_path.open("xb") as f:
                torch.save(bank, f)
        hashes = {key: hashlib.sha256(value.numpy()).hexdigest() for key, value in bank.items()}
        hashes["ordinals"] = hashlib.sha256(self.used_ordinals.cpu().numpy()).hexdigest()
        return dict(first_epoch=first_epoch, calls=len(epochs), **hashes)


def register_long_rpc(worker_class, get_buffer, modules):
    from attention_order import reset_order, order_coverage
    from stream_checkpoint import prepare_worker

    @torch.inference_mode()
    def begin(self, phase):
        from vllm.models.deepseek_v41.nvidia.flashmla import DeepseekV4FlashMLAAttention

        buf = get_buffer()
        torch.cuda.synchronize()
        rank, tape = int(buf.rank_idx), buf.fixed_routes
        if phase == "stats":
            # All captured and prefill work is finished. Clear the actual table
            # consumed by rearrange_now (some registered methods share it).
            for m in modules:
                if m._eplb is not None:
                    m._eplb["stats"].zero_()
            self._long_initial = {}
        elif phase == "measure":
            if not getattr(self, "_long_stats_done", False):
                raise ValueError("missing complete statistics phase")
            ref = self._gc_reference_tokens
            ref.mark_replay_start()
            self._long_initial = prepare_worker(
                self, buf, modules, ref, os.environ["GC_MOE_STRICT_STATE_DIR"], publishes_checkpoint()
            )
        else:
            raise ValueError(phase)
        tape.start_phase(phase, rank)
        if phase == "measure":
            reset_order(tape, self.model_runner.model.modules(), DeepseekV4FlashMLAAttention)
        else:
            # The full-batch decode graph may first be instantiated in this phase.
            # Ordering stays enabled; arm its 1000 coverage counters after warmup.
            tape.replaying.fill_(False)
        buf.replay_stream_mode.fill_(1)
        buf.long_clocks.zero_()
        buf.long_clocks[:, :2].fill_(-1)  # unsigned 64-bit minimum identity
        self._long_first = int(buf.header[0].item())
        return dict(rank=rank, epoch=self._long_first, checkpoint=self._long_initial)

    @torch.inference_mode()
    def finish(self, phase, directory):
        torch.cuda.synchronize()
        buf = get_buffer()
        tape, rank = buf.fixed_routes, int(buf.rank_idx)
        audit = tape.finish_phase(self._long_first)
        modes = order_coverage(tape) if phase == "measure" else {}
        if (
            (phase == "measure" and not modes)
            or any(values != [1] * tape.steps for values in modes.values())
            or buf.replay_stream_mode.item() != 1
            or buf.header[0].item() != self._long_first + len(tape.ids)
        ):
            raise ValueError("incomplete ordered stream window")
        audit.update(rank=rank, compressed_layers=sorted(map(int, modes)))
        counts = tape.counts.cpu().numpy()
        audit["total_rows_by_layer"] = counts.reshape(tape.steps, tape.layers, -1).sum(axis=(0, 2)).tolist()
        if phase == "stats":
            self._long_stats_done = True
        elif phase == "measure":
            idx = tape.epochs.long().remainder(len(tape.ids))
            clocks = buf.long_clocks.index_select(0, idx).cpu().numpy()
            ref = self._gc_reference_tokens
            if publishes_checkpoint():
                ref.lengths.copy_(ref.positions)
            samples = np.asarray(ref.native_history(tape.steps)["replay_top1"], dtype=np.int32)
            audit["batch_order"] = ref.batch_hash.hexdigest()
            path = Path(directory) / f"rank{rank}.npz"
            with path.open("xb") as f:
                np.savez_compressed(
                    f, clocks=clocks, counts=counts, epochs=tape.epochs.cpu().numpy(), native=samples
                )
        else:
            raise ValueError(phase)
        return audit

    worker_class._gc_moe_long_begin = begin
    worker_class._gc_moe_long_finish = finish
