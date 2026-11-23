"""Short, untimed native trace and paired expert replay with actual decode operands."""
import hashlib
import json
from functools import wraps
from types import SimpleNamespace

import torch

from stream_checkpoint import checkpoint, named_caches, runner_boundary, tensor_hash


def install():
    from vllm.models.deepseek_v4.nvidia.model import DeepseekV4DecoderLayer, DeepseekV4MegaMoEExperts

    init = DeepseekV4DecoderLayer.__init__

    @wraps(init)
    def decoder(layer, *args, **kw):
        init(layer, *args, **kw)
        layer.attn._fair_samples = {}

        def attention(module, inputs, out):
            x = inputs[1]
            if x.shape[0] == 512 and out.shape[0] == 512:
                for name, v in (("attn_in", x), ("attn_out", out)):
                    save(module._fair_samples, name, v)

        layer.attn.register_forward_hook(attention)

    DeepseekV4DecoderLayer.__init__ = decoder
    forward = DeepseekV4MegaMoEExperts.forward

    @wraps(forward)
    def expert(module, hidden_states, topk_weights, topk_ids, *, activation_clamp, fast_math=True):
        record = hidden_states.shape[0] == 64 and not getattr(module, "_fair_replay", False)
        if record:
            if not hasattr(module, "_fair_samples"):
                module._fair_samples = {}
            for name, value in (("hidden_states", hidden_states), ("topk_weights", topk_weights),
                                ("topk_ids", topk_ids)):
                save(module._fair_samples, name, value)
            module._fair_options = dict(activation_clamp=activation_clamp, fast_math=fast_math)
        y = forward(module, hidden_states, topk_weights, topk_ids,
                    activation_clamp=activation_clamp, fast_math=fast_math)
        if record:
            save(module._fair_samples, "y", y)
        return y

    DeepseekV4MegaMoEExperts.forward = expert


def save(samples, name, value):
    if name not in samples:
        samples[name] = torch.empty_like(value)
    samples[name].copy_(value)


def boundary(m):
    states = m.model_runner.req_states
    keys = sorted(states.req_id_to_index)
    indices = [states.req_id_to_index[k] for k in keys]
    ref = SimpleNamespace(keys=keys, positions=states.total_len.gpu[indices],
                          batch_hash=hashlib.sha256(json.dumps(indices).encode()))
    controls = runner_boundary(m.model_runner, ref)
    folder = m.root / "common-prefill"
    folder.mkdir(exist_ok=True)
    m.fair_kv = checkpoint(folder, m.rank, m.publisher, named_caches(m.model_runner),
                           controls, audit_prior=True)


@torch.inference_mode()
def capture(m, step):
    records = []
    for layer in m.fair_layers:
        expert = layer.ffn.experts
        tensors = {k: v.cpu() for k, v in expert._fair_samples.items()}
        attention = layer.attn._fair_samples
        if not {"attn_in", "attn_out"} <= attention.keys():
            raise RuntimeError("missing native attention observations")
        records.append(dict(tensors=tensors, options=expert._fair_options,
                            attention={k: tensor_hash(v) for k, v in attention.items()}))
    torch.save(records, m.folder / f"native-{step}-{m.rank}.pt")
    return True


@torch.inference_mode()
def replay(m):
    # Run after all native steps so diagnostic calls cannot modify the native trajectory.
    evidence = []
    for step in range(3):
        rows = torch.load(m.root / "mega" / f"native-{step}-{m.rank}.pt", weights_only=True)
        for index, (layer, row) in enumerate(zip(m.fair_layers, rows, strict=True)):
            expert = layer.ffn.experts
            expert._fair_replay = True
            inputs = row["tensors"]
            kwargs = {k: v.to(device=m.model_runner.device) for k,v in inputs.items() if k != "y"}
            if row["options"] != expert._fair_options:
                raise ValueError("expert activation options differ")
            actual = expert(**kwargs, **row["options"]).cpu()
            ref = inputs["y"]
            if actual.dtype != ref.dtype or actual.shape != ref.shape:
                raise ValueError("expert output layout differs")
            mask = actual.view(torch.int16) != ref.view(torch.int16)
            different = int(mask.sum())
            item = dict(step=step, layer=index, different=different, finite=bool(actual.isfinite().all()),
                        inputs={k:tensor_hash(v) for k,v in kwargs.items()})
            if different:
                pos = tuple(mask.nonzero()[0].tolist())
                item.update(first=pos, reference=ref[pos].item(), actual=actual[pos].item(),
                            max_abs=float((actual.float()-ref.float()).abs().max()))
            buf = expert.get_symm_buffer()
            n = actual.shape[0]
            sf = buf.l2_acts_sf
            start = sf.data_ptr() - buf.buffer.data_ptr() + sf.stride(1) * sf.shape[1] * sf.element_size()
            slots = buf.num_topk + buf.num_shared_experts
            size = slots * buf.num_max_tokens_per_rank * buf.hidden * 2
            partials = buf.buffer.narrow(0, start, size).view(torch.bfloat16).reshape(
                slots, buf.num_max_tokens_per_rank, buf.hidden)[:, :n]
            # Reconstruct shared-first, then routed-slot order. This guards the
            # raw layout calculation before interpreting any per-slot hash.
            total = partials[-1].float()
            for partial in partials[:-1]:
                total = total + partial.float()
            item["partials_verified"] = torch.equal(total.bfloat16().cpu(), actual)
            item["partial_hashes"] = [tensor_hash(v) for v in partials]
            if different:
                row = item["first"][0]
                item["slot_values_at_first"] = partials[:,row,item["first"][1]].float().cpu().tolist()
            block_m = m.bm
            row = torch.arange(n, device=buf.x.device) % block_m
            sf_row = (torch.arange(n, device=buf.x.device) // block_m * ((block_m+127)//128*128)
                      + row//128*128 + row%32*4 + row%128//32)
            for k, v in dict(x=buf.x[:n], sf=buf.x_sf[:n], shared_sf=buf.shared_l1_acts_sf[sf_row],
                             ids=buf.topk_idx[:n], weights=buf.topk_weights[:n]).items():
                item["inputs"][k] = tensor_hash(v)
            evidence.append(item)
    (m.folder / f"replay-{m.rank}.json").write_text(json.dumps(evidence))
    return m.fair_kv
