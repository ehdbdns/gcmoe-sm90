"""Capture exactly the third post-placement step inside the original replay."""
import json
import os
from pathlib import Path
from uuid import uuid4

from .aligned_step import resolve_epochs

LAYERS = (0, 8, 20, 32, 39)


def capture_step(engine, arm):
    raw = Path(os.environ['GC_BS256_TRACE_DIR'])
    raw.mkdir()
    rpc = engine.collective_rpc
    before = rpc('_gc_moe_profile_snapshot')
    starts = {row['next_epoch'] for row in before}
    assert len(starts) == 1 and {row['rank'] for row in before} == set(range(8))
    start = starts.pop()
    tag = uuid4().hex
    rpc('_gc_moe_prof_start', args=(str(raw), tag, True))
    try:
        outputs = engine.step()
    finally:
        status = rpc('_gc_moe_prof_stop')
    assert len(status) == 8 and all(r['ok'] for r in status)
    rpc('_gc_moe_profile_park_fine')
    assert len({o.request_id for o in outputs if o.outputs and len(o.outputs[0].token_ids) == 1}) == 256
    manifest = dict(id=tag, arm=arm, label=arm, step=2, layers=40, first_epoch=start,
                    dispatch_transport=('source_push' if os.getenv('GC_MOE_SM90_SINGLE_PUSH') == '1' else 'pull'),
                    decode_tokens=256, gemm_start='first_valid_cta_claim',
                    ranks=rpc('_gc_moe_profile_snapshot', args=(start,)))
    capture = resolve_epochs(raw, manifest, 20 if arm == "eco" and os.getenv("GC_MOE_SM90_PLAN_DISPATCH", "1") == "1" else 24)
    assert capture['first_epoch'] == start
    (raw/'capture.json').write_text(json.dumps(capture))
    epochs = [start + layer for layer in LAYERS]
    for epoch in epochs:
        rows = rpc('_gc_moe_profile_fine_dump', args=([epoch],))
        assert len(rows) == 8 and all(not r.get('error') and r['epochs'][str(epoch)] for r in rows)
        (raw/f'fine-epoch-{epoch}.json').write_text(json.dumps(dict(id=tag, epochs=[epoch], ranks=rows)))
    (raw/'fine.json').write_text(json.dumps(dict(id=tag, epochs=epochs, per_epoch_files=True)))
    return outputs
