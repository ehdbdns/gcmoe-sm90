"""Check the calibrated constants, plan score arithmetic and SM90 cache identity."""
import os
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'source'), str(ROOT/'scripts/trace_layer')]
from fit_valid_cost import fit, load_trace, score
from gc_moe.eco.h20_valid_cost import FULL, INTERCEPTS, ROW_COST, block_cost, slot_cost
from gc_moe.apis.sm90_gc_moe import ValidatedCall


def main():
    cfg = (8,384,2048,6,5120,2304,256,2048,10.,True)
    with patch.dict(os.environ, {'GC_MOE_ECO_COSTFN':'0'}):
        linear = ValidatedCall(*cfg)
    with patch.dict(os.environ, {'GC_MOE_ECO_COSTFN':'2'}):
        calibrated = ValidatedCall(*cfg)
    assert len({linear, calibrated}) == 2
    assert block_cost(0) == 0
    assert all(block_cost(v) < block_cost(v+1) for v in range(64))
    for q in range(193):
        assert slot_cost(q) == sum(block_cost(min(q-i,64)) for i in range(0,q,64))
    assert slot_cost(128) == 2 * FULL
    folder = ROOT/'docs/trace_executor_20261119/batch_claim_results/after/stream'
    _, training = load_trace(next(folder.glob('*layer08*.detail.json')))
    _, holdout = load_trace(next(folder.glob('*layer20*.detail.json')))
    offsets, slope, _ = fit(training)
    assert (offsets, slope) == (INTERCEPTS, ROW_COST)
    result = score(holdout, offsets, slope)
    assert result['block_mae_cta_us']['valid'] < 11
    print('H20_VALID_COST_CPU PASS')


if __name__ == '__main__':
    main()
