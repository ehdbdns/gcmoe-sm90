"""Check the sampled EP8 export, including separate CTA completions and real ready events."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/sm100"))
from eco_trace import absolute, export, metrics


class TestTrace(unittest.TestCase):
    def test_absolute(self):
        # The leader issues one cluster MMA, and two CTAs complete the two N tiles.
        rows = np.array([[15, 0, 1, 0, 2, 3, 4, 5, 6, 7, 30, 0],
                         [25, 0, 5, 0, 4, 3, 4, 5, 6, 7, 30, 0],
                         [28, 0, 5, 1, 4, 3, 4, 5, 6, 7, 31, 0],
                         [12, 0, 12, 2, 3, 4, 5, 6, 7, 8, 32, 0]], np.int32)
        ev = absolute(rows, 0, float, {})
        self.assertEqual(sorted(e["dur"] for e in ev if e["ph"] == "X"), [10, 13])
        self.assertEqual(ev[0]["name"], "M block ready")
        with self.assertRaises(ValueError):
            absolute(rows[[0, 2]], 0, float, {})

    def test_rank_and_layer_boundaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for r in range(8):
                prof = dict(baseTimeNanoseconds=1000+r, traceEvents=[
                    dict(name="sm100_phase_clock_anchor", ph="X", cat="kernel", ts=t, dur=.2)
                    for t in (2, 22)])
                (root/f"step0002-rank{r}.kineto.json").write_text(json.dumps(prof))
                end = 17000 + r
                clocks = [3000, 3500, 5000, 7200, 7100, 9000, 9200, 15000,
                          3600, 6200, 6500, 8700, 15200, end, 6800, 2500, 5200, 3, 7, 1, 0]
                np.savez(root/f"step0002-layer20-rank{r}.npz", clocks=clocks, counts=[1, 2],
                         quota=[1, 2], ticks=[1000, 21000], step=2, layer=20, rank=r, block_m=16)
            ranks = export(root, "step0002-layer20")
            row = metrics(ranks)
            self.assertEqual(row["moe_us"], 14.507)
            self.assertEqual(row["total_m"], 16)
            self.assertEqual(row["accepted"], 3)
            self.assertEqual(row["routed_mae_pp"], 0)
            trace = json.loads((root/"step0002-layer20.perfetto.json").read_text())
            disp = [e for e in trace["traceEvents"] if e.get("pid") == 0 and
                    e["name"] in ("count exchange", "pull token")]
            self.assertEqual(len(disp), 2)
            self.assertEqual(disp[0]["tid"], disp[1]["tid"])
            entry, = [e for e in trace["traceEvents"] if e.get("pid") == 0 and e["name"] == "compute kernel entry"]
            mma, = [e for e in trace["traceEvents"] if e.get("pid") == 0 and e["name"] == "shared expert L1"]
            self.assertEqual(entry["ph"], "i")
            self.assertAlmostEqual(mma["ts"] - entry["ts"], 0.1)


if __name__ == "__main__":
    unittest.main()
