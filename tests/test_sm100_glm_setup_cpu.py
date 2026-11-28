"""Check live worker RPC and archived JSON against the same GLM launch contract."""
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/sm100"))
from glm53_matrix import validate_refit_setup
from glm53_price import ARMS, NEW, PUBLIC


class TestGlmSetup(unittest.TestCase):
    @patch.dict("os.environ", GLM_SLOT_BUDGETS="1")
    def test_each_slot_budget(self):
        native = json.loads((PUBLIC / "vanilla/result.json").read_text())["setup"]
        for label, width in (("vanilla", 32), ("rr", 33), ("eco8", 40), ("eco12", 44)):
            eco = label.startswith("eco")
            source = "eco" if eco else label
            workers = json.loads((PUBLIC / source / "result.json").read_text())["setup"]
            for row in workers:
                roles = {r[0]: r for r in row["roles"]}
                row["roles"] = [roles[i] for i in ((2, 1, 4, 3) if eco else (2, 1, 3))]
                row["slots"] = width
                row["eco_unit"] = (16, *NEW, 1, 6, 0) if eco else None
            with self.subTest(label=label):
                validate_refit_setup(label, workers, native)
                workers[-1]["slots"] = 36
                with self.assertRaisesRegex(ValueError, "rank7 slots"):
                    validate_refit_setup(label, workers, native)

    def test_live_and_archived_roles(self):
        native = json.loads((PUBLIC / "vanilla/result.json").read_text())["setup"]
        for arm in ARMS:
            setup = json.loads((PUBLIC / arm / "result.json").read_text())["setup"]
            for worker in setup:
                roles = {node[0]: tuple(node) for node in worker["roles"]}
                order = (2, 1, 4, 3) if arm == "eco" else (2, 1, 3)
                worker["roles"] = [roles[role] for role in order]
                worker["slots"] = 32 if arm == "vanilla" else 36
                worker["eco_unit"] = (16, *NEW, 1, 6, 0) if arm == "eco" else None
            with self.subTest(arm=arm):
                validate_refit_setup(arm, setup, native)
                validate_refit_setup(arm, json.loads(json.dumps(setup)), native)
                if arm == "eco":
                    setup[2]["eco_unit"] = (16, 249, 140, 1, 6, 0)
                    with self.assertRaisesRegex(ValueError, "rank2 eco_unit"):
                        validate_refit_setup(arm, setup, native)
                    setup[2]["eco_unit"] = (16, *NEW, 1, 6, 0)
                setup[4]["roles"][1] = (1, 0, 10)
                with self.assertRaisesRegex(ValueError, "rank4 roles"):
                    validate_refit_setup(arm, setup, native)


if __name__ == "__main__":
    unittest.main()
