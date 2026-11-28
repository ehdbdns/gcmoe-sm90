"""Check live worker RPC and archived JSON against the same GLM launch contract."""
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/sm100"))
from glm53_matrix import validate_refit_setup
from glm53_price import ARMS, NEW, PUBLIC


class TestGlmSetup(unittest.TestCase):
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
