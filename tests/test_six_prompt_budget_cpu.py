"""Drive BS256 prompt selection up to the vLLM constructor, without a GPU."""
import hashlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

DRIVER = Path(__file__).resolve().parents[1] / "scripts/real_decode_100_100.py"


class SixPromptBudget(unittest.TestCase):
    def test_publisher_and_consumers(self):
        # The long source rows reproduce the old second-pass budget failure.
        original = [[i] * (2048 - 7 * i) for i in range(256)]
        prefixes = [ids[:256] for ids in sorted(original, key=len)]
        prompt_hash = hashlib.sha256(json.dumps(prefixes, separators=(",", ":")).encode()).hexdigest()

        class EngineReached(Exception):
            pass

        engine = Mock(side_effect=EngineReached)
        modules = {
            "vllm_warmup_compat": Mock(), "deepep_compare": Mock(),
            "vllm": SimpleNamespace(LLM=engine, SamplingParams=object),
            "vllm.sampling_params": SimpleNamespace(RequestOutputKind=object),
            "vllm.config.compilation": SimpleNamespace(CUDAGraphMode=SimpleNamespace(FULL_DECODE_ONLY=1)),
        }
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, modules):
            root = Path(tmp)
            dataset = root / "prompts.jsonl"
            dataset.write_text("".join(json.dumps({"input_ids": ids}) + "\n" for ids in original))
            (root / "tokens.json").write_text(json.dumps({
                "prompt_hash": prompt_hash, "tokens": {f"req{i}": [] for i in range(256)}
            }))
            env = dict(DATASET=str(dataset), MODEL="fixture", MAX_BATCHED="512", GC_PREFILL_MARGIN="64",
                       SM90_SIX_ARM="1", GC_KV_BUDGET="740000", GC_MOE_KV_BYTES=str(10 * 1024**3),
                       GC_MOE_MATRIX_BS="256", GC_MOE_MATRIX_SIZE="24", GC_MOE_ECO_COSTFN="2",
                       GC_MOE_LONG_COST="1", GC_MOE_ORDER_EXTRA="1", GC_MOE_STEP_PROFILE="1",
                       GC_MOE_SM90_STREAM_M_BLOCKS="1", GC_MOE_FIXED_DIR=tmp, GC_MOE_STRICT_STATE_DIR=tmp,
                       GC_MOE_FINE_TRACE="0", GC_MEASURE_FLUSH="0")
            args = [str(DRIVER), "--output", str(root/"run.json"), "--label", "six", "--long-cost-study",
                    "--warm-steps", "1000", "--measure-steps", "1000", "--prefill-hi", "2048"]
            with patch.dict(os.environ, env, clear=True), patch.object(sys, "argv", args):
                spec = importlib.util.spec_from_file_location("decode", DRIVER)
                driver = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(driver)
                for arm in range(7):
                    os.environ.update(SM90_EIGHT_ARM=str(arm), GC_MOE_MATRIX_ARM="vanilla" if arm < 4 else "rr")
                    os.environ.pop("GC_KV_OVERHEAD", None)
                    log = io.StringIO()
                    with redirect_stdout(log), self.assertRaises(EngineReached):
                        driver.main()
                    self.assertIn("per_stream_overhead=2449 required=692480 limit=740000", log.getvalue())
                    kw = engine.call_args.kwargs
                    self.assertEqual(kw["max_model_len"], 2705)
                    self.assertEqual(kw["kv_cache_memory_bytes"], 10 * 1024**3)
                    self.assertEqual(driver.select_decode_prompts(256, 256, 2048), prefixes)
                os.environ.pop("GC_KV_OVERHEAD")
                os.environ["GC_KV_BUDGET"] = "692479"
                with self.assertRaisesRegex(ValueError, "requires 692480"):
                    driver.main()
                self.assertEqual(engine.call_count, 7)


if __name__ == "__main__":
    unittest.main()
