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

        class Inproc:
            get_output = Mock()

        core = Inproc()
        cache = SimpleNamespace(kv_cache_size_tokens=None, kv_cache_max_concurrency=None)
        core.engine_core = SimpleNamespace(vllm_config=SimpleNamespace(cache_config=cache))
        engine = Mock(return_value=SimpleNamespace(llm_engine=SimpleNamespace(engine_core=core)))
        modules = {
            "vllm_warmup_compat": Mock(), "deepep_compare": Mock(),
            "vllm": SimpleNamespace(LLM=engine, SamplingParams=Mock(side_effect=EngineReached)),
            "vllm.v1.engine.core_client": SimpleNamespace(InprocClient=Inproc),
            "vllm.sampling_params": SimpleNamespace(RequestOutputKind=SimpleNamespace(DELTA=1)),
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
                       SM90_SIX_ARM="1",
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
                for steps, gib, limit in ((1000, 7, 740000), (500, 5, 480000)):
                    max_len = 705 + 2 * steps
                    required = 256 * max_len
                    args[args.index("--warm-steps") + 1] = str(steps)
                    args[args.index("--measure-steps") + 1] = str(steps)
                    os.environ.update(SM90_SIX_STEPS=str(steps), GC_KV_BUDGET=str(limit),
                                      GC_MOE_KV_BYTES=str(gib * 1024**3))
                    cache.kv_cache_size_tokens, cache.kv_cache_max_concurrency = required, 256
                    for arm in range(7):
                        os.environ.update(SM90_EIGHT_ARM=str(arm), GC_MOE_MATRIX_ARM="vanilla" if arm < 4 else "rr")
                        os.environ.pop("GC_KV_OVERHEAD", None)
                        log = io.StringIO()
                        with redirect_stdout(log), self.assertRaises(EngineReached):
                            driver.main()
                        self.assertIn(f"per_stream_overhead={max_len-256} required={required} limit={limit}", log.getvalue())
                        self.assertIn(f"[six-capacity] tokens={required} required={required}", log.getvalue())
                        self.assertEqual(engine.call_args.kwargs["max_model_len"], max_len)
                        self.assertEqual(engine.call_args.kwargs["kv_cache_memory_bytes"], gib * 1024**3)
                        self.assertEqual(driver.select_decode_prompts(256, 256, 2048), prefixes)
                    with redirect_stdout(io.StringIO()):
                        for tokens, concurrency in ((required-1, 256), (required, 255)):
                            cache.kv_cache_size_tokens, cache.kv_cache_max_concurrency = tokens, concurrency
                            with self.assertRaisesRegex(ValueError, "insufficient KV"):
                                driver.main()
                        os.environ.pop("GC_KV_OVERHEAD")
                        os.environ["GC_KV_BUDGET"] = str(required-1)
                        with self.assertRaisesRegex(ValueError, f"requires {required}"):
                            driver.main()
                self.assertEqual(engine.call_count, 18)


if __name__ == "__main__":
    unittest.main()
