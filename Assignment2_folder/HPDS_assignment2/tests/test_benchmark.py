#!/usr/bin/env python3
"""Test measurement collection with synthetic process output, without a GPU."""
import contextlib
import csv
import importlib.util
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / os.environ.get("RESULTS_ROOT", "results")
spec = importlib.util.spec_from_file_location("benchmark", ROOT / "benchmark_baseline.py")
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


class BenchmarkTests(unittest.TestCase):
    def test_result_parser_rejects_failed_missing_and_invalid_results(self):
        OUTPUT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=OUTPUT) as folder:
            log = Path(folder) / "process.log"
            programs = [
                "raise SystemExit(1)",
                "print('no result')",
                "print('RESULT end_to_end_ms=nan accuracy=50')",
                "print('RESULT end_to_end_ms=0 accuracy=50')",
                "print('RESULT end_to_end_ms=1 accuracy=101')",
            ]
            with contextlib.redirect_stdout(io.StringIO()):
                for program in programs:
                    with self.assertRaises((ValueError, benchmark.subprocess.CalledProcessError)):
                        benchmark.run([sys.executable, "-c", program], log)
            self.assertTrue(log.exists())

    def test_summary_and_prediction_failure(self):
        OUTPUT.mkdir(parents=True, exist_ok=True)
        for mismatch in (False, True):
            with self.subTest(mismatch=mismatch), tempfile.TemporaryDirectory(dir=OUTPUT) as folder:
                root = Path(folder).resolve()

                def fake_run(command, log):
                    gpu = command[0] == "./cuda"
                    Path(command[-1]).write_text("1\n" if mismatch and gpu else "0\n")
                    log.write_text("Synthetic fixture, no GPU execution\n")
                    return {"strategy": "global" if gpu else "serial", "end_to_end_ms": "5" if gpu else "20",
                            "accuracy": "100", **({"kernel_ms": "1"} if gpu else {})}

                with patch.object(benchmark, "ROOT", root), patch.object(benchmark, "run", fake_run), \
                        patch.dict(os.environ, {"K": "3", "REPS": "3", "DATASETS": "small", "BLOCK_SIZES": "64", "RESULTS_ROOT": "results"}), \
                        contextlib.redirect_stdout(io.StringIO()):
                    if mismatch:
                        with self.assertRaisesRegex(ValueError, "Prediction mismatch"):
                            benchmark.main()
                    else:
                        benchmark.main()
                outputs = list((root / "results").glob("baseline-*"))
                self.assertEqual(len(outputs), 1)
                summary = outputs[0] / "summary.csv"
                if mismatch:
                    self.assertFalse(summary.exists())
                    with (outputs[0] / "raw.csv").open() as stream:
                        self.assertEqual(len(list(csv.DictReader(stream))), 3)
                else:
                    with summary.open() as stream:
                        rows = list(csv.DictReader(stream))
                    gpu = next(row for row in rows if row["strategy"] == "global")
                    self.assertEqual(float(gpu["serial_speedup"]), 4)
                    self.assertEqual(int(gpu["runs"]), 3)


if __name__ == "__main__":
    unittest.main()
