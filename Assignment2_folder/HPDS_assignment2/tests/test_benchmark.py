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
                    strategy = "shared" if "--strategy=shared" in command else "global" if gpu else "serial"
                    return {"strategy": strategy, "end_to_end_ms": "5" if gpu else "20",
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


    def test_shared_strategy_sweep(self):
        OUTPUT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=OUTPUT) as folder:
            root = Path(folder).resolve()
            commands = []

            def fake_run(command, log):
                commands.append(command)
                gpu = command[0] == "./cuda"
                Path(command[-1]).write_text("0\n")
                log.write_text("Synthetic fixture, no GPU execution\n")
                strategy = "shared" if "--strategy=shared" in command else "global" if gpu else "serial"
                return {"strategy": strategy, "end_to_end_ms": "4" if strategy == "shared" else "5" if gpu else "20",
                        "accuracy": "100", **({"kernel_ms": "1"} if gpu else {})}

            with patch.object(benchmark, "ROOT", root), patch.object(benchmark, "run", fake_run), \
                    patch.dict(os.environ, {"K": "3", "REPS": "3", "DATASETS": "small", "BLOCK_SIZES": "64 128",
                                            "STRATEGIES": "global shared", "TILE_ROWS": "128", "RESULTS_ROOT": "results"}), \
                    contextlib.redirect_stdout(io.StringIO()):
                benchmark.main()
            with (next((root / "results").glob("baseline-*")) / "summary.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 5)
            shared = [row for row in rows if row["strategy"] == "shared"]
            self.assertEqual([int(row["block_size"]) for row in shared], [64, 128])
            self.assertTrue(all(float(row["serial_speedup"]) == 5 and int(row["runs"]) == 3 for row in shared))
            shared_commands = [c for c in commands if "--strategy=shared" in c]
            self.assertEqual(len(shared_commands), 6)
            self.assertTrue(all("--tile=128" in c and not c[-1].startswith("--") for c in shared_commands))


if __name__ == "__main__":
    unittest.main()
