#!/usr/bin/env python3
"""Test MPI+CUDA measurement collection with synthetic process output, without a GPU."""
import contextlib
import csv
import importlib.util
import io
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / os.environ.get("RESULTS_ROOT", "results")
spec = importlib.util.spec_from_file_location("benchmark_mpi", ROOT / "benchmark_mpi_cuda.py")
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


class MpiBenchmarkTests(unittest.TestCase):
    def run_sweep(self, mismatch=False, wrong_ranks=False):
        OUTPUT.mkdir(parents=True, exist_ok=True)
        folder = tempfile.mkdtemp(dir=OUTPUT)
        self.addCleanup(shutil.rmtree, folder, True)
        root = Path(folder).resolve()
        commands = []

        def fake_run(command, log):
            commands.append(command)
            Path(command[-1]).write_text("1\n" if mismatch and "./mpi_cuda" in command and "2" in command else "0\n")
            log.write_text("Synthetic fixture, no GPU execution\n")
            if command[0] == "./serial":
                return {"strategy": "serial", "end_to_end_ms": "40", "accuracy": "100"}
            if "./mpi_cuda" in command:
                ranks = int(command[command.index("-n") + 1])
                return {"strategy": "shared", "end_to_end_ms": str(20 / ranks), "kernel_ms": "1", "accuracy": "100",
                        "mpi_ranks": str(ranks + 1 if wrong_ranks else ranks), "nodes": "1"}
            return {"strategy": "shared", "end_to_end_ms": "20", "kernel_ms": "1", "accuracy": "100"}

        with patch.object(benchmark, "ROOT", root), patch.object(benchmark.base, "run", fake_run), \
                patch.dict(os.environ, {"K": "3", "REPS": "3", "DATASETS": "small", "LAYOUTS": "1x1 1x2",
                                        "MPI_LAUNCH": "mpirun -n {ranks}", "RESULTS_ROOT": "results"}), \
                contextlib.redirect_stdout(io.StringIO()):
            try:
                benchmark.main()
            finally:
                outputs = list((root / "results").glob("mpi-benchmark-*"))
        return root, outputs, commands

    def test_scaling_summary(self):
        root, outputs, commands = self.run_sweep()
        self.assertEqual(len(outputs), 1)
        with (outputs[0] / "raw.csv").open() as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 12)
        with (outputs[0] / "summary.csv").open() as stream:
            rows = {r["layout"]: r for r in csv.DictReader(stream)}
        self.assertEqual(float(rows["1x2"]["speedup_vs_1_mpi_gpu"]), 2)
        self.assertEqual(float(rows["1x2"]["efficiency_vs_1_mpi_gpu"]), 1)
        self.assertEqual(float(rows["1x2"]["serial_speedup"]), 4)
        self.assertEqual(sum("./mpi_cuda" in c for c in commands), 6)

    def test_prediction_mismatch_and_wrong_rank_count_are_rejected(self):
        for kwargs, message in ((dict(mismatch=True), "Prediction mismatch"), (dict(wrong_ranks=True), "ran with")):
            with self.subTest(**kwargs):
                with self.assertRaisesRegex(ValueError, message):
                    self.run_sweep(**kwargs)


if __name__ == "__main__":
    unittest.main()
