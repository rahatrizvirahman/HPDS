#!/usr/bin/env python3
"""Test CPU measurement collection with synthetic process output."""
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
spec = importlib.util.spec_from_file_location("benchmark_cpu", ROOT / "benchmark_cpu.py")
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


class CpuBenchmarkTests(unittest.TestCase):
    def sweep(self, mismatch=False):
        OUTPUT.mkdir(parents=True, exist_ok=True)
        folder = tempfile.mkdtemp(dir=OUTPUT)
        self.addCleanup(shutil.rmtree, folder, True)
        root = Path(folder).resolve()

        def fake_run(command, log):
            program = {"./serial": "serial", "./threaded": "pthreads", "./openmp": "openmp"}.get(command[0], "mpi")
            workers = 1 if program == "serial" else int(command[-2] if program != "mpi" else command[command.index("-n") + 1])
            Path(command[-1]).write_text("1\n" if mismatch and program == "openmp" and workers == 2 else "0\n")
            log.write_text("Synthetic fixture, no execution\n")
            time = 40.0 / workers if program != "serial" else 40.0
            return {"strategy": program, "end_to_end_ms": str(time), "accuracy": "100", "workers": str(workers)}

        with patch.object(benchmark, "ROOT", root), patch.object(benchmark.base, "run", fake_run), \
                patch.dict(os.environ, {"K": "3", "REPS": "3", "DATASETS": "small", "WORKERS": "1 2",
                                        "MPI_LAUNCH": "mpirun -n {ranks}", "RESULTS_ROOT": "results"}), \
                contextlib.redirect_stdout(io.StringIO()):
            benchmark.main()
        return next((root / "results").glob("cpu-benchmark-*"))

    def test_summary(self):
        directory = self.sweep()
        with (directory / "raw.csv").open() as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 21)
        with (directory / "summary.csv").open() as stream:
            rows = {(r["program"], r["workers"]): r for r in csv.DictReader(stream)}
        self.assertEqual(float(rows[("openmp", "2")]["speedup_vs_serial"]), 2)
        self.assertEqual(float(rows[("mpi", "2")]["efficiency_vs_1_worker"]), 1)

    def test_prediction_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Prediction mismatch"):
            self.sweep(mismatch=True)


if __name__ == "__main__":
    unittest.main()
