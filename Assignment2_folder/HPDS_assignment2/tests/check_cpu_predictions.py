#!/usr/bin/env python3
"""Synthetic checks for the pthreads, OpenMP and MPI CPU programs.

Worker counts above the number of test rows leave some workers idle, which exercises empty ranges.
MPI_LAUNCH is a template such as "srun --mpi=pmix --ntasks={ranks}". Set MPI_LAUNCH empty to skip the MPI program.
"""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
WORKERS = os.environ.get("WORKERS", "1 2 3 4 8").split()
LAUNCH = os.environ.get("MPI_LAUNCH", "mpirun -n {ranks}")


def main():
    output_root = ROOT / os.environ.get("RESULTS_ROOT", "results")
    output_root.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix="cpu-correctness-", dir=output_root))
    header = "@relation ties\n@attribute feature numeric\n@attribute class numeric\n@data\n"
    train, test, output = folder / "train.arff", folder / "test.arff", folder / "predictions.txt"
    train.write_text(header + "-1,1\n1,0\n3,2\n5,0\n")
    test.write_text(header + "0,1\n2,0\n4,2\n")
    sequence = 0

    def run(command):
        nonlocal sequence
        sequence += 1
        if output.exists():
            output.unlink()
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=120)
        (folder / f"case-{sequence}.command.txt").write_text(" ".join(command) + f"\nexit={result.returncode}\n")
        (folder / f"case-{sequence}.stdout.txt").write_text(result.stdout)
        (folder / f"case-{sequence}.stderr.txt").write_text(result.stderr)
        if output.exists():
            (folder / f"case-{sequence}.predictions.txt").write_bytes(output.read_bytes())
        return result

    programs = [("threaded", lambda n: [str(ROOT / "threaded")], True), ("openmp", lambda n: [str(ROOT / "openmp")], True)]
    if LAUNCH.strip():
        programs.append(("mpi", lambda n: shlex.split(LAUNCH.format(ranks=n)) + [str(ROOT / "mpi")], False))
    cases = 0
    for name, prefix, takes_workers in programs:
        for workers in WORKERS:
            command = prefix(workers)
            tail = [workers] if takes_workers else []
            for k, expected in [(1, [1, 0, 2]), (2, [0, 0, 0]), (4, [0, 0, 0])]:
                args = command + [str(train), str(test), str(k)] + tail + [str(output)]
                result = run(args)
                if result.returncode != 0:
                    raise AssertionError(f"{name} workers={workers} k={k} failed\n{result.stderr}")
                actual = [int(v) for v in output.read_text().splitlines()]
                if actual != expected:
                    raise AssertionError(f"{name} workers={workers} k={k}, expected {expected}, got {actual}")
                cases += 1
            for bad_k in ("0", "-1", "5", "abc", "3x"):
                if run(command + [str(train), str(test), bad_k] + tail).returncode == 0:
                    raise AssertionError(f"{name} workers={workers} accepted invalid k={bad_k}")
        if takes_workers:
            for bad_workers in ("0", "-2", "x"):
                if run(command + [str(train), str(test), "1", bad_workers]).returncode == 0:
                    raise AssertionError(f"{name} accepted invalid worker count {bad_workers}")
        print(f"PASS {name} workers {' '.join(WORKERS)}, ties, k values, invalid input", flush=True)
    print(f"PASS {cases} CPU parallel synthetic predictions; evidence in {folder}")


if __name__ == "__main__":
    main()
