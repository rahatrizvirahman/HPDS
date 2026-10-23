#!/usr/bin/env python3
"""Synthetic MPI+CUDA checks. Run inside a Slurm allocation with enough GPUs for the chosen layouts.

LAYOUTS lists NODESxRANKS_PER_NODE values. MPI_LAUNCH is a template such as
"srun --mpi=pmix --nodes={nodes} --ntasks-per-node={per_node} --gpus-per-task=1".
"""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
LAUNCH = os.environ.get("MPI_LAUNCH", "mpirun -n {ranks}")
LAYOUTS = os.environ.get("LAYOUTS", "1x1 1x2 1x3 1x4").split()


def launcher(layout):
    nodes, per_node = (int(v) for v in layout.split("x"))
    return shlex.split(LAUNCH.format(nodes=nodes, per_node=per_node, ranks=nodes * per_node))


def main():
    output_root = ROOT / os.environ.get("RESULTS_ROOT", "results")
    output_root.mkdir(parents=True, exist_ok=True)
    run_id = os.environ.get("RUN_ID", "")
    folder = Path(tempfile.mkdtemp(prefix=f"mpi-correctness-{run_id + '.' if run_id else ''}", dir=output_root))
    header = "@relation ties\n@attribute feature numeric\n@attribute class numeric\n@data\n"
    train, test, output = folder / "train.arff", folder / "test.arff", folder / "predictions.txt"
    train.write_text(header + "-1,1\n1,0\n3,2\n5,0\n")
    test.write_text(header + "0,1\n2,0\n4,2\n")
    sequence = 0

    def run(command, env=None, timeout=120):
        nonlocal sequence
        sequence += 1
        if output.exists():
            output.unlink()
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                timeout=timeout, env={**os.environ, **(env or {})})
        (folder / f"case-{sequence}.command.txt").write_text(" ".join(command) + f"\nexit={result.returncode}\n")
        (folder / f"case-{sequence}.stdout.txt").write_text(result.stdout)
        (folder / f"case-{sequence}.stderr.txt").write_text(result.stderr)
        if output.exists():
            (folder / f"case-{sequence}.predictions.txt").write_bytes(output.read_bytes())
        return result

    strategies = [[], ["--strategy=shared", "--tile=1"], ["--strategy=shared", "--tile=3"], ["--strategy=shared", "--tile=256"]]
    cases = 0
    for layout in LAYOUTS:
        launch = launcher(layout)
        for flags in strategies:
            for block in ("1", "2", "128"):
                for k, expected in [(1, [1, 0, 2]), (2, [0, 0, 0]), (4, [0, 0, 0])]:
                    result = run(launch + [str(ROOT / "mpi_cuda"), str(train), str(test), str(k), block, str(output)] + flags)
                    if result.returncode != 0:
                        raise AssertionError(f"layout {layout} {flags} block {block} k={k} failed\n{result.stderr}")
                    actual = [int(v) for v in output.read_text().splitlines()]
                    if actual != expected:
                        raise AssertionError(f"layout {layout} {flags} block {block} k={k}, expected {expected}, got {actual}")
                    cases += 1
        for bad_k in ("0", "-1", "5", "abc", "3x"):
            result = run(launch + [str(ROOT / "mpi_cuda"), str(train), str(test), bad_k, "64"])
            if result.returncode == 0:
                raise AssertionError(f"layout {layout} accepted invalid k={bad_k}")
        for flags in (["--strategy=other"], ["--strategy=shared", "--tile=0"], ["--strategy=shared", "--tile=100000000"]):
            result = run(launch + [str(ROOT / "mpi_cuda"), str(train), str(test), "1", "64"] + flags)
            if result.returncode == 0:
                raise AssertionError(f"layout {layout} accepted invalid options {flags}")
        if layout != "1x1":
            # One failing rank must abort the whole job promptly. A timeout here means other ranks hung.
            result = run(launch + [str(ROOT / "mpi_cuda"), str(train), str(test), "1", "64"],
                         env={"MPI_CUDA_FAIL_RANK": "1"}, timeout=60)
            if result.returncode == 0:
                raise AssertionError(f"layout {layout} did not fail when rank 1 failed")
        print(f"PASS layout {layout}, uneven and empty rank ranges, tiles, blocks, invalid input and rank failure", flush=True)
    print(f"PASS {cases} MPI+CUDA synthetic predictions across layouts {' '.join(LAYOUTS)}; evidence in {folder}")


if __name__ == "__main__":
    main()
