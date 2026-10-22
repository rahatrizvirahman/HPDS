#!/usr/bin/env python3
"""Exercise neighbor ties, vote ties, runtime k, and partial query blocks."""
import argparse
import contextlib
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", action="store_true", help="Also exercise the actual CUDA executable")
    args = parser.parse_args()
    output = ROOT / os.environ.get("RESULTS_ROOT", "results")
    output.mkdir(parents=True, exist_ok=True)
    run_id = os.environ.get("RUN_ID", "")
    with contextlib.nullcontext(tempfile.mkdtemp(prefix=f"correctness-{run_id + '.' if run_id else ''}", dir=output)) as scratch:
        folder = Path(scratch)
        header = "@relation ties\n@attribute feature numeric\n@attribute class numeric\n@data\n"
        train, test, output = (folder / name for name in ("train.arff", "test.arff", "predictions.txt"))
        train.write_text(header + "-1,1\n1,0\n3,2\n5,0\n")
        test.write_text(header + "0,1\n2,0\n4,2\n")
        sequence = 0

        def run(command, **kwargs):
            nonlocal sequence
            sequence += 1
            check = kwargs.pop("check", False)
            kwargs.pop("stdout", None)
            kwargs.pop("stderr", None)
            if output.exists():
                output.unlink()
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kwargs)
            for name, value in (("stdout", result.stdout), ("stderr", result.stderr)):
                path = folder / f"case-{sequence}.{name}.txt"
                path.write_bytes(value.encode() if isinstance(value, str) else value)
            (folder / f"case-{sequence}.command.txt").write_text(repr(command) + f"\nexit={result.returncode}\n")
            if output.exists():
                (folder / f"case-{sequence}.predictions.txt").write_bytes(output.read_bytes())
            if check:
                result.check_returncode()
            return result

        configurations = [("serial", [])]
        if args.cuda:
            configurations += [("cuda", [str(block)]) for block in (1, 64, 128, 256)]
        for binary, extra in configurations:
            for k, expected in [(1, [1, 0, 2]), (2, [0, 0, 0]), (4, [0, 0, 0])]:
                run([str(ROOT / binary), str(train), str(test), str(k)] + extra + [str(output)],
                               check=True, stdout=subprocess.PIPE, text=True)
                actual = [int(v) for v in output.read_text().splitlines()]
                if actual != expected:
                    raise AssertionError(f"{binary} {extra} k={k}, expected {expected}, got {actual}")
            for bad_k in ("0", "-1", "5", "abc", "3x"):
                result = run([str(ROOT / binary), str(train), str(test), bad_k] + extra,
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if result.returncode == 0:
                    raise AssertionError(f"{binary} accepted invalid k={bad_k}")
        # A test set may omit classes present in training.
        test.write_text(header + "4,0\n")
        run([str(ROOT / "serial"), str(train), str(test), "1", str(output)],
                       check=True, stdout=subprocess.PIPE)
        if output.read_text() != "2\n":
            raise AssertionError("Prediction must remain valid when test labels omit a training class")
    print("PASS neighbor ties, vote ties, valid and invalid k, and prediction output")
    if args.cuda:
        print("PASS CUDA predictions with partial blocks")


if __name__ == "__main__":
    main()
